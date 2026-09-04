"""The core batch stream: a framework-neutral iterable of numpy ``Batch`` objects.

:class:`InSituDataset` ties the pieces together and yields assembled numpy batches.
It inherits nothing framework-specific -- parallelism lives in :class:`Scheduler`'s
event loop, not a worker pool. Framework handoff (torch / JAX / TF) is a thin,
optional DLPack adapter layer in :mod:`insitubatch.frameworks`; the core never
imports a framework, so ``import insitubatch`` works on a box without any installed.
For PyTorch::

    from insitubatch.frameworks import as_torch
    loader = DataLoader(as_torch(InSituDataset(...)), batch_size=None, num_workers=0)

``batch_size=None`` because the dataset already yields assembled batches;
``num_workers=0`` because forking workers would re-introduce exactly the
redundant-read / nested-parallelism problems we set out to avoid. JAX iterates the
dataset directly (``frameworks.to_jax`` per batch); TF wraps it
(``frameworks.as_tf_dataset``).

The engine is the fetch scheduler: one event loop streams stored-chunk reads under
a single ``max_inflight`` budget and delivers decoded tiles into a
:class:`ChunkPool`, which holds them by reference. This producer walks the shuffle
order, waits on each block's chunks, gathers coalesced batches, and unpins the block
(making it
LRU-evictable / retainable for reuse). Read concurrency (``max_inflight``) and the
residency budget are independent dials. See [docs/architecture.md] for the pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TextIO

import numpy as np
from zarr.abc.store import Store

from .buffers import HostAllocator
from .pool import ChunkPool, output_geometry
from .runtime import Depths, PassStats, StatsCollector, format_pass
from .scheduler import Scheduler, SchedulerConfig
from .shuffle import block_shuffled_order, sequential_order
from .split import SplitManifest, valid_anchor_range
from .store import close_store, open_geometries
from .summary import DatasetReport, describe, print_summary, working_set_bytes
from .types import ArrayGeometry, Batch, DecodedChunk, SplitName, StoredChunkRead

logger = logging.getLogger(__name__)

# Default for the single concurrency dial when the caller does not pin it.
# max_inflight is independent of the shuffle window -- sized to saturate the
# network, not the buffer.
DEFAULT_MAX_INFLIGHT = 32


def _partition_blocks(order: np.ndarray, block_chunks: int) -> list[tuple[int, int, np.ndarray]]:
    """Split a draw ``order`` into shuffle-blocks: ``(row_start, row_stop, chunk_ids)``.

    ``block_shuffled_order`` shuffles samples *within* a block of ``block_chunks``
    chunks and concatenates blocks in chunk-permutation order, so every block is a
    contiguous row range over disjoint chunks. We recover the blocks from the chunks'
    first-appearance order (vectorized: O(chunks) of Python, not O(samples)), which
    is robust to short final chunks where fixed-stride slicing would misalign.
    """
    if not len(order):
        return []
    cids = order[:, 0].astype(np.int64)
    _, first_pos = np.unique(cids, return_index=True)
    appearance = cids[np.sort(first_pos)]  # chunk ids in order of first appearance
    block_of = np.full(int(cids.max()) + 1, -1, dtype=np.int64)
    block_of[appearance] = np.arange(len(appearance)) // block_chunks
    block_per_row = block_of[cids]
    starts = [0, *(np.flatnonzero(np.diff(block_per_row) != 0) + 1).tolist(), len(order)]
    return [
        (starts[k], starts[k + 1], appearance[k * block_chunks : (k + 1) * block_chunks])
        for k in range(len(starts) - 1)
    ]


class InSituDataset:
    """A framework-neutral source of shuffled numpy batches from Zarr, split-aware.

    The dataset is *not* itself iterated -- you iterate one of its split views:
    :attr:`train` (shuffled), :attr:`val`, :attr:`test`, :attr:`all` (deterministic).
    All four share **one** :class:`ChunkPool`, so a chunk that two splits both read --
    e.g. a windowed read spilling across a split boundary -- is decoded once::

        ds = InSituDataset(store, manifest, geometries=geoms, batch_size=32)
        for batch in ds.train: ...   # one epoch; ds.set_epoch(e) reshuffles
        for batch in ds.val: ...

    One epoch over a view = permute the split's chunks -> walk shuffle-blocks, stream-fetching
    each block's stored chunks into the pool -> gather coalesced batches cut over the whole
    epoch (so only the last is short, and one may span a block boundary) -> evict.
    Batches are numpy :class:`Batch`; convert to a framework with
    :mod:`insitubatch.frameworks` (``as_torch`` / ``to_jax`` / ``as_tf_dataset``). A
    different per-split configuration (e.g. train-only augmentation) is a separate dataset.

    Two preprocessing hooks, placed by cost (full model in the docs, "Transforms"):

    - ``chunk_transforms`` -- ``(DecodedChunk) -> DecodedChunk``, run per chunk *before*
      shuffle, seeing **one variable**. The cacheable home for elementwise, per-variable,
      deterministic work (scaling, unit conversion, dtype cast); amortized over every sample
      in the chunk and reused across epochs.
    - ``batch_transforms`` -- ``(Batch) -> Batch``, run per assembled batch, seeing **all
      variables** aligned on the sample axis. For cross-variable derived fields and
      per-sample random augmentation; runs after the cache, so it is **never cached**.

    Runnable side-by-side example: ``examples/transforms.py``.

    **One writer per ``cache_dir``.** The cache directory (and ``persist=True`` on top of
    it) is arbitrated across processes by an advisory lock held for the dataset's
    lifetime: one writer at a time, and a second fails fast rather than silently
    corrupting the first's chunks. Several jobs may read one warm cache at once with
    ``readonly_cache=True``, which takes the lock shared, never writes, and **raises on a
    miss** -- it asserts the cache is complete for what this run reads, rather than
    quietly falling back to fetching. Put ``cache_dir`` on local NVMe: over a network
    filesystem the lock may be emulated per client and we can only warn. See
    ``docs/tuning.md``, "Sharing a cache_dir between processes".
    """

    def __init__(
        self,
        store: Store,
        manifest: SplitManifest,
        geometries: dict[str, ArrayGeometry] | None = None,
        *,
        batch_size: int = 32,
        block_chunks: int = 16,
        max_inflight: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
        prefetch_depth: int = 2,
        cache_dir: str | None = None,
        cache_budget_bytes: int | None = None,
        persist: bool = False,
        readonly_cache: bool = False,
        reset_stale_cache: bool = False,
        on_bad_chunk: str = "raise",
        chunk_transforms: Sequence[Callable[[DecodedChunk], DecodedChunk]] = (),
        batch_transforms: Sequence[Callable[[Batch], Batch]] = (),
    ) -> None:
        self.store = store
        self.geometries = geometries if geometries is not None else open_geometries(store)
        self.manifest = manifest
        self.variables = list(self.geometries)

        # Variables must share the sample-axis *length* (samples are paired row-for-row);
        # they may chunk that axis differently (raw Z-chunk 1 + label mask Z-chunk 30). The
        # manifest defines the reference anchor grid; each variable maps anchors onto its own
        # chunks. n_samples must match the manifest so the anchor grid indexes every variable.
        lengths = {g.n_samples for g in self.geometries.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"All variables must share the same sample-axis length; got "
                f"n_samples={sorted(lengths)}."
            )
        if lengths and manifest.n_samples != next(iter(lengths)):
            raise ValueError(
                f"manifest sample-axis length {manifest.n_samples} does not match the "
                f"variables' n_samples {next(iter(lengths))}."
            )
        self._ref_spc = manifest.sample_chunk_size  # the anchor grid (shuffle/split/gather)

        self.batch_size = batch_size
        # A batch is cut over the whole epoch order, so it draws from every shuffle-block
        # it spans -- and a block is released only once a batch has consumed its last row.
        # The residency floor below covers two blocks (the current one plus a read-ahead),
        # which is the invariant the scheduler's release bookkeeping is written to, so a
        # batch has to fit inside one block. Widen the block rather than reject the
        # configuration: block_chunks is a shuffle-span knob, and a block too narrow to
        # hold a single batch is not a meaningful span. Finely chunked archives are where
        # the default bites -- at one sample per chunk (SDSS spPlate fibers, Hubble frames)
        # 16 chunks is a 16-sample block, and any larger batch would park admission
        # forever. Capped at the array's chunk count: no batch can span more blocks than
        # there are chunks.
        min_block_chunks = min(-(-batch_size // self._ref_spc), manifest.n_chunks)
        self._block_chunks_requested = block_chunks  # reported by describe()
        self.block_chunks = max(block_chunks, min_block_chunks)
        if self.block_chunks != block_chunks:
            logger.info(
                "block_chunks widened %d -> %d so a %d-sample batch fits one shuffle-block "
                "(%d sample(s) per chunk)",
                block_chunks,
                self.block_chunks,
                batch_size,
                self._ref_spc,
            )
        self.seed = seed
        self.shuffle = shuffle
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.chunk_transforms = tuple(chunk_transforms)
        self.batch_transforms = tuple(batch_transforms)
        # Geometry each variable presents *after* the chunk_transform pipeline (regrid / dtype
        # recast). What the consumer and the framework adapters see, and what sizes the cache.
        self._out_geometries = {
            label: output_geometry(g, self.chunk_transforms) for label, g in self.geometries.items()
        }
        self._epoch = 0
        self._persist = persist
        self._readonly_cache = readonly_cache
        self._cache_dir = cache_dir
        self.resident_peak = 0  # peak resident outer chunks (observability)
        self.cache_hits = 0  # chunks served without a fetch this epoch (cross-epoch/run)
        self.cache_misses = 0  # chunks fetched + decoded this epoch
        # Stored tiles NaN-filled in the last epoch (when on_bad_chunk="nan") -- which
        # (array, chunk_index, inner_coord) reads were corrupt/truncated. len() is the
        # count. Inspect after iterating to log/quarantine bad chunks.
        self.bad_chunks: list[StoredChunkRead] = []
        self.inflight_peak = 0  # peak in-flight tiles (observability); see `last_pass`
        # The last completed pass, or None before one finishes. Per pass, not per epoch: a
        # training epoch iterates train then val over one pool, and averaging two passes
        # with different shapes into one report is how a number stops meaning anything.
        self.last_pass: PassStats | None = None
        self._on_bad_chunk = on_bad_chunk

        # The pool is the assembly buffer AND the cache, owned here so it persists
        # across epochs. The byte budget is the single residency knob: the floor is
        # the working set -- the current block plus one read-ahead block, all
        # variables, must be co-resident (a batch draws across a whole block) -- and
        # a larger budget (cache_budget_bytes) retains drained chunks for cross-epoch
        # decode-once reuse. cache_dir spills slots to NVMe (mmap) instead of heap.
        # Two blocks is a floor only because a batch fits inside one -- which is what
        # `self.block_chunks` (widened above) guarantees, and what the scheduler's
        # release bookkeeping assumes.
        #
        # One formula, shared with `describe()`: see summary.working_set_bytes for how
        # windows and shuffle widen that floor, and why it charges slot_charge_bytes rather
        # than the assembled shape. A report that predicted a different number from the one
        # the engine uses would be worse than no report.
        working_set = working_set_bytes(
            [self.geometries[label] for label in self.variables],
            [self._out_geometries[label] for label in self.variables],
            manifest,
            block_chunks=self.block_chunks,
            ref_spc=self._ref_spc,
            shuffle=self.shuffle,
            assembles=bool(self.chunk_transforms),
        )
        # Sized for ONE iteration, deliberately. Every active iteration shares this pool and
        # holds its own references, so N concurrent iterations need ~N x this -- but the engine
        # cannot know N, and guessing high would cost memory in the single-iteration case that
        # is almost every case. Running several is an explicit choice, so sizing for it is the
        # caller's: pass `cache_budget_bytes`. Under-sizing is not silent -- admission raises
        # and names how many iterations are sharing the pool (`Scheduler._starvation`).
        # See docs/tuning.md, "Several iterations at once multiply the budget".
        self.cache_budget_bytes = max(int(cache_budget_bytes or 0), working_set)
        # persist turns the cache_dir mmap tier into a cross-run cache (files + manifest
        # survive close; reopen revives them as hits). It needs a dir to keep files in;
        # the dir path is the dataset+pipeline identity (bury a version in it).
        if persist and cache_dir is None:
            raise ValueError("persist=True requires cache_dir to keep the cache files in")
        # One writer per cache_dir, arbitrated by an advisory lock the pool holds for its
        # lifetime; `readonly_cache=True` opens it shared, serves only what is already
        # cached, and raises on a miss. See ChunkPool._lock and docs/tuning.md.
        self._pool = ChunkPool(
            self.geometries,
            chunk_transforms=self.chunk_transforms,
            backing_dir=cache_dir,
            budget_bytes=self.cache_budget_bytes,
            persist=persist,
            readonly_cache=readonly_cache,
            reset_stale_cache=reset_stale_cache,
        )

        # One concurrency dial (max_inflight, network), independent of the shuffle
        # window (block_chunks) and the cache budget. Exposed so the probe can tune
        # decode_threads at iteration time.
        self.scheduler_config = SchedulerConfig(
            max_inflight=max_inflight or DEFAULT_MAX_INFLIGHT, on_bad_chunk=on_bad_chunk
        )

    def set_epoch(self, epoch: int) -> None:
        """Call from the training loop so each epoch reshuffles deterministically."""
        self._epoch = epoch

    def describe(self, *, iterations: int = 1) -> DatasetReport:
        """What this dataset will do, from geometry and configuration -- no store access.

        ``iterations`` is how many passes will share the pool at once (``zip(ds.train,
        ds.val)`` is two), since each holds its own chunk references and the automatic
        budget covers one. See :meth:`print_summary` for the formatted view, and
        :mod:`insitubatch.summary` for what each number means.
        """
        return describe(self, iterations=iterations)

    def print_summary(self, *, iterations: int = 1, file: TextIO | None = None) -> None:
        """Print :meth:`describe` for a human. On demand -- construction stays quiet."""
        print_summary(self.describe(iterations=iterations), file=file)

    # -- the splits, as iterables (one dataset, one shared pool) -------------

    @property
    def train(self) -> _SplitView:
        """Iterable over the train split, shuffled per the dataset's ``shuffle`` flag."""
        return _SplitView(self, SplitName.TRAIN, self.shuffle)

    @property
    def val(self) -> _SplitView:
        """Iterable over the val split, in deterministic (sequential) order."""
        return _SplitView(self, SplitName.VAL, False)

    @property
    def test(self) -> _SplitView:
        """Iterable over the test split, in deterministic (sequential) order."""
        return _SplitView(self, SplitName.TEST, False)

    @property
    def all(self) -> _SplitView:
        """Iterable over every split's chunks (deterministic) -- e.g. full-archive inference."""
        return _SplitView(self, None, False)

    _SENTINEL = object()

    def _chunk_ids(self, split: SplitName | None) -> np.ndarray:
        """Sample-axis chunk indices for a split (``None`` = every split's chunks)."""
        if split is None:
            ids = sorted(set().union(*(set(self.manifest.chunks[s.value]) for s in SplitName)))
        else:
            ids = self.manifest.chunks[split.value]
        return np.asarray(ids, dtype=np.int64)

    def _draw_order(self, split: SplitName | None, shuffle: bool) -> np.ndarray:
        chunk_ids = self._chunk_ids(split)
        spc = self._ref_spc  # the manifest's anchor grid, shared by every variable
        n_samples = self.manifest.n_samples
        if shuffle:
            order = block_shuffled_order(
                chunk_ids,
                spc,
                n_samples,
                block_chunks=self.block_chunks,
                seed=self.seed,
                epoch=self._epoch,
            )
        else:
            order = sequential_order(chunk_ids, spc, n_samples)
        return self._drop_edge_anchors(order, spc, n_samples)

    def _drop_edge_anchors(self, order: np.ndarray, spc: int, n_samples: int) -> np.ndarray:
        """Keep only anchors whose every windowed read ``anchor + offset`` is on the
        array. Offset 0 (no window) keeps the whole order. Anchors are dropped, not
        their chunks, so an edge chunk still contributes its interior anchors."""
        offsets = [g.offset for g in self.geometries.values()]
        lo, hi = valid_anchor_range(offsets, n_samples)
        if lo == 0 and hi == n_samples:
            return order
        anchor = order[:, 0] * spc + order[:, 1]
        return order[(anchor >= lo) & (anchor < hi)]

    def _block_read_keys(self, block_rows: np.ndarray, spc: int) -> set[tuple[str, int]]:
        """The ``(path, chunk)`` slots a block's anchors read across all variables --
        the residency set to pin while draining it. A windowed variable reads
        ``anchor + offset``, so its read chunks may spill into neighbouring blocks'
        chunks; the refcounted pins let those shared chunks be held by both blocks."""
        anchor = block_rows[:, 0].astype(np.int64) * spc + block_rows[:, 1].astype(np.int64)
        keys: set[tuple[str, int]] = set()
        for geom in self.geometries.values():
            read_cid = (anchor + geom.offset) // geom.sample_chunk_size  # variable's own grid
            keys.update((geom.path, int(c)) for c in np.unique(read_cid))
        return keys

    def _iterate(self, split: SplitName | None, shuffle: bool) -> Iterator[Batch]:
        """Drain assembled batches for one split from a background producer (prefetch).

        Called by the ``.train`` / ``.val`` / ``.test`` / ``.all`` views, all sharing the
        one :class:`ChunkPool` -- so a chunk a windowed read pulls across a split boundary
        is decoded once and reused by both splits.

        A producer thread starts the scheduler over the split's chunks (in draw order), then
        walks the epoch's batches: waiting each shuffle-block a batch draws from, gathering,
        and unpinning a block once a batch has consumed its last row. Batches are cut over the
        whole epoch order, so only the last one is short -- a batch may span a block boundary,
        and the two frontiers in :func:`produce` are what makes that safe. This consumer pops
        from a bounded queue (depth ``prefetch_depth``) that provides backpressure and
        inter-batch overlap. The scheduler keeps ``max_inflight`` tiles continuously in flight
        and fetches one block ahead, so block-boundary IO overlaps the per-batch compute.
        Chunks the pool already holds (cross-epoch or cross-split hits) cost no fetch.
        """
        spc = self._ref_spc  # the manifest anchor grid, shared by every variable
        order = self._draw_order(split, shuffle)
        blocks = _partition_blocks(order, self.block_chunks)
        ordered_chunks = [int(c) for _rstart, _rstop, cids in blocks for c in cids]

        # One owner per pass, minted here and threaded through this iteration's scheduler
        # (admission pins) and its producer (block pins). Scoping references this way is
        # what lets two iterations share a pool: the prologue below releases only *our*
        # leftovers, and `wait_ready` is satisfied only by *our* reference.
        owner = self._pool.new_owner()

        # Per-pass observability starts clean. Releasing this pass's references is NOT
        # done here: it happens in our own teardown below, so an abandoned pass cleans up
        # after itself instead of relying on the next pass's prologue -- which, now that
        # references are owner-scoped, would be a different owner and could not.
        self._pool.reset_epoch_counters()

        out_q: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        stop = threading.Event()
        # One collector for the pass, shared with the Scheduler. It takes no lock: every
        # field has a single writing thread (see StatsCollector), and this pass reads it
        # only after `producer.join()` and the scheduler's `close()` have published them.
        stats = StatsCollector(queue_capacity=self.prefetch_depth)
        wall0 = time.perf_counter()
        batches = 0
        depths = Depths(
            batch_queue_capacity=self.prefetch_depth,
            max_inflight=self.scheduler_config.max_inflight,
        )

        # Per-block read-union keys (path, chunk) -- what each block reads across all
        # variables' offsets. A windowed read can spill into chunks owned by any other
        # block (shuffle permutes chunk order), and the driver fetches each chunk only
        # once, so a chunk must stay resident from admit until its *last* referencing
        # block drains. Release each chunk exactly there (last_use).
        block_keys = [self._block_read_keys(order[rs:re], spc) for rs, re, _ in blocks]
        release: list[set[tuple[str, int]]] = [set() for _ in blocks]
        last_use: dict[tuple[str, int], int] = {}
        for bi, keys in enumerate(block_keys):
            for key in keys:
                last_use[key] = bi  # later block overwrites -> ends on the max index
        for key, bi in last_use.items():
            release[bi].add(key)

        def produce(sched: Scheduler) -> None:
            # Batches are cut over the WHOLE epoch order, not per block. Cutting them per
            # block (`range(rstart, rstop, bs)`) made every block whose row count was not a
            # multiple of batch_size end in a short batch -- one per block rather than one
            # per epoch, and up to *three* distinct shapes once the epoch's own short final
            # block is counted. Sample-once held either way, so nothing failed; what broke
            # was any fixed-shape consumer (torch.compile / jax.jit retrace per shape, and
            # with three shapes the retrace cache never settles).
            #
            # Blocks stay exactly what they were -- contiguous row ranges over disjoint
            # chunks -- and `order` was always one flat array for the epoch, so a batch that
            # crosses a boundary is just `order[start : start + bs]` not being truncated.
            # What the boundary needs is bookkeeping, tracked as two monotone frontiers over
            # the block list: `ready` (waited) and `freed` (released). Both only advance, and
            # each block passes through each exactly once, so this stays O(blocks) of Python.
            bs = self.batch_size
            ready = freed = 0
            try:
                sched.start(ordered_chunks, spc)
                for start in range(0, len(order), bs):
                    if stop.is_set():
                        return
                    stop_row = min(start + bs, len(order))
                    # Wait every block this batch draws from. A batch spans at most two
                    # (bs <= a block's rows in any sane configuration, and the loop is
                    # correct regardless): a block's batches draw across its whole
                    # read-union, so it must be assembled -- and claimed by the driver, see
                    # ChunkPool.wait_ready -- before gathering. Each wait is cheap once ready.
                    while ready < len(blocks) and blocks[ready][0] < stop_row:
                        for path, cid in block_keys[ready]:
                            sched.pool.wait_ready(path, cid, owner)
                        ready += 1
                    g0 = time.thread_time()
                    batch = sched.pool.gather(order[start:stop_row], self.variables, spc)
                    # thread_time, not perf_counter: gather is work this thread does, and a
                    # wall clock here would bill it for whatever else held the GIL.
                    stats.gather_s += time.thread_time() - g0
                    # Release the driver's reference on chunks whose *last* use is a block now
                    # fully behind the frontier -- a batch has consumed its last row, so it is
                    # done. Now LRU-evictable (retained for reuse if budget allows), unblocking
                    # the read-ahead. Chunks a later block reads again keep their reference.
                    #
                    # Here, not after the queue put: `out_q.put` blocks when full, so releasing
                    # after it keeps a whole block pinned for as long as the consumer is slow,
                    # and pinned chunks cannot be evicted -- consumer slowness would propagate
                    # into a read-ahead stall via `try_admit` parking on a full budget. Safe
                    # this early because `gather` COPIES out of the slots, so the batch never
                    # aliases chunk memory and the reference is already dead weight.
                    while freed < len(blocks) and blocks[freed][1] <= stop_row:
                        sched.unpin_block(release[freed])
                        freed += 1
                    b0 = time.thread_time()
                    for transform in self.batch_transforms:
                        batch = transform(batch)
                    stats.batch_transform_s += time.thread_time() - b0
                    out_q.put(batch)  # blocks when full -> backpressure
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                out_q.put(exc)
            finally:
                out_q.put(self._SENTINEL)

        try:
            with Scheduler(
                self.store,
                self.geometries,
                self._pool,
                self.scheduler_config,
                owner=owner,
                stats=stats,
            ) as sched:
                producer = threading.Thread(
                    target=produce, args=(sched,), name="insitu-prefetch", daemon=True
                )
                producer.start()
                try:
                    while True:
                        # Sample before the get: qsize() after it has already been decremented
                        # by our own take, which would report every healthy queue as one short.
                        stats.sample_queue(out_q.qsize())
                        item = out_q.get()
                        if item is self._SENTINEL:
                            break
                        if isinstance(item, Exception):
                            raise item
                        batches += 1
                        # perf_counter around the yield: the caller's loop body is wall
                        # time to us, whatever it spends it on (a GPU step is mostly not
                        # CPU). Consumer thread only, and no await between load and store.
                        t_yield = time.perf_counter()
                        yield item
                        stats.consumer_s += time.perf_counter() - t_yield
                finally:
                    # Signal stop, then drain so a producer parked on a full queue can
                    # proceed and exit before the scheduler (context manager) is closed.
                    stop.set()
                    while producer.is_alive():
                        with contextlib.suppress(queue.Empty):
                            out_q.get(timeout=0.05)
                    producer.join(timeout=10)  # publishes the producer's stage timers
                    pool = sched.pool
                    self.resident_peak = pool.max_resident  # peak residency this epoch
                    self.cache_hits = pool.hits
                    self.cache_misses = pool.misses
                    self.bad_chunks = list(sched.bad_chunks)  # tiles NaN-filled this epoch
                    # Was unreachable before: it lives on the Scheduler, which is created inside
                    # this method and torn down with the pass, so unlike the three above nothing
                    # ever copied it out.
                    self.inflight_peak = sched.inflight_peak
                    depths = Depths(
                        batch_queue_capacity=stats.queue_capacity,
                        batch_queue_peak=stats.queue_peak,
                        batch_queue_samples=stats.queue_samples,
                        batch_queue_empty=stats.queue_empty,
                        batch_queue_full_enough=stats.queue_full_enough,
                        inflight_peak=sched.inflight_peak,
                        max_inflight=self.scheduler_config.max_inflight,
                        decode_queue_peak=stats.decode_queue_peak,
                        decode_threads=sched.decode_threads,
                        resident_peak=pool.max_resident,
                        resident_peak_bytes=pool.max_resident_bytes,
                        budget_bytes=pool.budget_bytes,
                    )
                    self._log_epoch_summary(pool, split)
                    # Persistence was asked for but served nothing, and the cache *was*
                    # consulted (entries existed and every revive failed) -> almost certainly
                    # a stale cache_dir or changed data/transforms. Loud once per epoch; a
                    # plain miss (no persisted entry for a chunk) is silent (normal).
                    failed_revives = pool.revive_mismatch + pool.revive_missing
                    if self._persist and pool.hits == 0 and failed_revives:
                        logger.warning(
                            "persist=True but 0 of %d persisted chunks were served this epoch "
                            "(%d shape/dtype mismatches, %d missing/unreadable) -- stale cache_dir "
                            "or changed data/transforms?",
                            pool.manifest_entries,
                            pool.revive_mismatch,
                            pool.revive_missing,
                        )
                    # Last: drop this pass's references (and any partial its cancelled
                    # fetches abandoned), after every counter above has been read. An
                    # early `break` finalizes this generator, so this runs then too --
                    # which is what keeps an abandoned pass from leaking budget.
                    self._pool.release_owner(owner)
        finally:
            # A `finally` around the `with`, not a statement after it: an early `break`
            # throws GeneratorExit at the yield, which unwinds through the `with` (so the
            # scheduler still closes) but would skip anything that merely followed it. The
            # abandoned pass is the one most worth a report -- it is usually abandoned
            # *because* it was slow.
            #
            # Here rather than in the inner teardown because only the scheduler's close()
            # publishes the counters the event loop owns; the producer's are already
            # published by its join(). Reading them inside would race the tiles still
            # finishing.
            self.last_pass = PassStats(
                split="all" if split is None else split.value,
                epoch=self._epoch,
                batches=batches,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                bad_chunks=len(self.bad_chunks),
                wall_s=time.perf_counter() - wall0,
                times=stats.times(),
                depths=depths,
            )
            if logger.isEnabledFor(logging.INFO):
                logger.info("%s", format_pass(self.last_pass))

    def _log_epoch_summary(self, pool: ChunkPool, split: SplitName | None) -> None:
        """One INFO line per epoch *per split*: what the chunk cache and the batch buffers did.

        Tagged with the split because a training run iterates more than one of them per epoch
        (train, then val) over the same pools, and two lines reading "epoch 1" with different
        numbers is a puzzle rather than a report.

        Enabled the standard way -- ``logging.getLogger("insitubatch").setLevel(logging.INFO)``
        -- rather than through a constructor flag, so there is nothing to thread through the
        API and nothing to leave switched on by accident. It is off by default because
        libraries do not configure logging for their callers.

        The two numbers worth watching are ``allocated`` and the hit rate. Allocations should
        fall to zero once the pool has converged on the in-flight batch count; a nonzero count
        in a later epoch means buffers are not coming back -- retained batches (which is
        legitimate, but it is a memory floor) or a changing batch geometry. The ``x pinned``
        term is the only confirmation available from a training log that page-locked buffers
        are actually in use, since the fallback to pageable memory is otherwise silent here.
        """
        if not logger.isEnabledFor(logging.INFO):
            return  # skip the snapshot, which takes the buffer pool's lock
        reads = pool.hits + pool.misses
        logger.info(
            "epoch %d (%s): chunks %d/%d hit (%.0f%%), peak resident %d%s; batch buffers %s",
            self._epoch,
            split or "all",
            pool.hits,
            reads,
            100 * pool.hits / reads if reads else 0.0,
            pool.max_resident,
            f", {len(self.bad_chunks)} bad chunks" if self.bad_chunks else "",
            pool.buffer_stats().summary(),
        )

    def close(self) -> None:
        """Release the cache pool's backing (mmap handles, cached chunks) and any async
        store session.

        The pool persists across epochs, so close it when done training -- not per
        epoch. With ``persist=True`` the cache files + manifest are kept on disk for a
        future run (only the in-memory handles are released); otherwise the mmap spill
        files are unlinked. An fsspec/gcsfs store's aiohttp session is closed on its own
        loop here (a no-op for obstore) so it does not leak or spew a teardown traceback
        at GC; gcsfs recreates it lazily if the store is reused. Idempotent; also called
        on GC.
        """
        self._pool.close()
        close_store(self.store)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # best-effort on GC
            self.close()


class _SplitView:
    """A lazy, re-iterable view of one split, returned by ``InSituDataset.train`` /
    ``.val`` / ``.test`` / ``.all``. Iterating it streams that split's batches through the
    dataset's *shared* pool, so a chunk two splits both read (a windowed read spilling
    across a split boundary) is decoded once. Re-iterable: a fresh pass each ``iter()``.
    ``geometries`` is exposed so the framework adapters can infer tensor shapes.
    """

    def __init__(self, dataset: InSituDataset, split: SplitName | None, shuffle: bool) -> None:
        self._dataset = dataset
        self._split = split
        self._shuffle = shuffle

    @property
    def geometries(self) -> dict[str, ArrayGeometry]:
        # Post-transform geometry: the adapters infer *output* tensor shapes/dtypes from
        # this, which a reshaping chunk_transform (regrid / dtype recast) changes.
        return self._dataset._out_geometries

    def __iter__(self) -> Iterator[Batch]:
        return self._dataset._iterate(self._split, self._shuffle)

    def _use_host_allocator(self, allocator: HostAllocator) -> None:
        """Point the batch-buffer pool at a different host allocator (adapter-internal).

        How ``frameworks.as_torch(..., device=...)`` installs page-locked buffers without the
        core importing torch. Deliberately not public and deliberately not a constructor
        argument: pinning is only *safe* alongside an adapter that owns the H2D copy, so the
        two arrive together or not at all -- pinned buffers under a caller's own
        ``non_blocking`` copy would reintroduce exactly the use-after-recycle this avoids.
        """
        self._dataset._pool.set_host_allocator(allocator)

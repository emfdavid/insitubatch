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
a single ``max_inflight`` budget and scatters decoded tiles into a
:class:`ChunkPool`. This producer walks the shuffle order, waits on each block's
assembled chunks, gathers coalesced batches, and unpins the block (making it
LRU-evictable / retainable for reuse). Read concurrency (``max_inflight``) and the
residency budget are independent dials. See [docs/architecture.md] for the pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from .pool import ChunkPool
from .scheduler import Scheduler, SchedulerConfig
from .shuffle import block_shuffled_order, sequential_order
from .split import SplitManifest, valid_anchor_range
from .store import StoreLike, open_geometries
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

    One epoch over a view = permute the split's chunks -> walk shuffle-blocks -> per
    block, stream-fetch its stored chunks into the pool, gather coalesced batches, evict.
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
    """

    def __init__(
        self,
        store: StoreLike,
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
        on_bad_chunk: str = "raise",
        chunk_transforms: Sequence[Callable[[DecodedChunk], DecodedChunk]] = (),
        batch_transforms: Sequence[Callable[[Batch], Batch]] = (),
        **store_kwargs: Any,
    ) -> None:
        self.store = store
        self.store_kwargs = store_kwargs
        self.geometries = (
            geometries if geometries is not None else open_geometries(store, **store_kwargs)
        )
        self.manifest = manifest
        self.variables = list(self.geometries)

        # Variables must share the sample axis (length + chunk size): the draw order
        # and gather use a single chunk size for all variables.
        spcs = {g.sample_chunk_size for g in self.geometries.values()}
        lengths = {g.n_samples for g in self.geometries.values()}
        if len(spcs) > 1 or len(lengths) > 1:
            raise ValueError(
                "All variables must share the same sample-axis length and chunk "
                f"size; got sample_chunk_size={sorted(spcs)}, "
                f"n_samples={sorted(lengths)}."
            )

        self.batch_size = batch_size
        self.block_chunks = block_chunks
        self.seed = seed
        self.shuffle = shuffle
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.chunk_transforms = tuple(chunk_transforms)
        self.batch_transforms = tuple(batch_transforms)
        self._epoch = 0
        self._persist = persist
        self.resident_peak = 0  # peak resident outer chunks (observability)
        self.cache_hits = 0  # chunks served without a fetch this epoch (cross-epoch/run)
        self.cache_misses = 0  # chunks fetched + decoded this epoch
        # Stored tiles NaN-filled in the last epoch (when on_bad_chunk="nan") -- which
        # (array, chunk_index, inner_coord) reads were corrupt/truncated. len() is the
        # count. Inspect after iterating to log/quarantine bad chunks.
        self.bad_chunks: list[StoredChunkRead] = []
        self._on_bad_chunk = on_bad_chunk

        # The pool is the assembly buffer AND the cache, owned here so it persists
        # across epochs. The byte budget is the single residency knob: the floor is
        # the working set -- the current block plus one read-ahead block, all
        # variables, must be co-resident (a batch draws across a whole block) -- and
        # a larger budget (cache_budget_bytes) retains drained chunks for cross-epoch
        # decode-once reuse. cache_dir spills slots to NVMe (mmap) instead of heap.
        #
        # Windows widen the floor. A windowed read (any nonzero offset) crosses a chunk
        # boundary, so an anchor chunk's read-union spans up to 2 + ceil(span/spc)
        # chunks per variable (span = max offset - min offset); with every offset 0 the
        # factor is 1 -- the plain 2 * block_chunks working set.
        spc0 = next(iter(self.geometries.values())).sample_chunk_size
        offsets = [g.offset for g in self.geometries.values()]
        span = max(offsets) - min(offsets)
        windowed = any(o != 0 for o in offsets)
        window_factor = 2 + (-(-span // spc0)) if windowed else 1  # 2 + ceil(span/spc)
        per_chunk_all_vars = int(
            sum(
                g.sample_chunk_size * int(np.prod(g.inner_shape)) * g.dtype.itemsize
                for g in self.geometries.values()
            )
        )
        working_set = 2 * block_chunks * window_factor * per_chunk_all_vars
        if windowed and self.shuffle:
            # Shuffle permutes chunk order, so a windowed read can spill into chunks
            # owned by any other block: a chunk admitted early may be needed late. Until
            # bounded residency (re-fetch the spill) lands, hold the whole split resident
            # -- decode-once, the accepted memory cost of windows (spill to NVMe via
            # cache_dir on large splits). Only `.train` shuffles (eval views are
            # sequential and spill only locally), so size to the train split.
            n_train_chunks = len(self.manifest.chunks[SplitName.TRAIN.value])
            working_set = max(working_set, n_train_chunks * per_chunk_all_vars)
        self.cache_budget_bytes = max(int(cache_budget_bytes or 0), working_set)
        # persist turns the cache_dir mmap tier into a cross-run cache (files + manifest
        # survive close; reopen revives them as hits). It needs a dir to keep files in;
        # the dir path is the dataset+pipeline identity (bury a version in it).
        if persist and cache_dir is None:
            raise ValueError("persist=True requires cache_dir to keep the cache files in")
        self._pool = ChunkPool(
            self.geometries,
            chunk_transforms=self.chunk_transforms,
            backing_dir=cache_dir,
            budget_bytes=self.cache_budget_bytes,
            persist=persist,
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
        geom = self.geometries[self.variables[0]]
        chunk_ids = self._chunk_ids(split)
        spc = geom.sample_chunk_size
        if shuffle:
            order = block_shuffled_order(
                chunk_ids,
                spc,
                geom.n_samples,
                block_chunks=self.block_chunks,
                seed=self.seed,
                epoch=self._epoch,
            )
        else:
            order = sequential_order(chunk_ids, spc, geom.n_samples)
        return self._drop_edge_anchors(order, spc, geom.n_samples)

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
            read_cid = (anchor + geom.offset) // spc
            keys.update((geom.path, int(c)) for c in np.unique(read_cid))
        return keys

    def _iterate(self, split: SplitName | None, shuffle: bool) -> Iterator[Batch]:
        """Drain assembled batches for one split from a background producer (prefetch).

        Called by the ``.train`` / ``.val`` / ``.test`` / ``.all`` views, all sharing the
        one :class:`ChunkPool` -- so a chunk a windowed read pulls across a split boundary
        is decoded once and reused by both splits.

        A producer thread starts the scheduler over the split's chunks (in draw order),
        then for each shuffle-block waits the block assembled, gathers its batches, and
        unpins it; this consumer pops from a bounded queue (depth ``prefetch_depth``) that
        provides backpressure and inter-batch overlap. The scheduler keeps ``max_inflight``
        tiles continuously in flight and fetches one block ahead, so block-boundary IO
        overlaps the per-batch compute. Chunks the pool already holds (cross-epoch or
        cross-split hits) cost no fetch.
        """
        geom = self.geometries[self.variables[0]]
        spc = geom.sample_chunk_size
        order = self._draw_order(split, shuffle)
        blocks = _partition_blocks(order, self.block_chunks)
        ordered_chunks = [int(c) for _rstart, _rstop, cids in blocks for c in cids]

        # Start clean: release any pins a prior epoch leaked (an early break leaves its
        # read-ahead pinned in the persistent pool -> would shrink this epoch's budget
        # until admission deadlocks). The prior scheduler is fully closed here, so no
        # pin/unpin can race. Resident chunks stay (unpinned) for cross-epoch reuse.
        self._pool.unpin_all()

        out_q: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        stop = threading.Event()

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
            bs = self.batch_size
            try:
                sched.start(ordered_chunks)
                for bi, (rstart, rstop, _cids) in enumerate(blocks):
                    # A block's batches draw across its whole read-union, so wait it all
                    # assembled (and claimed by the driver -- see ChunkPool.wait_ready)
                    # before gathering; each wait is cheap once ready.
                    for path, cid in block_keys[bi]:
                        sched.pool.wait_ready(path, cid)
                    for start in range(rstart, rstop, bs):
                        if stop.is_set():
                            return
                        rows = order[start : min(start + bs, rstop)]
                        batch = sched.pool.gather(rows, self.variables, spc)
                        for transform in self.batch_transforms:
                            batch = transform(batch)
                        out_q.put(batch)  # blocks when full -> backpressure
                    # Release the driver's reference on chunks whose *last* use is this
                    # block: now LRU-evictable (retained for reuse if budget allows),
                    # unblocking the read-ahead. Chunks read again later keep their
                    # reference until then.
                    sched.unpin_block(release[bi])
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                out_q.put(exc)
            finally:
                out_q.put(self._SENTINEL)

        with Scheduler(
            self.store,
            self.geometries,
            self._pool,
            self.scheduler_config,
            **self.store_kwargs,
        ) as sched:
            producer = threading.Thread(
                target=produce, args=(sched,), name="insitu-prefetch", daemon=True
            )
            producer.start()
            try:
                while True:
                    item = out_q.get()
                    if item is self._SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                # Signal stop, then drain so a producer parked on a full queue can
                # proceed and exit before the scheduler (context manager) is closed.
                stop.set()
                while producer.is_alive():
                    with contextlib.suppress(queue.Empty):
                        out_q.get(timeout=0.05)
                producer.join(timeout=10)
                pool = sched.pool
                self.resident_peak = pool.max_resident  # peak residency this epoch
                self.cache_hits = pool.hits
                self.cache_misses = pool.misses
                self.bad_chunks = list(sched.bad_chunks)  # tiles NaN-filled this epoch
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

    def close(self) -> None:
        """Release the cache pool's backing (mmap handles, cached chunks).

        The pool persists across epochs, so close it when done training -- not per
        epoch. With ``persist=True`` the cache files + manifest are kept on disk for a
        future run (only the in-memory handles are released); otherwise the mmap spill
        files are unlinked. Idempotent; also called on GC.
        """
        self._pool.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # best-effort on GC
            self._pool.close()


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
        return self._dataset.geometries

    def __iter__(self) -> Iterator[Batch]:
        return self._dataset._iterate(self._split, self._shuffle)

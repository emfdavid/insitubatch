"""What will this dataset actually do? -- a static report, before anything runs.

``InSituDataset.describe()`` answers from **geometry and configuration only**: it opens no
store, fetches nothing, and runs no pass. That is the point. The facts it reports -- a
360-byte gather run, a 2x ragged residency multiplier, a budget sized for one iteration
when you meant to run two -- are exactly the ones you want *before* waiting an hour to
discover them, and a report that had to touch the store would be unusable in the situation
it exists for.

It is deliberately on demand. Construction stays quiet (bar the two warnings that already
fire), because layout advice on every dataset you build teaches people to skim our logs.

Runtime facts -- queue depths, per-stage timing, cache hits, peak residency -- are a
different surface with a different cost model, and live in #45.

:func:`working_set_bytes` is shared with :class:`~insitubatch.source.InSituDataset`, which
sizes its automatic budget with it. One formula, called twice: a report that predicted a
different number from the one the engine uses would be worse than no report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TextIO, TypedDict

import numpy as np

from .pool import slot_charge_bytes
from .shuffle import shuffle_quality
from .split import SplitManifest
from .transforms import transform_scope, unwrap_transform
from .types import ArrayGeometry, SplitName

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps summary importable from source
    from .source import InSituDataset

# The measured cliff sits between ~1.4 KiB and ~360 B of contiguous gather run (#43): at
# 1440 B chunked gather is at parity or faster, at 360 B it costs 1.84-2.20x. 1 KiB is
# inside that band, so it flags the losing side without crying wolf at the parity point.
MIN_GATHER_RUN_BYTES = 1024

# Peak RssAnon divided by the accounted total, measured on the chunked-slot pool (#36,
# re-measured 2026-09-01 after #41: 1.22-1.27x over four runs, MALLOC_ARENA_MAX 32 and 64).
# One geometry, so it is a rule of thumb for the estimated row -- not a per-dataset model.
ALLOCATOR_RETENTION = 1.25


class VariableReport(TypedDict):
    """One variable's geometry, and the per-chunk bytes derived from it."""

    label: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    sample_axis: int
    sample_chunk_size: int
    n_samples: int
    n_chunks: int
    offset: int
    inner_shape: tuple[int, ...]
    inner_chunks: tuple[int, ...]
    stored_chunks_per_chunk: int
    stored_chunk_bytes: int
    logical_chunk_bytes: int
    slot_bytes: int
    ragged_multiplier: float
    gather_run_bytes: int
    output_inner_shape: tuple[int, ...]
    output_dtype: str


class MemoryReport(TypedDict):
    """The accounted pieces of the memory model, evaluated for this configuration."""

    iterations: int
    residency_bytes: int
    inflight_bytes: int
    queue_bytes: int
    scratch_bytes: int
    accounted_bytes: int
    estimated_peak_bytes: int
    budget_is_automatic: bool
    working_set_bytes: int


class ConfigReport(TypedDict):
    """What the dataset resolved to -- derived values, not the arguments passed in."""

    batch_size: int
    block_chunks: int
    block_chunks_requested: int
    max_inflight: int
    prefetch_depth: int
    shuffle: bool
    seed: int
    cache_budget_bytes: int
    cache_dir: str | None
    persist: bool
    readonly_cache: bool
    chunk_transforms: tuple[str, ...]
    batch_transforms: tuple[str, ...]
    assembles: bool
    windowed: bool
    offsets: dict[str, int]
    samples_per_chunk: int
    n_chunks: dict[str, int]
    split_chunks: dict[str, int]
    shuffle_quality: float | None


class Note(TypedDict):
    """One finding. ``severity`` separates "this will hurt" from "worth knowing"."""

    code: str
    severity: Literal["warn", "note"]
    subject: str
    message: str


class DatasetReport(TypedDict):
    """The whole static picture. Returned by :meth:`InSituDataset.describe`."""

    variables: dict[str, VariableReport]
    config: ConfigReport
    memory: MemoryReport
    notes: list[Note]


def gather_run_bytes(geom: ArrayGeometry) -> int:
    """Bytes written per contiguous run when ``gather`` places a stored chunk.

    A stored chunk is contiguous in itself, but it lands in a *sub-rectangle* of the batch,
    so a run breaks at the first axis where the chunk is narrower than the array. Walking
    inward from the last axis, the run extends while the chunk spans the full extent; the
    first axis that does not span ends it. When every inner axis spans (one stored chunk per
    outer chunk), the whole tile is one run.

    This is the quantity behind the short-run warning: it is set by the innermost extent,
    which is why full-width slabs ``(1, k, W)`` beat square tiles of the same byte size.
    """
    inner_shape, inner_chunks = geom.inner_shape, geom.inner_chunks
    run = geom.dtype.itemsize
    for extent, chunk in zip(reversed(inner_shape), reversed(inner_chunks), strict=True):
        run *= int(chunk)
        if chunk != extent:
            break
    return run


def working_set_bytes(
    geoms: list[ArrayGeometry],
    out_geoms: list[ArrayGeometry],
    manifest: SplitManifest,
    *,
    block_chunks: int,
    ref_spc: int,
    shuffle: bool,
    assembles: bool,
) -> int:
    """The residency floor: what must be co-resident for one iteration to make progress.

    The current block plus one read-ahead block, every variable, because a batch draws
    across a whole block. Windows widen it (a windowed read crosses chunk boundaries), and
    shuffle plus windows widen it to the whole split, because a shuffled windowed read can
    spill into a chunk owned by any other block.

    Charged with :func:`~insitubatch.pool.slot_charge_bytes`, which is what the pool will
    actually charge -- sizing from the assembled shape while the pool charges stored tiles
    under-provisions the budget, and the pool then starves mid-epoch, which is a hang-shaped
    failure rather than a slow one.

    Sized from the **output** geometry: the pool caches post-transform chunks, so a regrid
    that grows (or shrinks) the data changes the footprint the budget must hold.
    """
    offsets = [g.offset for g in geoms]
    span = max(offsets) - min(offsets)
    windowed = any(o != 0 for o in offsets)
    uniform_spc = len({g.sample_chunk_size for g in geoms}) == 1

    def bytes_per_chunk(g: ArrayGeometry, o: ArrayGeometry) -> int:
        return slot_charge_bytes(g, o, assembles=assembles)

    pairs = list(zip(geoms, out_geoms, strict=True))
    if uniform_spc:
        # Every variable's chunk aligns to the reference grid, so a block reads exactly its
        # own chunks. A windowed read straddles at most one boundary, so an anchor chunk's
        # read-union spans 2 + ceil(span/spc) chunks per variable; with every offset 0 the
        # factor is 1 -- the plain 2 * block_chunks working set.
        spc0 = geoms[0].sample_chunk_size
        window_factor = 2 + (-(-span // spc0)) if windowed else 1
        per_chunk_all_vars = sum(bytes_per_chunk(g, o) for g, o in pairs)
        working_set = 2 * block_chunks * window_factor * per_chunk_all_vars
        if windowed and shuffle:
            # Shuffle permutes chunk order, so a windowed read can spill into chunks owned by
            # any other block: a chunk admitted early may be needed late. Until bounded
            # residency (re-fetch the spill) lands, hold the whole split resident -- decode-
            # once, the accepted memory cost of windows (spill to NVMe via cache_dir on large
            # splits). Only `.train` shuffles (eval views are sequential and spill only
            # locally), so size to the train split.
            n_train_chunks = len(manifest.chunks[SplitName.TRAIN.value])
            working_set = max(working_set, n_train_chunks * per_chunk_all_vars)
        return working_set

    # Non-uniform chunk size: a variable maps the reference anchor grid onto its own chunks,
    # so a block touches a variable-specific chunk count. A 2-block read-ahead window of
    # 2*block_chunks*ref_spc anchor samples (plus the offset span) covers, per variable,
    # ceil(window/spc)+1 of its chunks (the +1 for boundary misalignment).
    def var_bytes(g: ArrayGeometry, o: ArrayGeometry, samples: int) -> int:
        n_chunks = -(-(samples + span) // g.sample_chunk_size) + 1
        return n_chunks * bytes_per_chunk(g, o)

    window_samples = 2 * block_chunks * ref_spc
    working_set = sum(var_bytes(g, o, window_samples) for g, o in pairs)
    if shuffle:
        # Under shuffle a variable chunk can be needed by scattered blocks (a coarse chunk
        # straddling reference-block boundaries) -- like a window spill -- so hold the train
        # split resident per variable (decode-once). A tighter bound is future work;
        # different-axis-chunking is today a modest-sized microscopy case.
        train_samples = len(manifest.chunks[SplitName.TRAIN.value]) * ref_spc
        working_set = max(working_set, sum(var_bytes(g, o, train_samples) for g, o in pairs))
    return working_set


def _variable_report(
    label: str, src: ArrayGeometry, out: ArrayGeometry, *, assembles: bool
) -> VariableReport:
    stored = int(np.prod(src.tile_shape(), dtype=np.int64)) * src.dtype.itemsize
    logical = int(np.prod(out.slot_shape(0), dtype=np.int64)) * out.dtype.itemsize
    slot = slot_charge_bytes(src, out, assembles=assembles)
    return VariableReport(
        label=label,
        path=src.path,
        dtype=str(src.dtype),
        shape=src.shape,
        chunks=src.chunks,
        sample_axis=src.sample_axis,
        sample_chunk_size=src.sample_chunk_size,
        n_samples=src.n_samples,
        n_chunks=src.n_chunks,
        offset=src.offset,
        inner_shape=src.inner_shape,
        inner_chunks=src.inner_chunks,
        stored_chunks_per_chunk=src.n_inner_chunks(0),
        stored_chunk_bytes=stored,
        logical_chunk_bytes=logical,
        slot_bytes=slot,
        ragged_multiplier=(slot / logical) if logical else 1.0,
        gather_run_bytes=gather_run_bytes(src),
        output_inner_shape=out.inner_shape,
        output_dtype=str(out.dtype),
    )


def _notes(variables: dict[str, VariableReport], cfg: ConfigReport) -> list[Note]:
    """Say when a layout is a problem, and why. This is the point of the report.

    Per-variable findings come first: they name the thing you can change. Only findings
    live here -- the two facts that hold for every dataset (the budget covers one
    iteration, glibc retains freed slots) are footnotes on the memory block, because a
    "note" that always fires is not a finding and teaches people to skip the section.
    """
    notes: list[Note] = []
    globals_: list[Note] = []

    for label, v in variables.items():
        if v["gather_run_bytes"] < MIN_GATHER_RUN_BYTES:
            notes.append(
                Note(
                    code="short-gather-run",
                    severity="warn",
                    subject=label,
                    message=(
                        f"{_bytes(v['gather_run_bytes'])} contiguous run per gather write "
                        f"(inner chunks {v['inner_chunks']} of {v['inner_shape']}). Below "
                        f"~{_bytes(MIN_GATHER_RUN_BYTES)}, gather costs 1.8-2.2x. The run is "
                        "set by the INNERMOST extent, so full-width slabs (1, k, W) give long "
                        "runs "
                        "and fewer decode tasks; square tiles lose on both."
                    ),
                )
            )
        if v["ragged_multiplier"] > 1.001:
            notes.append(
                Note(
                    code="ragged-grid",
                    severity="note",
                    subject=label,
                    message=(
                        f"residency is {v['ragged_multiplier']:.3f}x the logical chunk: the "
                        f"chunk grid {v['inner_chunks']} does not divide {v['inner_shape']}, "
                        "and stored chunks are kept whole (padding included) rather than "
                        "clipped. The budget charges the padded size; this is real memory."
                    ),
                )
            )
        if v["stored_chunks_per_chunk"] == 1:
            notes.append(
                Note(
                    code="single-inner-fat-chunk",
                    severity="note",
                    subject=label,
                    message=(
                        f"one stored chunk per outer chunk at "
                        f"{_bytes(v['stored_chunk_bytes'])}: concurrency and memory are "
                        f"coupled here, so each of max_inflight={cfg['max_inflight']} slots "
                        "costs a whole chunk. Inner-chunk the field to separate them."
                    ),
                )
            )

    if cfg["block_chunks"] != cfg["block_chunks_requested"]:
        globals_.append(
            Note(
                code="block-chunks-widened",
                severity="note",
                subject="config",
                message=(
                    f"block_chunks widened {cfg['block_chunks_requested']} -> "
                    f"{cfg['block_chunks']} so a {cfg['batch_size']}-sample batch fits one "
                    f"shuffle block at {cfg['samples_per_chunk']} samples/chunk. Below that, "
                    "a batch spans more blocks than the budget holds and the loader stalls."
                ),
            )
        )

    return notes + globals_


def describe(ds: InSituDataset, *, iterations: int = 1) -> DatasetReport:
    """The static report. ``iterations`` is how many passes will share the pool at once."""
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    assembles = bool(ds.chunk_transforms)
    variables = {
        label: _variable_report(
            label, ds.geometries[label], ds._out_geometries[label], assembles=assembles
        )
        for label in ds.variables
    }

    order = ds._draw_order(SplitName.TRAIN, ds.shuffle)
    quality = shuffle_quality(order, ds._ref_spc) if len(order) > 1 else None

    cfg = ConfigReport(
        batch_size=ds.batch_size,
        block_chunks=ds.block_chunks,
        block_chunks_requested=ds._block_chunks_requested,
        max_inflight=ds.scheduler_config.max_inflight,
        prefetch_depth=ds.prefetch_depth,
        shuffle=ds.shuffle,
        seed=ds.seed,
        cache_budget_bytes=ds.cache_budget_bytes,
        cache_dir=str(ds._cache_dir) if ds._cache_dir is not None else None,
        persist=ds._persist,
        readonly_cache=ds._readonly_cache,
        chunk_transforms=tuple(_name(t) for t in ds.chunk_transforms),
        batch_transforms=tuple(_name(t) for t in ds.batch_transforms),
        assembles=assembles,
        windowed=any(g.offset != 0 for g in ds.geometries.values()),
        offsets={label: g.offset for label, g in ds.geometries.items()},
        samples_per_chunk=ds._ref_spc,
        n_chunks={label: v["n_chunks"] for label, v in variables.items()},
        split_chunks={name: len(ids) for name, ids in ds.manifest.chunks.items()},
        shuffle_quality=quality,
    )

    geoms = [ds.geometries[label] for label in ds.variables]
    out_geoms = [ds._out_geometries[label] for label in ds.variables]
    floor = working_set_bytes(
        geoms,
        out_geoms,
        ds.manifest,
        block_chunks=ds.block_chunks,
        ref_spc=ds._ref_spc,
        shuffle=ds.shuffle,
        assembles=assembles,
    )
    inflight = cfg["max_inflight"] * max(v["stored_chunk_bytes"] for v in variables.values())
    batch_bytes = sum(
        int(np.prod(ds._out_geometries[label].inner_shape, dtype=np.int64))
        * ds._out_geometries[label].dtype.itemsize
        * ds.batch_size
        for label in ds.variables
    )
    queue = (ds.prefetch_depth + 1) * batch_bytes
    scratch = (
        sum(v["stored_chunk_bytes"] * v["stored_chunks_per_chunk"] for v in variables.values())
        if assembles
        else 0
    )
    residency = ds.cache_budget_bytes * iterations
    mem = MemoryReport(
        iterations=iterations,
        residency_bytes=residency,
        inflight_bytes=inflight,
        queue_bytes=queue,
        scratch_bytes=scratch,
        accounted_bytes=residency + inflight + queue + scratch,
        estimated_peak_bytes=int((residency + inflight + queue + scratch) * ALLOCATOR_RETENTION),
        budget_is_automatic=ds.cache_budget_bytes == floor,
        working_set_bytes=floor,
    )

    return DatasetReport(variables=variables, config=cfg, memory=mem, notes=_notes(variables, cfg))


def _name(fn: object) -> str:
    """A transform's display name, with its scope when it has one.

    ``kelvin_to_celsius[2m_temperature]`` -- the report exists to say what a configuration
    will do, and "which variables does this transform touch" is now part of that."""
    scope = transform_scope(fn) if callable(fn) else None
    base = unwrap_transform(fn) if scope is not None else fn  # type: ignore[arg-type]
    label = getattr(base, "__name__", type(base).__name__)
    return label if scope is None else f"{label}[{','.join(sorted(scope))}]"


def _bytes(n: int) -> str:
    """Bytes in the unit people think in at that size -- B, KiB, MiB, GiB."""
    if n < 1024:
        return f"{n} B"
    if n < 1 << 20:
        return f"{n / 1024:.1f} KiB"
    if n < 1 << 30:
        return f"{n / (1 << 20):.1f} MiB"
    return f"{n / (1 << 30):.2f} GiB"


def _dims(t: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in t)


def print_summary(report: DatasetReport, file: TextIO | None = None) -> None:
    """Format :func:`describe` for a terminal. Numbers first, then what to do about them."""
    out: list[str] = []
    cfg, mem = report["config"], report["memory"]

    out.append("insitubatch dataset")
    out.append("")
    out.append("variables")
    for label, v in report["variables"].items():
        window = f"  window +{v['offset']}" if v["offset"] else ""
        out.append(f"  {label}{window}")
        out.append(
            f"    {_dims(v['shape'])} {v['dtype']}   chunks {_dims(v['chunks'])}"
            f"   sample axis {v['sample_axis']}"
        )
        out.append(
            f"    {v['n_chunks']} chunks of {v['sample_chunk_size']} sample(s)"
            f"   field {_dims(v['inner_shape'])} in {v['stored_chunks_per_chunk']} stored"
            f" chunk(s) of {_dims(v['inner_chunks'])}"
        )
        ragged = (
            f"   ragged {v['ragged_multiplier']:.3f}x" if v["ragged_multiplier"] > 1.001 else ""
        )
        out.append(
            f"    stored chunk {_bytes(v['stored_chunk_bytes'])}"
            f"   resident per chunk {_bytes(v['slot_bytes'])}{ragged}"
            f"   gather run {_bytes(v['gather_run_bytes'])}"
        )
        if v["output_inner_shape"] != v["inner_shape"] or v["output_dtype"] != v["dtype"]:
            out.append(
                f"    after chunk_transform: field {_dims(v['output_inner_shape'])}"
                f" {v['output_dtype']}"
            )

    spc = cfg["samples_per_chunk"]
    out.append("")
    out.append("configuration (resolved)")
    widened = (
        f" (raised from {cfg['block_chunks_requested']})"
        if cfg["block_chunks"] != cfg["block_chunks_requested"]
        else ""
    )
    out.append(
        f"  batch_size {cfg['batch_size']}   block_chunks {cfg['block_chunks']}{widened}"
        f"   max_inflight {cfg['max_inflight']}   prefetch_depth {cfg['prefetch_depth']}"
    )
    quality = "n/a" if cfg["shuffle_quality"] is None else f"{cfg['shuffle_quality']:.2f}"
    out.append(
        f"  shuffle {'on' if cfg['shuffle'] else 'off'} (seed {cfg['seed']}, quality {quality})"
        f"   window {'yes' if cfg['windowed'] else 'no'}"
        f"   shuffle pool {cfg['block_chunks'] * spc} samples"
        f" for a {cfg['batch_size']}-sample batch"
    )
    splits = "  ".join(f"{k} {n}" for k, n in cfg["split_chunks"].items())
    out.append(f"  splits (chunks)  {splits}")
    cache = cfg["cache_dir"] or "heap"
    # read-only is worth a line of its own: it changes what the run *does* (serves only what
    # is already cached, and raises on a miss rather than fetching), which is exactly what a
    # "what will this cost" report exists to tell you before it costs it.
    mode = ", read-only" if cfg["readonly_cache"] else (", persist" if cfg["persist"] else "")
    out.append(f"  cache backing {cache}{mode}")
    if cfg["chunk_transforms"]:
        out.append(f"  chunk_transforms {', '.join(cfg['chunk_transforms'])} (slot assembles)")
    if cfg["batch_transforms"]:
        out.append(f"  batch_transforms {', '.join(cfg['batch_transforms'])}")

    auto = " (automatic: the working-set floor)" if mem["budget_is_automatic"] else ""
    out.append("")
    out.append(f"memory, accounted, for {mem['iterations']} concurrent iteration(s)")
    out.append(f"  residency   {_bytes(mem['residency_bytes']):>12}   budget{auto}")
    out.append(f"  in flight   {_bytes(mem['inflight_bytes']):>12}   max_inflight x stored chunk")
    out.append(f"  batch queue {_bytes(mem['queue_bytes']):>12}   (prefetch_depth + 1) x batch")
    if mem["scratch_bytes"]:
        out.append(f"  scratch     {_bytes(mem['scratch_bytes']):>12}   chunk_transform assembly")
    out.append(f"  accounted   {_bytes(mem['accounted_bytes']):>12}   sum of the rows above")
    out.append(
        f"  ESTIMATED   {_bytes(mem['estimated_peak_bytes']):>12}"
        f"   accounted x {ALLOCATOR_RETENTION} -- plan for this"
    )
    out.append("")
    footnotes = [
        f"The x{ALLOCATOR_RETENTION} is allocator retention, not slack we can spend: slot "
        "buffers are freed per chunk and glibc keeps them under its dynamic mmap threshold "
        "instead of returning them to the kernel. It is a rule of thumb measured on one "
        "geometry (#36); your own ratio can differ.",
    ]
    if mem["iterations"] == 1:
        footnotes.append(
            "Sized for ONE iteration. Every active iteration shares this pool and holds its "
            "own chunk references, so zip(ds.train, ds.val) or two DataLoaders need ~N x the "
            "residency. Pass describe(iterations=N) to size it, cache_budget_bytes to buy it."
        )
    for note in footnotes:
        out.append(f"  {_wrap(note, width=88, indent=4)}")

    warns = [n for n in report["notes"] if n["severity"] == "warn"]
    notes = [n for n in report["notes"] if n["severity"] == "note"]
    for heading, group in (("warnings", warns), ("notes", notes)):
        if not group:
            continue
        out.append("")
        out.append(heading)
        for n in group:
            body = _wrap(n["message"], width=88, indent=6)
            out.append(f"  [{n['subject']}] {body.lstrip()}")

    print("\n".join(out), file=file)


def _wrap(text: str, *, width: int, indent: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    pad = " " * indent
    return f"\n{pad}".join(lines)


__all__ = ["DatasetReport", "describe", "print_summary", "working_set_bytes"]

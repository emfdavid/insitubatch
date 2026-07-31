"""Probe: what a pool of reused (and pinned) batch buffers could actually buy.

``pool.gather`` allocates a fresh ``np.empty`` per variable per batch, and that heap
memory is never page-locked, so H2D copies are pageable. Replacing it with a pool of
pre-pinned buffers fixes both at once -- but only if either half is on the critical
path. This probe measures the **ceiling** for both, before any engine change, so the
decision is a number and not an instinct.

Four arms, each isolating one claim:

* **alloc** (no GPU needed) -- fresh ``np.empty`` vs a reused buffer, running
  ``gather``'s exact inner loop (one coalesced fancy-index per chunk). ``np.empty``
  itself is nearly free (it reserves address space, it does not touch pages), so what
  this really measures is **first-touch page faults** on the scatter-write. glibc's
  *dynamic* mmap threshold (128 KiB, growing to at most 32 MiB as blocks are freed)
  means a batch under ~32 MiB gets recycled on the heap and re-faults nothing, while a
  larger one is ``mmap``/``munmap``'d every batch and faults its whole page set again.
  Expect a cliff, not a curve -- and expect ~0 for a WB2-sized batch.
* **h2d** (needs CUDA) -- pinned vs pageable ``.to('cuda', non_blocking=True)``,
  ``torch.cuda.Event``-timed. Gives the raw bandwidth ratio: the *upper bound* on what
  pinning can save per batch.
* **overlap** (needs CUDA) -- the question that actually decides it. Issue the copy on a
  side stream against a busy compute stream and compare wall time to the serialized
  sum. Only a pinned source can overlap; a pageable one is staged through a driver
  bounce buffer and serializes. If the copy is already hidden by prefetch, both arms
  land at the same wall time and pinning buys nothing end to end.
* **roundtrip** (needs CUDA) -- a feasibility check, not a measurement. The core is numpy
  and imports no framework, so pinning would arrive via an allocator injected by the torch
  adapter, and every batch would travel ``pinned tensor -> .numpy() -> base[:k] view ->
  DLPack -> .to(cuda)``. This asks whether pinning actually survives that path. It is the
  gate on the whole design: if it fails, pinning needs a torch-owned buffer instead.

Read the result as: *alloc* says whether buffer reuse alone pays, *h2d* bounds the pinning
prize, *overlap* says whether that prize is real or already collected, and *roundtrip* says
whether we can collect it without putting torch in the core.

    uv run python -m bench.probe_batch_buffers                 # host arm only
    uv run python -m bench.probe_batch_buffers --arms all      # on a GPU box
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch

# Coalesced groups per batch: gather does one fancy-index per distinct source chunk
# (``np.unique(read_cid)``), so the write is a handful of strided scatters, not one
# contiguous memcpy. Page-fault cost is the same either way; keep the shape honest.
CHUNKS = 4

ARMS = ("alloc", "h2d", "overlap", "roundtrip")


class Case(NamedTuple):
    """One batch payload to sweep: ``(batch, *inner)`` of ``dtype``."""

    name: str
    batch: int
    inner: tuple[int, ...]
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.batch, *self.inner)

    @property
    def mib(self) -> float:
        return self.batch * int(np.prod(self.inner)) * self.dtype.itemsize / 2**20


# Chosen to bracket glibc's 32 MiB dynamic mmap ceiling: the first two land under it
# (heap-recycled), the last two above it (mmap churn + a full re-fault every batch).
CASES = (
    Case("wb2 16x3x128x64", 16, (3, 128, 64), np.dtype("f4")),
    Case("vit 32x3x224x224", 32, (3, 224, 224), np.dtype("f4")),
    Case("vit-big 64x3x224x224", 64, (3, 224, 224), np.dtype("f4")),
    Case("microscopy 8x2x32x512x512", 8, (2, 32, 512, 512), np.dtype("f4")),
)


def _median_ms(fn: Callable[[], None], iters: int) -> float:
    """Median wall time of *iters* calls, in ms.

    Median, not mean: page-fault and allocator noise is one-sided, and a single unlucky
    ``munmap`` should not set the number.
    """
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def probe_alloc(case: Case, iters: int) -> tuple[float, float]:
    """``(fresh_ms, reuse_ms)`` for one batch assembly.

    The arms differ only in where ``out`` comes from -- identical scatter-write from
    identical sources -- so the delta is allocation plus first touch, nothing else.
    """
    src = [np.ones(case.shape, dtype=case.dtype) for _ in range(CHUNKS)]
    row_chunk = np.arange(case.batch) % CHUNKS
    masks = [row_chunk == k for k in range(CHUNKS)]
    pooled = np.empty(case.shape, dtype=case.dtype)

    def assemble(out: np.ndarray) -> None:
        for k in range(CHUNKS):
            out[masks[k]] = src[k][masks[k]]

    def fresh() -> None:
        assemble(np.empty(case.shape, dtype=case.dtype))

    def reuse() -> None:
        assemble(pooled)

    # Warm both arms: the pooled buffer's own first touch must not be timed.
    fresh()
    reuse()
    return _median_ms(fresh, iters), _median_ms(reuse, iters)


def _time_h2d(src: torch.Tensor, iters: int) -> float:
    """Median ms for one ``src -> cuda`` copy, CUDA-event timed.

    Events, not the wall clock: ``non_blocking=True`` returns before the copy lands, so
    a host-side timer would measure the launch and not the transfer.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        src.to("cuda", non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def probe_h2d(case: Case, iters: int) -> tuple[float, float]:
    """``(pageable_ms, pinned_ms)`` for one H2D copy of a batch."""
    import torch

    dtype = getattr(torch, str(case.dtype))
    pageable = torch.empty(case.shape, dtype=dtype)
    pinned = torch.empty(case.shape, dtype=dtype, pin_memory=True)

    _time_h2d(pageable, iters)  # warm the context and the caching allocator
    _time_h2d(pinned, iters)
    return _time_h2d(pageable, iters), _time_h2d(pinned, iters)


class Roundtrip(NamedTuple):
    """Whether pinned-ness survives the pool's hand-out path, and what it costs."""

    shares: bool  # the numpy base aliases the pinned tensor (no silent copy)
    full_pinned: bool  # torch sees a full-batch view as pinned
    prefix_pinned: bool  # ...and a ragged `base[:k]` prefix too
    ms: float  # H2D through the full path
    pageable_ms: float
    pinned_ms: float

    @property
    def verdict(self) -> str:
        """Which reference the round-trip timing actually matches.

        ``is_pinned()`` alone is not enough to trust: the failure that matters is a
        ``non_blocking`` copy silently degrading to a synchronous pageable one, and that
        shows up in the clock, not the flag. Require both to agree.
        """
        if not self.shares:
            return "BROKEN: .numpy() copied"
        midpoint = (self.pinned_ms + self.pageable_ms) / 2
        fast = self.ms <= midpoint
        if self.full_pinned and self.prefix_pinned and fast:
            return "ok"
        if fast:
            return "fast but flag says pageable"
        return "DEAD: degrades to pageable"


def probe_roundtrip(case: Case, iters: int) -> Roundtrip:
    """Does host memory stay pinned along the pool's actual hand-out path?

    The pool cannot call ``torch.empty(pin_memory=True)`` itself -- the core is numpy and
    imports no framework -- so pinning would arrive via an allocator injected by the torch
    adapter, and every batch would travel::

        torch pinned tensor -> .numpy() -> pool base -> base[:k] view -> DLPack -> .to(cuda)

    Three ways that silently fails, all checked here: ``.numpy()` could copy instead of
    alias (the pool would then pin memory nobody reads from); CUDA could fail to recognise
    the pages through the round-trip, so ``non_blocking=True`` degrades to a synchronous
    pageable copy with no error; or the ragged-tail ``base[:k]`` prefix could behave
    differently from the full-batch view. If this arm does not come back ``ok``, the
    injected-allocator design is dead and pinning needs a torch-owned buffer instead.
    """
    import torch

    dtype = getattr(torch, str(case.dtype))
    owner = torch.empty(case.shape, dtype=dtype, pin_memory=True)
    base = owner.numpy()  # must alias, not copy
    shares = base.__array_interface__["data"][0] == owner.data_ptr()

    full = torch.from_dlpack(base[: case.batch])
    prefix = torch.from_dlpack(base[: max(1, case.batch // 2)])  # the ragged tail
    pageable = torch.empty(case.shape, dtype=dtype)

    _time_h2d(full, iters)  # warm
    return Roundtrip(
        shares=shares,
        full_pinned=full.is_pinned(),
        prefix_pinned=prefix.is_pinned(),
        ms=_time_h2d(full, iters),
        pageable_ms=_time_h2d(pageable, iters),
        pinned_ms=_time_h2d(owner, iters),
    )


class Load(NamedTuple):
    """A fixed synthetic training step: ``reps`` matmuls of ``a @ b`` on the GPU.

    Built **once** and shared by every case so the compute baseline is identical across
    rows. Sizing it per case would let the measured single-matmul time jitter change
    ``reps`` between rows, drifting the baseline and making the absolute wall times
    incomparable (the within-row page-vs-pin delta would stay valid, but nothing else).
    """

    a: torch.Tensor
    b: torch.Tensor
    reps: int


def make_load(compute_ms: float) -> Load:
    """Size the synthetic compute load to roughly ``compute_ms`` per step."""
    import torch

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    return Load(a, b, max(1, round(compute_ms / _gpu_ms(lambda: a @ b))))


def probe_overlap(case: Case, iters: int, load: Load) -> tuple[float, float, float]:
    """``(compute_only_ms, pageable_ms, pinned_ms)`` wall time for a copy *issued
    alongside compute*.

    The copy goes on a side stream while ``load``'s matmuls occupy the default stream --
    a stand-in for the training step the loader is meant to hide behind. A pinned source
    DMAs straight off the page-locked pages and overlaps; a pageable one is staged
    through a driver bounce buffer and serializes.

    ``compute_only_ms`` is the floor: the same step with no copy at all. An arm sitting
    at the floor has its transfer fully hidden, so pinning cannot help it -- that
    comparison is the point of the arm, and it needs the floor reported, not assumed.
    """
    import torch

    dtype = getattr(torch, str(case.dtype))
    pageable = torch.empty(case.shape, dtype=dtype)
    pinned = torch.empty(case.shape, dtype=dtype, pin_memory=True)
    stream = torch.cuda.Stream()

    def run(src: torch.Tensor | None) -> float:
        def once() -> None:
            if src is not None:
                with torch.cuda.stream(stream):
                    src.to("cuda", non_blocking=True)
            for _ in range(load.reps):
                load.a @ load.b
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        return _median_ms(once, iters)

    run(pageable)
    run(pinned)
    return run(None), run(pageable), run(pinned)


def _gpu_ms(fn: Callable[[], object]) -> float:
    """Median GPU time of one op, in ms -- used to size the synthetic compute load."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return max(statistics.median(samples), 1e-3)


def _device_name() -> str:
    """Assert a usable CUDA device and name it, or exit with the reason.

    A silent CPU fallback would report a meaningless "pinning does nothing", so the GPU
    arms fail fast rather than degrade.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise SystemExit("the GPU arms need PyTorch: uv sync --extra torch") from exc
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- run the GPU arms on a GPU box")
    name: str = torch.cuda.get_device_name(0)
    return name


def _report(header: str, columns: str, rows: list[str]) -> None:
    print(header)
    print(columns)
    for row in rows:
        print(row)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--arms",
        default="alloc",
        help="comma-separated: alloc,h2d,overlap,roundtrip (or 'all'). Default: alloc.",
    )
    ap.add_argument("--iters", type=int, default=50, help="timed repetitions per measurement")
    ap.add_argument(
        "--compute-ms",
        type=float,
        default=20.0,
        help="synthetic per-step GPU compute the overlap arm hides the copy behind",
    )
    args = ap.parse_args()
    arms = set(ARMS) if args.arms == "all" else set(args.arms.split(","))
    unknown = arms - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arm(s): {sorted(unknown)}; choose from {list(ARMS)}")

    if arms - {"alloc"}:
        print(f"device: {_device_name()}\n")

    if "alloc" in arms:
        rows = []
        for case in CASES:
            iters = args.iters if case.mib < 50 else max(10, args.iters // 5)
            fresh, reuse = probe_alloc(case, iters)
            pct = 100 * (fresh - reuse) / fresh if fresh else 0.0
            rows.append(
                f"{case.name:<28}{case.mib:>9.1f}{fresh:>10.3f}{reuse:>10.3f}"
                f"{fresh - reuse:>10.3f}{pct:>7.1f}%"
            )
        _report(
            "alloc -- batch assembly, fresh np.empty vs reused buffer",
            f"{'case':<28}{'MiB':>9}{'fresh ms':>10}{'reuse ms':>10}{'saved':>10}{'saved':>8}",
            rows,
        )

    if "h2d" in arms:
        rows = []
        for case in CASES:
            page_ms, pin_ms = probe_h2d(case, args.iters)
            gib = case.mib / 1024
            rows.append(
                f"{case.name:<28}{case.mib:>9.1f}{page_ms:>10.3f}{pin_ms:>10.3f}"
                f"{gib / (page_ms / 1e3):>11.1f}{gib / (pin_ms / 1e3):>10.1f}"
            )
        _report(
            "h2d -- one batch copied to device, CUDA-event timed",
            f"{'case':<28}{'MiB':>9}{'page ms':>10}{'pin ms':>10}{'GB/s page':>11}{'GB/s pin':>10}",
            rows,
        )

    if "overlap" in arms:
        load = make_load(args.compute_ms)  # once, so every row shares one baseline
        rows = []
        for case in CASES:
            base_ms, page_ms, pin_ms = probe_overlap(case, max(10, args.iters // 5), load)
            pct = 100 * (page_ms - pin_ms) / page_ms if page_ms else 0.0
            rows.append(
                f"{case.name:<28}{case.mib:>9.1f}{base_ms:>10.3f}{page_ms:>10.3f}"
                f"{pin_ms:>10.3f}{page_ms - pin_ms:>10.3f}{pct:>7.1f}%"
            )
        _report(
            f"overlap -- copy issued against ~{args.compute_ms:.0f}ms of compute (wall time);"
            " an arm at 'compute' has its copy fully hidden",
            f"{'case':<28}{'MiB':>9}{'compute':>10}{'page ms':>10}{'pin ms':>10}"
            f"{'saved':>10}{'saved':>8}",
            rows,
        )

    if "roundtrip" in arms:
        rows = []
        for case in CASES:
            rt = probe_roundtrip(case, args.iters)
            flags = f"{str(rt.shares):>7}{str(rt.full_pinned):>8}{str(rt.prefix_pinned):>8}"
            rows.append(
                f"{case.name:<28}{case.mib:>9.1f}{flags}{rt.ms:>10.3f}"
                f"{rt.pinned_ms:>10.3f}{rt.pageable_ms:>10.3f}  {rt.verdict}"
            )
        _report(
            "roundtrip -- pinned tensor -> .numpy() -> base[:k] view -> DLPack -> .to(cuda);"
            " does pinning survive the pool's hand-out path?",
            f"{'case':<28}{'MiB':>9}{'alias':>7}{'pinned':>8}{'ragged':>8}"
            f"{'via ms':>10}{'pin ms':>10}{'page ms':>10}  verdict",
            rows,
        )


if __name__ == "__main__":
    main()

"""Thin, optional framework adapters: numpy ``Batch`` -> torch / JAX / TF via DLPack.

The core (:mod:`insitubatch.source`) yields numpy :class:`Batch` objects and imports
no framework. These adapters convert a batch's arrays to a framework's tensors with
DLPack (zero-copy on CPU where the framework supports it). The wrapping differs per
ecosystem -- there is no single cross-framework "dataset" base class:

  * **torch** has one. ``DataLoader`` requires a ``Dataset`` / ``IterableDataset``
    subclass (it ``isinstance``-checks), so :func:`as_torch` wraps the stream in one::

        DataLoader(as_torch(ds), batch_size=None, num_workers=0)

    ``batch_size=None`` (the stream already yields assembled batches) and
    ``num_workers=0`` (parallelism is in our event loop; forking re-introduces the
    redundant-read problem).
  * **JAX** has none -- it is loader-agnostic. Iterate the dataset and call
    :func:`to_jax` per batch.
  * **TF** adapts via a factory, not a base class: :func:`as_tf_dataset` wraps the
    stream in ``tf.data.Dataset.from_generator``.

Each framework is imported lazily inside its function, so importing this module costs
nothing and a missing framework raises a clear, actionable error. ``sample_indices``
(provenance) stays on the numpy ``Batch``; only the model-input arrays are converted.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .buffers import HostAllocator, aligned_empty
from .source import _SplitView
from .types import Batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # annotations only; these are optional at runtime
    import tensorflow as tf
    import torch
    from torch.utils.data import IterableDataset


def _missing(name: str, extra: str) -> ImportError:
    return ImportError(
        f"insitubatch.frameworks needs {name}; install it with: pip install 'insitubatch[{extra}]'"
    )


_DEFAULT_PIN_FRACTION = 8  # of physical RAM
_FALLBACK_PIN_BUDGET = 1 << 30  # where physical RAM is not discoverable


def _default_pin_budget() -> int:
    """An eighth of physical RAM, or 1 GiB where that cannot be read.

    Page-locked memory is a *global* resource the kernel cannot reclaim, so over-pinning
    degrades the whole machine rather than just this process. The default is sized to never
    fire for weather or ViT payloads and to catch a microscopy-sized pool on a small host.
    """
    try:
        total: int = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
        return _FALLBACK_PIN_BUDGET
    return total // _DEFAULT_PIN_FRACTION


def pinned_allocator(budget_bytes: int | None = None) -> HostAllocator:
    """A host allocator handing out page-locked buffers, up to ``budget_bytes``.

    Past the budget it returns ordinary aligned heap memory instead. Degrading rather than
    raising or blocking is deliberate: blocking could deadlock (a consumer legitimately holds
    batches, so the producer could wait forever), raising would turn a performance feature
    into a crash, and the degraded state is simply what the loader shipped before pinning
    existed. It warns once so "pinning did nothing" is diagnosable rather than mysterious.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise _missing("PyTorch", "torch") from exc

    budget = _default_pin_budget() if budget_bytes is None else int(budget_bytes)
    used = 0
    warned = False

    def allocate(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        nonlocal used, warned
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if used + nbytes > budget:
            if not warned:
                warned = True
                logger.warning(
                    "pinned-buffer budget exhausted (%.0f MiB used, %.0f MiB needed for the "
                    "next buffer, budget %.0f MiB) -- further batch buffers are pageable, so "
                    "H2D transfers stop overlapping. Raise pin_budget_bytes or lower "
                    "batch_size/prefetch_depth.",
                    used / 2**20,
                    nbytes / 2**20,
                    budget / 2**20,
                )
            return aligned_empty(shape, dtype)
        # numpy view of a page-locked tensor: aliases (never copies), and CUDA still
        # recognises the pages through the DLPack round-trip back into torch, which is what
        # keeps `non_blocking` genuinely asynchronous. Page-locked memory is page-aligned, so
        # the pool's 128-byte XLA alignment comes free.
        pinned = torch.empty(tuple(shape), dtype=getattr(torch, str(dtype)), pin_memory=True)
        used += nbytes
        return pinned.numpy()

    # Read back by `BatchBuffers.kind` for the epoch summary: without it there is no way to
    # confirm from a training log that `--pin` / `as_torch(device=...)` actually took effect.
    allocate.label = "pinned"  # type: ignore[attr-defined]
    return allocate


def pin_host_buffers(view: _SplitView, *, budget_bytes: int | None = None) -> None:
    """Switch a split's batch buffers to page-locked memory, for the duration of the dataset.

    Only safe when **we** issue the H2D copy -- that is, when every batch goes through
    :func:`to_torch` with ``device=`` (or :func:`as_torch`, which calls this for you). Pinning
    is what makes a ``non_blocking`` copy genuinely asynchronous, and Python refcounts cannot
    see an in-flight DMA, so pinned buffers under a *caller's own* ``non_blocking`` copy would
    let the pool recycle a buffer mid-transfer. With ``to_torch(..., device=...)`` the source
    is held until its event fires and the hazard is closed.

    Call before iterating. ``as_torch(view, device=...)`` is the one-liner for the
    ``DataLoader`` path; this is the equivalent for code that iterates the dataset itself and
    converts per batch.
    """
    view._use_host_allocator(pinned_allocator(budget_bytes))


class _InFlight:
    """Sources held until their asynchronous H2D copy has actually landed.

    The reclaim hazard pinning introduces: a ``non_blocking`` copy returns before it has
    finished reading its source, and Python refcounts cannot see an in-flight DMA, so a buffer
    whose view the consumer has dropped could be recycled and overwritten mid-copy. Pageable
    memory hides this (the driver stages through a bounce buffer, so the copy is effectively
    synchronous with respect to the host) -- pinning is what makes it real.

    The guard needs no new machinery in the pool: holding the *source arrays* until
    ``event.query()`` keeps their views referenced, so the pool's existing liveness poll
    defers reclaim on its own. Nothing crosses into the core, and ``Batch`` is untouched.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[dict[str, np.ndarray], torch.cuda.Event]] = []
        # Prune-then-append is a read-modify-write, and this object is process-wide: two
        # threads converting batches at once (two DataLoaders, `zip(ds.train, ds.val)`) both
        # run it. Without the lock a thread can finish its comprehension, be preempted before
        # the slice assignment lands, and then overwrite a hold another thread appended in the
        # gap -- releasing a buffer whose DMA is still reading it. The window is only the few
        # bytecodes between the comprehension and the store, and it does not reproduce
        # unaided (the comprehension iterates the live list by index, so it picks up a
        # concurrent append); widened with a sleep at exactly that point it loses a live hold
        # 20 times out of 20. One uncontended acquire per batch against a megabyte-scale copy
        # is not a cost worth reasoning about, and the failure it prevents is silent.
        self._lock = threading.Lock()

    def hold(self, arrays: dict[str, np.ndarray], event: torch.cuda.Event) -> None:
        """Retire finished holds, then take this one -- in that order, never the reverse.

        Pruning after appending lets a hold evaporate in the microseconds between issuing the
        copy and testing it: a small or fast DMA can already be complete, and the batch we were
        asked to protect is released before the caller has done anything with it. Even when
        that is *safe* (a complete event means the copy landed) it makes reuse depend on a
        race, which is untestable by construction -- the buffer is retained or not depending on
        who wins. Retiring first means a fresh hold always survives until at least the next
        transfer, so the guarantee is structural rather than probabilistic. The cost is
        deferring one buffer's reclaim by one hand-out.

        Both steps happen under the lock, because together they are one atomic replacement of
        the pending set -- see :meth:`__init__`.
        """
        with self._lock:
            self._pending[:] = [(a, e) for a, e in self._pending if not e.query()]
            self._pending.append((arrays, event))


_in_flight = _InFlight()
"""Process-wide, because CUDA streams are.

Bounded: every hand-out retires the holds whose copies have landed, so the set is whatever is
genuinely in flight plus the one deliberately-deferred entry. The tail is that single entry --
after the last transfer of a run, one batch's source arrays stay referenced until either the
next transfer prunes them or the process exits. Bounded at one batch, and not reachable from
``InSituDataset.close()``: this lives in the torch adapter, the core cannot import it, and
draining a process-wide set on one dataset's close would be wrong anyway.
"""


def to_torch(batch: Batch, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    """Convert a numpy ``Batch`` to a dict of torch tensors (DLPack; zero-copy on CPU).

    With ``device`` set, the copy is issued here rather than by the caller, and the batch's
    source arrays are held until it lands. That is what makes pinned batch buffers safe to
    recycle -- see :class:`_InFlight`. Without it the behaviour is unchanged: CPU tensors
    aliasing the batch, and the caller's own ``.to(...)`` afterwards.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise _missing("PyTorch", "torch") from exc
    tensors = {k: torch.from_dlpack(v) for k, v in batch.arrays.items()}
    if device is None:
        return tensors
    out = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
    if torch.device(device).type != "cuda":
        # No async DMA to guard: a CPU target copies synchronously (or not at all), and
        # torch.cuda.Event() raises outright on a machine without a driver.
        return out
    # Bind the event to the *target* device: torch.cuda.Event() and record() attach to the
    # current device's current stream, so for `cuda:1` without a set_device the event would
    # belong to device 0, query True immediately, and release a buffer mid-DMA.
    with torch.cuda.device(device):
        event = torch.cuda.Event()
        event.record()
    _in_flight.hold(batch.arrays, event)
    return out


def to_jax(batch: Batch) -> dict[str, Any]:
    """Convert a numpy ``Batch`` to a dict of ``jax.Array`` (DLPack).

    Zero-copy only when the batch buffer sits on a 128-byte boundary: XLA:CPU requires that
    alignment and silently falls back to a copy otherwise. numpy guarantees only 16, so with
    today's per-batch ``np.empty`` roughly half of batches are copied (measured 20/40 —
    ``bench/probe_batch_buffers.py --arms jax``). Correct either way, just not free.
    """
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - jax-less installs
        raise _missing("JAX", "jax") from exc
    return {k: jnp.from_dlpack(v) for k, v in batch.arrays.items()}


def to_tf(batch: Batch) -> dict[str, Any]:
    """Convert a numpy ``Batch`` to a dict of ``tf.Tensor`` (one CPU copy per variable).

    Unlike torch/JAX -- whose array-accepting ``from_dlpack`` manages the exported buffer's
    lifetime correctly -- TensorFlow only exposes the *experimental* ``from_dlpack(capsule)``,
    which mishandles ownership of the exported numpy buffer: under the concurrent allocation
    of the prefetch decode threads it double-frees that buffer and aborts the process
    (SIGABRT, no message). ``convert_to_tensor`` copies into a TF-owned tensor instead, so TF
    never touches insitu-managed memory; the batch is already an owned array, so this is a
    single CPU copy. (torch stays zero-copy, JAX when aligned -- see :func:`to_jax`; TF's is a
    DLPack limitation, not ours.)
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - tf-less installs
        raise _missing("TensorFlow", "tf") from exc
    return {k: tf.convert_to_tensor(v) for k, v in batch.arrays.items()}


def as_torch(
    view: _SplitView,
    *,
    device: str | torch.device | None = None,
    pin_budget_bytes: int | None = None,
) -> IterableDataset:
    """Wrap a split view (e.g. ``ds.train``) as a torch ``IterableDataset`` for ``DataLoader``.

    Each yielded item is a ``dict[str, torch.Tensor]`` (via :func:`to_torch`). Use
    ``DataLoader(as_torch(ds.train), batch_size=None, num_workers=0)``.

    Passing ``device`` moves batches here instead of in the training loop, and switches the
    batch buffers to **page-locked** memory. The two are one option rather than two on
    purpose: pinning is what makes an H2D copy genuinely asynchronous, so it is only safe
    when whoever issues the copy also knows when it landed. A ``pin_memory=True`` flag the
    caller could set without ``device`` would hand out pinned buffers under the caller's own
    ``non_blocking`` copy and reintroduce a use-after-recycle we would then have to document
    instead of prevent.

    Worth it only above ~32 MiB per batch. Measured on an L4, pinning takes a 37 MiB batch's
    transfer from +4.0 ms to +0.3 ms against a 25 ms step, and a weather-sized batch from
    +1.5 ms to +0.7 ms -- real but marginal. ``pin_budget_bytes`` caps the page-locked total
    (default: an eighth of RAM); past it buffers are pageable again, with a warning.
    """
    try:
        from torch.utils.data import IterableDataset
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise _missing("PyTorch", "torch") from exc

    class _TorchStream(IterableDataset):
        def __init__(self, stream: _SplitView) -> None:
            self._stream = stream
            # Only a CUDA target: page-locking is a CUDA allocation, so pinning for a CPU
            # device would raise "no NVIDIA driver" from inside the producer thread on the
            # first gather -- and buys nothing even where a driver exists.
            if device is not None and torch.device(device).type == "cuda":
                pin_host_buffers(stream, budget_bytes=pin_budget_bytes)

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for batch in self._stream:
                yield to_torch(batch, device=device)

    return _TorchStream(view)


def as_tf_dataset(view: _SplitView, *, prefetch: int = 2) -> tf.data.Dataset:
    """Wrap a split view (e.g. ``ds.val``) as a ``tf.data.Dataset`` via ``from_generator``.

    ``output_signature`` is inferred from the view's geometries: each variable is
    ``(None, *inner)`` (None = the variable last-batch size) with the variable's dtype.
    Both ``from_generator`` here and :func:`to_tf` copy into the TF runtime -- TF has no
    reliable zero-copy path from insitu's buffers (its experimental DLPack mishandles buffer
    ownership; see :func:`to_tf`). Call :func:`to_tf` on the raw stream when you want plain
    ``dict[str, tf.Tensor]`` batches instead of a ``tf.data.Dataset``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - tf-less installs
        raise _missing("TensorFlow", "tf") from exc

    signature = {
        name: tf.TensorSpec(shape=(None, *geom.inner_shape), dtype=geom.dtype)
        for name, geom in view.geometries.items()
    }

    def gen() -> Iterator[dict[str, Any]]:
        for batch in view:
            yield batch.arrays

    tfds = tf.data.Dataset.from_generator(gen, output_signature=signature)
    return tfds.prefetch(prefetch) if prefetch else tfds

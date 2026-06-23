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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .source import InSituDataset
from .types import Batch

if TYPE_CHECKING:  # annotations only; these are optional at runtime
    import tensorflow as tf
    import torch
    from torch.utils.data import IterableDataset


def _missing(name: str, extra: str) -> ImportError:
    return ImportError(
        f"insitubatch.frameworks needs {name}; install it with: pip install 'insitubatch[{extra}]'"
    )


def to_torch(batch: Batch) -> dict[str, torch.Tensor]:
    """Convert a numpy ``Batch`` to a dict of torch tensors (DLPack; zero-copy on CPU)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise _missing("PyTorch", "torch") from exc
    return {k: torch.from_dlpack(v) for k, v in batch.arrays.items()}


def to_jax(batch: Batch) -> dict[str, Any]:
    """Convert a numpy ``Batch`` to a dict of ``jax.Array`` (DLPack)."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - jax-less installs
        raise _missing("JAX", "jax") from exc
    return {k: jnp.from_dlpack(v) for k, v in batch.arrays.items()}


def to_tf(batch: Batch) -> dict[str, Any]:
    """Convert a numpy ``Batch`` to a dict of ``tf.Tensor`` (DLPack; zero-copy on CPU)."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - tf-less installs
        raise _missing("TensorFlow", "tf") from exc
    return {k: tf.experimental.dlpack.from_dlpack(v.__dlpack__()) for k, v in batch.arrays.items()}


def as_torch(ds: InSituDataset) -> IterableDataset:
    """Wrap an :class:`InSituDataset` as a torch ``IterableDataset`` for ``DataLoader``.

    Each yielded item is a ``dict[str, torch.Tensor]`` (via :func:`to_torch`). Use
    ``DataLoader(as_torch(ds), batch_size=None, num_workers=0)``.
    """
    try:
        from torch.utils.data import IterableDataset
    except ImportError as exc:  # pragma: no cover - torch-less installs
        raise _missing("PyTorch", "torch") from exc

    class _TorchStream(IterableDataset):
        def __init__(self, stream: InSituDataset) -> None:
            self._stream = stream

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            for batch in self._stream:
                yield to_torch(batch)

    return _TorchStream(ds)


def as_tf_dataset(ds: InSituDataset, *, prefetch: int = 2) -> tf.data.Dataset:
    """Wrap an :class:`InSituDataset` as a ``tf.data.Dataset`` via ``from_generator``.

    ``output_signature`` is inferred from the dataset's geometries: each variable is
    ``(None, *inner)`` (None = the variable last-batch size) with the variable's dtype.
    Note ``from_generator`` *copies* into the TF runtime; for a zero-copy handoff call
    :func:`to_tf` on the raw stream instead.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - tf-less installs
        raise _missing("TensorFlow", "tf") from exc

    signature = {
        name: tf.TensorSpec(shape=(None, *geom.inner_shape), dtype=geom.dtype)
        for name, geom in ds.geometries.items()
    }

    def gen() -> Iterator[dict[str, Any]]:
        for batch in ds:
            yield batch.arrays

    tfds = tf.data.Dataset.from_generator(gen, output_signature=signature)
    return tfds.prefetch(prefetch) if prefetch else tfds

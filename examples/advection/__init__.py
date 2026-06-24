"""Advected-field 24-hour forecast: one insitu dataset, three framework training loops.

The same :class:`~insitubatch.source.InSituDataset` (a numpy ``Batch``) feeds torch, JAX,
and TensorFlow via the thin DLPack adapters in :mod:`insitubatch.frameworks` -- see
``train_torch.py`` / ``train_jax.py`` / ``train_tf.py`` (near-identical but for the
framework calls). ``data.py`` builds the store, the dataset, and the shared eval.
"""

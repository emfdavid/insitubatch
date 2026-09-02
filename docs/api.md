# API reference

The public surface is the top-level `insitubatch` package — **everything in its
`__all__`**. `InSituDataset` and the framework adapters are re-exported there, so
import them from the package root (`from insitubatch import InSituDataset, to_torch`),
not from submodules. The adapters are optional: they import torch / JAX / TF **lazily**,
only when called, so importing `insitubatch` never pulls a framework in.

## `insitubatch`

::: insitubatch
    options:
      show_root_heading: false

## `InSituDataset`

::: insitubatch.InSituDataset

## Debugging

::: insitubatch.debug
    options:
      show_root_heading: false
      members:
        - print_debug_info
        - debug_info

## The static report

The entry points are the two `InSituDataset` methods above — `describe()` for the dict,
`print_summary()` for the rendering. These are the shapes `describe()` returns.

::: insitubatch.summary
    options:
      show_root_heading: false
      members:
        - DatasetReport
        - VariableReport
        - ConfigReport
        - MemoryReport
        - Note

## Framework adapters

::: insitubatch.frameworks
    options:
      show_root_heading: false
      members:
        - to_torch
        - as_torch
        - to_jax
        - to_tf
        - as_tf_dataset

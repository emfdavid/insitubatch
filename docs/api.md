# API reference

The public surface is everything re-exported from the top-level `insitubatch`
package (its `__all__`), plus the framework-neutral `InSituDataset` source from
`insitubatch.source` and the optional DLPack adapters in `insitubatch.frameworks`.

## `insitubatch`

::: insitubatch
    options:
      show_root_heading: false

## `insitubatch.source`

::: insitubatch.source.InSituDataset

## `insitubatch.frameworks`

::: insitubatch.frameworks
    options:
      show_root_heading: false
      members:
        - to_torch
        - as_torch
        - to_jax
        - to_tf
        - as_tf_dataset

# API reference

The public surface is everything re-exported from the top-level `insitubatch`
package, plus `InSituDataset` (the torch source) from `insitubatch.source`.

## `insitubatch`

::: insitubatch
    options:
      show_root_heading: false
      members:
        - open_geometries
        - store_from_url
        - ensure_local_dir
        - split_by_chunk
        - SplitManifest
        - SplitName
        - ArrayGeometry
        - Batch
        - ChunkRead
        - DecodedChunk
        - StoredChunkRead
        - build_read_plan
        - build_stored_chunk_reads
        - dedup_ratio
        - ReadPlan
        - block_shuffled_order
        - sequential_order
        - chunk_permutation
        - shuffle_quality
        - Scheduler
        - SchedulerConfig
        - ChunkPool
        - AsyncChunkReader
        - IOConfig
        - StandardScaler
        - fit_standard_scaler
        - ChunkTransform
        - BatchTransform

## `insitubatch.source`

::: insitubatch.source.InSituDataset

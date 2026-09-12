"""Statistical downscaling on a dynamical.org Icechunk archive -- 1440 samples per chunk.

The **deep-chunk** companion to ``examples/advection``. A sample is one hourly global field of
surface solar irradiance, and on ``noaa-gfs-analysis`` 1440 of them live inside a single stored
chunk: the geometry where fetch-and-decode-once is worth the most, and where a per-sample
``__getitem__`` would re-read the same object for every sample it holds.

The task is single-variable by necessity, and that is the lesson. A stored chunk here is
**6.87 GiB resident**, so the residency floor is ~13.7 GiB *per variable* -- set by the shard,
not by the batch size. ``print_summary()`` reports it before a byte moves, which is the only
reason the archive is approachable at all.

``data.py`` builds the store (synthetic irradiance, or the real GFS analysis), the dataset and
the shared eval; ``train_torch.py`` trains a tiny CNN that beats bilinear upsampling by reading
structure the coarse field's neighbourhood implies.
"""

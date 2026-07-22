# Pangeo Showcase — submission draft

Venue notes (from the announcement format): the organizers prepend
**"Pangeo Showcase:"** to the title, so supply only the descriptive part. Talk is
~15 min presentation + 10–30 min discussion + community check-in. Abstracts run
~250 words / 3 paragraphs, technical-but-accessible, structured *why it matters →
the solution → specifics*. Submit via the Google Form linked from
<https://pangeo.io/showcase>.

ORCID 0009-0000-2804-7191
@emfdavid.bsky.social

## Title

**insitubatch: Streaming ML Batches from Cloud Zarr, No Reshard**

Alternatives:
- insitubatch: chunk-aware ML batching that scales with chunks, not samples
- Train in Place: a Streaming Batch Loader for Cloud-Native Zarr

## Abstract

The Pangeo community already has an elegant way to *define* machine-learning
batches over an n-dimensional archive: xbatcher generates xarray-native ndim
batches with no reshard required. What has stayed hard is the *engine* underneath.
However a batch is defined, feeding it to a GPU at scale usually runs through the
classic PyTorch `DataLoader`, which spreads work across worker *processes*, each
running a synchronous `__getitem__` — so there is no shared chunk cache across
workers, no way to drive async object-store IO, and the same Zarr chunk is
re-decoded once per worker whose samples land in it. The common escape hatch —
resharding the whole store into a sample-oriented format (MDS, WebDataset, tar) —
trades that for a full ETL copy that throws away the chunk locality the archive
already has.

But the IO half of this problem is now solved: obstore, Zarr v3's async store, and
Icechunk already saturate the NIC. `insitubatch` is the loader-orchestration layer
built directly on top of that solved IO. It keeps the same in-place, ndim batch
semantics but stays at the numpy/tensor level on a *different* engine: one async
event loop streams stored chunks under a single concurrency budget into a bounded
pool that doubles as a decode-once cache. Splits, shuffle, and batches live in
coordinate space over the existing Zarr — no second copy — and the Python hot path
scales with **chunks, not samples**, at memory bounded by a residency budget rather
than the working set. You still define windows in xarray; labels and coordinates
ride along as planning metadata rather than through the hot path. Handoff to
PyTorch, JAX, or TensorFlow is a thin DLPack adapter — `num_workers=0`, batches
arrive framework-ready.

The sample axis is a *role*, not a fixed dimension, so one engine spans domains
without special-casing. I'll walk through the runnable examples: a windowed
24-hour ERA5 forecast trained identically in PyTorch, JAX, and TensorFlow;
per-plane cell segmentation over the Z axis of an OME-NGFF microscopy volume,
co-batching two variables chunked differently; and real telescope archives that
were never Zarr — Hubble WFC3/IR frames and SDSS spectra indexed straight from
their FITS objects as VirtualiZarr references and streamed in place, no pixels
copied. Along the way I'll show where insitubatch matches a hand-tuned worker/
xbatcher pool, where it pulls far ahead (shared chunks decoded once), where it
honestly trails — and the flat-memory curve where the dense alternative OOMs. It's
built to ride the ecosystem's roadmap, including the GPU-Zarr codec path shown in
an earlier showcase.

## Speaker bio

David Stuebe is a staff machine-learning engineer at ThinkLabs AI, building ML
infrastructure for electric grid utilities. With a background in
physical oceanography (MIT/WHOI Joint Program) and years of operational
cloud-native weather-data work — including the Kerchunk/Zarr optimizations for NODD
GRIB forecasts presented in an earlier Pangeo Showcase — David works on the data
plumbing that keeps large models fed directly from cloud archives. insitubatch
grows out of that work: the loader-orchestration layer on top of solved async
Zarr IO.

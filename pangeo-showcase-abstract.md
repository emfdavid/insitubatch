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

**insitubatch: Streaming ML Batches from Cloud Zarr Without Resharding**

## Abstract

Between a dataset small enough to hold in memory and one worth building a
purpose-built ETL pipeline for, there is a wide middle: archives too large to load,
not yours to rewrite, or read in ways that keep changing. The usual move is to
reshard into a sample-oriented format, a full ETL copy that throws away the
archive's chunk locality, must be rebuilt as the archive grows, and only pays for
itself if you run the same job over the same data many times. Keeping the data in
place runs into the loader: PyTorch's `DataLoader` puts parallelism in worker
processes, so each carries its own cache and IO budget, and a chunk feeding four
workers is fetched and decoded four times.

The IO is no longer the hard part: obstore and Icechunk over Zarr v3's async store
already saturate the NIC. `insitubatch` is the loader-orchestration layer on top of
it, reading any zarr `Store`, whether obstore, fsspec or Icechunk backs it. It plans
reads chunk-first, so Python work scales with the **chunks a batch
touches, not the samples it contains**, and one async event loop streams them under
a single concurrency budget into a bounded pool that is residency tier and
decode-once cache at once. A stored chunk is fetched and decoded exactly once,
however many samples, batches or epochs reference it. Splits and shuffle live in
coordinate space over the existing store, so there is no second copy of the archive,
and memory is a budget you set rather than the working set.

The same ETL amortization argument applies to process parallelism: a worker pool
earns back its startup over a long training run but never over a single inference
pass. With no processes to launch, insitubatch reaches its first batch while a
worker stack is still starting, which makes it as much an inference loader as a
training loader. Handoff to PyTorch, JAX or TensorFlow is a thin DLPack adapter,
and the sample axis is a role rather than a fixed dimension, so one engine spans
domains: ERA5 forecasting, OME-NGFF microscopy segmentation, and Hubble and SDSS
telescope archives that were never Zarr, streamed in place via VirtualiZarr. I'll
be equally clear about the price: a block-local shuffle rather than a global one,
and the regimes where a tuned worker pool still wins.

## Speaker bio

David Stuebe is a staff machine-learning engineer at ThinkLabs AI, building ML
infrastructure for electric grid utilities. With a background in
physical oceanography (MIT/WHOI Joint Program) and years of operational
cloud-native weather-data work, including the Kerchunk/Zarr optimizations for NODD
GRIB forecasts presented in an earlier Pangeo Showcase, David works on the data
plumbing that keeps large models fed directly from cloud archives. insitubatch
grows out of that work: the loader-orchestration layer on top of async
Zarr IO.

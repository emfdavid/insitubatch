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

## Material to fold in (from the Ilan Gold / annbatch work, PR #19)

Backing links (Pangeo abstracts render Markdown links, and long ones are fine —
the Icechunk showcase runs ~450 words with a bullet list):

- annbatch paper — [arXiv:2604.01949](https://arxiv.org/abs/2604.01949) (scverse;
  Gold et al.), docs <https://annbatch.readthedocs.io>
- the read-once/sample-once contract + archive→batch figure —
  [docs/architecture.md#read-once-and-sample-once](https://github.com/emfdavid/insitubatch/blob/main/docs/architecture.md#read-once-and-sample-once)
- the shuffle convergence argument —
  [docs/architecture.md#why-the-block-local-shuffle-is-enough](https://github.com/emfdavid/insitubatch/blob/main/docs/architecture.md#why-the-block-local-shuffle-is-enough)
- the neighbor table (annbatch row) —
  [DESIGN.md#what-it-is-by-contrast](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md#what-it-is-by-contrast)

The two ideas worth adding:

1. **annbatch as the sharp mirror.** Same premise (loaders, not GPUs, are the
   bottleneck; large contiguous reads feed the accelerator), **opposite bet on
   where batch randomness comes from.** annbatch buys a clean *global* shuffle by
   rewriting a pre-randomized Zarr copy and streaming big slices through an
   in-memory buffer that dissolves each chunk's rows across many batches — the
   currency is a one-time rewrite (which also freezes the sampling policy into byte
   order). insitubatch refuses the rewrite and pays with an *approximate* shuffle
   instead. Two points on one spectrum, not competitors: annbatch owns
   *rewrite-is-cheap, 1-D observation rows, local disk* (single-cell anndata);
   insitubatch owns *immutable, remote, PB-scale, n-D-windowed, multi-variable*
   archives you can't or won't rewrite. Both carry a single bounded-randomness knob
   (their buffer `m` ≈ our `block_chunks`).
2. **The honest shuffle compromise (the interesting part).** An exact global
   shuffle wants a random chunk per sample — which defeats reading a chunk once and
   gathering every sample in it. So: permute *which* chunks are scheduled each
   epoch, then shuffle samples within a window of `block_chunks` chunks. Over the
   epochs training actually runs the permutation re-pairs chunks, so it **converges
   toward a global shuffle at `O(block_chunks)` memory, not `O(dataset)`.** That is
   the price of training in place, stated plainly.

## Abstract — maximal draft (everything, ~350 words; cut from here)

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

The sharpest way to see the bet is against
[annbatch](https://arxiv.org/abs/2604.01949) (scverse), the nearest neighbor and
its mirror image: same premise, opposite answer to *where batch randomness comes
from.* annbatch buys a clean global shuffle by rewriting a pre-randomized Zarr copy
and streaming slices through a buffer that dissolves each chunk's rows across many
batches; insitubatch refuses the rewrite and keeps chunks whole and resident,
paying instead with an approximate, block-local shuffle that provably converges
toward a global shuffle over the epochs training runs — at `O(block_chunks)`
memory, not `O(dataset)`. Two points on one spectrum: annbatch owns
rewrite-is-cheap, 1-D observation rows, local disk (single-cell anndata);
insitubatch owns immutable, remote, PB-scale, n-D-windowed, multi-variable archives
you can't or won't rewrite.

The sample axis is a *role*, not a fixed dimension, so one engine spans domains
without special-casing. I'll walk through the runnable examples: a windowed
24-hour ERA5 forecast trained identically in PyTorch, JAX, and TensorFlow;
per-plane cell segmentation over the Z axis of an OME-NGFF microscopy volume,
co-batching two variables chunked differently; and real telescope archives that
were never Zarr — Hubble WFC3/IR frames and SDSS spectra indexed straight from
their FITS objects as VirtualiZarr references and streamed in place, no pixels
copied. Along the way I'll show where insitubatch matches a hand-tuned worker/
xbatcher pool, where it pulls far ahead (shared chunks decoded once), where it
honestly trails — and the flat-memory curve where the dense alternative OOMs.

## Abstract — punchy cut (~200 words, the submission candidate)

The Pangeo community already has an elegant way to *define* ML batches over an
n-dimensional archive — xbatcher, xarray-native, no reshard. What stays hard is the
*engine*: feeding those batches to a GPU usually runs through PyTorch's
process-based `DataLoader`, with no shared chunk cache, no async object-store IO,
and the same Zarr chunk re-decoded once per worker. The usual escape — resharding
into a sample format (MDS, WebDataset) — is a full ETL copy that throws away the
archive's chunk locality.

But the IO half is solved: obstore, Zarr v3's async store, and Icechunk already
saturate the NIC. `insitubatch` is the loader-orchestration layer on top. One async
event loop streams chunks under a single concurrency budget into a bounded pool that
doubles as a decode-once cache; splits, shuffle, and batches live in coordinate
space over the existing Zarr, so the hot path scales with **chunks, not samples**.
The sharpest contrast is [annbatch](https://arxiv.org/abs/2604.01949): it buys a
clean global shuffle by *rewriting* a randomized copy; insitubatch keeps chunks
whole and pays with a block-local shuffle that converges to global over epochs at
`O(block)` memory — training in place, no second copy.

The sample axis is a *role*, not a fixed dimension, so one engine spans domains. I'll
walk through runnable examples — ERA5 forecasting in PyTorch/JAX/TensorFlow, OME-NGFF
microscopy segmentation, and Hubble/SDSS telescope archives that were never Zarr,
streamed in place via VirtualiZarr — showing where insitubatch matches a tuned
xbatcher pool, where it pulls ahead, and where the dense alternative OOMs.

## Abstract — lead-with-the-thesis cut (~230 words, differentiation first)

`insitubatch` trains machine-learning models **in place** on n-dimensional cloud
Zarr — no reshard, no second copy of the archive. That is the whole bet, and it
buys two operational properties nothing else in the loader field delivers together.
**First, the Python hot path scales with chunks, not samples:** planning and gather
are vectorized over the deduplicated set of chunk reads, so a batch costs
`O(chunks-touched)` regardless of how many samples it holds. **Second, memory is
flat** — a single async event loop streams stored chunks under one concurrency
budget into a bounded pool that doubles as a decode-once cache, so a shared chunk is
fetched and decoded exactly once however many samples, batches, or epochs touch it,
and total memory is a residency budget you set, not the working set. There are no
worker processes (`num_workers=0`), no per-worker chunk re-decode, no ETL step
before training — the loader drives obstore / Zarr-v3 async IO directly (the
[architecture](https://emfdavid.github.io/insitubatch/architecture/)) and hands off
to your ML framework via DLPack — PyTorch, JAX, or TensorFlow. That
framework-agnostic handoff is one axis of reach; the other is domain: because the
sample axis is a *role* with flexible batch geometry, not a fixed dimension, one
engine spans weather, microscopy, and astronomy without special-casing.

This is a crowded, capable field, and insitubatch is built to be a good neighbor in
it — and honest about where it is *not* the right tool.
[xbatcher](https://xbatcher.readthedocs.io) already *defines* xarray-native ndim
batches elegantly, and at the GRIB end of the spectrum — one sample per chunk, its
worker pool's sweet spot — it works well and insitubatch has little to add;
insitubatch is the streaming *engine* that keeps up as chunks fatten to hold many
samples — not a replacement for it. [annbatch](https://arxiv.org/abs/2604.01949)
(scverse) is the sharpest mirror — same premise, opposite bet: it buys a clean
global shuffle by *rewriting* a randomized copy to local disk; insitubatch keeps
chunks whole and remote, and pays with a block-local shuffle that converges to global
over epochs at `O(block)` memory. Two points on one spectrum: annbatch owns
single-cell, rewrite-is-cheap, local disk; insitubatch owns immutable, PB-scale,
remote, n-D-windowed, multi-variable archives you can't or won't rewrite.

I'll close on the runnable
[examples](https://emfdavid.github.io/insitubatch/examples/) that prove the "one
engine" claim — ERA5 forecasting across PyTorch/JAX/TensorFlow, OME-NGFF microscopy
segmentation, and Hubble/SDSS archives that were never Zarr, streamed in place via
VirtualiZarr — and walk the
[benchmarks](https://emfdavid.github.io/insitubatch/benchmarks/) that map the
spectrum honestly: where insitubatch pulls ahead (shared chunks decoded once, flat
memory where the dense alternative OOMs) and where a tuned worker pool or a
rewrite-once loader still outperforms it.

## Speaker bio

David Stuebe is a staff machine-learning engineer at ThinkLabs AI, building ML
infrastructure for electric grid utilities. With a background in
physical oceanography (MIT/WHOI Joint Program) and years of operational
cloud-native weather-data work — including the Kerchunk/Zarr optimizations for NODD
GRIB forecasts presented in an earlier Pangeo Showcase — David works on the data
plumbing that keeps large models fed directly from cloud archives. insitubatch
grows out of that work: the loader-orchestration layer on top of solved async
Zarr IO.

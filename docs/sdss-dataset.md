# SDSS spectra as virtual references — a public dataset

A read-anonymous [Icechunk](https://icechunk.io) store of byte-range pointers into the original
SDSS DR17 FITS files. Tens of kilobytes of references, no reshard, no download step: open it and
stream 3,840 galaxy spectra.

<style>
  .spec-wrap {
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 2px;
    overflow: hidden;
    margin: 1.2em 0 0.4em;
  }
  svg.spec { display: block; width: 100%; height: 16rem; }
  .spec-trace { fill: none; stroke: var(--md-typeset-color); stroke-width: 1.1;
                vector-effect: non-scaling-stroke; stroke-linejoin: round; }
  .spec-area  { fill: var(--md-typeset-color); opacity: 0.07; stroke: none; }
  .spec-tick  { stroke: var(--md-default-fg-color--lightest); stroke-width: 1;
                vector-effect: non-scaling-stroke; }
  .spec-tl, .spec-ml { font-family: var(--md-code-font-family, monospace); font-size: 11px;
                       fill: var(--md-default-fg-color--light); letter-spacing: 0.04em; }
  .spec-tl { text-anchor: middle; }
  .spec-ml { text-anchor: start; font-size: 10.5px; }
  .spec-mark { stroke-width: 1.2; stroke-dasharray: 3 3; vector-effect: non-scaling-stroke; }
  line.spec-ha   { stroke: #c0453b; }
  line.spec-oiii { stroke: #2b8a8d; }
  text.spec-ha   { fill: #c0453b; }
  text.spec-oiii { fill: #2b8a8d; }
</style>

<div class="spec-wrap" markdown="0">
--8<-- "docs/figures/sdss_spectrum.svg"
</div>

*Fiber 900 of the six-plate store — one row of `flux[3840, 3841]`, median-binned 5:1 for
drawing. Dashed rules mark the **rest** wavelengths of Hα (6563 Å) and [O III] (5007 Å); this
object is redshifted, so its own lines sit to the right of them.*

## What it is

SDSS `spPlate` frames hold ~640 spectra apiece on a shared log-wavelength grid, stored
fiber-major. That means a block of fibers is already a *contiguous byte range* — so you can hand
out chunks of an archival FITS file by doing arithmetic on offsets, without rewriting a pixel.

That is what these stores are: kilobyte-sized Icechunk repositories holding a chunk manifest of
byte ranges, built by pointing [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) at
the FITS headers. Open one and you get an ordinary zarr-v3 array; read it and the bytes come out
of the FITS files themselves.

| Store | Shape | Chunk | Fibers/chunk | Pixels from |
|---|---|---|---|---|
| `sdss_dr17_p0266_refs` | 640 × 3864 | 64 × 3864 | 64 | `data.sdss.org` |
| `sdss_dr17_6plate_refs` | 3840 × 3841 | 1 × 3841 | 1 | `data.sdss.org` |
| **`sdss_dr17_6plate_mirror_refs`** | 3840 × 3841 | 1 × 3841 | 1 | **this bucket** |

All three expose one `float32` variable, `flux`, as `(fiber, wavelength)`, under
`gs://insitubatch-bench-insitubatch/astronomy/`. Start with the mirror store: same geometry as
`6plate`, but its references point at our copy of the FITS files in the same bucket, so it is one
provider and in-region from GCP.

## Open one

No credentials, no requester-pays, no build step. `pip install icechunk zarr` is the whole
dependency list — the FITS stack was only needed to *write* the references.

```python
import icechunk, zarr

BUCKET = "insitubatch-bench-insitubatch"
PREFIX = "astronomy/sdss_dr17_6plate_mirror_refs"
PIXELS = f"https://storage.googleapis.com/{BUCKET}/astronomy/sdss/dr17/"

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(PIXELS, icechunk.http_store())
)
repo = icechunk.Repository.open(
    icechunk.gcs_storage(bucket=BUCKET, prefix=PREFIX, anonymous=True),
    config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials({PIXELS: None}),
)

flux = zarr.open_array(repo.readonly_session("main").store, path="flux", mode="r")
flux.shape        # (3840, 3841) float32, chunks (1, 3841)
flux[0]           # one spectrum
```

Each store records the prefix its virtual chunks resolve against — `print(repo.config.virtual_chunk_containers)`
if you need it. For the two `data.sdss.org`-backed stores, set `PIXELS = "https://data.sdss.org/"`.

### Wavelengths

Uniform in log₁₀(λ), the SDSS convention: bin `i` is `10**(COEFF0 + i*COEFF1)` Å. Each store
carries its own solution as array attributes, so read them off `flux.attrs` rather than
hard-coding the table below:

```python
import numpy as np

c0, c1, n_bins = 3.5797, 0.0001, 3841      # == flux.attrs["COEFF0"], ["COEFF1"], shape[1]
wavelength_angstrom = 10.0 ** (c0 + c1 * np.arange(n_bins))
wavelength_angstrom[[0, -1]]               # array([3799.3, 9198.1])
```

| Store | `COEFF0` | `COEFF1` | Bins | Range |
|---|---|---|---|---|
| `p0266` | 3.5785 | 0.0001 | 3864 | 3789 – 9221 Å |
| `6plate`, `6plate-mirror` | 3.5797 | 0.0001 | 3841 | 3799 – 9198 Å |

The six-plate `COEFF0` is the *shared* window start — the largest of the six plates' own start
wavelengths, which is also the crop offset into each source frame. Fluxes are in SDSS units of
10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹; bad pixels are `NaN`.

## Two layouts, one set of bytes

The same FITS files produce two genuinely different chunk geometries, and you cannot have both
at once:

- **One plate → 64 fibers per chunk.** Full wavelength width. One read decodes 64 spectra, so
  per-read overhead amortizes across many samples.
- **Six plates → one fiber per chunk.** Plates cover slightly different wavelength ranges, so
  combining them means cropping each to the shared window — and that crop breaks the fiber
  contiguity the first layout depends on. You get archive-scale concatenation and pay one ranged
  read per spectrum.

The crop is exact rather than interpolated: every plate uses the same `dloglam` step and their
start wavelengths differ by whole bins, so the windows land on one grid with no resampling. Six
plates is 3,840 spectra; the same construction runs out to the ~2,800-plate archive with the
manifest still measured in kilobytes.

## What it costs to read

One fiber per chunk means one ranged GET per spectrum, so the six-plate store is latency-bound,
not bandwidth-bound. A full pass over all 3,840 spectra (59 MB in 3,840 reads), from a VM in
`us-central1`:

| Reader | Concurrency | Full pass | Spectra/s |
|---|---|---|---|
| zarr, row at a time | — | ~154 s | 25 |
| zarr, 64-row slices | 10 *(default)* | 22.3 s | 172 |
| zarr, 64-row slices | 32 | 12.6 s | 305 |
| **insitubatch, shuffled batches** | 32 | **10.7 s** | **360** |

Medians of five interleaved runs. The knee is around 32 concurrent reads for both readers and
neither improves past it. If you use plain zarr, raise `async.concurrency` — the default of 10
leaves about half the throughput on the table for a store shaped like this.

!!! note "Why the loader is only ~18% ahead here"

    One sample per chunk is the geometry where a batch loader has least to offer: nothing to
    amortize across samples within a chunk, no redundant reads to de-duplicate. What the 18% buys
    is shuffled, split-aware, transformed batches rather than raw slices. The
    many-fibers-per-chunk layout (`p0266`) is where amortization actually pays — see
    [Tuning](tuning.md).

## As training batches

Any loader works on the array above. This is insitubatch, streaming shuffled, chunk-aligned
splits straight from the store — complete and standalone, so it runs as-is:

```python
import icechunk
from insitubatch import InSituDataset, open_geometries, split_by_chunk

BUCKET = "insitubatch-bench-insitubatch"
PREFIX = "astronomy/sdss_dr17_6plate_mirror_refs"
PIXELS = f"https://storage.googleapis.com/{BUCKET}/astronomy/sdss/dr17/"

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(PIXELS, icechunk.http_store())
)
repo = icechunk.Repository.open(
    icechunk.gcs_storage(bucket=BUCKET, prefix=PREFIX, anonymous=True),
    config=config,
    authorize_virtual_chunk_access=icechunk.containers_credentials({PIXELS: None}),
)
store = repo.readonly_session("main").store

geoms = open_geometries(store, variables=["flux"], sample_axis=0)
ds = InSituDataset(
    store,
    split_by_chunk(geoms["flux"], fractions=(0.8, 0.1, 0.1)),
    geometries=geoms,
    batch_size=64,
    max_inflight=32,   # concurrent ranged GETs -- the knob that matters here
)
for batch in ds.train:
    x = batch.arrays["flux"]        # (64, 3841) float32
```

A worked example — spectral reconstruction with a conv autoencoder against a PCA baseline,
mirroring astroML's `compute_sdss_pca` workflow — ships in the repo:

```bash
uv run python -m examples.sdss.train_torch --source published
```

See [Examples](examples.md#sdss-reconstructing-galaxy-spectra-streamed-in-place) for the flags
and the build-it-yourself path.

## Provenance and licence

SDSS DR17, public domain: *"All SDSS data released in our public data releases are considered in
the public domain."* Work using these data should cite SDSS per the
[survey's guidance](https://www.sdss4.org/collaboration/citing-sdss/). Sources, the mirrored file
list and citation pointers are in
[`PROVENANCE.md`](https://storage.googleapis.com/insitubatch-bench-insitubatch/astronomy/sdss/dr17/PROVENANCE.md)
beside the data; bucket-level docs in
[`astronomy/README.md`](https://storage.googleapis.com/insitubatch-bench-insitubatch/astronomy/README.md).

Reading big-endian FITS correctly through the virtual chain needs **VirtualiZarr ≥ 2.7.2** and
**zarr ≥ 3.3**. Those releases are why this dataset exists in a usable form — thanks to the
VirtualiZarr and zarr-python maintainers for turning the fixes around quickly, and to Martin
Durant, whose `kerchunk.fits` does the header parsing underneath.

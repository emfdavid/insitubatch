# insitubatch — working notes for Claude

Train in place on n-dimensional cloud tensors: the data-loader orchestration
layer on top of solved async IO (obstore / zarr v3). See [DESIGN.md](DESIGN.md)
for the thesis and [docs/architecture.md](docs/architecture.md) for the pipeline.

## Working principles

- **PEP 20**, especially *"there should be one — and preferably only one —
  obvious way to do it."* Prefer a single clear path over configurable cleverness;
  when you find a second way to do something, remove one.
- **TDD where practical**, and *always* for bugs: first write the failing test
  that reproduces the bug, then fix until green. New behavior ships with tests.
- **Fail fast.** Do not catch-and-continue on errors that cannot be genuinely
  recovered — let them propagate with context. Validate at boundaries and raise
  early. Use explicit exceptions for runtime contracts; reserve `assert` for
  internal dev invariants (it is stripped under `-O`).

## Interaction style

- **No praise for input** ("great question", "sharp", etc.). Lead with analysis.
- **Present options with trade-offs**, give a recommendation, and state the
  reasoning — do not merely validate the user's framing.
- **Do not reinforce bias.** Push back with evidence when the analysis disagrees;
  surface the counter-case and risks even when unprompted.

## Toolchain

- **`uv`** manages everything (env, deps, running tools).
- Verify: `uv run ruff check src tests bench examples`, `uv run mypy src bench examples`,
  `uv run pytest -q`.
- Pre-commit (ruff + mypy): `uv run pre-commit install` once; runs on every commit.
- Build: `uv build`. Sync env: `uv sync` (extras: `--extra torch`, `--extra gpu`).
- Python ≥ 3.12, src layout (`src/insitubatch/`), build backend `uv_build`.
- mypy is clean and enforced — keep it that way (use precise types, not `Any`,
  except for genuine third-party passthrough kwargs).

## Load-bearing invariants (do not break)

- **`Batch` is numpy.** Frameworks (torch/JAX/TF) are thin DLPack adapters, never
  core dependencies. The core engine imports torch only optionally.
- **Python hot path is O(chunks), not O(samples).** Never loop per-sample in
  Python; planning/gather are vectorized. This is the whole performance thesis.
- **Parallelism lives in the async event loop**, not worker processes. The torch
  surface runs `num_workers=0`, `batch_size=None`.
- **chunk transforms must be vectorized numpy** (so they release the GIL and
  overlap IO on the decode path). Pure-Python per-element transforms are a bug.
- **Transform stages by cost:** `chunk_transform` (per-chunk, deterministic,
  cacheable — scaling/regrid) → `batch_transform` (cross-variable / per-sample
  random — uncached) → `device_transform` (GPU, in the adapter).
- **Sample geometry v1:** a sample is a slice of the outer (sample) axis that does
  not cross a chunk boundary. No cross-chunk samples; cross-variable derived
  fields are batch-stage only.
- **One URL, any backend:** `store_from_url` (obstore) gives `file://` locally,
  `s3://`/`gs://` in the cloud — no hot-path change.

## Do not

- Put **dask** on the hot path (its nested worker thread pools are the problem we
  route around).
- Make insitubatch **build `xr.DataArray`** (we deliver tensors; see the
  Earth2Studio section in docs/architecture.md).
- **Reshard** data into a sample format (the no-reshard, train-in-place stance).

## Status / roadmap

Single source of truth: [DESIGN.md](DESIGN.md) (Status + Roadmap sections). Do
not mirror milestone state here — it goes stale.

## Commits

Commit only when asked. End commit messages with the Co-Authored-By trailer.

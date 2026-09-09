# insitubatch — working notes for Claude

Train in place on n-dimensional cloud tensors: the data-loader orchestration
layer on top of solved async IO (obstore / zarr v3). See [DESIGN.md](DESIGN.md)
for the thesis and [docs/architecture.md](docs/architecture.md) for the pipeline.
Contributor-facing rules (scope limits, dev setup, AI policy) live in
[docs/contributing.md](docs/contributing.md); governance in [GOVERNANCE.md](GOVERNANCE.md).

## Working principles

- **PEP 20**, especially *"there should be one — and preferably only one —
  obvious way to do it."* Prefer a single clear path over configurable cleverness;
  when you find a second way to do something, remove one.
- **TDD where practical**, and *always* for bugs: first write the failing test
  that reproduces the bug, then fix until green. New behavior ships with tests.
- **Validate the instrument before the subject.** Point any new measurement — a
  counter, a probe, a benchmark harness — at a case whose answer you already know
  and require it to produce that answer, *before* trusting it on the case you
  don't. A miswired instrument almost never raises; it returns a plausible number,
  and often a flattering one. (Opening a zarr store read-only *copies* it, so a
  read-counter kept on the instance you hold counts nothing and reports "no
  redundant reads at all" — indistinguishable from a real, favorable result.)
  This is the same discipline as the control/NULL run required of performance
  claims, applied one step earlier: to the tool, not the treatment.
- **Fail fast.** Do not catch-and-continue on errors that cannot be genuinely
  recovered — let them propagate with context. Validate at boundaries and raise
  early. Use explicit exceptions for runtime contracts; reserve `assert` for
  internal dev invariants (it is stripped under `-O`).
- **Copy-paste-safe command blocks.** Interactive zsh (macOS default) has
  `interactive_comments` off, so a pasted `#` line runs as a command
  (`zsh: command not found: #`); bash honors in-block comments by default. Docs/runbooks
  that keep `#` comments inside fenced blocks (`bench/ops_*.md`) must carry a
  `setopt interactive_comments` note near the top so zsh readers enable it once. Prefer
  that one escape-hatch note over sprinkling the caveat per block.

## Interaction style

- **No praise for input** ("great question", "sharp", etc.). Lead with analysis.
- **Present options with trade-offs**, give a recommendation, and state the
  reasoning — do not merely validate the user's framing.
- **Do not reinforce bias.** Push back with evidence when the analysis disagrees;
  surface the counter-case and risks even when unprompted.

## Toolchain

- **`uv`** manages everything (env, deps, running tools).
- Verify: `uv run ruff check src tests bench examples`, `uv run ruff format --check src
  tests bench examples`, `uv run mypy src bench examples`, `uv run pytest -q`. Docs:
  `uv run --extra docs mkdocs build --strict` (CI runs it, so a broken link fails the
  build). Those five are exactly what CI enforces, and docs/contributing.md plus the PR
  checklist list the same set — keep the three in step. Pages under `docs/` **cannot**
  relatively link root files (`DESIGN.md`, `GOVERNANCE.md`) — use the GitHub URL, as the
  existing pages do.
- **One framework per env.** torch/JAX/TF cannot share a process (Keras 3 pulls JAX in when
  installed), so a `.venv` carrying several **segfaults mid-suite** — the documented failure
  mode (README, "One framework per environment"), not a regression. If yours has them all:
  `pytest -q --ignore=tests/test_tf.py`, then `pytest -q tests/test_tf.py`. CI gives each
  framework its own job and so never hits it.
- Pre-commit (ruff + mypy): `uv run pre-commit install` once; runs on every commit.
- Build: `uv build`. Sync env: `uv sync` (extras: `--extra torch`, `--extra gpu`).
- Python ≥ 3.12, src layout (`src/insitubatch/`), build backend `uv_build`.
- mypy is clean and enforced — keep it that way (use precise types, not `Any`,
  except for genuine third-party passthrough kwargs).

## Load-bearing invariants (do not break)

> **Paired with [docs/contributing.md](docs/contributing.md).** That page carries the same
> rule set contributor-facing, *with the reasoning*: this is the terse working copy, that is
> the published contract, and neither may be weakened alone.
>
> The *organization* differs on purpose. Its scannable "hard no" list repeats the
> frameworks-are-never-a-core-dep rule from the invariants below — the one contributors try
> hardest to breach — so it has **four** entries where "Do not" has **three**. Same rules, not
> the same line count; don't reconcile by deleting.

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
- **One contract, any backend:** the engine reads a zarr-v3 `Store`; constructors
  build one per backend (`obstore_store` for `file://`/`s3://`/`gs://`,
  `fsspec_store` for what obstore can't reach — GCS Rapid/zonal, requester-pays —,
  `arraylake_store` for Icechunk sessions). No str-vs-Store dispatch, no hot-path
  change. obstore is the default URL path today; fsspec is under evaluation as a
  co-equal fast path for GCS.

## Do not

- Put **dask** on the hot path (its nested worker thread pools are the problem we
  route around).
- Make insitubatch **build `xr.DataArray`** (we deliver tensors; see the
  Earth2Studio section in docs/architecture.md).
- **Reshard** data into a sample format (the no-reshard, train-in-place stance).

## Docs: what goes where

**Living docs describe the code as it is now.** README, `docs/architecture.md`,
`docs/tuning.md`, the other `docs/*.md` pages and every docstring document the
*current* implementation only — the pattern a reader should use today, and why it
works. **`DESIGN.md` and `CHANGELOG.md` own the history**: how the design got
here, which alternatives were weighed and rejected, and what changed in each
release.

The reason is that a living doc is read by someone deciding what to do next, so it
owes them the current rule and the reason for it — and nothing else. Two things
impersonate a reason without being one:

- **A superseded design.** Prose about what the code used to do is indistinguishable
  from prose about what it does, at the moment of reading.
- **Provenance — how we came to know.** *Keep the reason a thing exists, not how we
  found it.* "A local fixture cannot reproduce a backend-specific defect" is the
  reason; "the sharded-decode bug lived in an Icechunk store and no synthetic fixture
  would have found it" is the incident that taught us the reason, and it does not
  make the rule any truer to a reader meeting it fresh.

Neither is lost by this rule, only relocated to where it is read deliberately rather
than accidentally: rejected alternatives and design evolution to `DESIGN.md`, and the
incident to `CHANGELOG.md`, whose house style (*what broke, who it bit, how it
presented*) is exactly the war story's home.

The test is mechanical: **"used to", "previously", "no longer", "instead of", "we
considered", "we found", "turned out", "while investigating", "was discovered" — each
is a signal the sentence belongs in `DESIGN.md` or `CHANGELOG.md`**, not in the living
doc. Rationale for a *rejected* alternative goes to `DESIGN.md`; rationale for the
*implemented* pattern stays with the code, stated positively. Linking an issue as a
pointer (`(#56)`) is fine — retelling it is not. When a change makes a living doc's
claim false, fix the claim; do not append the correction to it.

Status and roadmap follow the same rule: single source of truth is
[DESIGN.md](DESIGN.md) (Status + Roadmap sections). Do not mirror milestone state
here or in the docs pages — it goes stale.

## Commits & pull requests

Commit only when asked. End commit messages with the Co-Authored-By trailer.

PRs are squash-merged, so the **PR title becomes the commit on `main`** — use a descriptive
Conventional-Commits-style title. Drafted PR descriptions follow
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).

- **Never tick the author-attestation checkbox.** It asserts that a human reviewed every
  change, can explain why each is correct, and has verified the claims in the description.
  That is the human author's to tick, and an assistant ticking it on their behalf is precisely
  the failure the policy exists to prevent. Leave it unchecked and say so.
- **Every PR adds a bullet under `## Unreleased` in CHANGELOG.md.** House style: bold lead-in,
  then *why it mattered* — what broke, who it bit, how it presented — not just what changed.
- **Performance claims carry their setup**: hardware (incl. vCPU count), store + region,
  chunk geometry, loader config, and a **control/NULL run** measured in the same session.
  Read the baseline's own defaults before publishing a number against it.
- Anything crossing a load-bearing invariant is an issue for discussion, not a PR.

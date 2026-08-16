# Contributing

insitubatch is built to be worked on by more than one person. It is small enough that you
can read the whole engine in an afternoon, and opinionated enough that a change lands
cleanly or not at all — this page exists so you can tell which one yours is *before* you
write it.

If you only remember one thing: **for anything larger than a bug fix, open an issue first.**
The scope limits below are load-bearing, not stylistic, and a change that crosses one is
rejected no matter how good the code is.

## Ways to contribute

- **Bug reports.** The most valuable thing you can send. Use the
  [bug report form](https://github.com/emfdavid/insitubatch/issues/new/choose).
- **Performance reports.** This is a performance project, so "it is slower than I expected"
  is a first-class bug — but only if it comes with the store geometry and the loader config.
  There is a [dedicated form](https://github.com/emfdavid/insitubatch/issues/new/choose) that
  asks for exactly what triage needs.
- **Documentation.** Including "this paragraph is wrong" and "I could not follow this".
- **A new-domain example.** insitubatch already trains on weather, microscopy and astronomy
  ([examples](examples.md)); the sample axis is a *role*, so a new domain is usually a new
  store and a short script, not engine work. This is the best-scoped substantial first
  contribution — it exercises the general contract without touching the hot path.
- **Code.** Bug fixes are always welcome. For features, see the scope section below.

## Where to ask

| For | Go to |
| --- | --- |
| "How do I do X with insitubatch?" | [GitHub Discussions](https://github.com/emfdavid/insitubatch/discussions) |
| A defect, or a performance surprise | [Issues](https://github.com/emfdavid/insitubatch/issues) |
| Zarr-ecosystem questions | the [Zarr Zulip](https://ossci.zulipchat.com/) |
| Wider scientific-data-on-the-cloud questions | [Pangeo Discourse](https://discourse.pangeo.io/) |

## Before you write code: what this project will and will not take

These are the invariants the design rests on. Each one is a constraint we accepted
deliberately; the reason matters more than the rule, because it tells you when your change
is compatible with it.

**`Batch` is numpy.** Frameworks (torch / JAX / TensorFlow) are thin DLPack adapters, never
core dependencies — the core engine imports no framework, and `import insitubatch` never
pulls one in. *Why:* the value is one loader contract across frameworks; a framework in the
core would fork the engine three ways and make the package uninstallable for people who use
the fourth.

**The Python hot path is O(chunks), not O(samples).** Planning and gather are vectorized;
nothing loops per sample in Python. *Why:* this is the entire performance thesis. A
per-sample Python loop reintroduces exactly the cost the project exists to remove, and it
will not show up in a small test — it shows up as a lost benchmark six months later.

**Parallelism lives in the async event loop, not in worker processes.** The torch surface
runs `num_workers=0, batch_size=None`. *Why:* worker processes cannot share a chunk cache,
cannot drive async obstore, and nest thread pools inside forks.

**`chunk_transform`s must be vectorized numpy that releases the GIL.** A pure-Python
per-element transform is a bug, not a slow path — it serializes the decode pool and kills IO
overlap. Check yours against one chunk of your real store before you train:
`insitubatch-check-transform` gives a GIL-release verdict and a non-zero exit you can gate a
hook on.

**Transform stages are ordered by cost.** `chunk_transform` (per chunk, deterministic,
cacheable — scaling, regrid) → `batch_transform` (cross-variable, per-sample random —
uncached) → `device_transform` (GPU, in the adapter). Putting per-sample randomness in the
chunk stage silently poisons the cache.

**Sample geometry v1: a sample is a slice of the sample axis that does not cross a chunk
boundary.** Cross-variable derived fields are batch-stage only. The deferred
generalizations, and *why* they were deferred, are enumerated in
[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md) so they do not get
relitigated from scratch — read that section before proposing one.

And four hard "no"s:

1. **No dask on the hot path.** Its nested worker thread pools are the problem we route
   around, not a tool we can borrow.
2. **insitubatch does not build `xr.DataArray`.** We deliver tensors plus a light coords
   dict. See the Earth2Studio discussion in [architecture](architecture.md).
3. **No resharding into a sample format.** Train in place is the thesis; a "just rewrite the
   store" path would make every benchmark on this site meaningless.
4. **No framework as a core dependency** (the first invariant, restated because it is the
   one people try hardest to breach).

If your idea touches one of these, that is not automatically the end of the conversation —
but it is a design discussion on an issue, not a pull request. The real backlog is the
**Known limitations & defects** and the ⏳ roadmap entries in
[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md), which is the single
source of truth for status; nothing here mirrors it.

## Development setup

Everything goes through [`uv`](https://docs.astral.sh/uv/) — environment, dependencies, and
tool versions, so nothing drifts between your machine and CI.

```bash
git clone https://github.com/emfdavid/insitubatch.git
cd insitubatch
uv sync                    # core engine + dev tools
uv run pre-commit install  # ruff + mypy on every commit
```

Extras, one at a time (see the framework caveat below):

```bash
uv sync --extra torch      # torch handoff (frameworks.as_torch)
uv sync --extra jax        # JAX handoff (frameworks.to_jax)
uv sync --extra tf         # TF handoff (frameworks.as_tf_dataset)
uv sync --extra bench      # benchmark suite (xbatcher baseline + plotly + py-spy)
uv sync --extra docs       # MkDocs site
uv sync --extra gpu        # CUDA box only: cupy + kvikio zero-copy path
```

Code lives in `src/insitubatch/`, tests in `tests/`, the benchmark suite in `bench/`, and
runnable examples in `examples/`. DESIGN.md's
[**Module map**](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md#module-map) is the
fastest way to find which file owns what.

## Running the tests

The three commands CI runs, and the ones a reviewer will expect to be green:

```bash
uv run ruff check src tests bench examples
uv run mypy src bench examples
uv run pytest -q
```

**One framework per environment.** This is the single most common way a new contributor's
run breaks, and it is not an insitubatch limitation. torch, JAX and TensorFlow cannot
coexist in one Python process — together they load duplicate OpenMP / XLA / protobuf
runtimes and the process dies with `SIGSEGV` or an abort. Separate pytest *processes* in one
environment are not enough either: TensorFlow (via its bundled Keras 3) transitively imports
JAX whenever JAX is *installed*, so the two collide even when only the TF tests are selected.
Install one adapter at a time:

```bash
uv sync --extra torch && uv run pytest -q   # torch adapter + core
uv sync --extra jax   && uv run pytest -q   # JAX adapter (others importorskip-skip)
uv sync --extra tf    && uv run pytest -q   # TF adapter
```

CI does exactly this: one job per framework, plus a lint/types job that installs every extra
but runs no pytest (mypy does not import the frameworks, so co-installation is harmless
there). Tests for an absent framework `importorskip`-skip, so a core-only run is a valid run.

**Free-threaded (3.13t).** If you touch `ChunkPool`, the scheduler, or anything that
publishes chunk readiness across threads, run the free-threaded job too — the pool's
lock-free disjoint scatter is exactly what it exercises. The full recipe (a separate
`.venv-ft`, and why you must assert the GIL is actually off before trusting the run) is in
the [README](https://github.com/emfdavid/insitubatch#free-threaded-313t). CI mirrors it.

## Code standards

- **PEP 20**, especially *"there should be one — and preferably only one — obvious way to do
  it."* Prefer a single clear path over a configuration knob; if you find a second way to do
  something, remove one. A PR that adds an option where a decision would do will be asked to
  pick.
- **Fail fast.** Do not catch-and-continue on errors that cannot be genuinely recovered — let
  them propagate with context. Validate at boundaries and raise early. Use explicit exceptions
  for runtime contracts; reserve `assert` for internal dev invariants (it is stripped
  under `-O`).
- **mypy is clean and enforced.** Keep it that way: precise types, not `Any`, except for
  genuine third-party passthrough kwargs.
- **ruff** with `E, F, I, UP, B, SIM` at line length 100. `uv run pre-commit install` makes
  this automatic.
- **Comments explain *why*.** The codebase is dense with rationale for decisions that look
  arbitrary until you know what they route around. Match that; a comment restating the code
  is noise, a comment naming the failure mode is the point.

## Tests

New behavior ships with tests. **For bugs, the failing test comes first** — write the test
that reproduces the report, watch it fail, then fix until green. This is a project rule, not
a suggestion: a bug fix without a reproducing test is a fix we cannot prove and cannot keep.

Tests live in `tests/`, one file per concern (`test_pool.py`, `test_scheduler.py`,
`test_window.py`, …). Prefer **structural** assertions over timeouts — the deadlock guard in
`test_residency.py` detects the *provably* terminal state rather than waiting, so a merely
slow machine is never mistaken for a bug.

## Performance claims

If your PR asserts a speedup, the description has to carry enough for someone else to
reproduce or refute it:

- **Hardware**, including vCPU count. Small boxes have an A/B resolution floor of a few
  percent; a "5% win" measured on four cores is usually noise.
- **Store** URL or scheme, region, and whether the run was cold or warm.
- **Geometry**: array shape, chunk shape, sample axis, samples per chunk.
- **Loader config**: `batch_size`, `block_chunks`, `max_inflight`, `prefetch_depth`, cache
  budget, `persist`.
- **A control run.** Measure a NULL configuration — the unchanged code, same box, same
  session, interleaved — alongside the treatment. Without it you cannot distinguish your
  change from the machine.
- **The baseline's own defaults**, if you are comparing against another library. Read its
  constructor before you publish a number against it; a misconfigured baseline produces a
  flattering result that will not survive review.

[`bench/benchmark_plan.md`](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md)
has the methodology, and [Benchmarks](benchmarks.md) has the published numbers and their
caveats — including where insitubatch *loses*, which is stated on purpose. Keep it that way.

## Documentation and changelog

- User-facing changes get docstrings (Google style; they render into the
  [API reference](api.md)) and an update to the relevant `docs/*.md` page.
- Every PR adds a bullet under `## Unreleased` in `CHANGELOG.md`. The house style is a bold
  lead-in and then *why it mattered* — what broke, who it bit, how it presented — not just
  what changed. Read the existing entries before writing yours.
- Docs are built with `uv run mkdocs build --strict` in CI, so a broken link fails the build.
  Note that pages under `docs/` cannot link to root-level files like `DESIGN.md` relatively —
  use the GitHub URL, as the existing pages do.

## AI-assisted contributions

AI assistance is welcome here, and stating otherwise would be dishonest: this project is
built with it, including drafted pull-request descriptions. The line is drawn at
**accountability, not at who typed the prose.**

1. **You are responsible for every change in your PR.** You must be able to explain why each
   one is correct, in conversation, without going back to a model. If you cannot, the PR is
   not ready — that is true whether the code came from an assistant, a tutorial, or Stack
   Overflow.
2. **A model may draft the description; you must have read it and verified every factual
   claim in it** — what it fixes, what it measures, what it leaves alone. An unverified
   description is worse than no description, because it costs the reviewer the time to
   discover it is wrong.
3. **Review every line of the diff yourself** before you open it.
4. **Keep PRs reviewable.** A large diff spends someone else's attention. For substantial
   AI-assisted work, open an issue first so scope can be agreed and the change can be split
   into pieces a human can actually review.

One project-specific warning. insitubatch has domain semantics that language models get
confidently wrong: chunk-versus-sample cardinality, the read-once / sample-once shuffle
contract, the no-reshard stance, and which of the transform stages is cacheable. A
plausible-sounding wrong sentence in the docs costs more than a bug — a bug fails a test,
whereas wrong prose gets believed and repeated. Documentation contributions get the same
scrutiny as code for exactly this reason.

## Review and merge

- Pull requests are squash-merged, so **the PR title becomes the commit on `main`.** Use a
  descriptive Conventional-Commits-style title (`fix: …`, `feat(geometry): …`, `docs: …`).
- Every PR needs a green CI run and approval from at least one maintainer.
- Expect a first response within about a week. If it has been longer, a nudge on the PR is
  welcome and not rude.
- Reviews go at the change, never the person. Both directions — the
  [Code of Conduct](https://github.com/emfdavid/insitubatch/blob/main/CODE_OF_CONDUCT.md)
  applies to maintainers first.

How decisions get made — lazy consensus, when a change needs an explicit one, and who
decides — is written down in
[GOVERNANCE.md](https://github.com/emfdavid/insitubatch/blob/main/GOVERNANCE.md).

## Becoming a maintainer

insitubatch is currently maintained by one person, and that is a transitional state rather
than the intent. The governance is deliberately already written down so that the second and
third maintainers have a framework to build on.

The path is merit-based and short: sustained, quality contribution — code, review,
documentation, triage, or supporting other users — makes you eligible for nomination to the
core developer group by an existing core developer. There is no application form and no
minimum PR count; what counts is a track record someone can point at. If you want to grow
into that, saying so on an issue is a perfectly good start.

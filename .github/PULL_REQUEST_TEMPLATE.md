<!-- The PR title becomes the squash-merge commit on main. Please use a descriptive,
     Conventional Commits-style title, e.g. "fix: raise block_chunks so a block always holds
     a batch". See https://www.conventionalcommits.org/en/v1.0.0/ -->

## Summary

<!-- What does this change, and why? If it fixes an issue, link it: "Closes #123". -->

## For reviewers

<!-- What would you most value a second look at, and what are you already confident in?
     For a refactor, say whether behavior is meant to be unchanged. -->

## Author attestation

- [ ] I have reviewed every change in this PR, I can explain why each one is correct, and I
      have verified the claims made in this description.

<!-- AI assistance is welcome, including drafting this description — provided the box above
     genuinely holds. See "AI-assisted contributions":
     https://emfdavid.github.io/insitubatch/contributing/#ai-assisted-contributions -->

## Checklist

<!-- Remove any line that does not apply to this change. -->

- [ ] Tests added or updated — for a **bug fix**, a test that reproduces it and failed before
      this change
- [ ] `uv run ruff check src tests bench examples`, `uv run mypy src bench examples` and
      `uv run pytest -q` are green locally
- [ ] Docstrings and API docs for any new or changed public surface
- [ ] User-facing behavior documented in `docs/*.md`
- [ ] A bullet added under `## Unreleased` in `CHANGELOG.md`
- [ ] No [load-bearing invariant](https://emfdavid.github.io/insitubatch/contributing/#before-you-write-code-what-this-project-will-and-will-not-take)
      is broken (numpy `Batch`, O(chunks) hot path, parallelism in the event loop, vectorized
      GIL-releasing chunk transforms, no dask / no reshard / no `xr.DataArray`)
- [ ] Touches `ChunkPool`, the scheduler, or cross-thread readiness → the free-threaded
      (3.13t) run passes
- [ ] Performance claim? The hardware (incl. vCPU count), store, chunk geometry, loader
      config, **and a control run** are in the description above

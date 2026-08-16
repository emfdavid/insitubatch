# Contributing

insitubatch welcomes contributions from anyone — bug reports, performance reports,
documentation, examples in a new domain, and code.

The full guide lives with the docs:
**<https://emfdavid.github.io/insitubatch/contributing/>**. It covers the development setup,
the project's load-bearing scope limits (read these before writing a feature), the
one-framework-per-environment test caveat, what a performance claim has to carry, and the
policy on AI-assisted contributions.

The three commands CI runs:

```bash
uv run ruff check src tests bench examples
uv run mypy src bench examples
uv run pytest -q
```

For anything larger than a bug fix, please
[open an issue](https://github.com/emfdavid/insitubatch/issues/new/choose) first.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). How decisions get
made, and how to become a maintainer, is in [GOVERNANCE.md](GOVERNANCE.md).

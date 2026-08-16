# Security policy

## Supported versions

insitubatch is pre-1.0. Only the **latest release** on
[PyPI](https://pypi.org/project/insitubatch/) receives security fixes; there are no
maintained backport branches. Fixes ship in the next release, and in an out-of-band patch
release if the issue warrants one.

## Reporting a vulnerability

Please report privately, not on the public issue tracker:

1. **Preferred:** open a
   [private security advisory](https://github.com/emfdavid/insitubatch/security/advisories/new)
   on this repository.
2. **Fallback:** email David Stuebe at <stu3b3@gmail.com> with `insitubatch security` in the
   subject.

Include what you were doing, what you observed, and — if you have one — a reproducer. The
[`print_debug_info()`](https://emfdavid.github.io/insitubatch/api/) output is useful, since
most of the attack surface here belongs to the versions of the storage stack you have
installed.

Expect an acknowledgement within **five working days**. This is a small project maintained by
volunteers; if you have heard nothing after that, please send a reminder rather than assuming
the report was ignored. We will keep you informed while the issue is investigated, and credit
you in the advisory unless you would rather not be named.

Please give us a reasonable window to ship a fix before disclosing publicly. We will not take
legal action against anyone who reports in good faith under this policy.

## What is in scope

insitubatch is a data loader. It has no authentication surface, no network server, and no
persistence format of its own. The things that are genuinely ours:

- Reading untrusted store metadata or chunk data into a crash, an unbounded allocation, or
  out-of-bounds behaviour in the planner, `ChunkPool`, or gather path.
- The **cross-run persistent cache** — it writes and later reads back decoded chunks from a
  local path you configure, so a cache-poisoning or path-traversal issue there is in scope.
- Executing more than you asked for: `chunk_transform` / `batch_transform` targets and the
  `insitubatch-check-transform` CLI resolve `module:attr` or `path.py:attr` and import them.
  That is by design — you are pointing it at your own code — but a way to make it load
  something you did not name is a bug.

## What belongs upstream

Credentials, transport security, signing and object-store authorization are handled by the
libraries underneath us, not by insitubatch. Report those to the relevant project:

- [obstore](https://github.com/developmentseed/obstore) — the default URL path
- [zarr-python](https://github.com/zarr-developers/zarr-python) and
  [numcodecs](https://github.com/zarr-developers/numcodecs) — store contract and codecs
- [icechunk](https://github.com/earth-mover/icechunk),
  [fsspec](https://github.com/fsspec/filesystem_spec) / gcsfs / s3fs — the other backends

If you are unsure which layer owns an issue, report it to us and we will route it.

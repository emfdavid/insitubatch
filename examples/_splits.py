"""Refuse a run whose train or val split has nothing to draw -- at setup, naming the fix.

Both facts are fixed by the arguments, so neither is worth an epoch to discover: a run with an
empty val split would train, validate, and only then fail, keeping none of it. A split can come
up empty two ways that look identical from outside -- it rounds to zero chunks, or it gets
chunks whose every anchor a windowed view drops off the end of the array -- and
:func:`~insitubatch.summary.drawable_samples` is the one number that separates them.

The remedy is *solved* against the same :func:`~insitubatch.split_by_chunk` and window
arithmetic the run uses rather than quoted as a constant, so the advice cannot drift from the
check. It is pure integer work on a stand-in geometry: no store is opened, which is also what
lets a driver (``bench.advection_sweep``) apply the child's exact rule before launching it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from insitubatch import ArrayGeometry, split_by_chunk
from insitubatch.summary import drawable_samples

SCORED = ("train", "val")  # the splits a run trains on and is scored against


def _drawable(
    geom: ArrayGeometry,
    fractions: tuple[float, float, float],
    offsets: Sequence[int],
    sample_range: tuple[int, int] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """``(chunks, drawable samples)`` per split, for ``geom`` read at each of ``offsets``."""
    manifest = split_by_chunk(geom, fractions=fractions, sample_range=sample_range)
    views = {f"offset{k}": geom.shift(k) for k in offsets}
    chunks = {name: len(ids) for name, ids in manifest.chunks.items()}
    return chunks, drawable_samples(views, manifest)


def minimum_samples(
    geom: ArrayGeometry, fractions: tuple[float, float, float], offsets: Sequence[int] = (0,)
) -> int:
    """The smallest sample-axis length, in whole chunks, for which every scored split draws.

    Returns 0 when nothing up to 64 chunks works; the caller then reports what it has.
    """
    step = geom.sample_chunk_size
    for n in range(step, step * 64 + 1, step):
        shape = list(geom.shape)
        shape[geom.sample_axis] = n
        _, drawable = _drawable(replace(geom, shape=tuple(shape)), fractions, offsets)
        if all(drawable[name] > 0 for name in SCORED):
            return n
    return 0


def require_drawable_splits(
    geom: ArrayGeometry,
    fractions: tuple[float, float, float],
    *,
    knob: str,
    offsets: Sequence[int] = (0,),
    sample_range: tuple[int, int] | None = None,
) -> None:
    """Raise ``ValueError`` if the train or val split of ``geom`` has nothing to draw.

    ``knob`` names what the reader changes to lengthen the sample axis (e.g. ``--n-steps``);
    ``offsets`` are the windowed views the run reads (``(0, horizon)`` for a forecast target).
    """
    chunks, drawable = _drawable(geom, fractions, offsets, sample_range)
    empty = [name for name in SCORED if not drawable[name]]
    if not empty:
        return
    counts = ", ".join(
        f"{name}={chunks[name]} chunk(s)/{drawable[name]} drawable samples" for name in SCORED
    )
    need = minimum_samples(geom, fractions, offsets)
    remedy = (
        f"Use at least {need} samples on the sample axis ({need // geom.sample_chunk_size} "
        f"chunks of {geom.sample_chunk_size}) -- {knob}; you have {geom.n_samples}."
        if need
        else f"Widen the sample axis ({knob}) or change the split fractions."
    )
    ahead = max(offsets)
    window = (
        f" A view read {ahead} samples ahead also drops the last {ahead} anchors, which can "
        f"empty a split that did get chunks."
        if ahead
        else ""
    )
    raise ValueError(
        f"the {' and '.join(empty)} split has nothing to draw, so this run could train but "
        f"never be scored. {geom.n_samples} samples over chunks of {geom.sample_chunk_size} is "
        f"{geom.n_chunks} chunk(s); split {fractions} gives {counts}.{window} {remedy}"
    )

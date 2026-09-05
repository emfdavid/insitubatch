"""Per-variable transform scope (``applies``) and per-array cache identity (#42, part 2).

Two properties, one mechanism. Scoping a ``chunk_transform`` at the call site rather than
inside its body makes the scope **visible to the engine**, which fixes two things a name
gate in ``__call__`` cannot:

- **Geometry.** ``output_geometry`` folds every transform's ``output_inner`` into every
  variable. With an in-body gate a transform that halves ``a`` also makes the engine believe
  ``b`` is halved -- ``b`` is then gathered truncated and can never revive from cache. No
  exception is raised: right shape, right dtype, wrong numbers.
- **Cache identity.** The manifest stamped one fingerprint of the *whole* pipeline on every
  array's entries, so editing ``t2m``'s transform invalidated ``u10`` and ``v10`` too, whose
  bytes had not changed. The hash is now per array, over the transforms that apply *to it*.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import zarr
from test_pool import _decode_tiles, _fill_chunk  # shared tiled-store helpers (pytest rootdir)

from insitubatch import (
    StandardScaler,
    applies,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.pool import _TMP_SUFFIX, ChunkPool, output_geometry
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry

SHAPE = (4, 8)
VARS = ("a", "b", "c")


@pytest.fixture
def three_vars(tmp_path):
    """Three same-shaped single-tile arrays; return (url, {var: source})."""
    url = f"file://{tmp_path}/three.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    srcs = {}
    for i, var in enumerate(VARS):
        arr = group.create_array(var, shape=SHAPE, chunks=(2, 8), dtype="f4")
        arr[:] = np.arange(int(np.prod(SHAPE)), dtype="f4").reshape(SHAPE) + i * 100
        srcs[var] = np.asarray(arr[:])
    return url, srcs


class HalveLastAxis:
    """A reshaping transform: block-mean pairs on the last inner axis, ``(n, 8) -> (n, 4)``."""

    def __call__(self, chunk):
        chunk.data = chunk.data.reshape(chunk.data.shape[0], -1, 2).mean(axis=-1)
        return chunk

    def output_inner(self, geom: ArrayGeometry) -> tuple[tuple[int, ...], np.dtype]:
        return (geom.inner_shape[0] // 2,), geom.dtype


def _geoms(url, variables=VARS):
    return open_geometries(obstore_store(url), variables=list(variables))


def _gather_all(pool, var, geom, tiles, owner):
    _fill_chunk(pool, var, 0, geom, tiles, owner)
    spc = geom.sample_chunk_size
    rows = np.array([[0, w] for w in range(spc)], dtype=np.int64)
    return pool.gather(rows, [var], spc).arrays[var]


# -- geometry: the correctness half -----------------------------------------


def test_scoped_reshape_does_not_reshape_the_other_variables(three_vars):
    """The regression the issue reported: a reshaping transform scoped to ``a`` must leave
    ``b``'s declared geometry alone. Fold it into every variable and ``b`` is gathered as a
    truncated prefix of itself -- silent, plausible-looking corruption."""
    url, srcs = three_vars
    geoms = _geoms(url)
    scoped = [applies(["a"], HalveLastAxis())]
    assert output_geometry(geoms["a"], scoped).inner_shape == (4,)
    assert output_geometry(geoms["b"], scoped).inner_shape == (8,)

    pool = ChunkPool(geoms, chunk_transforms=scoped)
    owner = pool.new_owner()
    for var in ("a", "b"):
        tiles = asyncio.run(_decode_tiles(url, var))
        got = _gather_all(pool, var, geoms[var], tiles, owner)
        expect = srcs[var][:2]
        if var == "a":
            expect = expect.reshape(2, -1, 2).mean(axis=-1)
        np.testing.assert_allclose(got, expect)
    pool.close()


def test_scoped_transform_is_not_called_off_scope(three_vars):
    """Off-scope arrays skip the call entirely -- the wasted no-op pass is gone, and a
    transform that would raise on a variable it does not understand never sees it."""
    url, _ = three_vars
    geoms = _geoms(url)
    seen = []

    def only_a(chunk):
        seen.append(chunk.read.array)
        return chunk

    pool = ChunkPool(geoms, chunk_transforms=[applies(["a"], only_a)])
    owner = pool.new_owner()
    for var in ("a", "b"):
        tiles = asyncio.run(_decode_tiles(url, var))
        _fill_chunk(pool, var, 0, geoms[var], tiles, owner)
    assert seen == ["a"]
    pool.close()


def test_unscoped_transform_still_applies_to_every_variable(three_vars):
    """A bare transform is unchanged from today: every variable, as before."""
    url, srcs = three_vars
    geoms = _geoms(url)

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    pool = ChunkPool(geoms, chunk_transforms=[double])
    owner = pool.new_owner()
    for var in ("a", "b"):
        tiles = asyncio.run(_decode_tiles(url, var))
        got = _gather_all(pool, var, geoms[var], tiles, owner)
        np.testing.assert_allclose(got, srcs[var][:2] * 2.0)
    pool.close()


def test_scope_names_an_unknown_array_raises_at_construction(three_vars):
    """A misspelled variable name is caught before any work -- and before any cache file or
    lock exists. Silently applying to nothing is how a scaler quietly stops scaling."""
    url, _ = three_vars
    geoms = _geoms(url)

    def noop(chunk):
        return chunk

    with pytest.raises(ValueError, match="t2m.*not.*arrays|unknown"):
        ChunkPool(geoms, chunk_transforms=[applies(["t2m"], noop)])


def test_applies_rejects_a_bare_string(three_vars):
    """``applies("a", fn)`` would scope to the *characters* of the name. Reject it with the
    fix in the message rather than accepting a second spelling of the argument."""

    def noop(chunk):
        return chunk

    with pytest.raises(TypeError, match=r"\[.*\]|sequence"):
        applies("a", noop)  # type: ignore[arg-type]


# -- cache identity: the per-array fingerprint ------------------------------


def _persist_pool(geoms, backing, transforms, **kw):
    return ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=transforms, **kw)


def _warm(url, geoms, backing, transforms, variables=VARS, **kw):
    pool = _persist_pool(geoms, backing, transforms, **kw)
    owner = pool.new_owner()
    for var in variables:
        tiles = asyncio.run(_decode_tiles(url, var))
        _fill_chunk(pool, var, 0, geoms[var], tiles, owner)
    pool.close()
    return pool


def test_editing_one_variables_transform_keeps_the_others_entries(three_vars, tmp_path):
    """The headline. Editing the transform scoped to ``a`` invalidates ``a`` and nothing
    else: ``b`` and ``c``'s bytes did not change, so their entries stay valid."""
    url, _ = three_vars
    geoms = _geoms(url)
    backing = tmp_path / "cache"

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    def scale_a_edited(chunk):
        chunk.data = chunk.data * 3.0
        return chunk

    _warm(url, geoms, backing, [applies(["a"], scale_a)])

    # Only `a` is stale. Default is still raise -- but the message must name which.
    with pytest.raises(ValueError, match=r"stale.*'a'|'a'.*stale"):
        _persist_pool(geoms, backing, [applies(["a"], scale_a_edited)])

    reopened = _persist_pool(
        geoms, backing, [applies(["a"], scale_a_edited)], reset_stale_cache=True
    )
    owner = reopened.new_owner()
    assert not reopened.pin_if_ready("a", 0, owner), "a's entry must be dropped"
    assert reopened.pin_if_ready("b", 0, owner), "b's entry must survive"
    assert reopened.pin_if_ready("c", 0, owner), "c's entry must survive"
    reopened.close()


def test_reset_stale_deletes_only_the_stale_arrays_files(three_vars, tmp_path):
    """``reset_stale_cache`` is now precise: it unlinks the stale array's ``.npy`` and leaves
    the others on disk. Wiping the whole directory would throw away work that is still valid."""
    url, _ = three_vars
    backing = tmp_path / "cache"
    geoms = _geoms(url)

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    def scale_a_edited(chunk):
        chunk.data = chunk.data * 3.0
        return chunk

    warm = _warm(url, geoms, backing, [applies(["a"], scale_a)])
    files = {var: backing / warm._persisted[(var, 0)] for var in VARS}
    assert all(p.exists() for p in files.values())

    reset = _persist_pool(geoms, backing, [applies(["a"], scale_a_edited)], reset_stale_cache=True)
    reset.close()
    assert not files["a"].exists(), "the stale array's file must be GC'd"
    assert files["b"].exists() and files["c"].exists(), "valid arrays keep their files"


def test_a_new_variable_over_an_existing_cache_is_cold_not_stale(three_vars, tmp_path):
    """An array with **no entries** has nothing to disagree with: it is a cold start for that
    array, not a stale cache. Raising here would make adding a variable a wipe."""
    url, _ = three_vars
    backing = tmp_path / "cache"

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    _warm(url, _geoms(url, ["a", "b"]), backing, [double], variables=("a", "b"))

    all_three = _geoms(url)
    pool = _persist_pool(all_three, backing, [double])  # must NOT raise
    owner = pool.new_owner()
    assert pool.pin_if_ready("a", 0, owner) and pool.pin_if_ready("b", 0, owner)
    assert not pool.pin_if_ready("c", 0, owner)  # cold: fetched this run
    pool.close()


def test_an_array_this_run_does_not_read_keeps_its_files(three_vars, tmp_path):
    """Two configurations may share one cache_dir. A run that reads only ``a`` must not
    delete -- or forget -- ``b``'s entries just because it cannot check them.

    This is the ablation / feature-importance case: sweeping over variable subsets must not
    cost a re-decode of the variables each run leaves out. Dropping a variable from a run is
    not an edit to it."""
    url, _ = three_vars
    backing = tmp_path / "cache"

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    warm = _warm(url, _geoms(url), backing, [double])
    b_file = backing / warm._persisted[("b", 0)]

    only_a = _persist_pool(_geoms(url, ["a"]), backing, [double])
    only_a.close()
    assert b_file.exists(), "an untouched array's cache file must survive"

    back = _persist_pool(_geoms(url), backing, [double])
    owner = back.new_owner()
    assert back.pin_if_ready("b", 0, owner), "and its manifest entry must survive too"
    back.close()


def test_narrowing_a_scope_invalidates_only_the_variable_that_left(three_vars, tmp_path):
    """Scope is not folded into a transform's own token: the array that *stays* in scope
    hashes identically, and only the one that left is stale.

    Leaving a scope *is* a real invalidation, not bookkeeping -- ``b``'s cached chunks hold
    doubled bytes that the narrowed configuration must not serve -- so under
    ``reset_stale_cache`` its files are deleted and it re-decodes cold (asserted below).
    Contrast :func:`test_an_array_this_run_does_not_read_keeps_its_files`: an array the run
    stops *reading* keeps everything."""
    url, _ = three_vars
    backing = tmp_path / "cache"
    geoms = _geoms(url)

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    _warm(url, geoms, backing, [applies(["a", "b"], double)])

    with pytest.raises(ValueError, match=r"'b'") as exc:
        _persist_pool(geoms, backing, [applies(["a"], double)])
    assert "'a'" not in str(exc.value), "a's pipeline is unchanged -- it must not be stale"

    narrowed = _persist_pool(geoms, backing, [applies(["a"], double)], reset_stale_cache=True)
    owner = narrowed.new_owner()
    assert narrowed.pin_if_ready("a", 0, owner)
    assert not narrowed.pin_if_ready("b", 0, owner)
    narrowed.close()


def test_stale_reset_compacts_the_log(three_vars, tmp_path):
    """The log is rewritten on load when entries are dropped -- atomically, via a temp name
    and a rename. Without compaction the stale entries would be re-read (and re-rejected)
    on every subsequent open, and the log would grow without bound."""
    url, _ = three_vars
    backing = tmp_path / "cache"
    geoms = _geoms(url)
    log = backing / "insitu_cache.jsonl"

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    def scale_a_edited(chunk):
        chunk.data = chunk.data * 3.0
        return chunk

    _warm(url, geoms, backing, [applies(["a"], scale_a)])
    assert len(log.read_text().splitlines()) == 4  # header + 3 entries

    reset = _persist_pool(geoms, backing, [applies(["a"], scale_a_edited)], reset_stale_cache=True)
    reset.close()
    lines = log.read_text().splitlines()
    assert json.loads(lines[0])["format_version"] == 4
    arrays = [json.loads(line)["array"] for line in lines[1:]]
    assert sorted(arrays) == ["b", "c"], "the stale entry is gone; the valid ones survive"
    assert not list(backing.glob("*insitu-tmp")), "compaction must not leave a temp file behind"

    # And the compacted log is what the *next* open reads -- a's entry is not re-rejected.
    again = _persist_pool(geoms, backing, [applies(["a"], scale_a_edited)])  # no reset flag
    owner = again.new_owner()
    assert again.pin_if_ready("b", 0, owner) and not again.pin_if_ready("a", 0, owner)
    again.close()


def test_readonly_cache_cannot_reset_a_stale_array(three_vars, tmp_path):
    """A read-only opener may not delete files another process may be reading, so a stale
    array is a hard error there -- with the fix pointed at the run that *writes* the cache."""
    url, _ = three_vars
    backing = tmp_path / "cache"
    geoms = _geoms(url)

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    def scale_a_edited(chunk):
        chunk.data = chunk.data * 3.0
        return chunk

    _warm(url, geoms, backing, [applies(["a"], scale_a)])
    with pytest.raises(ValueError, match=r"readonly_cache.*stale.*'a'"):
        ChunkPool(
            geoms,
            backing_dir=backing,
            readonly_cache=True,
            chunk_transforms=[applies(["a"], scale_a_edited)],
        )


def test_scaler_scope_is_checked_against_its_own_stats(three_vars):
    """``applies(["b"], scaler_fitted_on_a)`` is caught by :func:`applies` itself, at
    construction -- not as a ``KeyError`` in a decode thread one chunk into training."""
    scaler = StandardScaler(mean={"a": np.float64(0.0)}, std={"a": np.float64(1.0)})
    with pytest.raises(ValueError, match=r"no mean/std for \['b'\]"):
        applies(["b"], scaler)
    applies(["a"], scaler)  # the scope it was fitted for is fine


def test_applies_rejects_an_empty_scope():
    """A scope of nothing means the transform never runs; that is a configuration mistake,
    not a way to disable one."""

    def noop(chunk):
        return chunk

    with pytest.raises(ValueError, match="empty scope"):
        applies([], noop)


def test_describe_names_each_transforms_scope(three_vars, tmp_path):
    """The static report says which variables a transform touches -- 'what will this
    configuration do' is incomplete without it once scope exists."""
    url, _ = three_vars
    geoms = _geoms(url)

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    ds = InSituDataset(
        obstore_store(url),
        split_by_chunk(geoms["a"], fractions=(1.0, 0.0, 0.0)),
        geometries=geoms,
        batch_size=2,
        chunk_transforms=[applies(["a", "b"], double)],
    )
    assert ds.describe()["config"]["chunk_transforms"] == ("double[a,b]",)
    ds.close()


def test_dataset_rejects_an_unknown_scope_name(three_vars):
    """The same check on the public surface, before any geometry is derived from the scope."""
    url, _ = three_vars
    geoms = _geoms(url)

    def noop(chunk):
        return chunk

    with pytest.raises(ValueError, match="not arrays in this dataset"):
        InSituDataset(
            obstore_store(url),
            split_by_chunk(geoms["a"], fractions=(1.0, 0.0, 0.0)),
            geometries=geoms,
            chunk_transforms=[applies(["nope"], noop)],
        )


def test_a_crashed_compaction_leaves_nothing_behind(three_vars, tmp_path):
    """Compaction writes a temp file and renames. A crash between the two must not leave
    litter or a second log: the temp name carries the sweep suffix, so the next writer to
    hold the cache lock clears it -- the same treatment a half-written chunk file gets."""
    url, _ = three_vars
    backing = tmp_path / "cache"
    geoms = _geoms(url)

    def double(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    _warm(url, geoms, backing, [double])
    orphan = backing / f"insitu_cache.jsonl.999999{_TMP_SUFFIX}"
    orphan.write_text('{"format_version": 4}\n')

    pool = _persist_pool(geoms, backing, [double])
    assert not orphan.exists(), "the next lock holder must sweep an orphaned temp file"
    assert pool.manifest_entries == 3, "and the real log is untouched"
    pool.close()

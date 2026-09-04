"""Cross-process arbitration of a shared ``cache_dir`` (#42, part 1).

Two processes pointed at one cache directory is the obvious thing to do, and unarbitrated
it corrupts data silently. The re-admission window is the sharp edge: a reader holds a
chunk's mapping while a writer re-allocates the same chunk, and ``open_memmap(mode="w+")``
truncates the file *under the reader*. Right shape, right dtype, wrong numbers; throughput
and smoke tests all pass.

The tests here are deliberately multi-process. A same-process pool arbitrates itself with
its own lock, so nothing below is observable within one, and neither half of the
contract -- atomic replace, and an advisory ``flock`` -- means anything until there are two.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries
from insitubatch.pool import _TMP_SUFFIX, ChunkPool

# Maps the chunk's .npy, reports its hash, waits for a line on stdin, reports it again.
# The second hash is read back through the SAME mapping -- which is the whole point.
_READER = """
import hashlib, sys
import numpy as np
a = np.lib.format.open_memmap(sys.argv[1], mode="r")
print(hashlib.sha256(a.tobytes()).hexdigest(), flush=True)
sys.stdin.readline()
print(hashlib.sha256(a.tobytes()).hexdigest(), flush=True)
"""


# Opens a writing pool over the cache dir and parks, holding the exclusive lock, until it
# is killed. A subprocess is the point: `flock` is per open file description, and only a
# real process death can demonstrate that the kernel -- not us -- releases it.
_HOLDER = """
import sys
from insitubatch import obstore_store, open_geometries
from insitubatch.pool import ChunkPool
cache, url = sys.argv[1], sys.argv[2]
pool = ChunkPool(open_geometries(obstore_store(url), variables=["t2m"]),
                 backing_dir=cache, persist=True)
print("locked", flush=True)
sys.stdin.readline()
"""


# Opens a *read-only* pool and parks, holding the shared lock, until it is killed.
_SHARED_HOLDER = """
import sys
from insitubatch import obstore_store, open_geometries
from insitubatch.pool import ChunkPool
cache, url = sys.argv[1], sys.argv[2]
pool = ChunkPool(open_geometries(obstore_store(url), variables=["t2m"]),
                 backing_dir=cache, readonly_cache=True)
assert pool.pin_if_ready("t2m", 0, pool.new_owner())
print("reading", flush=True)
sys.stdin.readline()
"""


def _geoms(url, var="t2m"):
    return open_geometries(obstore_store(url), variables=[var])


def _fill(pool, geom, var, cid, src):
    """Admit + deliver every stored tile of one outer chunk, then wait for READY."""
    owner = pool.new_owner()
    assert pool.try_admit(var, cid, owner)
    spc = geom.sample_chunk_size
    for coord in geom.inner_coords():
        pool.deliver_tile(var, cid, coord, src[cid * spc : (cid + 1) * spc])
    pool.wait_ready(var, cid, owner)
    return owner


def _warm(url, cache, srcs, var="t2m", cids=(0,)):
    """Populate a persist cache over ``cache`` and close it, leaving the files behind."""
    geoms = _geoms(url, var)
    pool = ChunkPool(geoms, backing_dir=cache, persist=True)
    for cid in cids:
        _fill(pool, geoms[var], var, cid, srcs[var])
    pool.close()
    return geoms


def _hash(path):
    return hashlib.sha256(np.lib.format.open_memmap(path, mode="r").tobytes()).hexdigest()


def test_readmission_does_not_corrupt_another_process_mapping(write_zarr, tmp_path):
    """Re-admitting a chunk another process has mapped must not truncate it.

    POSIX keeps the old inode alive for anyone holding it, so replacing the directory
    entry (write a temp file, ``rename``) is safe where ``open_memmap(mode="w+")`` --
    which truncates in place -- is not. The reader here takes no lock at all: it is a
    bare mapping, so this stays a test of the *file* mechanics rather than of the
    arbitration layered on top.
    """
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    npy = cache / "t2m__0.npy"
    expected = _hash(npy)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READER, str(npy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert reader.stdout.readline().strip() == expected, "reader mapped the wrong bytes"
        # A miss on a chunk whose .npy is on disk is reachable: an evicted (demoted) entry
        # whose revive a full budget declines, or -- where no lock can be taken -- any second
        # process with persist=False sharing the dir. try_admit IS that re-admission; going
        # through it directly keeps the test on the mechanic.
        pool = ChunkPool(geoms, backing_dir=cache, persist=True)
        owner = pool.new_owner()
        assert pool.try_admit("t2m", 0, owner)
        reader.stdin.write("go\n")
        reader.stdin.flush()
        after = reader.stdout.readline().strip()
        pool.close()
    finally:
        reader.stdin.close()
        reader.wait(timeout=30)
    assert after == expected, "re-admission truncated a mapping another process was holding"


def test_second_writer_fails_fast_with_guidance(write_zarr, tmp_path):
    """Two writers on one cache_dir is an error, not a warning -- and the error has to be
    actionable: who holds it, what to do either way, and the two commands that answer
    "who?". The alternative outcome is silent corruption, so a vague message wastes the
    one chance to say so."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _geoms(url)
    first = ChunkPool(geoms, backing_dir=cache, persist=True)
    try:
        with pytest.raises(RuntimeError) as exc:
            ChunkPool(geoms, backing_dir=cache, persist=True)
    finally:
        first.close()
    msg = str(exc.value)
    assert "already open for writing" in msg
    assert f"PID {os.getpid()}" in msg  # the holder, read back from the lockfile
    assert "readonly_cache=True" in msg  # what to do if two jobs on one cache was intended
    assert "separate cache_dirs" in msg  # ...and if it was not
    assert "fuser" in msg and "lsof" in msg
    assert "Do NOT delete the lockfile" in msg
    # Released with the pool, so the next run gets it.
    ChunkPool(geoms, backing_dir=cache, persist=True).close()


def test_the_lock_keys_on_cache_dir_not_on_persist(write_zarr, tmp_path):
    """A correction to #42 as filed: scoping the lock to persist mode leaves the hole open.

    ``_alloc`` writes ``{array}__{cid}.npy`` whenever a backing dir is set, so two
    processes sharing a spill dir with ``persist=False`` collide on identical filenames
    just as surely as two persist runs do."""
    url, _ = write_zarr()
    cache = tmp_path / "spill"
    geoms = _geoms(url)
    first = ChunkPool(geoms, backing_dir=cache)  # persist=False: ephemeral spill
    try:
        with pytest.raises(RuntimeError, match="already open for writing"):
            ChunkPool(geoms, backing_dir=cache)
    finally:
        first.close()


def test_a_reader_and_a_writer_do_not_coexist(write_zarr, tmp_path):
    """``readonly_cache`` takes the lock shared, so it cannot open a cache being warmed --
    which is the honest answer: a cache still being written is not complete, and
    completeness is the whole contract of the flag."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    writer = ChunkPool(geoms, backing_dir=cache, persist=True)
    try:
        with pytest.raises(RuntimeError, match="being written right now"):
            ChunkPool(geoms, backing_dir=cache, readonly_cache=True)
    finally:
        writer.close()


def test_many_readers_coexist(write_zarr, tmp_path):
    """The workload the single-writer rule is shaped around: one job warms a cache, several
    score against it at once. Shared locks do not exclude each other."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    readers = [ChunkPool(geoms, backing_dir=cache, readonly_cache=True) for _ in range(3)]
    try:
        for reader in readers:
            assert reader.pin_if_ready("t2m", 0, reader.new_owner()), "a warm entry must revive"
        # ...and a writer is still excluded while any of them is open.
        with pytest.raises(RuntimeError, match="already open for writing"):
            ChunkPool(geoms, backing_dir=cache, persist=True)
    finally:
        for reader in readers:
            reader.close()
    ChunkPool(geoms, backing_dir=cache, persist=True).close()  # ...and allowed once they go


def test_readonly_miss_raises_and_names_the_chunk(write_zarr, tmp_path):
    """A miss in read-only mode is the contract turning out to be false. Raising is what
    makes ``readonly_cache`` mean "this cache is complete for what I am about to read"
    rather than "quietly re-fetch whatever is missing"."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs, cids=(0,))  # only chunk 0 is warm
    reader = ChunkPool(geoms, backing_dir=cache, readonly_cache=True)
    try:
        owner = reader.new_owner()
        assert reader.try_admit("t2m", 0, owner), "the warm chunk is served"
        with pytest.raises(RuntimeError) as exc:
            reader.try_admit("t2m", 1, owner)
    finally:
        reader.close()
    msg = str(exc.value)
    assert "chunk 1 of 't2m'" in msg  # names what was missing
    assert str(cache) in msg
    assert "split, sample_range or transform set" in msg  # ...and the usual cause


def test_readonly_cache_never_writes(write_zarr, tmp_path):
    """A read-only opener must leave the directory byte-identical: no new .npy, no log
    entry, no header. Otherwise "read-only" is a name, not a property."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    before = {f.name: f.stat().st_mtime_ns for f in cache.iterdir() if f.name != ".insitu.lock"}
    log_before = (cache / "insitu_cache.jsonl").read_bytes()

    reader = ChunkPool(geoms, backing_dir=cache, readonly_cache=True)
    assert reader.pin_if_ready("t2m", 0, reader.new_owner())
    reader.close()

    after = {f.name: f.stat().st_mtime_ns for f in cache.iterdir() if f.name != ".insitu.lock"}
    assert after == before, "readonly_cache modified the cache directory"
    assert (cache / "insitu_cache.jsonl").read_bytes() == log_before


def test_readonly_cache_rejects_reset_stale_cache(write_zarr, tmp_path):
    """The two flags contradict each other: a read-only opener may not delete files another
    process may be reading. Rejected at construction rather than silently ignored."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    with pytest.raises(ValueError, match="contradict"):
        ChunkPool(geoms, backing_dir=cache, readonly_cache=True, reset_stale_cache=True)


def test_readonly_cache_requires_a_cache_dir(write_zarr, tmp_path):
    """...and requires the directory to already exist: a read-only opener reads a cache
    another run warmed, it never creates one."""
    url, _ = write_zarr()
    geoms = _geoms(url)
    with pytest.raises(ValueError, match="requires cache_dir"):
        ChunkPool(geoms, readonly_cache=True)
    with pytest.raises(ValueError, match="does not exist"):
        ChunkPool(geoms, backing_dir=tmp_path / "never-warmed", readonly_cache=True)


def test_stale_tmp_files_are_swept(write_zarr, tmp_path):
    """A writer killed between `open_memmap` and `rename` leaves a temp file. The next
    writer to hold the exclusive lock sweeps them -- which is the only moment it is safe,
    since holding LOCK_EX is what proves no other writer has one in flight."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    litter = cache / f"t2m__3.npy.99999{_TMP_SUFFIX}"
    litter.write_bytes(b"partial")
    keep = cache / "t2m__0.npy"

    pool = ChunkPool(geoms, backing_dir=cache, persist=True)
    pool.close()
    assert not litter.exists(), "a crashed writer's temp file must be swept"
    assert keep.exists(), "the sweep must not touch real cache entries"

    # A read-only opener never sweeps: it does not hold the exclusive lock, so it cannot
    # know whether a live writer is mid-rename.
    litter.write_bytes(b"partial")
    reader = ChunkPool(geoms, backing_dir=cache, readonly_cache=True)
    reader.close()
    assert litter.exists(), "readonly_cache must not delete anything"


def test_a_sigkilled_holders_lock_is_immediately_reacquirable(write_zarr, tmp_path):
    """There is no such thing as a stale ``flock``: the kernel releases it when the process
    dies, ``SIGKILL`` included. Pinned into CI rather than left as something verified once
    by hand, because the docs promise it and it is the first thing a user will doubt."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(cache), url],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already open for writing"):
            ChunkPool(geoms, backing_dir=cache, persist=True)
        holder.kill()
        holder.wait(timeout=30)
    finally:
        holder.stdin.close()
    lockfile = cache / ".insitu.lock"
    assert lockfile.exists(), "the lockfile outlives the holder -- it is the inode, not the lock"
    assert str(holder.pid) in lockfile.read_text(), "...still naming the dead PID"
    ChunkPool(geoms, backing_dir=cache, persist=True).close()  # ...and yet immediately free


def test_end_to_end_readonly_cache_serves_a_warm_cache(write_zarr, tmp_path):
    """The workload the whole feature is for: one run warms a cache, a second reads it
    without writing and gets byte-identical batches -- fetching nothing."""
    from insitubatch import split_by_chunk
    from insitubatch.source import InSituDataset

    url, _ = write_zarr(n=80, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    cache = str(tmp_path / "cache")

    def make(**kw):
        return InSituDataset(
            obstore_store(url),
            manifest,
            batch_size=4,
            block_chunks=4,
            shuffle=False,
            cache_dir=cache,
            cache_budget_bytes=10_000_000,
            **kw,
        )

    warm = make(persist=True)
    try:
        warm.set_epoch(0)
        expected = np.concatenate([b.arrays["t2m"] for b in warm.train])
    finally:
        warm.close()

    reader = make(readonly_cache=True)
    try:
        reader.set_epoch(0)
        got = np.concatenate([b.arrays["t2m"] for b in reader.train])
    finally:
        reader.close()
    np.testing.assert_array_equal(got, expected)
    assert reader.cache_misses == 0, "a read-only run must fetch nothing"
    assert reader.cache_hits > 0


def test_a_network_filesystem_warns_that_it_cannot_be_arbitrated(write_zarr, tmp_path, caplog):
    """NFS is where the lock may be emulated per client, so two hosts can both hold it.
    We cannot fix that, so we must say it -- this is the one configuration where two
    writers still corrupt each other."""
    url, _ = write_zarr()
    monkey = pytest.MonkeyPatch()
    monkey.setattr("insitubatch.pool._filesystem_type", lambda _p: "nfs4")
    try:
        with caplog.at_level("WARNING", logger="insitubatch.pool"):
            ChunkPool(_geoms(url), backing_dir=tmp_path / "cache").close()
    finally:
        monkey.undo()
    msg = caplog.text
    assert "nfs4" in msg and "unarbitrated" in msg
    assert "silently corrupt" in msg


def test_filesystem_type_reads_the_real_mount_table(tmp_path):
    """Longest-prefix match on /proc/self/mountinfo, on the resolved path. ``/proc`` is the
    one mount every Linux box has that differs from its parent, so it is the check that
    proves the match is by mount point rather than by luck."""
    if not os.path.exists("/proc/self/mountinfo"):
        pytest.skip("no /proc: not Linux")  # macOS/Windows -> None, by design
    from insitubatch.pool import _filesystem_type

    assert _filesystem_type("/proc") == "proc"
    assert _filesystem_type("/proc/self") == "proc"
    assert _filesystem_type(tmp_path) not in (None, "proc")


def test_log_entries_append_regardless_of_file_position(write_zarr, tmp_path):
    """Manifest entries go out as one ``os.write`` on an ``O_APPEND`` fd, so the seek to the
    end and the write are a single kernel operation and a sub-PIPE_BUF entry can never
    interleave with another writer's. Seeking the fd to the start proves it: with an
    ordinary handle the next entry would overwrite the header.

    The lock already gives us a single writer. This keeps the *format* robust independently
    of it -- which is what makes a log written on an unarbitrated platform readable."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _geoms(url)
    pool = ChunkPool(geoms, backing_dir=cache, persist=True)
    try:
        _fill(pool, geoms["t2m"], "t2m", 0, srcs["t2m"])
        os.lseek(pool._log_fd, 0, os.SEEK_SET)  # a writer that ignored O_APPEND would clobber
        _fill(pool, geoms["t2m"], "t2m", 1, srcs["t2m"])
    finally:
        pool.close()
    lines = (cache / "insitu_cache.jsonl").read_text().splitlines()
    assert len(lines) == 3, "header + one entry per completed chunk, none overwritten"
    assert "format_version" in lines[0]
    assert [json.loads(x)["chunk_index"] for x in lines[1:]] == [0, 1]


def test_the_lock_outlives_every_mapping_the_pool_holds(write_zarr, tmp_path, monkeypatch):
    """The lock must be released *last*, after the final mmap is closed.

    Otherwise ``close()`` opens a window where this process still has cache files mapped
    while another process is already free to take the write lock and replace them. Atomic
    replace means that window cannot corrupt data -- POSIX keeps our inode alive -- but the
    invariant the arbitration rests on is "hold the lock for as long as you hold a mapping",
    and a fix that leans on the other layer to cover it is a fix that stops working the day
    the other layer changes."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    pool = ChunkPool(geoms, backing_dir=cache, persist=True)
    assert pool.pin_if_ready("t2m", 0, pool.new_owner()), "need a live mapping to close"

    held: list[bool] = []
    real_free = ChunkPool._free
    monkeypatch.setattr(
        ChunkPool,
        "_free",
        lambda self, slot, *, keep_file: (
            held.append(self._lock_fd is not None),
            real_free(self, slot, keep_file=keep_file),
        )[1],
    )
    pool.close()
    assert held, "the pool freed no mapping -- the test proves nothing"
    assert all(held), "close() released the cache lock while mappings were still open"


def test_a_writer_cannot_start_while_another_process_is_reading(write_zarr, tmp_path):
    """The question the single-writer rule has to answer: what happens to active readers
    when a writer starts overwriting? Nothing -- because the writer never starts.

    A reader holds ``LOCK_SH``, which excludes ``LOCK_EX``, so the writer is refused at
    construction, before it can allocate, evict, or reset anything. Cross-process on
    purpose: the in-process case (``test_a_reader_and_a_writer_do_not_coexist``) proves the
    same thing, but a real reader in another process is the case people actually run."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _warm(url, cache, srcs)
    reader = subprocess.Popen(
        [sys.executable, "-c", _SHARED_HOLDER, str(cache), url],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert reader.stdout.readline().strip() == "reading"
        with pytest.raises(RuntimeError, match="already open for writing"):
            ChunkPool(geoms, backing_dir=cache, persist=True)
        # A second reader is still fine -- readers do not exclude each other.
        ChunkPool(geoms, backing_dir=cache, readonly_cache=True).close()
    finally:
        reader.stdin.close()
        reader.wait(timeout=30)
    ChunkPool(geoms, backing_dir=cache, persist=True).close()  # free once the reader exits


def test_deleting_a_cached_file_does_not_disturb_a_held_mapping(write_zarr, tmp_path):
    """Invalidation is deletion (``reset_stale_cache``), and deletion is safe for a reader
    that already holds the mapping: POSIX keeps the inode alive until the last reference
    goes. So even if the lock were bypassed -- a network filesystem, a platform without
    advisory locking, a user with ``rm`` -- an active reader keeps reading real data. What
    it can lose is a *future* open, which is a miss, never wrong numbers.

    This is the second layer, tested on its own so we know it is load-bearing rather than
    merely implied by the lock."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    _warm(url, cache, srcs)
    npy = cache / "t2m__0.npy"
    expected = _hash(npy)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READER, str(npy)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert reader.stdout.readline().strip() == expected
        npy.unlink()  # what a stale-cache reset does to every entry it listed
        reader.stdin.write("go\n")
        reader.stdin.flush()
        after = reader.stdout.readline().strip()
    finally:
        reader.stdin.close()
        reader.wait(timeout=30)
    assert after == expected, "unlinking the file corrupted a mapping already held on it"


def test_a_slot_file_is_named_by_where_it_landed_not_where_it_was_written(write_zarr, tmp_path):
    """The rename leaves the two consumers of a slot's path with a name that no longer
    exists unless it is corrected.

    ``_alloc`` opens the mapping on ``<name>.<pid>.insitu-tmp`` and renames it to
    ``<name>``. ``_record_completed`` names the manifest entry from that path and ``_free``
    unlinks through it, so if the temp name survived, persist would log an entry that cannot
    be reopened and spill would leak every file it was supposed to remove. Both failures are
    silent -- a leaked spill file is just disk, and a bad log entry reads as a cold cache --
    which is why this is asserted directly rather than left to the two tests that would
    happen to catch it."""
    url, srcs = write_zarr()
    cache = tmp_path / "cache"
    geoms = _geoms(url)

    # persist: the recorded entry must be the bare landed name, and reopenable.
    pool = ChunkPool(geoms, backing_dir=cache, persist=True)
    _fill(pool, geoms["t2m"], "t2m", 0, srcs["t2m"])
    assert pool._persisted[("t2m", 0)] == "t2m__0.npy"
    assert not list(cache.glob(f"*{_TMP_SUFFIX}")), "the temp name must not survive the rename"
    logged = json.loads((cache / "insitu_cache.jsonl").read_text().splitlines()[1])["file"]
    assert logged == "t2m__0.npy"
    np.lib.format.open_memmap(cache / logged, mode="r")  # the log entry actually opens
    pool.close()
    assert (cache / "t2m__0.npy").exists(), "persist must keep the landed file"

    # spill: _free unlinks through the same path, so the landed file must go.
    spill = tmp_path / "spill"
    pool = ChunkPool(geoms, backing_dir=spill)
    _fill(pool, geoms["t2m"], "t2m", 0, srcs["t2m"])
    assert (spill / "t2m__0.npy").exists()
    pool.close()
    assert not list(spill.glob("*.npy")), "spill leaked the file it meant to unlink"

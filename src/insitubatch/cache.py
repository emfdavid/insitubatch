"""Pluggable, byte-bounded caches of *prepped* chunks.

A cache stores decoded + chunk-transformed :class:`DecodedChunk` objects keyed by
``(array, chunk_index)``; a hit skips fetch + decode + chunk transforms (see
docs/architecture.md, "The caching continuum"). Owned by ``InSituDataset`` so it
persists across epochs.

Two implementations behind one `ChunkCache` protocol:

- **MemoryCache** — LRU on the Python heap. Simple and fastest on a hit, but
  competes for the very RAM we bound; use for small/hot working sets.
- **DiskCache** — LRU of mmap'd ``.npy`` files in a directory (point it at local
  NVMe). The prepped chunk lives in a file; a hit ``mmap``s it read-only and the
  gather faults only the pages it needs. The RAM footprint is then **reclaimable,
  kernel-managed page cache**, not anonymous heap — so the training working set
  (buffer + prefetch queue) stays bounded while the cache is bounded *on disk* by
  a byte budget. Files persist, enabling cross-*run* reuse.

Both bound by **bytes** (chunk sizes vary by variable/level, so bytes — not count
— is the right budget). Returned chunks are **shared, read-only** (transforms
produce new arrays; gather fancy-indexes into copies), which is what makes the
DiskCache mmap safe.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

import numpy as np

from .types import ChunkRead, DecodedChunk


@runtime_checkable
class ChunkCache(Protocol):
    """Cache of prepped chunks keyed ``(array, chunk_index)``, bounded by bytes."""

    hits: int
    misses: int

    def get(self, array: str, chunk_index: int) -> DecodedChunk | None: ...
    def put(self, array: str, chunk_index: int, chunk: DecodedChunk) -> None: ...


class MemoryCache:
    """Heap LRU of prepped chunks, bounded by total bytes."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self.max_bytes = max_bytes
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[tuple[str, int], DecodedChunk] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, array: str, chunk_index: int) -> DecodedChunk | None:
        key = (array, chunk_index)
        with self._lock:
            chunk = self._store.get(key)
            if chunk is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return chunk

    def put(self, array: str, chunk_index: int, chunk: DecodedChunk) -> None:
        key = (array, chunk_index)
        nbytes = int(chunk.data.nbytes)
        with self._lock:
            old = self._store.pop(key, None)
            if old is not None:
                self._bytes -= int(old.data.nbytes)
            self._store[key] = chunk
            self._bytes += nbytes
            while self._bytes > self.max_bytes and len(self._store) > 1:
                _, evicted = self._store.popitem(last=False)
                self._bytes -= int(evicted.data.nbytes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


@dataclass(slots=True)
class _Entry:
    path: Path
    nbytes: int
    sample_offset: int


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class DiskCache:
    """LRU of mmap'd ``.npy`` files (point ``cache_dir`` at local NVMe).

    This implementation keeps an in-process index, so reuse is within a process. A
    dir scan on init for cross-*run* reuse + a content fingerprint in the key are
    a small follow-up (see docs/architecture.md). Linux/POSIX assumed: evicting a
    file still referenced by an open mmap unlinks it but keeps it readable until
    that mmap is dropped.
    """

    def __init__(self, cache_dir: str | Path, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.hits = 0
        self.misses = 0
        self._index: OrderedDict[tuple[str, int], _Entry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, array: str, chunk_index: int) -> DecodedChunk | None:
        key = (array, chunk_index)
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._index.move_to_end(key)
            self.hits += 1
            data = np.load(entry.path, mmap_mode="r")  # read-only memmap
            return DecodedChunk(
                read=ChunkRead(array=array, chunk_index=chunk_index),
                data=data,
                sample_offset=entry.sample_offset,
            )

    def put(self, array: str, chunk_index: int, chunk: DecodedChunk) -> None:
        key = (array, chunk_index)
        nbytes = int(chunk.data.nbytes)
        path = self.dir / f"{_safe(array)}__{chunk_index}.npy"
        tmp = self.dir / f".tmp-{uuid4().hex}.npy"
        np.save(tmp, chunk.data)  # tmp ends .npy -> no double suffix
        tmp.replace(path)  # atomic publish
        with self._lock:
            old = self._index.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._index[key] = _Entry(path=path, nbytes=nbytes, sample_offset=chunk.sample_offset)
            self._bytes += nbytes
            while self._bytes > self.max_bytes and len(self._index) > 1:
                _, evicted = self._index.popitem(last=False)
                self._bytes -= evicted.nbytes
                evicted.path.unlink(missing_ok=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

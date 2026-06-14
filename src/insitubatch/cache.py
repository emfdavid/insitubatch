"""Bounded LRU cache of *prepped* chunks.

Stores decoded + chunk-transformed :class:`DecodedChunk` objects keyed by
``(array, chunk_index)``, so a hit skips fetch + decode + chunk transforms (see
docs/architecture.md, "The caching continuum"). Owned by ``InSituDataset`` so it
persists across epochs — that is where the multi-epoch / fat-chunk / scoring
reuse win comes from; the per-epoch shuffle-block buffer is the within-epoch tier
above it.

v1 is RAM-only with chunk-count LRU eviction. A spill-to-NVMe tier and a content
fingerprint in the key (for cross-*run* reuse) are deferred — within a process a
single cache instance is scoped to one fixed chunk-transform pipeline, so the
``(array, chunk_index)`` key is sufficient. See docs/architecture.md
("Future: persistent (NVMe) tier") for that design sketch (fingerprinting,
raw-vs-prepped tiers, two-tier eviction, GDS synergy).

Returned chunks are **shared, not copied** — treat their arrays as read-only.
That holds today: chunk transforms produce new arrays and ``gather_batch`` fancy-
indexes into copies, so nothing downstream mutates a cached array in place.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from .types import DecodedChunk


class ChunkCache:
    """Thread-safe bounded LRU cache of prepped chunks, keyed ``(array, chunk_index)``."""

    def __init__(self, capacity_chunks: int) -> None:
        if capacity_chunks < 1:
            raise ValueError(f"capacity_chunks must be >= 1, got {capacity_chunks}")
        self.capacity = capacity_chunks
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[tuple[str, int], DecodedChunk] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, array: str, chunk_index: int) -> DecodedChunk | None:
        key = (array, chunk_index)
        with self._lock:
            chunk = self._store.get(key)
            if chunk is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)  # mark most-recently-used
            self.hits += 1
            return chunk

    def put(self, array: str, chunk_index: int, chunk: DecodedChunk) -> None:
        key = (array, chunk_index)
        with self._lock:
            self._store[key] = chunk
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)  # evict least-recently-used

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict[str, int]:
        """Hits, misses, current size — for benchmarking and observability."""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}

"""M-W phase 1: offset variables (`ArrayGeometry.shift`) + anchor range validity.

Pure coordinate-space behavior -- no engine. A variable is ``(path, offset)``; a window
is a set of offsets around a shared anchor; the only validity the engine enforces is that
every read ``anchor + offset`` stays on the array.
"""

from __future__ import annotations

import numpy as np

from insitubatch import valid_anchor_range
from insitubatch.types import ArrayGeometry


def _geom(n: int = 20, offset: int = 0) -> ArrayGeometry:
    return ArrayGeometry(
        path="t2m", shape=(n, 4, 4), chunks=(4, 4, 4), dtype=np.dtype("f4"), offset=offset
    )


def test_offset_defaults_zero_and_shift_composes() -> None:
    g = _geom()
    assert g.offset == 0
    s = g.shift(1)
    assert s.offset == 1
    # same array, only the offset moves -- two views that will share decoded slots
    assert (s.path, s.shape, s.chunks, s.dtype) == (g.path, g.shape, g.chunks, g.dtype)
    assert s.shift(1).offset == 2  # composes relatively
    assert g.shift(-3).offset == -3
    assert g.offset == 0  # frozen: the original is untouched


def test_valid_anchor_range_drops_edge_anchors() -> None:
    T = 20
    assert valid_anchor_range([0], T) == (0, T)  # no window -> every anchor
    assert valid_anchor_range([0, 1], T) == (0, T - 1)  # +1 target -> drop last
    assert valid_anchor_range([-1, 0], T) == (1, T)  # -1 history -> drop first
    assert valid_anchor_range([-1, 0, 1], T) == (1, T - 1)  # both ends
    assert valid_anchor_range([-2, 3], T) == (2, T - 3)  # asymmetric
    assert valid_anchor_range([], T) == (0, T)  # no offsets


def test_valid_anchor_range_empty_when_window_exceeds_array() -> None:
    lo, hi = valid_anchor_range([0, 25], 20)  # window wider than the array
    assert lo >= hi  # no anchor can satisfy it

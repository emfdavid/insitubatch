"""The examples' shared split check: the size it names must hold on any sample axis."""

from __future__ import annotations

import numpy as np
import pytest

from examples._splits import minimum_samples
from insitubatch import ArrayGeometry

FRACTIONS = (0.8, 0.1, 0.1)


@pytest.mark.parametrize("horizon", [0, 24])
def test_minimum_samples_solves_the_sample_axis_not_axis_zero(horizon) -> None:
    """The IDR mask samples Z, a middle axis, 30 planes to a chunk behind a T axis chunked 1.

    The solver must lengthen and split *that* axis. The reference is the same array laid out
    Z-first, where axis 0 and the sample axis coincide -- the case the advection tests pin
    against real runs. Solving axis 0 instead reads 30 one-plane chunks where there is one
    thirty-plane chunk, and names a size whose val split is still empty.
    """
    middle = ArrayGeometry(
        "m", (1, 1, 236, 275, 271), (1, 1, 30, 69, 68), np.dtype("i8"), sample_axis=2
    )
    z_first = ArrayGeometry("m", (236, 1, 1, 275, 271), (30, 1, 1, 69, 68), np.dtype("i8"))

    expected = minimum_samples(z_first, FRACTIONS, (0, horizon))
    assert expected == 180  # six 30-plane chunks
    assert minimum_samples(middle, FRACTIONS, (0, horizon)) == expected

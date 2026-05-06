"""Tests for Geometry1D and BoundarySpec."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.geometry import Geometry1D


def test_geometry_dx_from_length_and_n_grid() -> None:
    geom = Geometry1D(n_grid=11, length=1.0)
    # 11 nodes, 10 intervals -> dx = 0.1
    assert geom.dx == pytest.approx(0.1)


def test_geometry_positions_are_uniform_from_zero_to_length() -> None:
    geom = Geometry1D(n_grid=5, length=2.0)
    np.testing.assert_allclose(geom.positions, [0.0, 0.5, 1.0, 1.5, 2.0])


def test_geometry_n_grid_must_be_at_least_2() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        Geometry1D(n_grid=1, length=1.0)


def test_geometry_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="length"):
        Geometry1D(n_grid=10, length=0.0)

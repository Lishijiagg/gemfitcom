"""Tests for SpatialState dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.state import SpatialState


def test_construct_from_arrays() -> None:
    C = np.zeros((3, 50))
    B = np.ones((2, 50))
    state = SpatialState(metabolites=C, biomass=B, t=0.5)
    assert state.n_metabolites == 3
    assert state.n_species == 2
    assert state.n_grid == 50
    assert state.t == 0.5


def test_default_t_is_zero() -> None:
    state = SpatialState(metabolites=np.zeros((1, 4)), biomass=np.zeros((1, 4)))
    assert state.t == 0.0


def test_metabolites_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        SpatialState(metabolites=np.zeros(10), biomass=np.zeros((1, 10)))


def test_biomass_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        SpatialState(metabolites=np.zeros((1, 10)), biomass=np.zeros(10))


def test_grid_dim_must_match() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        SpatialState(metabolites=np.zeros((1, 10)), biomass=np.zeros((1, 8)))


def test_from_arrays_classmethod_uses_kwargs() -> None:
    state = SpatialState.from_arrays(C=np.zeros((1, 5)), B=np.zeros((1, 5)), t=2.0)
    assert state.t == 2.0
    assert state.metabolites.dtype == np.float64

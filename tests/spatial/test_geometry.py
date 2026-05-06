"""Tests for Geometry1D and BoundarySpec."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.geometry import BoundarySpec, Geometry1D


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


def test_boundary_spec_reflecting_default_empty_values() -> None:
    bc = BoundarySpec(type="reflecting")
    assert bc.values == {}


def test_boundary_spec_flux_holds_source_dict() -> None:
    bc = BoundarySpec(type="flux", values={"EX_o2_e": 1.0e-3})
    assert bc.values["EX_o2_e"] == 1.0e-3


def test_boundary_spec_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="type"):
        BoundarySpec(type="bogus")  # type: ignore[arg-type]


def test_apply_boundary_sources_flux_adds_to_left_cell() -> None:
    geom = Geometry1D(
        n_grid=5,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"EX_o2_e": 2.0}),
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 5))
    geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=0.5)
    # 2.0 mmol/(L·h) * 0.5 h = 1.0 mmol/L added to left cell only
    np.testing.assert_allclose(C[0], [1.0, 0.0, 0.0, 0.0, 0.0])


def test_apply_boundary_sources_dirichlet_pins_right_cell() -> None:
    geom = Geometry1D(
        n_grid=5,
        length=1.0,
        bc_left=BoundarySpec(type="reflecting"),
        bc_right=BoundarySpec(type="dirichlet", values={"EX_glc__D_e": 5.0}),
    )
    C = np.full((1, 5), 1.0)
    geom.apply_boundary_sources(C, metabolite_ids=["glc__D_e"], dt=0.1)
    # Dirichlet pins the right cell to 5.0, leaves others alone
    np.testing.assert_allclose(C[0], [1.0, 1.0, 1.0, 1.0, 5.0])


def test_apply_boundary_sources_unknown_metabolite_skipped() -> None:
    geom = Geometry1D(
        n_grid=3,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"EX_xyz_e": 99.0}),
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 3))
    geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=1.0)
    # xyz_e not in metabolite_ids -> silently skipped
    np.testing.assert_allclose(C, np.zeros((1, 3)))


def test_apply_boundary_sources_rejects_non_exchange_key() -> None:
    geom = Geometry1D(
        n_grid=3,
        length=1.0,
        bc_left=BoundarySpec(type="flux", values={"o2_e": 1.0}),  # missing EX_ prefix
        bc_right=BoundarySpec(type="reflecting"),
    )
    C = np.zeros((1, 3))
    with pytest.raises(ValueError, match="EX_"):
        geom.apply_boundary_sources(C, metabolite_ids=["o2_e"], dt=1.0)


def test_geometry_default_boundaries_are_reflecting() -> None:
    geom = Geometry1D(n_grid=5, length=1.0)
    assert geom.bc_left.type == "reflecting"
    assert geom.bc_right.type == "reflecting"

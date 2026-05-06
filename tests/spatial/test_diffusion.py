"""Tests for the diffusion operator and step function."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from gemfitcom.spatial.diffusion import build_laplacian_1d, diffuse_step


def test_laplacian_returns_csr_matrix() -> None:
    L = build_laplacian_1d(n_grid=5, dx=0.1)
    assert sp.isspmatrix_csr(L)
    assert L.shape == (5, 5)


def test_laplacian_neumann_interior_row_is_standard_stencil() -> None:
    L = build_laplacian_1d(n_grid=5, dx=2.0)
    # Interior row i should be [..., 1, -2, 1, ...] / dx**2
    expected_row = np.array([0.0, 1.0, -2.0, 1.0, 0.0]) / 4.0
    np.testing.assert_allclose(L.toarray()[2], expected_row)


def test_laplacian_neumann_left_boundary_is_one_sided() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0, bc_left="neumann", bc_right="neumann")
    # Reflecting (ghost = interior): row 0 -> (C[1] - C[0]) / dx**2
    expected_row = np.array([-1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(L.toarray()[0], expected_row)


def test_laplacian_neumann_right_boundary_is_one_sided() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    expected_row = np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    np.testing.assert_allclose(L.toarray()[-1], expected_row)


def test_laplacian_dirichlet_boundary_row_is_zero() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0, bc_left="dirichlet", bc_right="neumann")
    # Dirichlet: external code pins the boundary, so the operator must not move it
    np.testing.assert_allclose(L.toarray()[0], np.zeros(5))


def test_laplacian_invalid_bc_raises() -> None:
    with pytest.raises(ValueError, match="bc_left"):
        build_laplacian_1d(n_grid=5, dx=1.0, bc_left="bogus")  # type: ignore[arg-type]


def test_diffuse_step_constant_field_unchanged() -> None:
    L = build_laplacian_1d(n_grid=10, dx=0.1)
    C = np.full((1, 10), 3.7)
    D = np.array([1e-3])
    out = diffuse_step(C, L, D, dt=0.5)
    np.testing.assert_allclose(out, C, atol=1e-12)


def test_diffuse_step_returns_new_array_does_not_mutate_input() -> None:
    L = build_laplacian_1d(n_grid=5, dx=0.1)
    C = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    D = np.array([1e-3])
    C_before = C.copy()
    _ = diffuse_step(C, L, D, dt=0.1)
    np.testing.assert_allclose(C, C_before)


def test_diffuse_step_pulse_spreads_to_neighbors() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    C = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    D = np.array([0.1])
    out = diffuse_step(C, L, D, dt=1.0)
    # Center loses to neighbors, edges still 0
    assert out[0, 2] < 1.0
    assert out[0, 1] > 0.0
    assert out[0, 3] > 0.0
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[0, 4], 0.0)


def test_diffuse_step_independent_metabolites() -> None:
    L = build_laplacian_1d(n_grid=5, dx=1.0)
    C = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],  # met 0: pulse in middle
            [1.0, 1.0, 1.0, 1.0, 1.0],  # met 1: uniform
        ]
    )
    D = np.array([0.1, 0.5])
    out = diffuse_step(C, L, D, dt=1.0)
    # Met 0 spreads; met 1 unchanged regardless of D
    assert out[0, 2] < 1.0
    np.testing.assert_allclose(out[1], C[1])


def test_diffuse_step_neumann_conserves_total_one_step() -> None:
    L = build_laplacian_1d(n_grid=20, dx=0.05)
    C = np.zeros((1, 20))
    C[0, 10] = 1.0
    D = np.array([1e-3])
    out = diffuse_step(C, L, D, dt=0.1)
    assert out.sum() == pytest.approx(C.sum(), abs=1e-12)

"""Tests for the diffusion operator and step function."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from gemfitcom.spatial.diffusion import build_laplacian_1d


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

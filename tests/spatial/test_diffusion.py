"""Tests for the diffusion operator and step function."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from gemfitcom.spatial.diffusion import build_laplacian_1d, cfl_dt_max, check_cfl, diffuse_step


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


def test_cfl_dt_max_formula() -> None:
    # safety=1 corresponds to the marginal CFL limit
    assert cfl_dt_max(dx=0.1, D_max=1e-3, safety=1.0) == pytest.approx(0.01 / 2e-3)


def test_cfl_dt_max_default_safety_is_0_4() -> None:
    assert cfl_dt_max(dx=0.1, D_max=1e-3) == pytest.approx(0.4 * 0.01 / 2e-3)


def test_check_cfl_passes_when_dt_within_limit() -> None:
    check_cfl(dt=1.0, dx=0.1, D_max=1e-3, safety=0.4)  # well below limit


def test_check_cfl_raises_when_dt_too_large() -> None:
    with pytest.raises(RuntimeError, match="CFL"):
        check_cfl(dt=100.0, dx=0.1, D_max=1.0, safety=0.4)


def test_check_cfl_error_message_includes_suggested_dt() -> None:
    with pytest.raises(RuntimeError, match=r"Reduce dt to"):
        check_cfl(dt=100.0, dx=0.1, D_max=1.0)


def test_diffuse_step_matches_analytical_gaussian() -> None:
    """Gaussian initial condition → Gaussian solution, sigma grows as sqrt(sigma0**2 + 2Dt).

    Domain is wide enough that the pulse never reaches the boundaries during
    the test window, so closed BCs don't pollute the comparison.
    """
    n_grid = 201
    length = 1.0
    dx = length / (n_grid - 1)
    D = 1.0e-3
    sigma0 = 0.05
    center = 0.5
    t_end = 0.1

    x = np.linspace(0.0, length, n_grid)
    C = np.exp(-((x - center) ** 2) / (2.0 * sigma0**2))[None, :]  # shape (1, n_grid)

    L = build_laplacian_1d(n_grid, dx)
    D_arr = np.array([D])
    dt = cfl_dt_max(dx, D, safety=0.4)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # exact match for t_end

    for _ in range(n_steps):
        C = diffuse_step(C, L, D_arr, dt)

    sigma_t = np.sqrt(sigma0**2 + 2.0 * D * t_end)
    expected = (sigma0 / sigma_t) * np.exp(-((x - center) ** 2) / (2.0 * sigma_t**2))

    rel_error = np.max(np.abs(C[0] - expected)) / np.max(expected)
    assert rel_error < 0.01, f"max relative error {rel_error:.4g} exceeds 1%"

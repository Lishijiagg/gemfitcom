"""Diffusion: sparse Laplacian + explicit FTCS step + CFL helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

LaplacianBC = Literal["neumann", "dirichlet"]
_VALID_LAPLACIAN_BCS: frozenset[str] = frozenset({"neumann", "dirichlet"})


def build_laplacian_1d(
    n_grid: int,
    dx: float,
    bc_left: LaplacianBC = "neumann",
    bc_right: LaplacianBC = "neumann",
) -> sp.csr_matrix:
    """Construct (1/dx**2) * second-difference operator for a 1D grid.

    Neumann (reflecting) BC: ghost cell equals the boundary cell; row reduces
    to a one-sided difference, giving zero net flux through the wall.

    Dirichlet BC: row is zeroed out, so the operator does NOT update the
    boundary cell (caller pins the value externally each step).
    """
    if bc_left not in _VALID_LAPLACIAN_BCS:
        raise ValueError(f"bc_left must be one of {sorted(_VALID_LAPLACIAN_BCS)}; got {bc_left!r}")
    if bc_right not in _VALID_LAPLACIAN_BCS:
        raise ValueError(
            f"bc_right must be one of {sorted(_VALID_LAPLACIAN_BCS)}; got {bc_right!r}"
        )

    main = -2.0 * np.ones(n_grid)
    off = np.ones(n_grid - 1)
    L = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n_grid, n_grid), format="lil")

    if bc_left == "neumann":
        L[0, 0] = -1.0
        L[0, 1] = 1.0
    else:  # dirichlet
        L[0, 0] = 0.0
        L[0, 1] = 0.0

    if bc_right == "neumann":
        L[-1, -1] = -1.0
        L[-1, -2] = 1.0
    else:  # dirichlet
        L[-1, -1] = 0.0
        L[-1, -2] = 0.0

    return (L / (dx * dx)).tocsr()


def diffuse_step(
    C: np.ndarray,
    L: sp.csr_matrix,
    D: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Explicit FTCS diffusion step.

    Args:
        C: shape (n_metabolites, n_grid), current concentrations.
        L: sparse (n_grid, n_grid) Laplacian from `build_laplacian_1d`.
        D: shape (n_metabolites,), per-metabolite diffusion coefficients.
        dt: time step.

    Returns:
        New concentration array of shape (n_metabolites, n_grid). Does not
        mutate the input.
    """
    # L @ C.T has shape (n_grid, n_metabolites); transpose back to match C.
    return C + dt * D[:, None] * (L @ C.T).T

"""Pure diffusion preserves nonnegativity within floating-point tolerance."""

from __future__ import annotations

import numpy as np

from gemfitcom.spatial.diffusion import (
    build_laplacian_1d,
    cfl_dt_max,
    diffuse_step,
)


def test_positivity_random_initial_conditions_neumann() -> None:
    n_grid = 30
    length = 1.0
    dx = length / (n_grid - 1)
    D = 1.0e-2
    L = build_laplacian_1d(n_grid, dx, bc_left="neumann", bc_right="neumann")

    rng = np.random.default_rng(seed=7)
    C = rng.uniform(low=0.0, high=1.0, size=(3, n_grid))
    D_arr = np.full(3, D)
    dt = cfl_dt_max(dx, D, safety=0.4)

    for _ in range(500):
        C = diffuse_step(C, L, D_arr, dt)
        assert C.min() >= -1e-12, f"negative concentration {C.min()} at step"

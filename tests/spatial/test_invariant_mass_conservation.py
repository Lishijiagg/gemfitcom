"""Long-run pure diffusion preserves total mass under reflecting BCs."""

from __future__ import annotations

import numpy as np

from gemfitcom.spatial.diffusion import (
    build_laplacian_1d,
    cfl_dt_max,
    diffuse_step,
)


def test_mass_conservation_1000_steps_neumann() -> None:
    n_grid = 50
    length = 1.0e-3
    dx = length / (n_grid - 1)
    D = 2.1e-9
    L = build_laplacian_1d(n_grid, dx, bc_left="neumann", bc_right="neumann")

    rng = np.random.default_rng(seed=42)
    C = rng.uniform(low=0.5, high=1.5, size=(2, n_grid))
    D_arr = np.array([D, D * 5])

    initial_mass = C.sum(axis=1) * dx
    dt = cfl_dt_max(dx, D_arr.max(), safety=0.4)

    for _ in range(1000):
        C = diffuse_step(C, L, D_arr, dt)

    final_mass = C.sum(axis=1) * dx
    rel_drift = np.abs(final_mass - initial_mass) / initial_mass
    assert np.max(rel_drift) < 1e-10, f"mass drift {rel_drift} exceeds 1e-10"

"""Build initial 1D fields (concentration or biomass) from an InitConfig."""

from __future__ import annotations

import numpy as np

from .config import InitConfig, SpeciesInitConfig

_InitLike = InitConfig | SpeciesInitConfig


def build_field_1d(cfg: _InitLike, n_grid: int) -> np.ndarray:
    """Build a 1D field of length ``n_grid`` from an init config.

    Supported modes:
        - ``uniform``:    constant ``value`` everywhere
        - ``gaussian``:   ``peak * exp(-((x - center)**2) / (2 * sigma**2))``,
                          x in [0, 1] (relative)
        - ``step``:       0 for ``x < center``, ``peak`` for ``x >= center``
        - ``from_array``: ``np.load(path)``; shape must equal ``(n_grid,)``
    """
    mode = cfg.mode
    if mode == "uniform":
        if cfg.value is None:
            raise ValueError("uniform mode requires 'value'")
        return np.full(n_grid, cfg.value, dtype=float)

    if mode == "gaussian":
        if cfg.center is None or cfg.sigma is None or cfg.peak is None:
            raise ValueError("gaussian mode requires 'center', 'sigma', and 'peak'")
        x = np.linspace(0.0, 1.0, n_grid)
        return cfg.peak * np.exp(-((x - cfg.center) ** 2) / (2.0 * cfg.sigma**2))

    if mode == "step":
        if cfg.center is None or cfg.peak is None:
            raise ValueError("step mode requires 'center' and 'peak'")
        x = np.linspace(0.0, 1.0, n_grid)
        return np.where(x >= cfg.center, cfg.peak, 0.0).astype(float)

    if mode == "from_array":
        if cfg.path is None:
            raise ValueError("from_array mode requires 'path'")
        arr = np.load(cfg.path)
        if arr.shape != (n_grid,):
            raise ValueError(f"from_array shape {arr.shape} does not match n_grid={n_grid}")
        return arr.astype(float)

    raise ValueError(f"unknown init mode {mode!r}")

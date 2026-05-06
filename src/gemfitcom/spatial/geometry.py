"""Geometry1D: 1D mucosa-lumen grid with boundary specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Geometry1D:
    """Uniform 1D grid from mucosa (index 0) to lumen (index n_grid - 1)."""

    n_grid: int
    length: float

    def __post_init__(self) -> None:
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2; got {self.n_grid}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0; got {self.length}")

    @property
    def dx(self) -> float:
        return self.length / (self.n_grid - 1)

    @property
    def positions(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.n_grid)

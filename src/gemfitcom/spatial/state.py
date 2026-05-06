"""SpatialState: concentration + biomass fields on a 1D grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialState:
    """Current state of a spatial simulation.

    Attributes:
        metabolites: ndarray (n_metabolites, n_grid), mmol/L
        biomass:     ndarray (n_species, n_grid), gDW/L
        t:           current simulation time in hours
    """

    metabolites: np.ndarray
    biomass: np.ndarray
    t: float = 0.0

    def __post_init__(self) -> None:
        if self.metabolites.ndim != 2:
            raise ValueError(
                f"metabolites must be 2D (n_metabolites, n_grid); got shape {self.metabolites.shape}"
            )
        if self.biomass.ndim != 2:
            raise ValueError(
                f"biomass must be 2D (n_species, n_grid); got shape {self.biomass.shape}"
            )
        if self.metabolites.shape[1] != self.biomass.shape[1]:
            raise ValueError(
                f"metabolites and biomass must share n_grid: "
                f"{self.metabolites.shape[1]} vs {self.biomass.shape[1]}"
            )

    @property
    def n_metabolites(self) -> int:
        return int(self.metabolites.shape[0])

    @property
    def n_species(self) -> int:
        return int(self.biomass.shape[0])

    @property
    def n_grid(self) -> int:
        return int(self.metabolites.shape[1])

    @classmethod
    def from_arrays(
        cls,
        *,
        C: np.ndarray,
        B: np.ndarray,
        t: float = 0.0,
    ) -> SpatialState:
        """Construct with explicit dtype coercion to float64."""
        return cls(
            metabolites=np.asarray(C, dtype=np.float64),
            biomass=np.asarray(B, dtype=np.float64),
            t=t,
        )

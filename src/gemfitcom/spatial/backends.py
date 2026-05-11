"""Strategy pattern for the reaction sub-step.

Backends iterate over grid cells × species and return per-step (mu, flux)
fields. ``SerialBackend`` is single-process. PR 4 will add ``JoblibBackend``
against the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .kinetics import ExchangeKinetics
from .reaction import build_exchange_index, solve_cell


class Backend(Protocol):
    """Reaction backend protocol.

    Returns:
        - ``mu``:   ``(n_species, n_grid)``, 1/h
        - ``flux``: ``(n_metabolites, n_species, n_grid)``, mmol/gDW/h
    """

    def step(
        self,
        *,
        models: list,
        kinetics: list[ExchangeKinetics],
        metabolite_ids: tuple[str, ...],
        C: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SerialBackend:
    """Single-process loop over grid cells × species.

    Attributes:
        empty_eps: Biomass threshold below which a (species, cell) pair is
            skipped (mu=0, flux=0 from zero-initialisation). Default 1e-12
            gDW/L.
    """

    empty_eps: float = 1.0e-12

    def step(
        self,
        *,
        models: list,
        kinetics: list[ExchangeKinetics],
        metabolite_ids: tuple[str, ...],
        C: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_species = len(models)
        if n_species != len(kinetics):
            raise ValueError(f"models length {n_species} != kinetics length {len(kinetics)}")
        n_metabolites, n_grid = C.shape
        if B.shape != (n_species, n_grid):
            raise ValueError(
                f"B shape {B.shape} does not match (n_species={n_species}, " f"n_grid={n_grid})"
            )

        indices = [
            build_exchange_index(models[i], kinetics[i], metabolite_ids) for i in range(n_species)
        ]

        mu = np.zeros((n_species, n_grid), dtype=float)
        flux = np.zeros((n_metabolites, n_species, n_grid), dtype=float)
        C_safe = np.maximum(C, 0.0)

        for i in range(n_species):
            for x in range(n_grid):
                if B[i, x] < self.empty_eps:
                    continue
                result = solve_cell(
                    model=models[i],
                    kinetics=kinetics[i],
                    exchange_index=indices[i],
                    C_local=C_safe[:, x],
                )
                mu[i, x] = result.mu
                flux[:, i, x] = result.flux

        return mu, flux

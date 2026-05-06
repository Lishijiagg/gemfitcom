"""Geometry1D: 1D mucosa-lumen grid with boundary specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BCType = Literal["flux", "dirichlet", "reflecting"]
_VALID_BC_TYPES: frozenset[str] = frozenset({"flux", "dirichlet", "reflecting"})


@dataclass
class BoundarySpec:
    """Boundary condition spec for one side of the 1D grid.

    type:
        - 'reflecting': no-flux Neumann (default; values must be empty)
        - 'flux':       add `value * dt` to the boundary cell each step
                        (value units: mmol/(L*h) for PR 1; physical conversion deferred)
        - 'dirichlet':  pin boundary cell to `value` (mmol/L)

    values: mapping from exchange-reaction ID (e.g. 'EX_o2_e') to numeric value.
            Keys must start with 'EX_'; the metabolite ID is the suffix ('o2_e').
    """

    type: BCType
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in _VALID_BC_TYPES:
            raise ValueError(
                f"BoundarySpec.type must be one of {sorted(_VALID_BC_TYPES)}; got {self.type!r}"
            )


def _ex_to_metabolite_id(key: str) -> str:
    if not key.startswith("EX_"):
        raise ValueError(f"Boundary key must start with 'EX_' (exchange reaction ID); got {key!r}")
    return key[3:]


@dataclass
class Geometry1D:
    """Uniform 1D grid from mucosa (index 0) to lumen (index n_grid - 1)."""

    n_grid: int
    length: float
    bc_left: BoundarySpec = field(default_factory=lambda: BoundarySpec(type="reflecting"))
    bc_right: BoundarySpec = field(default_factory=lambda: BoundarySpec(type="reflecting"))

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

    def apply_boundary_sources(
        self,
        C: np.ndarray,
        metabolite_ids: list[str],
        dt: float,
    ) -> None:
        """In-place: apply flux sources and Dirichlet pinning to boundary cells.

        Reflecting BCs are no-ops here; the diffusion operator (Task 5) handles
        the no-flux condition by construction.
        """
        for bc, idx in ((self.bc_left, 0), (self.bc_right, -1)):
            if bc.type == "reflecting":
                continue
            for ex_id, value in bc.values.items():
                met_id = _ex_to_metabolite_id(ex_id)
                if met_id not in metabolite_ids:
                    continue
                j = metabolite_ids.index(met_id)
                if bc.type == "flux":
                    C[j, idx] += value * dt
                elif bc.type == "dirichlet":
                    C[j, idx] = value

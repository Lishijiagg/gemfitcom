"""Pydantic schema for the spatial simulation YAML config (PR 1 surface).

Sections covered:
    - geometry
    - metabolites
    - simulation
    - output

Sections deferred to later PRs:
    - species   (PR 2)
    - kinetics  (PR 2)
    - backend   (PR 4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .diffusion import cfl_dt_max


class BoundaryConfig(BaseModel):
    """One side of the geometry boundary."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["flux", "dirichlet", "reflecting"]
    sources: dict[str, float] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim: Literal[1] = 1
    n_grid: int = Field(gt=1)
    length: float = Field(gt=0.0)
    boundary: dict[str, BoundaryConfig]


class InitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["uniform", "gaussian", "step", "from_array"]
    value: float | None = None
    center: float | None = None
    sigma: float | None = None
    peak: float | None = None
    path: Path | None = None


class MetaboliteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    diffusion: float = Field(ge=0.0)
    init: InitConfig


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_end: float = Field(gt=0.0)
    dt: float = Field(gt=0.0)
    snapshot_every: float = Field(gt=0.0)
    cfl_safety: float = Field(default=0.4, gt=0.0, le=1.0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["npz", "netcdf"] = "npz"
    precision: Literal["float32", "float64"] = "float32"


class SpatialConfig(BaseModel):
    """Top-level spatial config (PR 1 surface)."""

    model_config = ConfigDict(extra="forbid")

    geometry: GeometryConfig
    metabolites: list[MetaboliteConfig]
    simulation: SimulationConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> SpatialConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def check_cfl(self) -> None:
        """Raise RuntimeError if the configured dt violates CFL stability.

        Uses the largest diffusion coefficient across metabolites to set the
        bound. If all diffusion coefficients are zero, the check is a no-op.
        """
        d_max = max((m.diffusion for m in self.metabolites), default=0.0)
        if d_max <= 0.0:
            return
        dx = self.geometry.length / (self.geometry.n_grid - 1)
        dt_limit = cfl_dt_max(dx, d_max, self.simulation.cfl_safety)
        if self.simulation.dt > dt_limit:
            raise RuntimeError(
                f"dt={self.simulation.dt} violates CFL stability "
                f"(dx={dx}, D_max={d_max}, safety={self.simulation.cfl_safety}). "
                f"Reduce dt to <= {dt_limit:.4g} or coarsen the grid."
            )

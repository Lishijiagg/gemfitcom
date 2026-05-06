"""Spatial dFBA subpackage for gemfitcom.

PR 1 surface: numerical core (state, geometry, diffusion, recorder, config).
FBA coupling, simulator, CLI, and parallel backends added in subsequent PRs.
"""

from .config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
)
from .diffusion import build_laplacian_1d, cfl_dt_max, check_cfl, diffuse_step
from .geometry import BoundarySpec, Geometry1D
from .recorder import SnapshotRecorder
from .state import SpatialState

__all__ = [
    "BoundaryConfig",
    "BoundarySpec",
    "GeometryConfig",
    "Geometry1D",
    "InitConfig",
    "MetaboliteConfig",
    "OutputConfig",
    "SimulationConfig",
    "SnapshotRecorder",
    "SpatialConfig",
    "SpatialState",
    "build_laplacian_1d",
    "cfl_dt_max",
    "check_cfl",
    "diffuse_step",
]

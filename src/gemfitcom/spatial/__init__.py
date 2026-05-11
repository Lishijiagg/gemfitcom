"""Spatial dFBA subpackage for gemfitcom.

PR 1 surface: numerical core (state, geometry, diffusion, recorder, config).
PR 2 surface: kinetics + reaction + serial backend + species config + GEM URI.
PR 3 will add: Simulator + CLI.
PR 4 will add: JoblibBackend + cache.
PR 5 will add: viz.
"""

from ._init_fields import build_field_1d
from .backends import Backend, SerialBackend
from .config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
    SpeciesConfig,
    SpeciesInitConfig,
)
from .diffusion import build_laplacian_1d, cfl_dt_max, check_cfl, diffuse_step
from .geometry import BoundarySpec, Geometry1D
from .kinetics import (
    ExchangeEntry,
    ExchangeKinetics,
    load_kinetics_yaml,
    resolve_gem,
)
from .reaction import ReactionEngine, SingleCellResult, solve_cell
from .recorder import SnapshotRecorder
from .state import SpatialState

__all__ = [
    # PR 1
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
    # PR 2
    "Backend",
    "ExchangeEntry",
    "ExchangeKinetics",
    "ReactionEngine",
    "SerialBackend",
    "SingleCellResult",
    "SpeciesConfig",
    "SpeciesInitConfig",
    "build_field_1d",
    "load_kinetics_yaml",
    "resolve_gem",
    "solve_cell",
]

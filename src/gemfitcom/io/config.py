"""Configuration schema and YAML loader.

Two top-level schemas live here:

* :class:`Config` mirrors ``configs/example_strain.yaml`` — the single-strain
  input consumed by the ``gemfitcom fit`` CLI.
* :class:`CommunityConfig` mirrors ``configs/example_community.yaml`` — the
  multi-strain input consumed by the ``gemfitcom simulate`` CLI.

Nested dataclasses are constructed explicitly in the loaders rather than via
a generic dict-to-dataclass helper so validation errors stay clear. Fitted
MM parameters flow between the two stages through the small YAML format
implemented by :func:`save_fitted_params` / :func:`load_fitted_params`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from gemfitcom.kinetics.mm import MMParams

ModelSource = Literal["curated", "agora2", "carveme"]
SimulationMode = Literal["sequential_dfba", "micom", "fusion"]

_VALID_MODEL_SOURCES: tuple[str, ...] = ("curated", "agora2", "carveme")
_VALID_SIM_MODES: tuple[str, ...] = ("sequential_dfba", "micom", "fusion")


class ConfigError(ValueError):
    """Raised for invalid configuration values or missing required fields."""


@dataclass
class StrainConfig:
    name: str
    model_path: Path
    model_source: ModelSource

    def __post_init__(self) -> None:
        if self.model_source not in _VALID_MODEL_SOURCES:
            raise ConfigError(
                f"strain.model_source must be one of {_VALID_MODEL_SOURCES}, "
                f"got {self.model_source!r}"
            )
        self.model_path = Path(self.model_path)


@dataclass
class CarbonSource:
    exchange_id: str
    initial_concentration_mM: float

    def __post_init__(self) -> None:
        if self.initial_concentration_mM < 0:
            raise ConfigError(
                "medium.carbon_source.initial_concentration_mM must be >= 0, "
                f"got {self.initial_concentration_mM}"
            )


@dataclass
class MediumConfig:
    name: str
    carbon_source: CarbonSource


@dataclass
class ExperimentConfig:
    od_file: Path
    hplc_file: Path
    biomass_conversion: float
    initial_biomass_gDW_per_L: float | None = None

    def __post_init__(self) -> None:
        self.od_file = Path(self.od_file)
        self.hplc_file = Path(self.hplc_file)
        if self.biomass_conversion <= 0:
            raise ConfigError(
                f"experiment.biomass_conversion must be > 0, got {self.biomass_conversion}"
            )
        if self.initial_biomass_gDW_per_L is not None and self.initial_biomass_gDW_per_L <= 0:
            raise ConfigError(
                "experiment.initial_biomass_gDW_per_L must be > 0 when set, "
                f"got {self.initial_biomass_gDW_per_L}"
            )


@dataclass
class MICOMConfig:
    tradeoff_alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.tradeoff_alpha <= 1.0:
            raise ConfigError(
                f"simulation.micom.tradeoff_alpha must be in [0, 1], got {self.tradeoff_alpha}"
            )


@dataclass
class SimulationConfig:
    dt: float = 0.25
    total_time_h: float = 72.0
    mode: SimulationMode = "sequential_dfba"
    micom: MICOMConfig = field(default_factory=MICOMConfig)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ConfigError(f"simulation.dt must be > 0, got {self.dt}")
        if self.total_time_h <= self.dt:
            raise ConfigError(
                f"simulation.total_time_h ({self.total_time_h}) must exceed dt ({self.dt})"
            )
        if self.mode not in _VALID_SIM_MODES:
            raise ConfigError(
                f"simulation.mode must be one of {_VALID_SIM_MODES}, got {self.mode!r}"
            )


@dataclass
class KineticsFitConfig:
    vmax_bounds_mmol_per_gDW_per_h: tuple[float, float] = (0.001, 20.0)
    km_bounds_mM: tuple[float, float] = (0.001, 30.0)
    de_maxiter: int = 50
    de_popsize: int = 15
    grid_points: int = 21
    grid_span: float = 0.5

    def __post_init__(self) -> None:
        self.vmax_bounds_mmol_per_gDW_per_h = self._check_bounds(
            self.vmax_bounds_mmol_per_gDW_per_h, "kinetics_fit.vmax_bounds_mmol_per_gDW_per_h"
        )
        self.km_bounds_mM = self._check_bounds(self.km_bounds_mM, "kinetics_fit.km_bounds_mM")
        if self.de_maxiter <= 0:
            raise ConfigError(f"kinetics_fit.de_maxiter must be > 0, got {self.de_maxiter}")
        if self.de_popsize <= 0:
            raise ConfigError(f"kinetics_fit.de_popsize must be > 0, got {self.de_popsize}")
        if self.grid_points < 3:
            raise ConfigError(f"kinetics_fit.grid_points must be >= 3, got {self.grid_points}")
        if not 0.0 < self.grid_span < 1.0:
            raise ConfigError(f"kinetics_fit.grid_span must be in (0, 1), got {self.grid_span}")

    @staticmethod
    def _check_bounds(raw: Any, path: str) -> tuple[float, float]:
        try:
            lo, hi = (float(raw[0]), float(raw[1]))
        except (TypeError, IndexError, ValueError) as e:
            raise ConfigError(f"{path} must be a length-2 sequence of numbers") from e
        if lo < 0 or lo >= hi:
            raise ConfigError(f"{path} must satisfy 0 <= lo < hi, got ({lo}, {hi})")
        return (lo, hi)


@dataclass
class Config:
    strain: StrainConfig
    medium: MediumConfig
    experiment: ExperimentConfig
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kinetics_fit: KineticsFitConfig = field(default_factory=KineticsFitConfig)


def _require(data: dict[str, Any], key: str, parent: str) -> Any:
    if key not in data:
        raise ConfigError(f"missing required field: {parent}.{key}")
    return data[key]


def load_config(path: str | Path, *, validate_paths: bool = False) -> Config:
    """Load and validate a GemFitCom YAML configuration file.

    Args:
        path: path to the YAML config file.
        validate_paths: if True, check that referenced model and experiment
            files exist on disk.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        ConfigError: if any field is missing, invalid, or (when
            ``validate_paths=True``) any referenced file does not exist.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")

    strain = StrainConfig(**_require(raw, "strain", "config"))
    medium_raw = _require(raw, "medium", "config")
    medium = MediumConfig(
        name=_require(medium_raw, "name", "medium"),
        carbon_source=CarbonSource(**_require(medium_raw, "carbon_source", "medium")),
    )
    experiment = ExperimentConfig(**_require(raw, "experiment", "config"))

    sim_raw = dict(raw.get("simulation", {}))
    micom_raw = sim_raw.pop("micom", {})
    simulation = SimulationConfig(micom=MICOMConfig(**micom_raw), **sim_raw)

    kinetics_fit = KineticsFitConfig(**raw.get("kinetics_fit", {}))

    cfg = Config(
        strain=strain,
        medium=medium,
        experiment=experiment,
        simulation=simulation,
        kinetics_fit=kinetics_fit,
    )

    if validate_paths:
        for attr, p in (
            ("strain.model_path", cfg.strain.model_path),
            ("experiment.od_file", cfg.experiment.od_file),
            ("experiment.hplc_file", cfg.experiment.hplc_file),
        ):
            if not p.is_file():
                raise ConfigError(f"{attr} not found: {p}")
    return cfg


# =========================================================================
# Community (multi-strain) configuration
# =========================================================================


@dataclass
class CommunityStrainConfig:
    """One strain in a multi-strain simulation."""

    name: str
    model_path: Path
    initial_biomass: float
    fitted_params_path: Path | None = None
    mm_params: dict[str, MMParams] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError(f"strain.name must be a non-empty string, got {self.name!r}")
        self.model_path = Path(self.model_path)
        if self.initial_biomass <= 0:
            raise ConfigError(
                f"strain[{self.name!r}].initial_biomass must be > 0, got {self.initial_biomass}"
            )
        if self.fitted_params_path is not None:
            self.fitted_params_path = Path(self.fitted_params_path)


@dataclass
class CommunitySimulationConfig:
    """Simulation parameters for the community run."""

    mode: SimulationMode = "sequential_dfba"
    dt: float = 0.25
    total_time_h: float = 24.0
    save_fluxes: bool = True
    micom_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in _VALID_SIM_MODES:
            raise ConfigError(
                f"simulation.mode must be one of {_VALID_SIM_MODES}, got {self.mode!r}"
            )
        if self.dt <= 0:
            raise ConfigError(f"simulation.dt must be > 0, got {self.dt}")
        if self.total_time_h <= self.dt:
            raise ConfigError(
                f"simulation.total_time_h ({self.total_time_h}) must exceed dt ({self.dt})"
            )
        if not 0.0 <= self.micom_fraction <= 1.0:
            raise ConfigError(
                f"simulation.micom_fraction must be in [0, 1], got {self.micom_fraction}"
            )


@dataclass
class CommunityConfig:
    """Multi-strain simulation configuration.

    Attributes:
        name: Community label (used in output filenames).
        medium: Registered medium name or a path to a medium YAML.
        strains: One :class:`CommunityStrainConfig` per strain. Names must
            be unique.
        simulation: Simulation controls (mode, dt, horizon, ...).
    """

    name: str
    medium: str
    strains: list[CommunityStrainConfig]
    simulation: CommunitySimulationConfig = field(default_factory=CommunitySimulationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError(f"community.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.medium, str) or not self.medium:
            raise ConfigError("community.medium must be a non-empty string (name or path)")
        if not self.strains:
            raise ConfigError("community.strains must be a non-empty list")
        names = [s.name for s in self.strains]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(f"community.strains contains duplicate names: {dupes}")


def _parse_strain(raw: dict[str, Any]) -> CommunityStrainConfig:
    mm_raw = raw.get("mm_params", {}) or {}
    if not isinstance(mm_raw, dict):
        raise ConfigError("strain.mm_params must be a mapping")
    mm_params: dict[str, MMParams] = {}
    for ex_id, body in mm_raw.items():
        if not isinstance(body, dict) or "vmax" not in body or "km" not in body:
            raise ConfigError(f"strain.mm_params[{ex_id!r}] must contain 'vmax' and 'km'")
        mm_params[str(ex_id)] = MMParams(vmax=float(body["vmax"]), km=float(body["km"]))
    return CommunityStrainConfig(
        name=_require(raw, "name", "strain"),
        model_path=_require(raw, "model_path", "strain"),
        initial_biomass=float(_require(raw, "initial_biomass", "strain")),
        fitted_params_path=raw.get("fitted_params_path"),
        mm_params=mm_params,
    )


def load_community_config(path: str | Path, *, validate_paths: bool = False) -> CommunityConfig:
    """Load and validate a multi-strain community YAML.

    The schema has top-level keys ``community``, ``medium``, ``strains``,
    and ``simulation``. See ``configs/example_community.yaml`` for a
    reference file.

    Args:
        path: Path to the YAML config.
        validate_paths: If True, verify every strain ``model_path`` and
            (when set) ``fitted_params_path`` exists on disk. Medium
            paths are not checked here because the medium value may be a
            registered name.

    Raises:
        ConfigError: on schema violations or missing required fields.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")

    community_raw = _require(raw, "community", "config")
    strains_raw = _require(raw, "strains", "config")
    if not isinstance(strains_raw, list):
        raise ConfigError("'strains' must be a list")

    sim_raw = raw.get("simulation", {}) or {}
    simulation = CommunitySimulationConfig(**sim_raw)

    cfg = CommunityConfig(
        name=_require(community_raw, "name", "community"),
        medium=str(_require(raw, "medium", "config")),
        strains=[_parse_strain(s) for s in strains_raw],
        simulation=simulation,
    )

    if validate_paths:
        for s in cfg.strains:
            if not s.model_path.is_file():
                raise ConfigError(f"strain[{s.name!r}].model_path not found: {s.model_path}")
            if s.fitted_params_path is not None and not s.fitted_params_path.is_file():
                raise ConfigError(
                    f"strain[{s.name!r}].fitted_params_path not found: {s.fitted_params_path}"
                )
    return cfg


# =========================================================================
# Fitted-parameter exchange format (fit → simulate)
# =========================================================================


@dataclass
class FittedParams:
    """Contents of a fitted-params YAML emitted by ``gemfitcom fit``."""

    strain: str
    r_squared: float
    mm_params: dict[str, MMParams]


def save_fitted_params(
    path: str | Path,
    *,
    strain: str,
    r_squared: float,
    mm_params: dict[str, MMParams],
) -> Path:
    """Write an MMParams bundle to YAML for downstream simulation use."""
    path = Path(path)
    payload = {
        "strain": strain,
        "r_squared": float(r_squared),
        "mm_params": {k: asdict(v) for k, v in mm_params.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def load_fitted_params(path: str | Path) -> FittedParams:
    """Inverse of :func:`save_fitted_params`."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: fitted-params YAML must parse to a mapping")
    strain = raw.get("strain")
    r2 = raw.get("r_squared")
    mm_raw = raw.get("mm_params", {}) or {}
    if not isinstance(strain, str) or not strain:
        raise ConfigError(f"{path}: missing/invalid 'strain'")
    if not isinstance(r2, int | float):
        raise ConfigError(f"{path}: missing/invalid 'r_squared'")
    if not isinstance(mm_raw, dict):
        raise ConfigError(f"{path}: 'mm_params' must be a mapping")
    mm_params = {
        str(k): MMParams(vmax=float(v["vmax"]), km=float(v["km"])) for k, v in mm_raw.items()
    }
    return FittedParams(strain=strain, r_squared=float(r2), mm_params=mm_params)

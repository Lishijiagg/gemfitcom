"""Tests for the SpatialConfig pydantic schema (PR 1 surface)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gemfitcom.spatial.config import (
    BoundaryConfig,
    GeometryConfig,
    InitConfig,
    MetaboliteConfig,
    OutputConfig,
    SimulationConfig,
    SpatialConfig,
)

# ---- Component schemas ----


def test_geometry_config_minimum_valid() -> None:
    cfg = GeometryConfig(
        n_grid=50,
        length=1e-3,
        boundary={
            "mucosa": BoundaryConfig(type="reflecting"),
            "lumen": BoundaryConfig(type="reflecting"),
        },
    )
    assert cfg.n_grid == 50


def test_geometry_config_n_grid_must_be_above_one() -> None:
    with pytest.raises(ValidationError):
        GeometryConfig(n_grid=1, length=1e-3, boundary={})


def test_geometry_config_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        GeometryConfig(n_grid=10, length=0.0, boundary={})


def test_metabolite_config_requires_diffusion_and_init() -> None:
    cfg = MetaboliteConfig(
        id="o2_e",
        diffusion=2.1e-9,
        init=InitConfig(mode="uniform", value=0.21),
    )
    assert cfg.id == "o2_e"


def test_metabolite_config_negative_diffusion_rejected() -> None:
    with pytest.raises(ValidationError):
        MetaboliteConfig(
            id="o2_e",
            diffusion=-1.0,
            init=InitConfig(mode="uniform", value=0.0),
        )


def test_simulation_config_requires_positive_dt_and_t_end() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(t_end=0.0, dt=0.1, snapshot_every=1.0)
    with pytest.raises(ValidationError):
        SimulationConfig(t_end=1.0, dt=0.0, snapshot_every=1.0)


def test_simulation_config_default_cfl_safety_is_0_4() -> None:
    cfg = SimulationConfig(t_end=1.0, dt=0.1, snapshot_every=0.5)
    assert cfg.cfl_safety == 0.4


def test_output_config_defaults() -> None:
    cfg = OutputConfig()
    assert cfg.format == "npz"
    assert cfg.precision == "float32"


# ---- Top-level + YAML round-trip ----


def _minimal_config_dict() -> dict:
    return {
        "geometry": {
            "n_grid": 11,
            "length": 1.0e-3,
            "boundary": {
                "mucosa": {"type": "flux", "sources": {"EX_o2_e": 1.0e-3}},
                "lumen": {"type": "dirichlet", "values": {"EX_glc__D_e": 5.0}},
            },
        },
        "metabolites": [
            {"id": "o2_e", "diffusion": 2.1e-9, "init": {"mode": "uniform", "value": 0.21}},
            {"id": "glc__D_e", "diffusion": 6.7e-10, "init": {"mode": "uniform", "value": 0.0}},
        ],
        "simulation": {"t_end": 24.0, "dt": 0.1, "snapshot_every": 1.0},
    }


def test_spatial_config_parses_minimal_dict() -> None:
    cfg = SpatialConfig(**_minimal_config_dict())
    assert cfg.geometry.n_grid == 11
    assert len(cfg.metabolites) == 2
    assert cfg.simulation.t_end == 24.0
    assert cfg.output.format == "npz"  # default


def test_spatial_config_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.safe_dump(_minimal_config_dict()))
    cfg = SpatialConfig.from_yaml(yaml_path)
    assert cfg.geometry.n_grid == 11


def test_spatial_config_missing_required_field_raises() -> None:
    bad = _minimal_config_dict()
    del bad["simulation"]
    with pytest.raises(ValidationError):
        SpatialConfig(**bad)


def test_spatial_config_check_cfl_passes_when_safe() -> None:
    cfg_dict = _minimal_config_dict()
    cfg_dict["simulation"]["dt"] = 1e-6  # tiny dt -> safe
    SpatialConfig(**cfg_dict).check_cfl()


def test_spatial_config_check_cfl_raises_when_dt_too_large() -> None:
    cfg_dict = _minimal_config_dict()
    cfg_dict["simulation"]["dt"] = 1.0e6  # absurdly large dt
    with pytest.raises(RuntimeError, match="CFL"):
        SpatialConfig(**cfg_dict).check_cfl()


def test_spatial_config_check_cfl_no_op_when_all_diffusion_zero() -> None:
    cfg_dict = _minimal_config_dict()
    for met in cfg_dict["metabolites"]:
        met["diffusion"] = 0.0
    cfg_dict["simulation"]["dt"] = 1.0e6  # would violate CFL if any D > 0
    SpatialConfig(**cfg_dict).check_cfl()


class TestSpeciesConfig:
    def _base_yaml(self):
        """Return the geometry/simulation/metabolites portion shared by tests."""
        return """
            geometry:
              dim: 1
              n_grid: 10
              length: 1.0e-3
              boundary:
                mucosa: {type: reflecting}
                lumen:  {type: reflecting}
            simulation:
              t_end: 1.0
              dt: 0.1
              snapshot_every: 0.5
            metabolites:
              - id: glc__D_e
                diffusion: 6.7e-10
                init: {mode: uniform, value: 5.0}
            """

    def test_minimal_species_section(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(
            self._base_yaml()
            + """
            species:
              - name: ecoli
                gem: cobra://textbook
                kinetics: ./kinetics/ecoli.yaml
                init: {mode: uniform, value: 1.0e-3}
            """
        )
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        assert len(cfg.species) == 1
        sp = cfg.species[0]
        assert sp.name == "ecoli"
        assert sp.gem == "cobra://textbook"
        assert sp.kinetics.name == "ecoli.yaml"
        assert sp.init.mode == "uniform"
        assert sp.init.value == 1.0e-3

    def test_species_gaussian_init(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(
            self._base_yaml()
            + """
            species:
              - name: fprau
                gem: cobra://textbook
                kinetics: ./kinetics/fprau.yaml
                init:
                  mode: gaussian
                  center: 0.7
                  sigma: 0.1
                  peak: 1.0e-3
            """
        )
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        sp = cfg.species[0]
        assert sp.init.mode == "gaussian"
        assert sp.init.center == 0.7
        assert sp.init.sigma == 0.1
        assert sp.init.peak == 1.0e-3

    def test_species_empty_list_rejected(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(self._base_yaml().rstrip() + "\n            species: []\n")
        from gemfitcom.spatial.config import SpatialConfig

        with pytest.raises(ValueError, match="at least one"):
            SpatialConfig.from_yaml(cfg_path)

    def test_species_field_optional_for_pr1_compat(self, tmp_path):
        cfg_path = tmp_path / "sim.yaml"
        cfg_path.write_text(self._base_yaml())
        from gemfitcom.spatial.config import SpatialConfig

        cfg = SpatialConfig.from_yaml(cfg_path)
        assert cfg.species == []

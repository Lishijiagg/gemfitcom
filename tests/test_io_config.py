"""Tests for YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gemfitcom.io.config import (
    CommunityConfig,
    Config,
    ConfigError,
    SimulationConfig,
    StrainConfig,
    load_community_config,
    load_config,
    load_fitted_params,
    save_fitted_params,
)
from gemfitcom.kinetics.mm import MMParams

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "example_strain.yaml"


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLE_CONFIG)
    assert isinstance(cfg, Config)
    assert cfg.strain.name == "toy_acetogen"
    assert cfg.strain.model_source == "curated"
    assert cfg.medium.name == "YCFA"
    assert cfg.medium.carbon_source.exchange_id == "EX_glc__D_e"
    assert cfg.medium.carbon_source.initial_concentration_mM == 5.0
    assert cfg.experiment.biomass_conversion == 0.35
    assert cfg.experiment.initial_biomass_gDW_per_L == 0.01
    assert cfg.simulation.dt == 0.25
    assert cfg.simulation.mode == "sequential_dfba"
    assert cfg.simulation.micom.tradeoff_alpha == 0.5


def test_example_config_validate_paths_succeeds(tmp_path: Path) -> None:
    # The example YAML now ships with synthetic data alongside it; the paths
    # all resolve relative to the repo root, so validate_paths must succeed.
    cfg = load_config(EXAMPLE_CONFIG, validate_paths=True)
    assert cfg.strain.model_path.is_file()
    assert cfg.experiment.od_file.is_file()
    assert cfg.experiment.hplc_file.is_file()


def test_validate_paths_raises_for_missing_files(tmp_path: Path) -> None:
    cfg = {
        "strain": {
            "name": "test",
            "model_path": str(tmp_path / "missing.xml"),
            "model_source": "agora2",
        },
        "medium": {
            "name": "YCFA",
            "carbon_source": {"exchange_id": "EX_glc__D_e", "initial_concentration_mM": 5.0},
        },
        "experiment": {
            "od_file": str(tmp_path / "missing_od.csv"),
            "hplc_file": str(tmp_path / "missing_hplc.csv"),
            "biomass_conversion": 0.35,
        },
    }
    cfg_path = tmp_path / "fake.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path, validate_paths=True)


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _base_config() -> dict:
    return {
        "strain": {
            "name": "test",
            "model_path": "./nowhere.xml",
            "model_source": "agora2",
        },
        "medium": {
            "name": "YCFA",
            "carbon_source": {"exchange_id": "EX_glc__D_e", "initial_concentration_mM": 5.0},
        },
        "experiment": {
            "od_file": "./od.csv",
            "hplc_file": "./hplc.csv",
            "biomass_conversion": 0.35,
        },
    }


def test_invalid_model_source_rejected(tmp_path: Path) -> None:
    data = _base_config()
    data["strain"]["model_source"] = "not_a_source"
    with pytest.raises(ConfigError, match="model_source"):
        load_config(_write(tmp_path, data))


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    data = _base_config()
    del data["strain"]["model_path"]
    with pytest.raises(TypeError):
        # StrainConfig(**...) raises TypeError for missing kwarg; acceptable here.
        load_config(_write(tmp_path, data))


def test_missing_top_level_section_rejected(tmp_path: Path) -> None:
    data = _base_config()
    del data["medium"]
    with pytest.raises(ConfigError, match="medium"):
        load_config(_write(tmp_path, data))


def test_dt_must_be_positive(tmp_path: Path) -> None:
    data = _base_config()
    data["simulation"] = {"dt": 0.0}
    with pytest.raises(ConfigError, match="dt"):
        load_config(_write(tmp_path, data))


def test_total_time_must_exceed_dt(tmp_path: Path) -> None:
    data = _base_config()
    data["simulation"] = {"dt": 5.0, "total_time_h": 2.0}
    with pytest.raises(ConfigError, match="total_time_h"):
        load_config(_write(tmp_path, data))


def test_invalid_simulation_mode_rejected(tmp_path: Path) -> None:
    data = _base_config()
    data["simulation"] = {"mode": "not_a_mode"}
    with pytest.raises(ConfigError, match="mode"):
        load_config(_write(tmp_path, data))


def test_micom_alpha_out_of_range(tmp_path: Path) -> None:
    data = _base_config()
    data["simulation"] = {"micom": {"tradeoff_alpha": 1.5}}
    with pytest.raises(ConfigError, match="tradeoff_alpha"):
        load_config(_write(tmp_path, data))


def test_kinetics_bounds_validated(tmp_path: Path) -> None:
    data = _base_config()
    data["kinetics_fit"] = {"vmax_bounds_mmol_per_gDW_per_h": [5.0, 1.0]}
    with pytest.raises(ConfigError, match="vmax_bounds"):
        load_config(_write(tmp_path, data))


def test_negative_biomass_conversion_rejected(tmp_path: Path) -> None:
    data = _base_config()
    data["experiment"]["biomass_conversion"] = -1.0
    with pytest.raises(ConfigError, match="biomass_conversion"):
        load_config(_write(tmp_path, data))


def test_simulation_defaults_when_omitted(tmp_path: Path) -> None:
    data = _base_config()
    cfg = load_config(_write(tmp_path, data))
    assert isinstance(cfg.simulation, SimulationConfig)
    assert cfg.simulation.dt == 0.25
    assert cfg.simulation.total_time_h == 72.0
    assert cfg.simulation.mode == "sequential_dfba"


def test_strain_config_post_init_coerces_path() -> None:
    strain = StrainConfig(name="x", model_path="foo.xml", model_source="curated")
    assert isinstance(strain.model_path, Path)


# ---------- CommunityConfig ----------


def _base_community() -> dict:
    return {
        "community": {"name": "demo"},
        "medium": "YCFA",
        "strains": [
            {
                "name": "A",
                "model_path": "./A.xml",
                "initial_biomass": 0.01,
            },
            {
                "name": "B",
                "model_path": "./B.xml",
                "initial_biomass": 0.01,
                "mm_params": {"EX_glc__D_e": {"vmax": 5.0, "km": 2.0}},
            },
        ],
        "simulation": {"mode": "sequential_dfba", "dt": 0.25, "total_time_h": 24.0},
    }


def test_community_config_round_trip(tmp_path: Path) -> None:
    cfg = load_community_config(_write(tmp_path, _base_community()))
    assert isinstance(cfg, CommunityConfig)
    assert cfg.name == "demo"
    assert cfg.medium == "YCFA"
    assert [s.name for s in cfg.strains] == ["A", "B"]
    assert cfg.strains[1].mm_params["EX_glc__D_e"].vmax == 5.0


def test_community_rejects_duplicate_strain_names(tmp_path: Path) -> None:
    data = _base_community()
    data["strains"][1]["name"] = "A"
    with pytest.raises(ConfigError, match="duplicate"):
        load_community_config(_write(tmp_path, data))


def test_community_rejects_empty_strains(tmp_path: Path) -> None:
    data = _base_community()
    data["strains"] = []
    with pytest.raises(ConfigError, match="strains"):
        load_community_config(_write(tmp_path, data))


def test_community_rejects_bad_mode(tmp_path: Path) -> None:
    data = _base_community()
    data["simulation"]["mode"] = "bogus"
    with pytest.raises(ConfigError, match="mode"):
        load_community_config(_write(tmp_path, data))


def test_community_rejects_bad_initial_biomass(tmp_path: Path) -> None:
    data = _base_community()
    data["strains"][0]["initial_biomass"] = 0.0
    with pytest.raises(ConfigError, match="initial_biomass"):
        load_community_config(_write(tmp_path, data))


def test_community_mm_params_must_have_vmax_km(tmp_path: Path) -> None:
    data = _base_community()
    data["strains"][1]["mm_params"] = {"EX_x": {"vmax": 1.0}}  # missing km
    with pytest.raises(ConfigError, match=r"vmax.*km"):
        load_community_config(_write(tmp_path, data))


# ---------- fitted-params I/O ----------


def test_fitted_params_round_trip(tmp_path: Path) -> None:
    out = save_fitted_params(
        tmp_path / "fitted.yaml",
        strain="Bfrag",
        r_squared=0.987,
        mm_params={
            "EX_glc__D_e": MMParams(vmax=5.5, km=1.2),
            "EX_ac_e": MMParams(vmax=2.3, km=0.8),
        },
    )
    assert out.exists()
    loaded = load_fitted_params(out)
    assert loaded.strain == "Bfrag"
    assert loaded.r_squared == pytest.approx(0.987)
    assert loaded.mm_params["EX_glc__D_e"].vmax == pytest.approx(5.5)
    assert loaded.mm_params["EX_ac_e"].km == pytest.approx(0.8)


def test_fitted_params_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"mm_params": {}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="strain"):
        load_fitted_params(path)

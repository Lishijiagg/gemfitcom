"""Integration tests for the gemfitcom CLI."""

from __future__ import annotations

import json
from pathlib import Path

import cobra.io
import networkx as nx
import pandas as pd
import pytest
import yaml
from cobra import Metabolite, Model, Reaction
from typer.testing import CliRunner

from gemfitcom.cli import app

runner = CliRunner()

Y_GLC: float = 0.1


# =========================================================================
# Helpers — toy cobra models written to disk
# =========================================================================


def _glc_only_strain(name: str = "A", *, produces_ac_c: bool = False) -> Model:
    """Toy strain that grows on glucose.

    When ``produces_ac_c=True``, the biomass reaction also produces
    intracellular acetate (``ac_c``) but exposes no extracellular
    acetate exchange — exactly the state a KB-driven gap-fill should
    complete by adding transport + exchange for acetate.
    """
    m = Model(name)
    glc = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex.add_metabolites({glc: -1})
    biomass_rxn = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    mets = {glc: -1.0 / Y_GLC, bio: 1.0}
    if produces_ac_c:
        ac_c = Metabolite("ac_c", compartment="c", formula="C2H3O2", charge=-1)
        mets[ac_c] = 1.0
    biomass_rxn.add_metabolites(mets)
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, biomass_rxn, sink])
    m.objective = "BIOMASS"
    return m


def _write_sbml(model: Model, path: Path) -> Path:
    cobra.io.write_sbml_model(model, str(path))
    return path


# =========================================================================
# Smoke: --help for every subcommand
# =========================================================================


@pytest.mark.parametrize(
    "cmd", ["version", "solvers", "fit", "simulate", "interactions", "gapfill"]
)
def test_help_works_for_every_subcommand(cmd: str) -> None:
    r = runner.invoke(app, [cmd, "--help"])
    assert r.exit_code == 0, r.output
    assert "Usage" in r.output or "Options" in r.output


def test_version_prints_something() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert r.output.strip()


# =========================================================================
# interactions CLI (pure pandas, no cobra)
# =========================================================================


def test_interactions_command_writes_edges_and_graph(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "strain": ["A", "B", "C", "A", "B", "C"],
            "exchange_id": ["EX_ac_e"] * 6,
            "flux": [2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
        }
    )
    panel_csv = tmp_path / "panel.csv"
    panel.to_csv(panel_csv, index=False)

    out_dir = tmp_path / "out"
    r = runner.invoke(app, ["interactions", str(panel_csv), "--output", str(out_dir)])
    assert r.exit_code == 0, r.output

    xfeed = pd.read_csv(out_dir / "cross_feeding.csv")
    assert set(xfeed.columns) == {"donor", "recipient", "exchange_id", "cumulative_flow"}
    assert len(xfeed) == 2  # A->B and A->C
    comp = pd.read_csv(out_dir / "competition.csv")
    assert len(comp) == 1  # B and C compete for EX_ac_e

    g = nx.read_graphml(out_dir / "graph.graphml")
    assert set(g.nodes) == {"A", "B", "C"}


def test_interactions_can_disable_competition(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 1.0, 1.0],
            "strain": ["A", "B", "A", "B"],
            "exchange_id": ["EX_ac_e"] * 4,
            "flux": [-1.0, -2.0, 0.0, 0.0],
        }
    )
    panel_csv = tmp_path / "panel.csv"
    panel.to_csv(panel_csv, index=False)
    out_dir = tmp_path / "out"
    r = runner.invoke(
        app, ["interactions", str(panel_csv), "--output", str(out_dir), "--no-competition"]
    )
    assert r.exit_code == 0, r.output
    assert not (out_dir / "competition.csv").exists()


# =========================================================================
# gapfill CLI
# =========================================================================


def test_gapfill_agora2_adds_missing_secretion(tmp_path: Path) -> None:
    # Strain produces ac_c internally but has no EX_ac_e; gap-fill should add it.
    model = _glc_only_strain("Astrain", produces_ac_c=True)
    model_path = _write_sbml(model, tmp_path / "Astrain.xml")

    out_dir = tmp_path / "out"
    r = runner.invoke(
        app,
        [
            "gapfill",
            str(model_path),
            "--source",
            "agora2",
            "--observed",
            "EX_ac_e",
            "--output",
            str(out_dir),
        ],
    )
    assert r.exit_code == 0, r.output

    filled = cobra.io.read_sbml_model(str(out_dir / "Astrain_gapfilled.xml"))
    assert "EX_ac_e" in {rxn.id for rxn in filled.reactions}

    report = json.loads((out_dir / "Astrain_gapfill_report.json").read_text(encoding="utf-8"))
    assert report["source"] == "agora2"
    assert "EX_ac_e" in [o["exchange_id"] for o in report["outcomes"]]


def test_gapfill_curated_skips(tmp_path: Path) -> None:
    model = _glc_only_strain("C")
    model_path = _write_sbml(model, tmp_path / "C.xml")
    out_dir = tmp_path / "out"
    r = runner.invoke(
        app,
        [
            "gapfill",
            str(model_path),
            "--source",
            "curated",
            "--observed",
            "EX_ac_e",
            "--output",
            str(out_dir),
        ],
    )
    assert r.exit_code == 0, r.output
    report = json.loads((out_dir / "C_gapfill_report.json").read_text(encoding="utf-8"))
    assert report["skipped"] is True


def test_gapfill_requires_observed(tmp_path: Path) -> None:
    model = _glc_only_strain("X")
    model_path = _write_sbml(model, tmp_path / "X.xml")
    r = runner.invoke(
        app, ["gapfill", str(model_path), "--source", "agora2", "--output", str(tmp_path)]
    )
    assert r.exit_code != 0
    assert "observed" in r.output.lower()


def test_gapfill_observed_file_accepted(tmp_path: Path) -> None:
    model = _glc_only_strain("Y", produces_ac_c=True)
    model_path = _write_sbml(model, tmp_path / "Y.xml")
    # Pass only EX_ac_e — EX_but_e would fail strict verification because this
    # toy strain has no internal butyrate source.
    obs_file = tmp_path / "obs.yaml"
    obs_file.write_text(yaml.safe_dump({"observed": ["EX_ac_e"]}), encoding="utf-8")

    out_dir = tmp_path / "out"
    r = runner.invoke(
        app,
        [
            "gapfill",
            str(model_path),
            "--source",
            "agora2",
            "--observed-file",
            str(obs_file),
            "--output",
            str(out_dir),
        ],
    )
    assert r.exit_code == 0, r.output
    report = json.loads((out_dir / "Y_gapfill_report.json").read_text(encoding="utf-8"))
    seen = {o["exchange_id"] for o in report["outcomes"]}
    assert "EX_ac_e" in seen


# =========================================================================
# simulate CLI
# =========================================================================


def _make_community_files(tmp_path: Path) -> Path:
    a_path = _write_sbml(_glc_only_strain("A"), tmp_path / "A.xml")
    b_path = _write_sbml(_glc_only_strain("B"), tmp_path / "B.xml")

    medium_path = tmp_path / "mini.yaml"
    medium_path.write_text(
        yaml.safe_dump(
            {
                "name": "mini",
                "pool_components": {"EX_glc__D_e": 5.0},
                "unlimited_components": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "community": {"name": "testcomm"},
        "medium": str(medium_path),
        "strains": [
            {"name": "A", "model_path": str(a_path), "initial_biomass": 0.01},
            {"name": "B", "model_path": str(b_path), "initial_biomass": 0.01},
        ],
        "simulation": {
            "mode": "sequential_dfba",
            "dt": 0.5,
            "total_time_h": 2.0,
            "save_fluxes": True,
        },
    }
    cfg_path = tmp_path / "community.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_simulate_sequential_writes_artifacts(tmp_path: Path) -> None:
    cfg_path = _make_community_files(tmp_path)
    out_dir = tmp_path / "out"
    r = runner.invoke(app, ["simulate", str(cfg_path), "--output", str(out_dir)])
    assert r.exit_code == 0, r.output

    for fname in (
        "testcomm_biomass.csv",
        "testcomm_pool.csv",
        "testcomm_exchange_panel.csv",
        "testcomm_biomass_panel.csv",
    ):
        assert (out_dir / fname).exists(), fname

    panel = pd.read_csv(out_dir / "testcomm_exchange_panel.csv")
    assert set(panel.columns) == {"time_h", "strain", "exchange_id", "flux"}
    assert set(panel["strain"]) == {"A", "B"}


def test_simulate_then_interactions_chain(tmp_path: Path) -> None:
    cfg_path = _make_community_files(tmp_path)
    out_dir = tmp_path / "out"
    r1 = runner.invoke(app, ["simulate", str(cfg_path), "--output", str(out_dir)])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        app,
        [
            "interactions",
            str(out_dir / "testcomm_exchange_panel.csv"),
            "--biomass",
            str(out_dir / "testcomm_biomass_panel.csv"),
            "--output",
            str(out_dir / "net"),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert (out_dir / "net" / "cross_feeding.csv").exists()
    assert (out_dir / "net" / "graph.graphml").exists()


# =========================================================================
# fit CLI
# =========================================================================


def _glc_strain_with_fixed_vmax(name: str = "S", vmax_true: float = 4.0) -> Model:
    # Uptake of glc is bounded by Vmax in the model; fit should recover ~vmax_true.
    m = Model(name)
    glc = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_glc__D_e", lower_bound=-vmax_true, upper_bound=1000.0)
    ex.add_metabolites({glc: -1})
    biomass_rxn = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    biomass_rxn.add_metabolites({glc: -1.0 / Y_GLC, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, biomass_rxn, sink])
    m.objective = "BIOMASS"
    return m


def test_fit_cli_produces_all_artifacts(tmp_path: Path) -> None:
    # Create a toy strain, synthetic OD curve, minimal HPLC, and config.
    strain_name = "Toy"
    model = _glc_strain_with_fixed_vmax(strain_name, vmax_true=3.0)
    model_path = _write_sbml(model, tmp_path / f"{strain_name}.xml")

    # Synthetic OD curve: simple growth doubling every few hours.
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    ods = [0.02, 0.04, 0.08, 0.16, 0.3]
    od_rows = [
        {"time_h": t, "carbon_source": "glc__D", "replicate": rep, "od": od}
        for t, od in zip(times, ods, strict=True)
        for rep in (1,)
    ]
    od_path = tmp_path / "od.csv"
    pd.DataFrame(od_rows).to_csv(od_path, index=False)

    hplc_path = tmp_path / "hplc.csv"
    pd.DataFrame(
        [{"carbon_source": "glc__D", "metabolite": "acetate", "value_mM": 2.5, "replicate": 1}]
    ).to_csv(hplc_path, index=False)

    medium_path = tmp_path / "mini.yaml"
    medium_path.write_text(
        yaml.safe_dump(
            {
                "name": "mini",
                "pool_components": {"EX_glc__D_e": 10.0},
                "unlimited_components": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "strain": {
            "name": strain_name,
            "model_path": str(model_path),
            "model_source": "curated",  # skip gapfill for speed
        },
        "medium": {
            "name": str(medium_path),
            "carbon_source": {"exchange_id": "EX_glc__D_e", "initial_concentration_mM": 10.0},
        },
        "experiment": {
            "od_file": str(od_path),
            "hplc_file": str(hplc_path),
            "biomass_conversion": 0.35,
        },
        "simulation": {"dt": 0.5, "total_time_h": 4.0, "mode": "sequential_dfba"},
        "kinetics_fit": {
            "vmax_bounds_mmol_per_gDW_per_h": [0.1, 10.0],
            "km_bounds_mM": [0.1, 10.0],
            "de_maxiter": 3,
            "de_popsize": 4,
            "grid_points": 3,
            "grid_span": 0.5,
        },
    }
    cfg_path = tmp_path / "strain.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    r = runner.invoke(app, ["fit", str(cfg_path), "--output", str(out_dir), "--no-gapfill"])
    assert r.exit_code == 0, r.output

    for fname in (
        f"{strain_name}_fitted_params.yaml",
        f"{strain_name}_model_fitted.xml",
        f"{strain_name}_fit_grid.csv",
    ):
        assert (out_dir / fname).exists(), fname

    # Fitted params YAML is loadable and matches strain name.
    from gemfitcom.io.config import load_fitted_params

    loaded = load_fitted_params(out_dir / f"{strain_name}_fitted_params.yaml")
    assert loaded.strain == strain_name
    assert "EX_glc__D_e" in loaded.mm_params


# =========================================================================
# End-to-end: shipped example dataset
# =========================================================================


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_CONFIG = _REPO_ROOT / "configs" / "example_strain.yaml"
_EXAMPLE_VMAX_TRUE = 4.0  # set by scripts/build_example_data.py
_EXAMPLE_KM_TRUE = 2.0


def test_fit_cli_recovers_truth_on_shipped_example(tmp_path: Path) -> None:
    """`gemfitcom fit configs/example_strain.yaml` recovers the synthesis truth.

    The mini dataset under ``data/examples/`` was generated with
    ``scripts/build_example_data.py`` at known Vmax/Km. A successful fit on
    that data is the strongest end-to-end smoke test we have for the CLI
    path (config → preprocess → fit → artifact write).
    """
    if not _EXAMPLE_CONFIG.is_file():
        pytest.skip("example config not present")

    out_dir = tmp_path / "out"
    # Run from the repo root so the relative paths in the config resolve.
    import os

    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        r = runner.invoke(
            app,
            ["fit", str(_EXAMPLE_CONFIG), "--output", str(out_dir), "--no-gapfill"],
        )
    finally:
        os.chdir(cwd)
    assert r.exit_code == 0, r.output

    from gemfitcom.io.config import load_fitted_params

    loaded = load_fitted_params(out_dir / "toy_acetogen_fitted_params.yaml")
    fit = loaded.mm_params["EX_glc__D_e"]
    assert loaded.r_squared > 0.95
    assert fit.vmax == pytest.approx(_EXAMPLE_VMAX_TRUE, rel=0.2)
    assert fit.km == pytest.approx(_EXAMPLE_KM_TRUE, rel=0.5)

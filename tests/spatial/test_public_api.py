"""Public API: verify all PR 1 names import and a tiny end-to-end run works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gemfitcom import spatial


def test_public_api_exports_pr1_surface() -> None:
    expected = {
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
    }
    assert expected.issubset(set(spatial.__all__))
    for name in expected:
        assert hasattr(spatial, name), f"missing public symbol: {name}"


def test_pr1_integration_smoke(tmp_path: Path) -> None:
    """Load YAML config -> diffuse for 10 steps with boundary sources -> snapshot -> reload."""
    cfg_dict = {
        "geometry": {
            "n_grid": 21,
            "length": 1.0e-3,
            "boundary": {
                "mucosa": {"type": "flux", "sources": {"EX_o2_e": 1.0e-3}},
                "lumen": {"type": "dirichlet", "values": {"EX_glc__D_e": 5.0}},
            },
        },
        "metabolites": [
            {"id": "o2_e", "diffusion": 2.1e-9, "init": {"mode": "uniform", "value": 0.0}},
            {"id": "glc__D_e", "diffusion": 6.7e-10, "init": {"mode": "uniform", "value": 0.0}},
        ],
        "simulation": {"t_end": 1.0, "dt": 0.1, "snapshot_every": 0.5},
    }
    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg_dict))

    cfg = spatial.SpatialConfig.from_yaml(yaml_path)
    cfg.check_cfl()

    geom = spatial.Geometry1D(
        n_grid=cfg.geometry.n_grid,
        length=cfg.geometry.length,
        bc_left=spatial.BoundarySpec(type="flux", values=cfg.geometry.boundary["mucosa"].sources),
        bc_right=spatial.BoundarySpec(
            type="dirichlet", values=cfg.geometry.boundary["lumen"].values
        ),
    )

    metabolite_ids = [m.id for m in cfg.metabolites]
    n_grid = geom.n_grid
    C = np.zeros((len(metabolite_ids), n_grid))
    B = np.zeros((1, n_grid))  # placeholder; FBA wired in PR 2
    state = spatial.SpatialState(metabolites=C, biomass=B, t=0.0)

    L = spatial.build_laplacian_1d(n_grid, geom.dx, bc_left="neumann", bc_right="dirichlet")
    D = np.array([m.diffusion for m in cfg.metabolites])

    rec = spatial.SnapshotRecorder(
        output_dir=tmp_path / "snaps", every=cfg.simulation.snapshot_every
    )

    n_steps = int(round(cfg.simulation.t_end / cfg.simulation.dt))
    for _ in range(n_steps):
        # Reaction sub-step is a no-op in PR 1 (no FBA yet).
        new_C = spatial.diffuse_step(state.metabolites, L, D, cfg.simulation.dt)
        geom.apply_boundary_sources(new_C, metabolite_ids, cfg.simulation.dt)
        state = spatial.SpatialState(
            metabolites=new_C, biomass=state.biomass, t=state.t + cfg.simulation.dt
        )
        rec.maybe_save(state)

    # Mucosa cell received O2 flux; lumen cell pinned to 5 mmol/L of glucose.
    assert state.metabolites[0, 0] > 0.0  # O2 accumulated at mucosa
    assert state.metabolites[1, -1] == pytest.approx(5.0)  # glucose Dirichlet held

    # At least one snapshot landed on disk and is readable.
    snaps = sorted((tmp_path / "snaps").glob("snapshot_*.npz"))
    assert len(snaps) >= 1
    restored = spatial.SnapshotRecorder.load(snaps[-1])
    assert restored.t > 0.0


class TestPr2PublicApi:
    def test_pr2_kinetics_classes_exported(self):
        from gemfitcom.spatial import (
            ExchangeEntry,
            ExchangeKinetics,
            load_kinetics_yaml,
            resolve_gem,
        )

        assert ExchangeKinetics is not None
        assert ExchangeEntry is not None
        assert callable(load_kinetics_yaml)
        assert callable(resolve_gem)

    def test_pr2_reaction_classes_exported(self):
        from gemfitcom.spatial import ReactionEngine, SerialBackend

        assert ReactionEngine is not None
        assert SerialBackend is not None

    def test_pr2_config_classes_exported(self):
        from gemfitcom.spatial import SpeciesConfig, SpeciesInitConfig

        assert SpeciesConfig is not None
        assert SpeciesInitConfig is not None

    def test_pr2_init_field_helper_exported(self):
        from gemfitcom.spatial import build_field_1d

        assert callable(build_field_1d)

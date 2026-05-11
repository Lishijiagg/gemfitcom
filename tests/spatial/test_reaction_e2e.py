"""End-to-end smoke test for PR 2: example sim.yaml × 10 reaction steps.

Reaction-only (no diffusion). Full pipeline lands in PR 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gemfitcom.spatial._init_fields import build_field_1d
from gemfitcom.spatial.backends import SerialBackend
from gemfitcom.spatial.config import SpatialConfig
from gemfitcom.spatial.kinetics import load_kinetics_yaml, resolve_gem
from gemfitcom.spatial.reaction import ReactionEngine

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "spatial" / "ecoli_fprau_skeleton"


@pytest.fixture
def loaded_engine():
    cfg_path = EXAMPLE_DIR / "sim.yaml"
    if not cfg_path.is_file():
        pytest.skip(f"Example not found at {cfg_path}")
    cfg = SpatialConfig.from_yaml(cfg_path)
    models = [resolve_gem(s.gem) for s in cfg.species]
    kinetics = [
        load_kinetics_yaml(s.kinetics if s.kinetics.is_absolute() else EXAMPLE_DIR / s.kinetics)
        for s in cfg.species
    ]
    metabolite_ids = tuple(m.id for m in cfg.metabolites)
    engine = ReactionEngine(
        models=models,
        kinetics=kinetics,
        metabolite_ids=metabolite_ids,
        backend=SerialBackend(),
    )
    return cfg, engine


def _initial_state(cfg):
    n_grid = cfg.geometry.n_grid
    C = np.stack([build_field_1d(m.init, n_grid) for m in cfg.metabolites])
    B = np.stack([build_field_1d(s.init, n_grid) for s in cfg.species])
    return C, B


class TestExampleSmoke:
    def test_ten_step_run_completes_without_exception(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert C.shape == (len(cfg.metabolites), cfg.geometry.n_grid)
        assert B.shape == (len(cfg.species), cfg.geometry.n_grid)

    def test_ten_step_biomass_and_conc_non_negative(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert np.all(B >= 0)
        assert np.all(C >= 0)

    def test_ten_step_glucose_does_not_increase(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        glc_idx = next(i for i, m in enumerate(cfg.metabolites) if m.id == "glc__D_e")
        initial_total = float(C[glc_idx].sum())
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        final_total = float(C[glc_idx].sum())
        assert final_total <= initial_total + 1e-9

    def test_at_least_one_species_grows(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        B0 = B.copy()
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert np.any(B.sum(axis=1) > B0.sum(axis=1))

    def test_no_engine_warnings_at_default_clip(self, loaded_engine):
        cfg, engine = loaded_engine
        C, B = _initial_state(cfg)
        for _ in range(10):
            C, B = engine.apply_to_state(C, B, cfg.simulation.dt)
        assert engine.warnings == []

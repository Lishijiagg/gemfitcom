"""Tests for spatial/reaction.py — single-cell FBA + SerialBackend.

PR 2 layered tests: this file starts with build_exchange_index and solve_cell
(Task 7). Later tasks (8, 9, 10) append SerialBackend, ReactionEngine, and guard
tests to the same file.
"""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.kinetics import ExchangeEntry, ExchangeKinetics
from gemfitcom.spatial.reaction import (
    SingleCellResult,
    build_exchange_index,
    solve_cell,
)


class TestBuildExchangeIndex:
    def test_returns_indices_for_present_exchanges(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e", "ac_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        assert idx == {0: 0, 1: 1}

    def test_kinetics_entry_missing_from_model_raises(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_does_not_exist_e", 10.0, 0.5, "uptake_only"),),
        )
        with pytest.raises(KeyError, match="EX_does_not_exist_e"):
            build_exchange_index(fresh_textbook, ek, ("does_not_exist_e",))


class TestSolveCell:
    def test_solve_at_glucose_saturation_returns_positive_growth(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([100.0, 100.0])
        result = solve_cell(model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local)
        assert isinstance(result, SingleCellResult)
        assert result.mu > 0
        assert not result.infeasible

    def test_solve_with_zero_glucose(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        metabolite_ids = ("glc__D_e", "o2_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([0.0, 100.0])
        result = solve_cell(model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local)
        assert result.mu >= 0
        glc_met_idx = metabolite_ids.index("glc__D_e")
        assert np.isclose(result.flux[glc_met_idx], 0.0)

    def test_solve_matches_bare_cobra_call(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 8.0, 0.5, "uptake_only"),),
        )
        metabolite_ids = ("glc__D_e",)
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([1.0])
        result = solve_cell(model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local)

        ref = fresh_textbook.copy()
        mm = 8.0 * 1.0 / (0.5 + 1.0)
        ref.reactions.get_by_id("EX_glc__D_e").lower_bound = -mm
        bare = ref.optimize()
        assert np.isclose(result.mu, bare.objective_value, rtol=1e-8)

    def test_solve_bidirectional_caps_secretion(self, fresh_textbook):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_ac_e", 5.0, 0.1, "bidirectional"),
            ),
        )
        metabolite_ids = ("glc__D_e", "ac_e")
        idx = build_exchange_index(fresh_textbook, ek, metabolite_ids)
        C_local = np.array([10.0, 0.0])
        result = solve_cell(model=fresh_textbook, kinetics=ek, exchange_index=idx, C_local=C_local)
        ac_idx = metabolite_ids.index("ac_e")
        assert result.flux[ac_idx] <= 1e-9

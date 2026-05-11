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


class TestSerialBackend:
    def _make_setup(self):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        return ek, ("glc__D_e", "o2_e")

    def test_serial_step_shapes(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        n_grid = 3
        C = np.array([[10.0, 5.0, 1.0], [100.0, 100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook],
            kinetics=[ek],
            metabolite_ids=met_ids,
            C=C,
            B=B,
        )
        assert mu.shape == (1, n_grid)
        assert flux.shape == (2, 1, n_grid)
        assert np.all(mu > 0)

    def test_serial_skips_empty_cells(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend(empty_eps=1e-12)
        C = np.array([[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]])
        B = np.array([[1.0e-3, 0.0, 1.0e-15]])
        mu, flux = backend.step(
            models=[fresh_textbook],
            kinetics=[ek],
            metabolite_ids=met_ids,
            C=C,
            B=B,
        )
        assert mu[0, 0] > 0
        assert mu[0, 1] == 0.0
        assert mu[0, 2] == 0.0
        assert np.all(flux[:, 0, 1] == 0)
        assert np.all(flux[:, 0, 2] == 0)

    def test_serial_two_species(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek1, met_ids = self._make_setup()
        ek2 = ExchangeKinetics(
            species="fprau",
            entries=(ExchangeEntry("EX_glc__D_e", 8.0, 0.5, "uptake_only"),),
        )
        backend = SerialBackend()
        C = np.array([[10.0, 10.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3], [1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook, fresh_textbook.copy()],
            kinetics=[ek1, ek2],
            metabolite_ids=met_ids,
            C=C,
            B=B,
        )
        assert mu.shape == (2, 2)
        assert flux.shape == (2, 2, 2)

    def test_negative_concentration_clipped(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        C = np.array([[-1e-12, 10.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3]])
        mu, flux = backend.step(
            models=[fresh_textbook],
            kinetics=[ek],
            metabolite_ids=met_ids,
            C=C,
            B=B,
        )
        assert mu[0, 0] >= 0
        glc_idx = met_ids.index("glc__D_e")
        assert np.isclose(flux[glc_idx, 0, 0], 0.0, atol=1e-9)

    def test_serial_rejects_wrong_B_shape(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend

        ek, met_ids = self._make_setup()
        backend = SerialBackend()
        C = np.zeros((2, 3))
        B = np.zeros((1, 4))  # n_grid mismatch
        with pytest.raises(ValueError, match="n_grid"):
            backend.step(
                models=[fresh_textbook],
                kinetics=[ek],
                metabolite_ids=met_ids,
                C=C,
                B=B,
            )


class TestReactionEngine:
    def test_engine_step_returns_correct_shapes(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        engine = ReactionEngine(
            models=[fresh_textbook],
            kinetics=[ek],
            metabolite_ids=("glc__D_e", "o2_e"),
            backend=SerialBackend(),
        )
        C = np.array([[10.0, 5.0], [100.0, 100.0]])
        B = np.array([[1.0e-3, 1.0e-3]])
        mu, flux = engine.step(C, B, dt=0.1)
        assert mu.shape == (1, 2)
        assert flux.shape == (2, 1, 2)

    def test_engine_rejects_mismatched_lengths(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(ExchangeEntry("EX_glc__D_e", 10.0, 0.5),),
        )
        with pytest.raises(ValueError, match="models"):
            ReactionEngine(
                models=[fresh_textbook, fresh_textbook],
                kinetics=[ek],
                metabolite_ids=("glc__D_e",),
                backend=SerialBackend(),
            )

    def test_engine_apply_to_state_grows_biomass_consumes_substrate(self, fresh_textbook):
        from gemfitcom.spatial.backends import SerialBackend
        from gemfitcom.spatial.reaction import ReactionEngine

        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry("EX_glc__D_e", 10.0, 0.5, "uptake_only"),
                ExchangeEntry("EX_o2_e", 15.0, 0.005, "uptake_only"),
            ),
        )
        engine = ReactionEngine(
            models=[fresh_textbook],
            kinetics=[ek],
            metabolite_ids=("glc__D_e", "o2_e"),
            backend=SerialBackend(),
        )
        C = np.array([[10.0], [100.0]])
        B = np.array([[1.0e-3]])
        C_new, B_new = engine.apply_to_state(C.copy(), B.copy(), dt=0.1)
        assert B_new[0, 0] > B[0, 0]  # biomass grew
        assert C_new[0, 0] < C[0, 0]  # glucose dropped

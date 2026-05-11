"""Tests for spatial/kinetics.py — ExchangeEntry + ExchangeKinetics."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.kinetics import ExchangeEntry, ExchangeKinetics


class TestExchangeKineticsBasics:
    def test_construct_with_two_substrates(self):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry(exchange_id="EX_glc__D_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_o2_e", vmax=15.0, km=0.005, mode="uptake_only"),
            ),
        )
        assert ek.species == "ecoli"
        assert ek.n_exchanges == 2
        assert ek.exchange_ids == ("EX_glc__D_e", "EX_o2_e")

    def test_mm_upper_bound_at_saturation(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([1000.0]))
        assert bounds.shape == (1,)
        assert np.isclose(bounds[0], 10.0, rtol=1e-3)

    def test_mm_upper_bound_at_zero(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([0.0]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_monotonic(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        C = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
        bounds = np.array([ek.mm_upper_bound(np.array([c]))[0] for c in C])
        assert np.all(np.diff(bounds) > 0)

    def test_mm_upper_bound_negative_input_clipped(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([-1e-9]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_shape_mismatch_raises(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_b_e", vmax=20.0, km=1.0, mode="uptake_only"),
            ),
        )
        with pytest.raises(ValueError, match="does not match n_exchanges"):
            ek.mm_upper_bound(np.array([0.5]))

    def test_invalid_vmax_rejected(self):
        with pytest.raises(ValueError, match="vmax"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=0.0, km=0.5, mode="uptake_only")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="weird_mode")

    def test_bidirectional_mode_stored(self):
        e = ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="bidirectional")
        assert e.mode == "bidirectional"

        ek = ExchangeKinetics(species="x", entries=(e,))
        assert ek.entries[0].mode == "bidirectional"
        # mm_upper_bound does NOT consume mode; just returns MM magnitude
        bounds = ek.mm_upper_bound(np.array([1000.0]))
        assert np.isclose(bounds[0], 10.0, rtol=1e-3)

    def test_duplicate_exchange_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ExchangeKinetics(
                species="x",
                entries=(
                    ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                    ExchangeEntry(exchange_id="EX_a_e", vmax=20.0, km=1.0, mode="uptake_only"),
                ),
            )

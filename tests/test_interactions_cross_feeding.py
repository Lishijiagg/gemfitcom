"""Tests for interactions.cross_feeding."""

from __future__ import annotations

import pandas as pd
import pytest

from gemfitcom.interactions import cross_feeding_edges


def _panel(rows: list[tuple[float, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["time_h", "strain", "exchange_id", "flux"])


# ---------- basic schema / edge cases ----------


def test_empty_panel_returns_empty_frame_with_schema() -> None:
    empty = pd.DataFrame(columns=["time_h", "strain", "exchange_id", "flux"])
    out = cross_feeding_edges(empty)
    assert list(out.columns) == [
        "donor",
        "recipient",
        "exchange_id",
        "cumulative_flow",
    ]
    assert out.empty


def test_rejects_panel_missing_columns() -> None:
    bad = pd.DataFrame({"time_h": [0.0], "strain": ["A"], "flux": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        cross_feeding_edges(bad)


def test_only_donors_no_recipients_yields_no_edges() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", 2.0),
            (1.0, "A", "EX_x", 1.0),
            (1.0, "B", "EX_x", 2.0),
        ]
    )
    out = cross_feeding_edges(panel)
    assert out.empty


# ---------- single donor / single recipient ----------


def test_single_donor_single_recipient_full_min() -> None:
    # time_h diffs: step at 0 has dt=1, last step (time 1) dt=0.
    panel = _panel(
        [
            (0.0, "A", "EX_ac_e", 2.0),
            (0.0, "B", "EX_ac_e", -1.0),
            (1.0, "A", "EX_ac_e", 1.5),
            (1.0, "B", "EX_ac_e", -0.8),
        ]
    )
    out = cross_feeding_edges(panel)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["donor"] == "A"
    assert row["recipient"] == "B"
    assert row["exchange_id"] == "EX_ac_e"
    # time 0: exchanged = min(2, 1) = 1, flow = 1 * 1h = 1
    # time 1: dt = 0, contributes 0
    assert row["cumulative_flow"] == pytest.approx(1.0)


# ---------- double-proportional allocation ----------


def test_multi_donor_multi_recipient_double_proportional() -> None:
    # sum_sec = 3, sum_up = 3, exchanged = 3
    # shares: A 2/3, B 1/3, C 1/3, D 2/3
    # flows: A->C=2/3, A->D=4/3, B->C=1/3, B->D=2/3
    panel = _panel(
        [
            (0.0, "A", "EX_x", 2.0),
            (0.0, "B", "EX_x", 1.0),
            (0.0, "C", "EX_x", -1.0),
            (0.0, "D", "EX_x", -2.0),
            (1.0, "A", "EX_x", 2.0),
            (1.0, "B", "EX_x", 1.0),
            (1.0, "C", "EX_x", -1.0),
            (1.0, "D", "EX_x", -2.0),
        ]
    )
    out = cross_feeding_edges(panel)
    lookup = {(r.donor, r.recipient): r.cumulative_flow for r in out.itertuples()}
    assert lookup[("A", "C")] == pytest.approx(2 / 3)
    assert lookup[("A", "D")] == pytest.approx(4 / 3)
    assert lookup[("B", "C")] == pytest.approx(1 / 3)
    assert lookup[("B", "D")] == pytest.approx(2 / 3)
    # Total flow = 3 * dt(1) = 3
    assert out["cumulative_flow"].sum() == pytest.approx(3.0)


def test_output_sorted_descending_by_flow() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 2.0),
            (0.0, "B", "EX_x", 1.0),
            (0.0, "C", "EX_x", -1.0),
            (0.0, "D", "EX_x", -2.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
            (1.0, "C", "EX_x", 0.0),
            (1.0, "D", "EX_x", 0.0),
        ]
    )
    out = cross_feeding_edges(panel)
    flows = out["cumulative_flow"].tolist()
    assert flows == sorted(flows, reverse=True)


# ---------- biomass weighting ----------


def test_biomass_weighting_scales_fluxes_before_allocation() -> None:
    # Without biomass: A=+1, B=-1 → exchanged=1
    # With biomass A=2, B=3 → effective A=+2, B=-3 → exchanged=min(2,3)=2
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
            (1.0, "A", "EX_x", 1.0),
            (1.0, "B", "EX_x", -1.0),
        ]
    )
    bio = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 1.0, 1.0],
            "strain": ["A", "B", "A", "B"],
            "biomass": [2.0, 3.0, 2.0, 3.0],
        }
    )
    out_raw = cross_feeding_edges(panel)
    out_bio = cross_feeding_edges(panel, biomass=bio)
    assert out_raw.iloc[0]["cumulative_flow"] == pytest.approx(1.0)
    assert out_bio.iloc[0]["cumulative_flow"] == pytest.approx(2.0)


def test_biomass_missing_rows_raises() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
        ]
    )
    # Missing biomass for strain B
    bio = pd.DataFrame({"time_h": [0.0], "strain": ["A"], "biomass": [1.0]})
    with pytest.raises(ValueError, match="does not cover"):
        cross_feeding_edges(panel, biomass=bio)


# ---------- dt handling ----------


def test_single_time_point_warns_and_defaults_dt_to_one() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
        ]
    )
    with pytest.warns(UserWarning, match="single time point"):
        out = cross_feeding_edges(panel)
    assert out.iloc[0]["cumulative_flow"] == pytest.approx(1.0)


def test_single_time_point_explicit_dt_no_warning() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
        ]
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        out = cross_feeding_edges(panel, dt=2.5)
    assert out.iloc[0]["cumulative_flow"] == pytest.approx(2.5)


def test_multi_step_uses_per_step_dt() -> None:
    # Non-uniform spacing: 0, 0.5, 2.0
    # Steps: 0->0.5 (dt=0.5), 0.5->2.0 (dt=1.5), last dt=0
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
            (0.5, "A", "EX_x", 2.0),
            (0.5, "B", "EX_x", -2.0),
            (2.0, "A", "EX_x", 999.0),
            (2.0, "B", "EX_x", -999.0),
        ]
    )
    out = cross_feeding_edges(panel)
    # Expected = min(1,1)*0.5 + min(2,2)*1.5 + last*0 = 0.5 + 3.0 = 3.5
    assert out.iloc[0]["cumulative_flow"] == pytest.approx(3.5)

"""Tests for interactions.competition."""

from __future__ import annotations

import pandas as pd
import pytest

from gemfitcom.interactions import competition_edges


def _panel(rows: list[tuple[float, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["time_h", "strain", "exchange_id", "flux"])


def test_rejects_panel_missing_columns() -> None:
    bad = pd.DataFrame({"time_h": [0.0], "strain": ["A"], "flux": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        competition_edges(bad)


def test_empty_panel_returns_empty_frame_with_schema() -> None:
    empty = pd.DataFrame(columns=["time_h", "strain", "exchange_id", "flux"])
    out = competition_edges(empty)
    assert list(out.columns) == [
        "strain_a",
        "strain_b",
        "exchange_id",
        "competition_intensity",
    ]
    assert out.empty


def test_single_uptaker_no_competition() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", 1.0),  # secreting, not competing
            (1.0, "A", "EX_x", -1.0),
            (1.0, "B", "EX_x", 1.0),
        ]
    )
    out = competition_edges(panel)
    assert out.empty


def test_two_uptakers_min_intensity_integrated() -> None:
    # t=0: A=-1, B=-2 → min=1, dt=1 → +1
    # t=1: last step, dt=0 → +0
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
            (1.0, "A", "EX_x", -1.0),
            (1.0, "B", "EX_x", -2.0),
        ]
    )
    out = competition_edges(panel)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["strain_a"] == "A"  # lexicographic
    assert row["strain_b"] == "B"
    assert row["competition_intensity"] == pytest.approx(1.0)


def test_pair_order_is_lexicographic() -> None:
    # Put "Z" first in data to confirm ordering is by strain name, not order.
    panel = _panel(
        [
            (0.0, "Z", "EX_x", -1.0),
            (0.0, "A", "EX_x", -2.0),
            (1.0, "Z", "EX_x", 0.0),
            (1.0, "A", "EX_x", 0.0),
        ]
    )
    out = competition_edges(panel)
    row = out.iloc[0]
    assert row["strain_a"] == "A"
    assert row["strain_b"] == "Z"


def test_three_way_competition_yields_three_pairs() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
            (0.0, "C", "EX_x", -3.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
            (1.0, "C", "EX_x", 0.0),
        ]
    )
    out = competition_edges(panel)
    pairs = {(r.strain_a, r.strain_b): r.competition_intensity for r in out.itertuples()}
    assert pairs[("A", "B")] == pytest.approx(1.0)  # min(1, 2)
    assert pairs[("A", "C")] == pytest.approx(1.0)  # min(1, 3)
    assert pairs[("B", "C")] == pytest.approx(2.0)  # min(2, 3)


def test_biomass_weighting_scales_before_min() -> None:
    # Raw: A=-1, B=-1 → min=1
    # With biomass A=5, B=2 → effective A=-5, B=-2 → min=2
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -1.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
        ]
    )
    bio = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 1.0, 1.0],
            "strain": ["A", "B", "A", "B"],
            "biomass": [5.0, 2.0, 5.0, 2.0],
        }
    )
    out_raw = competition_edges(panel)
    out_bio = competition_edges(panel, biomass=bio)
    assert out_raw.iloc[0]["competition_intensity"] == pytest.approx(1.0)
    assert out_bio.iloc[0]["competition_intensity"] == pytest.approx(2.0)


def test_single_time_point_warns_and_defaults_dt_to_one() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
        ]
    )
    with pytest.warns(UserWarning, match="single time point"):
        out = competition_edges(panel)
    assert out.iloc[0]["competition_intensity"] == pytest.approx(1.0)


def test_output_sorted_descending_by_intensity() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
            (0.0, "A", "EX_y", -5.0),
            (0.0, "B", "EX_y", -4.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
            (1.0, "A", "EX_y", 0.0),
            (1.0, "B", "EX_y", 0.0),
        ]
    )
    out = competition_edges(panel)
    intensities = out["competition_intensity"].tolist()
    assert intensities == sorted(intensities, reverse=True)

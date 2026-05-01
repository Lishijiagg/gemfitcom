"""Tests for preprocess.hplc."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gemfitcom.preprocess.hplc import (
    average_replicates,
    hplc_long_to_wide,
)


def _make_hplc_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "carbon_source": ["GMC", "GMC", "GMC", "GMC", "LNT", "LNT"],
            "metabolite": ["acetate", "acetate", "butyrate", "butyrate", "acetate", "butyrate"],
            "value_mM": [5.0, 5.4, 2.0, 2.4, 6.0, 1.0],
            "replicate": [1, 2, 1, 2, 1, 1],
        }
    )


def test_average_replicates_mean_sd_n() -> None:
    df = _make_hplc_long()
    out = average_replicates(df)

    assert set(out.columns) == {"carbon_source", "metabolite", "mean_mM", "sd_mM", "n_replicates"}
    gmc_ace = out.query("carbon_source == 'GMC' and metabolite == 'acetate'").iloc[0]
    assert gmc_ace["mean_mM"] == pytest.approx(5.2)
    assert gmc_ace["sd_mM"] == pytest.approx(np.std([5.0, 5.4], ddof=1))
    assert gmc_ace["n_replicates"] == 2

    lnt_ace = out.query("carbon_source == 'LNT' and metabolite == 'acetate'").iloc[0]
    assert lnt_ace["n_replicates"] == 1
    assert pd.isna(lnt_ace["sd_mM"])


def test_hplc_long_to_wide_aggregates_replicates() -> None:
    df = _make_hplc_long()
    wide = hplc_long_to_wide(df, aggregate=True)

    assert wide.index.name == "carbon_source"
    assert set(wide.columns) == {"acetate", "butyrate"}
    assert wide.loc["GMC", "acetate"] == pytest.approx(5.2)
    assert wide.loc["GMC", "butyrate"] == pytest.approx(2.2)
    assert wide.loc["LNT", "acetate"] == pytest.approx(6.0)


def test_hplc_long_to_wide_no_aggregate_keeps_replicates() -> None:
    df = _make_hplc_long()
    wide = hplc_long_to_wide(df, aggregate=False)

    assert ("GMC", 1) in wide.index
    assert ("GMC", 2) in wide.index
    assert wide.loc[("GMC", 1), "acetate"] == pytest.approx(5.0)
    assert wide.loc[("GMC", 2), "acetate"] == pytest.approx(5.4)


def test_hplc_long_to_wide_no_aggregate_raises_on_duplicates() -> None:
    df = pd.DataFrame(
        {
            "carbon_source": ["GMC", "GMC"],
            "metabolite": ["acetate", "acetate"],
            "value_mM": [5.0, 5.1],
            "replicate": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        hplc_long_to_wide(df, aggregate=False)


def test_hplc_missing_columns_raise() -> None:
    bad = pd.DataFrame({"carbon_source": ["A"], "value_mM": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        average_replicates(bad)
    with pytest.raises(ValueError, match="missing required columns"):
        hplc_long_to_wide(bad)


def test_average_replicates_groups_by_time_when_present() -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 6.0, 6.0, 14.0, 14.0],
            "carbon_source": ["GMC"] * 6,
            "metabolite": ["acetate"] * 6,
            "value_mM": [0.0, 0.1, 1.4, 1.6, 5.0, 5.4],
            "replicate": [1, 2, 1, 2, 1, 2],
        }
    )
    out = average_replicates(df)

    assert "time_h" in out.columns
    assert len(out) == 3
    t14 = out.query("time_h == 14.0").iloc[0]
    assert t14["mean_mM"] == pytest.approx(5.2)
    assert t14["n_replicates"] == 2


def test_average_replicates_ignores_time_when_all_nan() -> None:
    df = pd.DataFrame(
        {
            "time_h": [pd.NA, pd.NA, pd.NA, pd.NA],
            "carbon_source": ["GMC", "GMC", "LNT", "LNT"],
            "metabolite": ["acetate", "acetate", "acetate", "acetate"],
            "value_mM": [5.0, 5.4, 6.0, 6.2],
            "replicate": [1, 2, 1, 2],
        }
    )
    out = average_replicates(df)

    assert "time_h" not in out.columns
    assert len(out) == 2

"""Tests for preprocess.od."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gemfitcom.preprocess.od import (
    average_replicates,
    floor_od,
    smooth_od,
    subtract_t0,
)


def _make_od_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "carbon_source": ["GMC", "GMC", "GMC", "GMC", "GMC", "GMC"],
            "replicate": [1, 1, 1, 2, 2, 2],
            "od": [0.05, 0.20, 0.60, 0.06, 0.22, 0.55],
        }
    )


def test_subtract_t0_aligns_each_replicate_to_zero() -> None:
    df = _make_od_frame()
    out = subtract_t0(df)

    rep1 = out[out["replicate"] == 1].sort_values("time_h")
    rep2 = out[out["replicate"] == 2].sort_values("time_h")
    assert rep1["od"].iloc[0] == pytest.approx(0.0)
    assert rep2["od"].iloc[0] == pytest.approx(0.0)
    assert rep1["od"].tolist() == pytest.approx([0.0, 0.15, 0.55])
    assert rep2["od"].tolist() == pytest.approx([0.0, 0.16, 0.49])


def test_subtract_t0_does_not_mutate_input() -> None:
    df = _make_od_frame()
    original = df.copy()
    _ = subtract_t0(df)
    pd.testing.assert_frame_equal(df, original)


def test_floor_od_clips_nonpositive_only() -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0],
            "carbon_source": ["A", "A", "A"],
            "replicate": [1, 1, 1],
            "od": [0.0, -0.01, 0.3],
        }
    )
    out = floor_od(df, floor=1e-4)
    assert out["od"].tolist() == pytest.approx([1e-4, 1e-4, 0.3])


def test_floor_od_invalid_floor() -> None:
    df = _make_od_frame()
    with pytest.raises(ValueError, match="floor must be positive"):
        floor_od(df, floor=0.0)


def test_average_replicates_computes_mean_sd_n() -> None:
    df = _make_od_frame()
    out = average_replicates(df)

    assert set(out.columns) == {"time_h", "carbon_source", "mean_od", "sd_od", "n_replicates"}
    row_t0 = out.query("time_h == 0.0").iloc[0]
    assert row_t0["mean_od"] == pytest.approx(0.055)
    assert row_t0["sd_od"] == pytest.approx(np.std([0.05, 0.06], ddof=1))
    assert row_t0["n_replicates"] == 2


def test_average_replicates_single_replicate_has_nan_sd() -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 1.0],
            "carbon_source": ["A", "A"],
            "replicate": [1, 1],
            "od": [0.1, 0.3],
        }
    )
    out = average_replicates(df)
    assert out["n_replicates"].tolist() == [1, 1]
    assert out["sd_od"].isna().all()


def test_smooth_od_flattens_noise() -> None:
    t = np.arange(0.0, 10.0, 0.5)
    rng = np.random.default_rng(0)
    signal = 0.1 + 0.05 * t
    noisy = signal + rng.normal(0.0, 0.05, size=t.shape)
    df = pd.DataFrame(
        {
            "time_h": t,
            "carbon_source": ["A"] * len(t),
            "replicate": [1] * len(t),
            "od": noisy,
        }
    )
    out = smooth_od(df, window=5)
    noisy_residual_var = np.var(noisy - signal)
    smoothed_residual_var = np.var(out["od"].to_numpy() - signal)
    assert smoothed_residual_var < noisy_residual_var


def test_smooth_od_invalid_window() -> None:
    df = _make_od_frame()
    with pytest.raises(ValueError, match="window"):
        smooth_od(df, window=0)


def test_missing_columns_raise() -> None:
    bad = pd.DataFrame({"time_h": [0.0, 1.0], "od": [0.1, 0.2]})
    with pytest.raises(ValueError, match="missing required columns"):
        subtract_t0(bad)
    with pytest.raises(ValueError, match="missing required columns"):
        average_replicates(bad.drop(columns=["od"]))

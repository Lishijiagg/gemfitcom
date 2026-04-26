"""Tests for preprocess.growth (fit_easylinear port)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gemfitcom.preprocess.growth import (
    GrowthFit,
    fit_easylinear,
    fit_growth_curves,
)


def _lag_exp_curve(
    t: np.ndarray, *, y0: float, mumax: float, lag: float, noise_sd: float = 0.0
) -> np.ndarray:
    """Flat-then-exponential curve: y = y0 for t<lag, y0*exp(mumax*(t-lag)) after."""
    y = np.where(t < lag, y0, y0 * np.exp(mumax * (t - lag)))
    if noise_sd > 0:
        rng = np.random.default_rng(0)
        y = y * np.exp(rng.normal(0.0, noise_sd, size=t.shape))
    return y


def test_fit_easylinear_recovers_known_mumax_and_lag_no_noise() -> None:
    t = np.arange(0.0, 24.0 + 0.25, 0.25)
    y = _lag_exp_curve(t, y0=0.01, mumax=0.5, lag=2.0)

    fit = fit_easylinear(t, y, h=8)

    assert isinstance(fit, GrowthFit)
    assert fit.mumax == pytest.approx(0.5, rel=1e-6)
    assert fit.lag == pytest.approx(2.0, rel=1e-6)
    assert fit.y0 == pytest.approx(0.01, rel=1e-6)
    assert fit.r_squared > 0.999
    assert fit.n_points >= 8


def test_fit_easylinear_pure_exponential_zero_lag() -> None:
    t = np.arange(0.0, 10.0 + 0.25, 0.25)
    y = 0.05 * np.exp(0.7 * t)

    fit = fit_easylinear(t, y, h=5)

    assert fit.mumax == pytest.approx(0.7, rel=1e-6)
    assert fit.lag == pytest.approx(0.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)


def test_fit_easylinear_with_noise_stays_close() -> None:
    t = np.arange(0.0, 24.0 + 0.25, 0.25)
    y = _lag_exp_curve(t, y0=0.02, mumax=0.3, lag=3.0, noise_sd=0.05)

    fit = fit_easylinear(t, y, h=10)

    assert fit.mumax == pytest.approx(0.3, rel=0.15)
    assert fit.lag == pytest.approx(3.0, abs=1.0)
    assert fit.r_squared > 0.9


def test_fit_easylinear_picks_highest_slope_window() -> None:
    t = np.linspace(0.0, 10.0, 41)
    y = np.where(t < 5.0, 0.01 * np.exp(0.1 * t), 0.01 * np.exp(0.5) * np.exp(0.6 * (t - 5.0)))

    fit = fit_easylinear(t, y, h=5)

    assert fit.mumax == pytest.approx(0.6, rel=1e-6)
    assert t[fit.window_start] >= 5.0 - 1e-9


def test_fit_easylinear_invalid_h_too_large() -> None:
    t = np.arange(0.0, 5.0, 0.5)
    y = np.exp(t)
    with pytest.raises(ValueError, match="exceeds number of samples"):
        fit_easylinear(t, y, h=100)


def test_fit_easylinear_invalid_h_too_small() -> None:
    t = np.arange(0.0, 5.0, 0.5)
    y = np.exp(t)
    with pytest.raises(ValueError, match="h must be >= 2"):
        fit_easylinear(t, y, h=1)


def test_fit_easylinear_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        fit_easylinear(np.arange(5), np.arange(4), h=3)


def test_fit_easylinear_invalid_quota() -> None:
    t = np.arange(0.0, 5.0, 0.5)
    y = np.exp(t)
    with pytest.raises(ValueError, match="quota"):
        fit_easylinear(t, y, h=3, quota=0.0)
    with pytest.raises(ValueError, match="quota"):
        fit_easylinear(t, y, h=3, quota=1.5)


def test_fit_easylinear_non_positive_values_are_floored() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, -0.01, 0.02, 0.05, 0.12, 0.3])

    fit = fit_easylinear(t, y, h=3, floor=1e-4)
    assert np.isfinite(fit.mumax)
    assert fit.y0 == pytest.approx(1e-4)


def test_fit_growth_curves_iterates_groups() -> None:
    t = np.arange(0.0, 10.0 + 0.25, 0.25)
    rows: list[dict[str, object]] = []
    for carbon, mu in [("GMC", 0.5), ("LNT", 0.3)]:
        y = _lag_exp_curve(t, y0=0.01, mumax=mu, lag=1.0)
        for rep in (1, 2):
            for ti, yi in zip(t, y, strict=True):
                rows.append(
                    {
                        "time_h": ti,
                        "carbon_source": carbon,
                        "replicate": rep,
                        "od": float(yi),
                    }
                )
    df = pd.DataFrame(rows)

    out = fit_growth_curves(df, h=8)

    assert set(out["carbon_source"]) == {"GMC", "LNT"}
    assert len(out) == 4
    gmc_mu = out.loc[out["carbon_source"] == "GMC", "mumax"].mean()
    lnt_mu = out.loc[out["carbon_source"] == "LNT", "mumax"].mean()
    assert gmc_mu == pytest.approx(0.5, rel=1e-4)
    assert lnt_mu == pytest.approx(0.3, rel=1e-4)


def test_fit_growth_curves_short_group_returns_nan_row() -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "carbon_source": ["A", "A", "B", "B", "B", "B", "B", "B", "B"],
            "replicate": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "od": [0.01, 0.02, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        }
    )
    out = fit_growth_curves(df, h=5)
    a_row = out.loc[out["carbon_source"] == "A"].iloc[0]
    b_row = out.loc[out["carbon_source"] == "B"].iloc[0]
    assert a_row["n_points"] == 0
    assert pd.isna(a_row["mumax"])
    assert b_row["n_points"] >= 5
    assert np.isfinite(b_row["mumax"])


def test_fit_growth_curves_missing_column_raises() -> None:
    df = pd.DataFrame({"time_h": [0.0, 1.0], "od": [0.1, 0.2]})
    with pytest.raises(ValueError, match="missing required columns"):
        fit_growth_curves(df)

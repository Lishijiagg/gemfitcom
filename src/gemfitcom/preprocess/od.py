"""OD growth-curve preprocessing.

Operates on the canonical long format produced by :mod:`gemfitcom.io.od`::

    time_h : float
    carbon_source : str
    replicate : int
    od : float

Typical pipeline::

    df = load_od(...)
    df = subtract_t0(df)
    df = floor_od(df)
    summary = average_replicates(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FLOOR: float = 1e-4
_GROUP_COLS: tuple[str, ...] = ("carbon_source", "replicate")


def subtract_t0(df: pd.DataFrame, *, time_column: str = "time_h") -> pd.DataFrame:
    """Subtract the earliest-time OD per ``(carbon_source, replicate)`` from all its points.

    This is the baseline-removal step used by the upstream R pipeline
    (``mutate(across(everything(), ~ . - first(.)))``): each curve is aligned
    to start at zero so differences between wells at t=0 do not contaminate
    later timepoints.

    Args:
        df: Long OD DataFrame.
        time_column: Column to sort by when determining t=0.

    Returns:
        A new DataFrame with the same schema and ``od`` baseline-corrected.
    """
    _require_columns(df, (*_GROUP_COLS, time_column, "od"))
    out = df.copy()
    out = out.sort_values([*_GROUP_COLS, time_column]).reset_index(drop=True)
    baseline = out.groupby(list(_GROUP_COLS))["od"].transform("first")
    out["od"] = out["od"] - baseline
    return out


def floor_od(df: pd.DataFrame, *, floor: float = DEFAULT_FLOOR) -> pd.DataFrame:
    """Clip non-positive OD values to ``floor`` so downstream ``log`` is safe.

    Args:
        df: Long OD DataFrame.
        floor: Minimum positive value. Must be > 0.

    Returns:
        A new DataFrame with ``od`` clipped.
    """
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")
    _require_columns(df, ("od",))
    out = df.copy()
    out["od"] = np.where(out["od"] > floor, out["od"], floor)
    return out


def average_replicates(
    df: pd.DataFrame,
    *,
    time_column: str = "time_h",
    group_column: str = "carbon_source",
) -> pd.DataFrame:
    """Collapse replicates to mean/sd/n per (time, carbon_source).

    Args:
        df: Long OD DataFrame.
        time_column: Column holding time in hours.
        group_column: Column identifying the condition (default
            ``"carbon_source"``).

    Returns:
        DataFrame with columns ``time_h, carbon_source, mean_od, sd_od,
        n_replicates`` (``sd_od`` is NaN when only one replicate is present).
    """
    _require_columns(df, (time_column, group_column, "od"))
    grouped = df.groupby([time_column, group_column], sort=True, as_index=False)
    summary = grouped["od"].agg(mean_od="mean", sd_od="std", n_replicates="count")
    return summary


def smooth_od(
    df: pd.DataFrame,
    *,
    window: int = 3,
    min_periods: int | None = None,
    time_column: str = "time_h",
) -> pd.DataFrame:
    """Apply a centered rolling-mean smoother to OD per ``(carbon_source, replicate)``.

    Args:
        df: Long OD DataFrame.
        window: Rolling window size in samples. Must be >= 1.
        min_periods: Passed to ``pd.DataFrame.rolling``; defaults to 1 so the
            curve edges are not dropped.
        time_column: Column to sort by before smoothing.

    Returns:
        A new DataFrame with the same schema and a smoothed ``od`` column.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    _require_columns(df, (*_GROUP_COLS, time_column, "od"))
    if min_periods is None:
        min_periods = 1
    out = df.sort_values([*_GROUP_COLS, time_column]).reset_index(drop=True)
    out["od"] = out.groupby(list(_GROUP_COLS))["od"].transform(
        lambda s: s.rolling(window, center=True, min_periods=min_periods).mean()
    )
    return out


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns {missing}")

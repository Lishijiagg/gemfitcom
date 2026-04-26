"""Maximum specific growth-rate extraction from OD curves.

Python port of ``growthrates::fit_easylinear`` (R). The algorithm:

1. Take ``log(y)`` (values below ``floor`` are clipped to ``floor`` to allow
   the logarithm on noisy baseline points).
2. Slide a window of length ``h`` across the log-linear-transformed curve
   and fit an OLS line in every window.
3. Retain every window whose slope is at least ``quota * max_slope``.
4. Merge the retained windows into a single contiguous index range and refit
   one regression across that range.
5. Report the refit slope as ``mumax`` and the lag time defined as
   ``lag = (log(y[0]) - intercept) / mumax``, i.e. the time at which the
   fitted exponential-phase line crosses the first observed value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

DEFAULT_H: int = 5
DEFAULT_QUOTA: float = 0.95
DEFAULT_FLOOR: float = 1e-4


@dataclass(frozen=True, slots=True)
class GrowthFit:
    """Result of a single log-linear growth-rate fit.

    Attributes:
        mumax: Maximum specific growth rate (slope of log(y) vs t) in 1/h.
        lag: Lag time in hours.
        y0: First observed value used as the lag-crossing reference.
        intercept: Intercept of the fitted ``log(y) = intercept + mumax * t``.
        r_squared: Coefficient of determination over the final window.
        window_start: Index (into the input arrays) of the first point.
        window_end: Index one past the last point (half-open, like slicing).
        n_points: Number of points in the final window.
    """

    mumax: float
    lag: float
    y0: float
    intercept: float
    r_squared: float
    window_start: int
    window_end: int
    n_points: int


def fit_easylinear(
    t: ArrayLike,
    y: ArrayLike,
    *,
    h: int = DEFAULT_H,
    quota: float = DEFAULT_QUOTA,
    floor: float = DEFAULT_FLOOR,
) -> GrowthFit:
    """Fit mumax / lag via sliding-window log-linear regression.

    Args:
        t: Time points (length ``n``).
        y: Observed values (e.g. OD), same length as ``t``.
        h: Window length in number of samples. Must satisfy ``2 <= h <= n``.
        quota: Proportion of the initial window's R² that must be retained
            when extending the window (``0 < quota <= 1``).
        floor: Minimum positive value substituted for non-positive ``y`` before
            the log transform.

    Returns:
        A :class:`GrowthFit`.

    Raises:
        ValueError: on invalid ``h``, ``quota``, length mismatch, or too-short
            input.
    """
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("t and y must be 1-D arrays")
    if t_arr.shape != y_arr.shape:
        raise ValueError(f"t and y must have equal length, got {t_arr.shape} vs {y_arr.shape}")
    n = t_arr.size
    if h < 2:
        raise ValueError(f"h must be >= 2, got {h}")
    if h > n:
        raise ValueError(f"h ({h}) exceeds number of samples ({n})")
    if not 0.0 < quota <= 1.0:
        raise ValueError(f"quota must be in (0, 1], got {quota}")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")

    log_y = np.log(np.maximum(y_arr, floor))

    n_windows = n - h + 1
    slopes = np.empty(n_windows, dtype=float)
    for i in range(n_windows):
        s, _, _, _, _ = stats.linregress(t_arr[i : i + h], log_y[i : i + h])
        slopes[i] = s

    max_slope = float(slopes.max())
    threshold = quota * max_slope
    candidates = np.flatnonzero(slopes >= threshold)
    first_window = int(candidates.min())
    last_window = int(candidates.max())
    start = first_window
    end = last_window + h

    slope, intercept, r, _, _ = stats.linregress(t_arr[start:end], log_y[start:end])

    y0 = float(max(y_arr[0], floor))
    lag = float((np.log(y0) - intercept) / slope) if slope != 0.0 else float("nan")

    return GrowthFit(
        mumax=float(slope),
        lag=lag,
        y0=y0,
        intercept=float(intercept),
        r_squared=float(r * r),
        window_start=int(start),
        window_end=int(end),
        n_points=int(end - start),
    )


def fit_growth_curves(
    df: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("carbon_source", "replicate"),
    time_column: str = "time_h",
    value_column: str = "od",
    h: int = DEFAULT_H,
    quota: float = DEFAULT_QUOTA,
    floor: float = DEFAULT_FLOOR,
) -> pd.DataFrame:
    """Apply :func:`fit_easylinear` per group of a long-format DataFrame.

    Args:
        df: Long-format DataFrame containing at least ``time_column``,
            ``value_column``, and every column in ``group_by``.
        group_by: Columns whose unique combinations define one growth curve.
        time_column: Name of the time column (hours).
        value_column: Name of the observation column (e.g. ``"od"``).
        h: Passed to :func:`fit_easylinear`.
        quota: Passed to :func:`fit_easylinear`.
        floor: Passed to :func:`fit_easylinear`.

    Returns:
        DataFrame indexed by the grouping columns with fit fields as columns
        (``mumax``, ``lag``, ``y0``, ``intercept``, ``r_squared``,
        ``window_start``, ``window_end``, ``n_points``). Curves shorter than
        ``h`` are skipped with a ``n_points = 0`` row so the caller sees which
        groups failed.
    """
    required = {time_column, value_column, *group_by}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns {sorted(missing)}")

    records: list[dict[str, object]] = []
    for key, grp in df.groupby(list(group_by), sort=True):
        grp_sorted = grp.sort_values(time_column)
        t = grp_sorted[time_column].to_numpy(dtype=float)
        y = grp_sorted[value_column].to_numpy(dtype=float)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_by, key_tuple, strict=True))
        if t.size < h:
            row.update(
                mumax=float("nan"),
                lag=float("nan"),
                y0=float("nan"),
                intercept=float("nan"),
                r_squared=float("nan"),
                window_start=0,
                window_end=0,
                n_points=0,
            )
        else:
            fit = fit_easylinear(t, y, h=h, quota=quota, floor=floor)
            row.update(
                mumax=fit.mumax,
                lag=fit.lag,
                y0=fit.y0,
                intercept=fit.intercept,
                r_squared=fit.r_squared,
                window_start=fit.window_start,
                window_end=fit.window_end,
                n_points=fit.n_points,
            )
        records.append(row)
    return pd.DataFrame.from_records(records)

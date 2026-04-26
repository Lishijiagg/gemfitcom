"""Competition edges from a long-form exchange-flux panel.

Two strains compete for a metabolite at a given time point if they both
take it up (flux < 0). Competition intensity for a pair ``(a, b)`` on
metabolite ``m`` at time ``t`` is defined as the *shared demand*::

    intensity(a, b, t, m) = min(|uptake_a|, |uptake_b|)

integrated over the horizon with a left-rectangle rule (last time
point contributes ``dt = 0``). Pairs are unordered: output columns are
``strain_a`` and ``strain_b`` with ``strain_a < strain_b``
lexicographically.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from gemfitcom.interactions.panel import PANEL_COLUMNS


def competition_edges(
    panel: pd.DataFrame,
    biomass: pd.DataFrame | None = None,
    *,
    dt: float | None = None,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Integrate pairwise shared-uptake demand across the horizon.

    Args:
        panel: Long-form flux table with columns
            ``(time_h, strain, exchange_id, flux)``.
        biomass: Optional long-form biomass with columns
            ``(time_h, strain, biomass)``. When provided, per-strain
            fluxes are weighted by biomass before the pairwise min.
        dt: Time step override (hours). Same conventions as
            :func:`cross_feeding_edges`.
        tol: Flux magnitudes below ``tol`` are treated as zero.

    Returns:
        DataFrame with columns ``(strain_a, strain_b, exchange_id,
        competition_intensity)``, sorted by descending intensity.
        ``strain_a < strain_b`` lexicographically.
    """
    required = set(PANEL_COLUMNS)
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    edge_columns = ["strain_a", "strain_b", "exchange_id", "competition_intensity"]
    if panel.empty:
        return pd.DataFrame(columns=edge_columns)

    work = panel[list(PANEL_COLUMNS)].copy()

    if biomass is not None:
        bcols = {"time_h", "strain", "biomass"}
        miss_b = bcols - set(biomass.columns)
        if miss_b:
            raise ValueError(f"biomass missing required columns: {sorted(miss_b)}")
        work = work.merge(
            biomass[["time_h", "strain", "biomass"]],
            on=["time_h", "strain"],
            how="left",
        )
        if work["biomass"].isna().any():
            raise ValueError("biomass does not cover all (time_h, strain) pairs in panel")
        work["flux"] = work["flux"] * work["biomass"]
        work = work.drop(columns="biomass")

    times = np.sort(work["time_h"].unique())
    if times.size == 1:
        if dt is None:
            warnings.warn(
                "competition_edges: panel has a single time point and dt is "
                "None; defaulting dt=1.0 (intensity = rate * 1h).",
                stacklevel=2,
            )
            step_dt = 1.0
        else:
            step_dt = float(dt)
        dt_map: dict[float, float] = {float(times[0]): step_dt}
    elif dt is not None:
        dt_map = dict.fromkeys((float(t) for t in times), float(dt))
        dt_map[float(times[-1])] = 0.0
    else:
        diffs = np.diff(times)
        dt_map = {float(t): float(d) for t, d in zip(times[:-1], diffs, strict=True)}
        dt_map[float(times[-1])] = 0.0

    accumulator: dict[tuple[str, str, str], float] = {}
    for (t, exch), sub in work.groupby(["time_h", "exchange_id"], sort=False):
        step_dt = dt_map.get(float(t), 0.0)
        if step_dt == 0.0:
            continue
        f = sub["flux"].to_numpy(dtype=float)
        s = sub["strain"].to_numpy()
        up_mask = f < -tol
        if up_mask.sum() < 2:
            continue
        up_flux = -f[up_mask]
        up_names = s[up_mask]
        for i, j in combinations(range(len(up_names)), 2):
            a, b = str(up_names[i]), str(up_names[j])
            if a > b:
                a, b = b, a
            intensity = float(min(up_flux[i], up_flux[j])) * step_dt
            if intensity <= 0.0:
                continue
            key = (a, b, str(exch))
            accumulator[key] = accumulator.get(key, 0.0) + intensity

    if not accumulator:
        return pd.DataFrame(columns=edge_columns)

    out = pd.DataFrame(
        [
            {
                "strain_a": k[0],
                "strain_b": k[1],
                "exchange_id": k[2],
                "competition_intensity": v,
            }
            for k, v in accumulator.items()
        ]
    )
    return out.sort_values("competition_intensity", ascending=False).reset_index(drop=True)


__all__ = ["competition_edges"]

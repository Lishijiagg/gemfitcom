"""Cross-feeding edges from a long-form exchange-flux panel.

At each time point and for each ``exchange_id`` we have a set of
strains with positive flux (secretors) and negative flux (uptakers).
The metabolite flow from donors to recipients over a time step is
attributed by *double-proportional allocation*::

    exchanged(t, m)   = min(sum_secretion, sum_uptake)
    flow(d, r, t, m)  = exchanged
                        * (secretion_d / sum_secretion)
                        * (uptake_r    / sum_uptake)

and integrated over the horizon with a left-rectangle rule. The last
time point contributes ``dt = 0`` because its flux has no forward step
to apply to (matches the way the simulate modules duplicate the final
row).

If ``biomass`` is provided, per-strain fluxes are multiplied by biomass
before allocation so cumulative flow is in ``mmol / L`` rather than
``mmol / gDW``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from gemfitcom.interactions.panel import PANEL_COLUMNS


def cross_feeding_edges(
    panel: pd.DataFrame,
    biomass: pd.DataFrame | None = None,
    *,
    dt: float | None = None,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Integrate donor->recipient metabolite flows across the horizon.

    Args:
        panel: Long-form flux table with columns
            ``(time_h, strain, exchange_id, flux)``. Sign convention:
            positive = secretion, negative = uptake.
        biomass: Optional long-form biomass with columns
            ``(time_h, strain, biomass)`` (e.g., from
            :func:`biomass_panel`). When provided, per-strain fluxes are
            weighted by biomass to get absolute rates.
        dt: Time step override (hours). If ``None`` and the panel has
            only one time point (e.g., a MICOM snapshot), ``dt``
            defaults to ``1.0`` with a :class:`UserWarning`. If ``None``
            and the panel has >=2 time points, per-step ``dt`` values
            are inferred from consecutive ``time_h`` differences.
        tol: Flux magnitudes below ``tol`` are treated as zero.

    Returns:
        DataFrame with columns ``(donor, recipient, exchange_id,
        cumulative_flow)``, sorted by descending flow. Donor != recipient
        by construction (a strain is never both secreting and uptaking
        the same metabolite at the same time point).
    """
    required = set(PANEL_COLUMNS)
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    edge_columns = ["donor", "recipient", "exchange_id", "cumulative_flow"]
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
                "cross_feeding_edges: panel has a single time point and dt "
                "is None; defaulting dt=1.0 (cumulative_flow = rate * 1h).",
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
        donor_mask = f > tol
        recip_mask = f < -tol
        if not donor_mask.any() or not recip_mask.any():
            continue
        d_flux = f[donor_mask]
        d_names = s[donor_mask]
        r_flux = -f[recip_mask]
        r_names = s[recip_mask]
        sum_sec = float(d_flux.sum())
        sum_up = float(r_flux.sum())
        exchanged = min(sum_sec, sum_up)
        if exchanged <= 0.0:
            continue
        flows = exchanged * np.outer(d_flux / sum_sec, r_flux / sum_up) * step_dt
        for i, dn in enumerate(d_names):
            for j, rn in enumerate(r_names):
                key = (str(dn), str(rn), str(exch))
                accumulator[key] = accumulator.get(key, 0.0) + float(flows[i, j])

    if not accumulator:
        return pd.DataFrame(columns=edge_columns)

    out = pd.DataFrame(
        [
            {
                "donor": k[0],
                "recipient": k[1],
                "exchange_id": k[2],
                "cumulative_flow": v,
            }
            for k, v in accumulator.items()
        ]
    )
    return out.sort_values("cumulative_flow", ascending=False).reset_index(drop=True)


__all__ = ["cross_feeding_edges"]

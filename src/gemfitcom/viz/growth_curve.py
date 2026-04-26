"""Growth-curve plot: observed biomass vs. fitted dFBA trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemfitcom.kinetics.fit import FitResult


def plot_growth_curve(
    t_obs: ArrayLike,
    biomass_obs: ArrayLike,
    fit_result: FitResult | None = None,
    *,
    ax: Axes | None = None,
    obs_label: str = "Observed",
    fit_label: str = "Fit",
    title: str | None = None,
    log_y: bool = False,
) -> Figure:
    """Plot observed biomass points and (optionally) the fitted simulation curve.

    Args:
        t_obs: Observation time points (hours).
        biomass_obs: Observed biomass at ``t_obs`` (gDW/L).
        fit_result: A :class:`gemfitcom.kinetics.fit.FitResult`. When supplied,
            the simulated biomass at the grid optimum is overlaid as a line and
            its R² is shown in the legend.
        ax: Optional axes to draw on; a new figure is created when omitted.
        obs_label: Legend label for the observed-data scatter.
        fit_label: Legend label for the fitted curve.
        title: Optional axes title. Default is ``"Growth curve fit"`` when
            ``fit_result`` is given, otherwise ``"Growth curve"``.
        log_y: If ``True``, use a log scale on the y-axis.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the plot.
    """
    import matplotlib.pyplot as plt

    t_obs_arr = np.asarray(t_obs, dtype=float)
    biomass_obs_arr = np.asarray(biomass_obs, dtype=float)
    if t_obs_arr.shape != biomass_obs_arr.shape or t_obs_arr.ndim != 1:
        raise ValueError("t_obs and biomass_obs must be 1-D arrays of the same length")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure

    ax.plot(
        t_obs_arr,
        biomass_obs_arr,
        marker="o",
        linestyle="",
        color="black",
        label=obs_label,
        zorder=3,
    )

    if fit_result is not None:
        fit_label_full = f"{fit_label} (R² = {fit_result.r_squared:.3f})"
        ax.plot(
            fit_result.sim_time_h,
            fit_result.sim_biomass,
            color="tab:red",
            linewidth=2.0,
            label=fit_label_full,
            zorder=2,
        )

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Biomass (gDW/L)")
    if log_y:
        ax.set_yscale("log")
    if title is None:
        title = "Growth curve fit" if fit_result is not None else "Growth curve"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


__all__ = ["plot_growth_curve"]

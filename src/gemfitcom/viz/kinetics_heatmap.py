"""Vmax × Km R² heatmap for the refinement grid of :func:`fit_kinetics`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemfitcom.kinetics.fit import FitResult


def plot_kinetics_heatmap(
    fit_result: FitResult,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    r2_floor: float | None = 0.0,
    mark_optimum: bool = True,
    title: str | None = None,
) -> Figure:
    """Render the Vmax × Km R² grid as a heatmap.

    Rows are Km (``fit_result.grid_km_axis``); columns are Vmax
    (``fit_result.grid_vmax_axis``). The grid optimum is marked with
    a white-edged red star when ``mark_optimum=True``.

    Args:
        fit_result: A :class:`gemfitcom.kinetics.fit.FitResult`.
        ax: Optional axes to draw on; a new figure is created when omitted.
        cmap: Matplotlib colormap name.
        r2_floor: If not ``None``, R² values below this floor are clipped to
            the floor before plotting (so a few catastrophic failures don't
            wash out the dynamic range). ``-inf`` cells are always clipped.
            Default ``0.0`` clips negative R² values to 0.
        mark_optimum: Overlay a star at ``fit_result.params``.
        title: Optional axes title.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the heatmap.
    """
    import matplotlib.pyplot as plt

    v_axis = np.asarray(fit_result.grid_vmax_axis, dtype=float)
    k_axis = np.asarray(fit_result.grid_km_axis, dtype=float)
    grid = np.asarray(fit_result.grid_r_squared, dtype=float)
    if grid.shape != (k_axis.size, v_axis.size):
        raise ValueError(
            f"grid_r_squared shape {grid.shape} does not match "
            f"(len(grid_km_axis), len(grid_vmax_axis)) = ({k_axis.size}, {v_axis.size})"
        )

    plot_grid = np.where(np.isfinite(grid), grid, np.nan)
    if r2_floor is not None:
        plot_grid = np.where(plot_grid < r2_floor, r2_floor, plot_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
    else:
        fig = ax.figure

    dv = (v_axis[-1] - v_axis[0]) / (v_axis.size - 1) if v_axis.size > 1 else 1.0
    dk = (k_axis[-1] - k_axis[0]) / (k_axis.size - 1) if k_axis.size > 1 else 1.0
    extent = (
        v_axis[0] - dv / 2,
        v_axis[-1] + dv / 2,
        k_axis[0] - dk / 2,
        k_axis[-1] + dk / 2,
    )
    im = ax.imshow(
        plot_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R²")

    if mark_optimum:
        ax.plot(
            fit_result.params.vmax,
            fit_result.params.km,
            marker="*",
            markersize=14,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.0,
            linestyle="",
            label=f"Best (R² = {fit_result.r_squared:.3f})",
            zorder=4,
        )
        ax.legend(loc="best", frameon=False)

    ax.set_xlabel("Vmax (mmol / gDW / h)")
    ax.set_ylabel("Km (mM)")
    ax.set_title(title or "Vmax × Km R² grid")
    fig.tight_layout()
    return fig


__all__ = ["plot_kinetics_heatmap"]

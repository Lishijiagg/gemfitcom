"""Tests for gemfitcom.viz (growth curve, kinetics heatmap, interaction network)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from gemfitcom.interactions import summary_graph
from gemfitcom.kinetics.fit import FitResult
from gemfitcom.kinetics.mm import MMParams
from gemfitcom.viz import plot_growth_curve, plot_interaction_network, plot_kinetics_heatmap


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _toy_fit_result(
    *,
    vmax_best: float = 4.0,
    km_best: float = 2.0,
    grid_n: int = 5,
    r2_best: float = 0.95,
) -> FitResult:
    v_axis = np.linspace(vmax_best * 0.5, vmax_best * 1.5, grid_n)
    k_axis = np.linspace(km_best * 0.5, km_best * 1.5, grid_n)
    vv, kk = np.meshgrid(v_axis, k_axis)
    grid = r2_best - 0.05 * ((vv - vmax_best) ** 2 + (kk - km_best) ** 2)
    return FitResult(
        params=MMParams(vmax=vmax_best, km=km_best),
        r_squared=float(grid.max()),
        sim_time_h=np.linspace(0.0, 6.0, 25),
        sim_biomass=0.01 * np.exp(0.4 * np.linspace(0.0, 6.0, 25)),
        de_params=MMParams(vmax=vmax_best * 1.05, km=km_best * 0.95),
        de_r_squared=float(grid.max()) - 0.01,
        grid_vmax_axis=v_axis,
        grid_km_axis=k_axis,
        grid_r_squared=grid,
    )


def test_growth_curve_returns_figure_with_obs_only() -> None:
    t = np.linspace(0.0, 6.0, 7)
    b = 0.01 * np.exp(0.3 * t)

    fig = plot_growth_curve(t, b)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time (h)"
    assert "Biomass" in ax.get_ylabel()
    assert ax.get_title() == "Growth curve"
    assert len(ax.lines) == 1


def test_growth_curve_overlays_fit_when_provided() -> None:
    t = np.linspace(0.0, 6.0, 7)
    b = 0.01 * np.exp(0.3 * t)
    fit = _toy_fit_result()

    fig = plot_growth_curve(t, b, fit_result=fit)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("R²" in label for label in labels)
    assert ax.get_title() == "Growth curve fit"


def test_growth_curve_log_y() -> None:
    t = np.linspace(0.0, 6.0, 7)
    b = 0.01 * np.exp(0.3 * t)

    fig = plot_growth_curve(t, b, log_y=True)
    assert fig.axes[0].get_yscale() == "log"


def test_growth_curve_validates_input_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_growth_curve(np.array([0.0, 1.0, 2.0]), np.array([0.01, 0.02]))


def test_growth_curve_uses_provided_axes() -> None:
    fig, ax = plt.subplots()
    t = np.linspace(0.0, 4.0, 5)
    b = np.linspace(0.01, 0.05, 5)
    out = plot_growth_curve(t, b, ax=ax)
    assert out is fig


def test_kinetics_heatmap_returns_figure() -> None:
    fit = _toy_fit_result(grid_n=7)

    fig = plot_kinetics_heatmap(fit)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert "Vmax" in ax.get_xlabel()
    assert "Km" in ax.get_ylabel()
    assert ax.get_title() == "Vmax × Km R² grid"
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (7, 7)


def test_kinetics_heatmap_marks_optimum() -> None:
    fit = _toy_fit_result()
    fig = plot_kinetics_heatmap(fit, mark_optimum=True)
    ax = fig.axes[0]
    star_lines = [ln for ln in ax.lines if ln.get_marker() == "*"]
    assert len(star_lines) == 1
    xs = star_lines[0].get_xdata()
    ys = star_lines[0].get_ydata()
    assert pytest.approx(xs[0], rel=1e-9) == fit.params.vmax
    assert pytest.approx(ys[0], rel=1e-9) == fit.params.km


def test_kinetics_heatmap_can_skip_optimum_marker() -> None:
    fit = _toy_fit_result()
    fig = plot_kinetics_heatmap(fit, mark_optimum=False)
    ax = fig.axes[0]
    star_lines = [ln for ln in ax.lines if ln.get_marker() == "*"]
    assert star_lines == []


def test_kinetics_heatmap_validates_grid_shape() -> None:
    fit = _toy_fit_result(grid_n=5)
    bad = FitResult(
        params=fit.params,
        r_squared=fit.r_squared,
        sim_time_h=fit.sim_time_h,
        sim_biomass=fit.sim_biomass,
        de_params=fit.de_params,
        de_r_squared=fit.de_r_squared,
        grid_vmax_axis=fit.grid_vmax_axis,
        grid_km_axis=fit.grid_km_axis,
        grid_r_squared=np.zeros((3, 5)),
    )
    with pytest.raises(ValueError, match="grid_r_squared shape"):
        plot_kinetics_heatmap(bad)


def _interaction_panel() -> pd.DataFrame:
    rows = [
        (0.0, "A", "EX_ac_e", 2.0),
        (0.0, "B", "EX_ac_e", -1.0),
        (0.0, "C", "EX_ac_e", -1.5),
        (0.0, "B", "EX_glc__D_e", -2.0),
        (0.0, "C", "EX_glc__D_e", -1.0),
        (1.0, "A", "EX_ac_e", 0.0),
        (1.0, "B", "EX_ac_e", 0.0),
        (1.0, "C", "EX_ac_e", 0.0),
        (1.0, "B", "EX_glc__D_e", 0.0),
        (1.0, "C", "EX_glc__D_e", 0.0),
    ]
    return pd.DataFrame(rows, columns=["time_h", "strain", "exchange_id", "flux"])


def test_interaction_network_returns_figure_with_legend() -> None:
    g = summary_graph(_interaction_panel())
    fig = plot_interaction_network(g)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Strain interaction network"
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert "Cross-feeding" in labels
    assert "Competition" in labels


def test_interaction_network_hides_competition_when_requested() -> None:
    g = summary_graph(_interaction_panel())
    fig = plot_interaction_network(g, show_competition=False)
    ax = fig.axes[0]
    legend = ax.get_legend()
    labels = {t.get_text() for t in legend.get_texts()} if legend else set()
    assert "Competition" not in labels


def test_interaction_network_handles_empty_graph() -> None:
    g = nx.MultiDiGraph()
    g.add_node("A")
    g.add_node("B")
    fig = plot_interaction_network(g)
    assert isinstance(fig, Figure)


def test_interaction_network_layouts() -> None:
    g = summary_graph(_interaction_panel())
    for layout in ("spring", "circular", "kamada_kawai", "shell"):
        fig = plot_interaction_network(g, layout=layout)
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_interaction_network_rejects_unknown_layout() -> None:
    g = summary_graph(_interaction_panel())
    with pytest.raises(ValueError, match="unknown layout"):
        plot_interaction_network(g, layout="not_a_layout")


def test_interaction_network_label_edges() -> None:
    g = summary_graph(_interaction_panel())
    fig = plot_interaction_network(g, label_edges=True)
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert any("EX_" in t for t in texts)

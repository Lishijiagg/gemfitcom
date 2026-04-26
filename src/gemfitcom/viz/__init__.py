"""Plotting utilities: growth curves, kinetics heatmaps, interaction networks.

All functions return a :class:`matplotlib.figure.Figure`. Saving to disk is
the caller's responsibility.
"""

from gemfitcom.viz.growth_curve import plot_growth_curve
from gemfitcom.viz.interaction_network import plot_interaction_network
from gemfitcom.viz.kinetics_heatmap import plot_kinetics_heatmap

__all__ = [
    "plot_growth_curve",
    "plot_interaction_network",
    "plot_kinetics_heatmap",
]

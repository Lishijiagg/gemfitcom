"""Michaelis-Menten kinetics fitting.

Two-stage fit:
    1. ``scipy.optimize.differential_evolution`` global coarse search.
    2. Local grid refinement around the DE optimum with a sensitivity heatmap.
"""

from gemfitcom.kinetics.fit import (
    DEFAULT_DE_MAXITER,
    DEFAULT_DE_POPSIZE,
    DEFAULT_GRID_POINTS,
    DEFAULT_GRID_SPAN,
    DEFAULT_KM_BOUNDS,
    DEFAULT_VMAX_BOUNDS,
    FitResult,
    fit_kinetics,
)
from gemfitcom.kinetics.mm import MMParams, michaelis_menten
from gemfitcom.kinetics.mono_dfba import DEFAULT_DT, MonoDFBAResult, simulate_mono_dfba

__all__ = [
    "DEFAULT_DE_MAXITER",
    "DEFAULT_DE_POPSIZE",
    "DEFAULT_DT",
    "DEFAULT_GRID_POINTS",
    "DEFAULT_GRID_SPAN",
    "DEFAULT_KM_BOUNDS",
    "DEFAULT_VMAX_BOUNDS",
    "FitResult",
    "MMParams",
    "MonoDFBAResult",
    "fit_kinetics",
    "michaelis_menten",
    "simulate_mono_dfba",
]

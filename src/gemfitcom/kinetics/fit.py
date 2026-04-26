"""Two-stage Vmax/Km fitting against an observed biomass time series.

Stage 1 (global): :func:`scipy.optimize.differential_evolution` searches the
full ``(vmax_bounds, km_bounds)`` rectangle, minimizing ``1 - R^2`` between
simulated and observed biomass on the raw (gDW/L) scale. Raw scale matters:
normalizing both trajectories by their own max destroys the absolute-scale
information and creates a structural identifiability ridge along which very
different ``(vmax, km)`` pairs all score near 1.0 — losing parameter
recovery on clean data.

Stage 2 (refinement): a uniform grid is laid down around the DE optimum
(spanning ``±grid_span * optimum`` in each dimension, clipped to the
original bounds) and R² is tabulated at every node. The grid best is
reported as the final estimate, and the 2-D grid is returned so the viz
module can render the parameter-sensitivity heatmap used in the original
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.kinetics.mono_dfba import DEFAULT_DT, simulate_mono_dfba
from gemfitcom.medium.medium import Medium

if TYPE_CHECKING:
    from cobra import Model


DEFAULT_VMAX_BOUNDS: tuple[float, float] = (0.001, 20.0)
DEFAULT_KM_BOUNDS: tuple[float, float] = (0.001, 30.0)
DEFAULT_DE_MAXITER: int = 50
DEFAULT_DE_POPSIZE: int = 15
DEFAULT_GRID_POINTS: int = 21
DEFAULT_GRID_SPAN: float = 0.5


@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of :func:`fit_kinetics`.

    Attributes:
        params: Best ``MMParams`` (from the refinement grid).
        r_squared: R² of simulated vs. observed biomass at the grid optimum.
        sim_time_h: Simulated time axis at the grid optimum.
        sim_biomass: Simulated biomass trajectory at the grid optimum (gDW/L).
        de_params: DE-stage optimum (stage 1).
        de_r_squared: R² at the DE optimum.
        grid_vmax_axis: Vmax axis of the refinement grid.
        grid_km_axis: Km axis of the refinement grid.
        grid_r_squared: R² tabulated on the grid, shape ``(len(km), len(vmax))``.
            Rows correspond to ``grid_km_axis``, columns to ``grid_vmax_axis``
            (row-major matches matplotlib/seaborn heatmap conventions).
    """

    params: MMParams
    r_squared: float
    sim_time_h: np.ndarray
    sim_biomass: np.ndarray
    de_params: MMParams
    de_r_squared: float
    grid_vmax_axis: np.ndarray
    grid_km_axis: np.ndarray
    grid_r_squared: np.ndarray


def fit_kinetics(
    model: Model,
    medium: Medium,
    carbon_exchange: str,
    t_obs: ArrayLike,
    biomass_obs: ArrayLike,
    *,
    initial_biomass: float | None = None,
    vmax_bounds: tuple[float, float] = DEFAULT_VMAX_BOUNDS,
    km_bounds: tuple[float, float] = DEFAULT_KM_BOUNDS,
    de_maxiter: int = DEFAULT_DE_MAXITER,
    de_popsize: int = DEFAULT_DE_POPSIZE,
    grid_points: int = DEFAULT_GRID_POINTS,
    grid_span: float = DEFAULT_GRID_SPAN,
    dt: float = DEFAULT_DT,
    other_mm_params: dict[str, MMParams] | None = None,
    seed: int | None = 0,
) -> FitResult:
    """Fit Vmax / Km for a single carbon source against an OD-derived biomass curve.

    Args:
        model: COBRA model (mutated during fit — pass a copy if that matters).
        medium: Medium definition; ``carbon_exchange`` must be in its pool.
        carbon_exchange: Exchange reaction ID of the carbon source whose MM
            parameters are being fit (e.g. ``"EX_glc__D_e"``).
        t_obs: Observation time points in hours; must be strictly increasing
            and start at 0 (first point is treated as the initial condition).
        biomass_obs: Observed biomass (gDW/L) at ``t_obs``. First value is
            used as ``initial_biomass`` unless overridden.
        initial_biomass: Override the initial biomass (gDW/L). Defaults to
            ``biomass_obs[0]``.
        vmax_bounds: ``(lo, hi)`` for vmax in mmol/gDW/h.
        km_bounds: ``(lo, hi)`` for km in mM.
        de_maxiter: ``differential_evolution`` ``maxiter``.
        de_popsize: ``differential_evolution`` ``popsize``.
        grid_points: Grid resolution per axis for stage 2 refinement.
        grid_span: Half-width of the refinement box as a fraction of the
            DE optimum (e.g. 0.5 spans ``[0.5*v_de, 1.5*v_de]``).
        dt: dFBA time step (hours).
        other_mm_params: Additional MM parameters applied to other pool
            substrates during the fit.
        seed: RNG seed passed to :func:`differential_evolution`.

    Returns:
        A :class:`FitResult`.
    """
    t_obs_arr = np.asarray(t_obs, dtype=float)
    biomass_obs_arr = np.asarray(biomass_obs, dtype=float)
    _validate_inputs(
        medium,
        carbon_exchange,
        t_obs_arr,
        biomass_obs_arr,
        vmax_bounds,
        km_bounds,
        grid_points,
        grid_span,
    )

    x0 = float(initial_biomass) if initial_biomass is not None else float(biomass_obs_arr[0])
    if x0 <= 0:
        raise ValueError(f"initial_biomass must be > 0, got {x0}")

    t_total = float(t_obs_arr[-1])
    if biomass_obs_arr.max() <= 0:
        raise ValueError("biomass_obs has no positive values")
    tss = float(np.sum((biomass_obs_arr - biomass_obs_arr.mean()) ** 2))
    if tss == 0.0:
        raise ValueError("biomass_obs is constant; R^2 is undefined")

    base_other = dict(other_mm_params) if other_mm_params else {}

    def r2_of(vmax: float, km: float) -> tuple[float, np.ndarray, np.ndarray]:
        mm = {**base_other, carbon_exchange: MMParams(vmax=vmax, km=km)}
        try:
            res = simulate_mono_dfba(
                model,
                medium,
                mm,
                initial_biomass=x0,
                t_total=t_total,
                dt=dt,
            )
        except Exception:
            return float("-inf"), np.array([]), np.array([])
        sim_b = np.interp(t_obs_arr, res.time_h, res.biomass)
        ss_res = float(np.sum((sim_b - biomass_obs_arr) ** 2))
        r2 = 1.0 - ss_res / tss
        return r2, res.time_h, res.biomass

    def de_objective(x: np.ndarray) -> float:
        r2, _, _ = r2_of(float(x[0]), float(x[1]))
        return 1.0 - r2

    de_result = differential_evolution(
        de_objective,
        bounds=[vmax_bounds, km_bounds],
        maxiter=de_maxiter,
        popsize=de_popsize,
        seed=seed,
        tol=1e-4,
        polish=False,
        updating="deferred",
    )
    vmax_de, km_de = float(de_result.x[0]), float(de_result.x[1])
    de_r2, _, _ = r2_of(vmax_de, km_de)

    v_lo = max(vmax_bounds[0], vmax_de * (1.0 - grid_span))
    v_hi = min(vmax_bounds[1], vmax_de * (1.0 + grid_span))
    k_lo = max(km_bounds[0], km_de * (1.0 - grid_span))
    k_hi = min(km_bounds[1], km_de * (1.0 + grid_span))
    if v_hi <= v_lo:
        v_lo, v_hi = vmax_bounds
    if k_hi <= k_lo:
        k_lo, k_hi = km_bounds

    v_axis = np.linspace(v_lo, v_hi, grid_points)
    k_axis = np.linspace(k_lo, k_hi, grid_points)
    r2_grid = np.full((grid_points, grid_points), -np.inf, dtype=float)

    best_r2 = -np.inf
    best_vmax = vmax_de
    best_km = km_de
    best_time: np.ndarray = np.array([])
    best_biomass: np.ndarray = np.array([])
    for i, k_val in enumerate(k_axis):
        for j, v_val in enumerate(v_axis):
            r2, t_sim, b_sim = r2_of(v_val, k_val)
            r2_grid[i, j] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_vmax = float(v_val)
                best_km = float(k_val)
                best_time = t_sim
                best_biomass = b_sim

    if best_time.size == 0:
        _, best_time, best_biomass = r2_of(best_vmax, best_km)

    return FitResult(
        params=MMParams(vmax=best_vmax, km=best_km),
        r_squared=float(best_r2),
        sim_time_h=best_time,
        sim_biomass=best_biomass,
        de_params=MMParams(vmax=vmax_de, km=km_de),
        de_r_squared=float(de_r2),
        grid_vmax_axis=v_axis,
        grid_km_axis=k_axis,
        grid_r_squared=r2_grid,
    )


def _validate_inputs(
    medium: Medium,
    carbon_exchange: str,
    t_obs: np.ndarray,
    biomass_obs: np.ndarray,
    vmax_bounds: tuple[float, float],
    km_bounds: tuple[float, float],
    grid_points: int,
    grid_span: float,
) -> None:
    if carbon_exchange not in medium.pool_components:
        raise KeyError(
            f"carbon_exchange {carbon_exchange!r} is not a pool component of medium {medium.name!r}"
        )
    if t_obs.ndim != 1 or biomass_obs.ndim != 1:
        raise ValueError("t_obs and biomass_obs must be 1-D arrays")
    if t_obs.shape != biomass_obs.shape:
        raise ValueError("t_obs and biomass_obs must have the same length")
    if t_obs.size < 3:
        raise ValueError("need at least 3 observation points to fit")
    if t_obs[0] != 0.0:
        raise ValueError(f"t_obs[0] must be 0, got {t_obs[0]}")
    if not np.all(np.diff(t_obs) > 0):
        raise ValueError("t_obs must be strictly increasing")
    if vmax_bounds[0] <= 0 or vmax_bounds[1] <= vmax_bounds[0]:
        raise ValueError(f"vmax_bounds must be (lo>0, hi>lo), got {vmax_bounds}")
    if km_bounds[0] <= 0 or km_bounds[1] <= km_bounds[0]:
        raise ValueError(f"km_bounds must be (lo>0, hi>lo), got {km_bounds}")
    if grid_points < 3:
        raise ValueError(f"grid_points must be >= 3, got {grid_points}")
    if not 0 < grid_span <= 1:
        raise ValueError(f"grid_span must be in (0, 1], got {grid_span}")

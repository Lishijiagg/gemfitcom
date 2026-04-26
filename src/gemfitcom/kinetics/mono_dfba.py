"""Single-strain dynamic FBA.

Fixed-step forward-Euler dFBA. At each step of length ``dt``:

1. For every pool metabolite with MM parameters, set the exchange's lower
   bound to ``-michaelis_menten(pool_conc, vmax, km)``. Pool metabolites
   without MM parameters are left at the model's current bound.
2. Call ``model.optimize()`` — the objective value is the specific growth
   rate μ (1/h) and ``solution.fluxes[ex_id]`` is the exchange flux per
   gDW of biomass.
3. Advance biomass by ``X_{t+dt} = X_t * exp(μ * dt)`` (analytic for the
   piecewise-constant μ over the step) and pool concentrations by
   ``C_{t+dt} = C_t + flux * X_avg * dt`` where ``X_avg`` is the mean
   biomass over the step. Clipped to 0 at the bottom.
4. If FBA fails or μ ≤ 0, biomass and pools are held fixed for the step.

This implementation is deliberately minimal; the ``simulate`` module will
offer a richer interface (MICOM / fusion modes) and share or supersede this
function once it lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gemfitcom.kinetics.mm import MMParams, michaelis_menten
from gemfitcom.medium.medium import Medium
from gemfitcom.utils.progress import progress_bar

if TYPE_CHECKING:
    from cobra import Model


DEFAULT_DT: float = 0.25


@dataclass(frozen=True, slots=True)
class MonoDFBAResult:
    """dFBA trajectory.

    Attributes:
        time_h: Time points (hours), shape ``(n,)``.
        biomass: Biomass trajectory (gDW / L), shape ``(n,)``.
        pool: Pool metabolite concentrations (mM), one column per exchange ID.
        growth_rate: Specific growth rate at each step (1/h), shape ``(n,)``.
            ``growth_rate[0]`` is the rate evaluated at the initial state.
    """

    time_h: np.ndarray
    biomass: np.ndarray
    pool: pd.DataFrame
    growth_rate: np.ndarray


def simulate_mono_dfba(
    model: Model,
    medium: Medium,
    mm_params: dict[str, MMParams],
    *,
    initial_biomass: float,
    t_total: float,
    dt: float = DEFAULT_DT,
    pool_init: dict[str, float] | None = None,
    progress: bool = False,
) -> MonoDFBAResult:
    """Integrate single-strain dFBA for ``t_total`` hours.

    Args:
        model: COBRA model. Its exchange bounds WILL be mutated during the
            call; pass ``model.copy()`` if you need the original preserved.
        medium: Sets ``pool_init`` defaults and identifies which exchanges
            are part of the dynamic pool.
        mm_params: Map ``exchange_id -> MMParams``. Exchanges listed here use
            MM-constrained uptake; unlisted pool exchanges keep the model's
            existing lower bound (typically 0 after :func:`apply_medium`).
        initial_biomass: Starting biomass in gDW / L. Must be > 0.
        t_total: Total simulation time (hours). Must be > 0.
        dt: Step size (hours). Must be > 0 and ≤ ``t_total``.
        pool_init: Override of initial pool concentrations (mM); merged over
            ``medium.pool_components``.
        progress: If True, show a tqdm progress bar over the time loop.
            Defaults to False so library calls stay silent; CLI and notebook
            callers should turn it on explicitly.

    Returns:
        A :class:`MonoDFBAResult`.

    Raises:
        ValueError: on invalid numeric arguments.
        KeyError: if an ``mm_params`` or ``pool_init`` key is not a pool
            component of ``medium`` or not an exchange in ``model``.
    """
    if initial_biomass <= 0:
        raise ValueError(f"initial_biomass must be > 0, got {initial_biomass}")
    if t_total <= 0:
        raise ValueError(f"t_total must be > 0, got {t_total}")
    if not 0 < dt <= t_total:
        raise ValueError(f"dt must satisfy 0 < dt <= t_total, got dt={dt}, t_total={t_total}")

    pool_ids = list(medium.pool_components)
    pool_id_set = set(pool_ids)

    for k in mm_params:
        if k not in pool_id_set:
            raise KeyError(f"mm_params key {k!r} is not a pool component of medium {medium.name!r}")
    if pool_init is not None:
        for k in pool_init:
            if k not in pool_id_set:
                raise KeyError(
                    f"pool_init key {k!r} is not a pool component of medium {medium.name!r}"
                )

    model_rxn_ids = {r.id for r in model.reactions}
    missing = [rid for rid in pool_ids if rid not in model_rxn_ids]
    if missing:
        raise KeyError(f"pool components not present as reactions in model: {sorted(missing)}")

    concs: dict[str, float] = dict(medium.pool_components)
    if pool_init is not None:
        concs.update(pool_init)

    n_steps = round(t_total / dt)
    n_points = n_steps + 1
    time_h = np.linspace(0.0, n_steps * dt, n_points)
    biomass = np.empty(n_points, dtype=float)
    growth_rate = np.zeros(n_points, dtype=float)
    pool_hist = np.empty((n_points, len(pool_ids)), dtype=float)

    biomass[0] = initial_biomass
    for j, rid in enumerate(pool_ids):
        pool_hist[0, j] = concs[rid]

    rxn_cache = {rid: model.reactions.get_by_id(rid) for rid in pool_ids}

    step_iter = progress_bar(
        range(n_steps),
        enabled=progress,
        desc="mono_dfba",
        total=n_steps,
        unit="step",
    )
    for i in step_iter:
        for j, rid in enumerate(pool_ids):
            if rid in mm_params:
                p = mm_params[rid]
                c = max(pool_hist[i, j], 0.0)
                rxn_cache[rid].lower_bound = -float(michaelis_menten(c, p.vmax, p.km))

        solution = model.optimize()
        if solution.status != "optimal":
            biomass[i + 1] = biomass[i]
            pool_hist[i + 1] = pool_hist[i]
            continue

        mu = float(solution.objective_value)
        growth_rate[i] = mu

        if mu <= 0:
            biomass[i + 1] = biomass[i]
            pool_hist[i + 1] = pool_hist[i]
            continue

        biomass[i + 1] = biomass[i] * np.exp(mu * dt)
        x_avg = (biomass[i + 1] - biomass[i]) / (mu * dt)

        for j, rid in enumerate(pool_ids):
            flux = float(solution.fluxes[rid])
            new_c = pool_hist[i, j] + flux * x_avg * dt
            pool_hist[i + 1, j] = max(new_c, 0.0)

    growth_rate[-1] = growth_rate[-2] if n_steps > 0 else 0.0

    pool_df = pd.DataFrame(pool_hist, columns=pool_ids)
    pool_df.insert(0, "time_h", time_h)

    return MonoDFBAResult(
        time_h=time_h,
        biomass=biomass,
        pool=pool_df,
        growth_rate=growth_rate,
    )

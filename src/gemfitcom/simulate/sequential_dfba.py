"""Sequential (independent-FBA) dynamic FBA for a microbial community.

Each strain is a separate COBRA model that shares one metabolite pool. At
every step of length ``dt``:

1. For every strain, set the lower bound of each pool exchange it has to
   ``-michaelis_menten(pool_conc, vmax, km)`` when MM parameters are
   supplied. Pool exchanges without MM params keep whatever bound the
   caller has already set (typically the placeholder from
   :func:`apply_medium`).
2. Run ``strain.model.optimize()`` independently for each strain; the
   objective value is the specific growth rate μ (1/h) of that strain and
   ``solution.fluxes[ex_id]`` is its per-gDW exchange flux.
3. Advance each strain's biomass analytically over the step:
   ``X_{t+dt} = X_t * exp(μ * dt)``. Per-strain mean biomass over the step
   is ``X_avg = (X_new - X_old) / (μ dt)`` (the exact integral of
   exponential growth at constant μ), so the pool update is mass-balanced
   against biomass growth: ``ΔC = Σ_strain flux_s * X_avg_s * dt``.
4. Pool concentrations are clipped to 0 at the bottom. Cross-feeding
   emerges naturally because a positive flux on an exchange (secretion)
   adds to the pool just as a negative flux (uptake) subtracts from it.
5. If a strain's FBA fails (non-optimal) or μ ≤ 0 over the step, that
   strain contributes nothing to the pool for the step and its biomass is
   held fixed.

This is the ``sequential_dfba`` mode of :mod:`gemfitcom.simulate`. It does
not enforce any cooperative objective across strains — for that, see the
``micom`` and ``fusion`` modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gemfitcom.kinetics.mm import MMParams, michaelis_menten
from gemfitcom.medium.medium import Medium
from gemfitcom.simulate._shared import flux_tensor_to_long
from gemfitcom.utils.progress import progress_bar

if TYPE_CHECKING:
    from cobra import Model


DEFAULT_DT: float = 0.25


@dataclass(frozen=True, slots=True)
class StrainSpec:
    """One strain in a sequential-dFBA community.

    Attributes:
        name: Unique strain identifier; used as column label in outputs.
        model: COBRA model. Bounds WILL be mutated during simulation; pass
            ``model.copy()`` if you need the original preserved.
        mm_params: Map ``exchange_id -> MMParams`` for substrates whose
            uptake bound should follow MM kinetics on the shared pool.
            Keys must be pool components of the community medium, but need
            not all be present in this strain's model (missing ones are
            silently skipped, which lets each strain declare the substrates
            it cares about).
        initial_biomass: Starting biomass in gDW / L. Must be > 0.
    """

    name: str
    model: Model
    mm_params: dict[str, MMParams]
    initial_biomass: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"StrainSpec.name must be a non-empty string, got {self.name!r}")
        if self.initial_biomass <= 0:
            raise ValueError(
                f"StrainSpec({self.name!r}).initial_biomass must be > 0, got {self.initial_biomass}"
            )


@dataclass(frozen=True, slots=True)
class SequentialDFBAResult:
    """Sequential-dFBA trajectory.

    All frames share the same time column so they align row-by-row.

    Attributes:
        time_h: Time points (hours), shape ``(n_points,)``.
        biomass: Per-strain biomass (gDW / L). Columns: ``time_h`` followed
            by one column per strain (in input order).
        pool: Shared pool concentrations (mM). Columns: ``time_h`` followed
            by one column per pool exchange (in ``medium.pool_components``
            iteration order).
        growth_rate: Per-strain specific growth rate (1/h). Columns:
            ``time_h`` followed by one column per strain. ``row 0`` is the
            rate evaluated at the initial state; the final row repeats the
            previous row.
        exchange_fluxes: Per-strain per-pool-exchange flux (mmol / gDW /
            h), in long form with columns ``[time_h, strain, exchange_id,
            flux]``. Positive = secretion, negative = uptake. Populated
            only when :func:`simulate_sequential_dfba` is called with
            ``save_fluxes=True``; otherwise ``None``. The final time
            point duplicates the previous step, matching ``growth_rate``.
    """

    time_h: np.ndarray
    biomass: pd.DataFrame
    pool: pd.DataFrame
    growth_rate: pd.DataFrame
    exchange_fluxes: pd.DataFrame | None = None


def simulate_sequential_dfba(
    strains: list[StrainSpec],
    medium: Medium,
    *,
    t_total: float,
    dt: float = DEFAULT_DT,
    pool_init: dict[str, float] | None = None,
    save_fluxes: bool = False,
    progress: bool = False,
) -> SequentialDFBAResult:
    """Integrate a multi-strain community via sequential dFBA.

    Args:
        strains: Ordered list of :class:`StrainSpec`. Must be non-empty with
            unique names. Each strain's ``model`` is mutated in place.
        medium: Defines the shared metabolite pool. Only exchanges listed
            in ``medium.pool_components`` are tracked dynamically;
            unlimited components keep whatever bound :func:`apply_medium`
            assigned.
        t_total: Total simulation time (hours). Must be > 0.
        dt: Step size (hours). Must be > 0 and ≤ ``t_total``.
        pool_init: Override of initial pool concentrations (mM); merged
            over ``medium.pool_components``. Every key must be a pool
            component of ``medium``.
        save_fluxes: If True, populate ``SequentialDFBAResult.exchange_fluxes``
            with a long-form per-step flux table. Off by default to save
            memory (shape is ``n_steps * n_strains * n_pool`` rows).
        progress: If True, show a tqdm progress bar over the time loop.
            Defaults to False so library calls stay silent; CLI and notebook
            callers should turn it on explicitly. Recommended for real
            runs with many strains or many steps — 8 AGORA2 strains at
            dt=0.25h over 72h is ~2300 FBA calls.

    Returns:
        A :class:`SequentialDFBAResult`.

    Raises:
        ValueError: on invalid numeric arguments or empty/duplicate strain
            lists.
        KeyError: if an ``mm_params`` or ``pool_init`` key is not a pool
            component of ``medium``.
    """
    if not strains:
        raise ValueError("strains must be a non-empty list")
    names = [s.name for s in strains]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"strain names must be unique; duplicates: {dupes}")

    if t_total <= 0:
        raise ValueError(f"t_total must be > 0, got {t_total}")
    if not 0 < dt <= t_total:
        raise ValueError(f"dt must satisfy 0 < dt <= t_total, got dt={dt}, t_total={t_total}")

    pool_ids = list(medium.pool_components)
    pool_id_set = set(pool_ids)
    pool_idx = {rid: j for j, rid in enumerate(pool_ids)}

    for s in strains:
        for k in s.mm_params:
            if k not in pool_id_set:
                raise KeyError(
                    f"strain {s.name!r}: mm_params key {k!r} is not a pool component "
                    f"of medium {medium.name!r}"
                )
    if pool_init is not None:
        for k in pool_init:
            if k not in pool_id_set:
                raise KeyError(
                    f"pool_init key {k!r} is not a pool component of medium {medium.name!r}"
                )

    concs: dict[str, float] = dict(medium.pool_components)
    if pool_init is not None:
        concs.update(pool_init)

    n_steps = round(t_total / dt)
    n_points = n_steps + 1
    time_h = np.linspace(0.0, n_steps * dt, n_points)

    n_strains = len(strains)
    biomass_hist = np.empty((n_points, n_strains), dtype=float)
    growth_hist = np.zeros((n_points, n_strains), dtype=float)
    pool_hist = np.empty((n_points, len(pool_ids)), dtype=float)
    flux_hist: np.ndarray | None = (
        np.zeros((n_points, n_strains, len(pool_ids)), dtype=float) if save_fluxes else None
    )

    for k, s in enumerate(strains):
        biomass_hist[0, k] = s.initial_biomass
    for j, rid in enumerate(pool_ids):
        pool_hist[0, j] = concs[rid]

    strain_pool_rxns: list[dict[str, object]] = []
    for s in strains:
        model_rxn_ids = {r.id for r in s.model.reactions}
        cache = {rid: s.model.reactions.get_by_id(rid) for rid in pool_ids if rid in model_rxn_ids}
        strain_pool_rxns.append(cache)

    step_iter = progress_bar(
        range(n_steps),
        enabled=progress,
        desc="sequential_dfba",
        total=n_steps,
        unit="step",
    )
    for i in step_iter:
        mu_step = np.zeros(n_strains, dtype=float)
        flux_step = np.zeros((n_strains, len(pool_ids)), dtype=float)
        solved_ok = np.zeros(n_strains, dtype=bool)

        for k, s in enumerate(strains):
            rxns = strain_pool_rxns[k]
            for rid, p in s.mm_params.items():
                rxn = rxns.get(rid)
                if rxn is None:
                    continue
                c = max(pool_hist[i, pool_idx[rid]], 0.0)
                rxn.lower_bound = -float(michaelis_menten(c, p.vmax, p.km))

            solution = s.model.optimize()
            if solution.status != "optimal":
                continue

            mu = float(solution.objective_value)
            mu_step[k] = mu
            solved_ok[k] = True
            for j, rid in enumerate(pool_ids):
                if rid in rxns:
                    flux_step[k, j] = float(solution.fluxes[rid])

        growth_hist[i] = mu_step
        if flux_hist is not None:
            flux_hist[i] = flux_step

        new_biomass = biomass_hist[i].copy()
        x_avg = np.zeros(n_strains, dtype=float)
        for k in range(n_strains):
            if solved_ok[k] and mu_step[k] > 0:
                new_biomass[k] = biomass_hist[i, k] * np.exp(mu_step[k] * dt)
                x_avg[k] = (new_biomass[k] - biomass_hist[i, k]) / (mu_step[k] * dt)
        biomass_hist[i + 1] = new_biomass

        delta = (flux_step * x_avg[:, None]).sum(axis=0) * dt
        pool_hist[i + 1] = np.maximum(pool_hist[i] + delta, 0.0)

    growth_hist[-1] = growth_hist[-2] if n_steps > 0 else growth_hist[-1]
    if flux_hist is not None and n_steps > 0:
        flux_hist[-1] = flux_hist[-2]

    biomass_df = pd.DataFrame(biomass_hist, columns=names)
    biomass_df.insert(0, "time_h", time_h)
    growth_df = pd.DataFrame(growth_hist, columns=names)
    growth_df.insert(0, "time_h", time_h)
    pool_df = pd.DataFrame(pool_hist, columns=pool_ids)
    pool_df.insert(0, "time_h", time_h)

    exchange_fluxes_df: pd.DataFrame | None = None
    if flux_hist is not None:
        exchange_fluxes_df = flux_tensor_to_long(flux_hist, time_h, names, pool_ids)

    return SequentialDFBAResult(
        time_h=time_h,
        biomass=biomass_df,
        pool=pool_df,
        growth_rate=growth_df,
        exchange_fluxes=exchange_fluxes_df,
    )

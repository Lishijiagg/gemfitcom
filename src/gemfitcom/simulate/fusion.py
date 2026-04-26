"""Dynamic MICOM (``fusion`` / dMICOM): cooperative-tradeoff QP per dFBA step.

The pipeline's core innovation: at each time step we run MICOM's cooperative
tradeoff — first maximize community growth, then redistribute by minimizing
``||μ||_2`` at a user-chosen fraction of that maximum — instead of doing
independent per-strain FBA (which the ``sequential_dfba`` mode does). The
effect is a time-resolved trajectory where every step respects
community-level cooperation.

Algorithm (per step of length ``dt``):

1. Update per-taxon external exchange bounds from the *current* pool. For
   each pool component ``j`` with MM parameters for member ``i``, set
   ``EX_<j>__<i>.lower_bound = -michaelis_menten(pool_j, vmax_ij, km_ij)``.
   Pool components without MM params use ``default_uptake``. Bounds are
   applied to the per-taxon exchange reactions, not the community-level
   medium setter, so each member can have its own Vmax/Km. (MICOM's
   ``set_abundance`` leaves these bounds untouched.)
2. Refresh relative abundances from current absolute biomass
   (``a_i = B_i / ΣB``) and call ``community.set_abundance(...)``. This
   rescales the cross-compartment stoichiometry (who dominates the
   community pool) without affecting per-taxon uptake caps.
3. Solve ``community.cooperative_tradeoff(fraction=..., fluxes=True,
   pfba=pfba)``. The per-taxon μ are read from ``solution.members`` and
   fluxes from ``solution.fluxes.loc[taxon, reaction_id]`` (per gDW of
   that taxon — verified empirically against single bounds).
4. Advance each member's biomass analytically:
   ``B_i^{t+dt} = B_i^t · exp(μ_i · dt)`` with
   ``X_avg_i = (B_i^{t+dt} − B_i^t) / (μ_i · dt)`` so the pool update is
   mass-balanced against exponential growth (same scheme as ``mono_dfba``
   and ``sequential_dfba``). Pool update:
   ``ΔC_j = Σ_i flux_ij · X_avg_i · dt``. Pools are clipped to 0.
5. Non-optimal QP steps: biomass and pool are held fixed for the step,
   ``fail_count`` is incremented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gemfitcom.kinetics.mm import MMParams, michaelis_menten
from gemfitcom.medium.medium import Medium
from gemfitcom.simulate._shared import flux_tensor_to_long
from gemfitcom.simulate.micom import (
    DEFAULT_UNLIMITED_UPTAKE,
    DEFAULT_UPTAKE,
    CommunityMember,
    _serialized_models,
    _silence_micom,
    _translate_to_community_ids,
)
from gemfitcom.utils.progress import progress_bar

DEFAULT_DT: float = 0.25
DEFAULT_FRACTION: float = 0.5


@dataclass(frozen=True, slots=True)
class FusionResult:
    """dMICOM trajectory.

    Attributes:
        time_h: Time points (hours), shape ``(n_points,)``.
        biomass: Per-member biomass (gDW / L). Columns: ``time_h`` then one
            column per member in input order.
        pool: Shared pool concentrations (mM). Columns: ``time_h`` then one
            column per pool exchange ID.
        growth_rate: Per-member μ (1/h). Columns: ``time_h`` then one
            column per member. Final row repeats the previous row.
        community_growth_rate: Community μ (1/h) per step, shape
            ``(n_points,)``. Final row repeats the previous row.
        fraction: Cooperative tradeoff fraction used.
        fail_count: Number of steps where the QP did not reach ``optimal``
            (state was held fixed for those steps).
        exchange_fluxes: Per-member per-pool-exchange flux (mmol / gDW /
            h) in long form with columns ``[time_h, strain, exchange_id,
            flux]``. Positive = secretion, negative = uptake. Populated
            only when :func:`simulate_fusion_dmicom` is called with
            ``save_fluxes=True``; otherwise ``None``. The final time
            point duplicates the previous step.
    """

    time_h: np.ndarray
    biomass: pd.DataFrame
    pool: pd.DataFrame
    growth_rate: pd.DataFrame
    community_growth_rate: np.ndarray
    fraction: float
    fail_count: int
    exchange_fluxes: pd.DataFrame | None = None


def simulate_fusion_dmicom(
    members: list[CommunityMember],
    medium: Medium,
    *,
    t_total: float,
    dt: float = DEFAULT_DT,
    fraction: float = DEFAULT_FRACTION,
    mm_params_by_member: dict[str, dict[str, MMParams]] | None = None,
    pool_init: dict[str, float] | None = None,
    default_uptake: float = DEFAULT_UPTAKE,
    unlimited_uptake: float = DEFAULT_UNLIMITED_UPTAKE,
    pfba: bool = True,
    solver: str | None = None,
    save_fluxes: bool = False,
    progress: bool = False,
) -> FusionResult:
    """Integrate a community via dynamic MICOM (cooperative tradeoff per step).

    Args:
        members: Non-empty list of :class:`CommunityMember` with unique
            names. ``member.abundance`` is interpreted as **initial biomass
            in gDW / L** here (fusion is dynamic, not steady-state).
        medium: Defines the shared metabolite pool and unlimited
            components.
        t_total: Total simulation time (hours). Must be > 0.
        dt: Step size (hours). Must be > 0 and ≤ ``t_total``.
        fraction: Cooperative tradeoff fraction in ``(0, 1]``. ``1.0`` is
            pure community max-growth each step; lower values spread
            growth across members.
        mm_params_by_member: ``{member_name: {exchange_id: MMParams}}``
            per-member MM kinetics. Different members can have different
            Vmax/Km for the same substrate. Omitted substrates use
            ``default_uptake``.
        pool_init: Override initial pool concentrations (mM). Merged over
            ``medium.pool_components``.
        default_uptake: Per-taxon lower-bound magnitude for pool exchanges
            without MM params. Must be > 0.
        unlimited_uptake: Cap for ``medium.unlimited_components`` at the
            community boundary. Must be > 0.
        pfba: If True, MICOM resolves fluxes by parsimonious FBA.
        solver: Override optlang solver. If None, MICOM picks.
        save_fluxes: If True, populate :attr:`FusionResult.exchange_fluxes`
            with a long-form per-step per-member flux table. Off by
            default to save memory.
        progress: If True, show a tqdm progress bar over the time loop.

    Returns:
        A :class:`FusionResult`.

    Raises:
        ValueError: on invalid numeric arguments or empty/duplicate member
            lists.
        KeyError: if an ``mm_params_by_member`` / ``pool_init`` key is not
            a pool component of ``medium``, or if ``mm_params_by_member``
            names a member not in ``members``.
    """
    if not members:
        raise ValueError("members must be a non-empty list")
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"member names must be unique; duplicates: {dupes}")
    if t_total <= 0:
        raise ValueError(f"t_total must be > 0, got {t_total}")
    if not 0 < dt <= t_total:
        raise ValueError(f"dt must satisfy 0 < dt <= t_total, got dt={dt}, t_total={t_total}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must satisfy 0 < fraction <= 1, got {fraction}")
    if default_uptake <= 0:
        raise ValueError(f"default_uptake must be > 0, got {default_uptake}")
    if unlimited_uptake <= 0:
        raise ValueError(f"unlimited_uptake must be > 0, got {unlimited_uptake}")

    pool_id_set = set(medium.pool_components)
    name_set = set(names)
    if mm_params_by_member is not None:
        unknown = sorted(set(mm_params_by_member) - name_set)
        if unknown:
            raise KeyError(f"mm_params_by_member names not in members: {unknown}")
        for mname, mm in mm_params_by_member.items():
            for rid in mm:
                if rid not in pool_id_set:
                    raise KeyError(
                        f"mm_params_by_member[{mname!r}] key {rid!r} is not a pool "
                        f"component of medium {medium.name!r}"
                    )
    if pool_init is not None:
        for k in pool_init:
            if k not in pool_id_set:
                raise KeyError(
                    f"pool_init key {k!r} is not a pool component of medium {medium.name!r}"
                )

    pool_ids = list(medium.pool_components)
    pool_idx = {rid: j for j, rid in enumerate(pool_ids)}

    concs: dict[str, float] = dict(medium.pool_components)
    if pool_init is not None:
        concs.update(pool_init)

    n_steps = round(t_total / dt)
    n_points = n_steps + 1
    time_h = np.linspace(0.0, n_steps * dt, n_points)
    n_members = len(members)

    biomass_hist = np.empty((n_points, n_members), dtype=float)
    growth_hist = np.zeros((n_points, n_members), dtype=float)
    community_mu_hist = np.zeros(n_points, dtype=float)
    pool_hist = np.empty((n_points, len(pool_ids)), dtype=float)
    flux_hist: np.ndarray | None = (
        np.zeros((n_points, n_members, len(pool_ids)), dtype=float) if save_fluxes else None
    )

    for k, m in enumerate(members):
        biomass_hist[0, k] = float(m.abundance)
    for j, rid in enumerate(pool_ids):
        pool_hist[0, j] = concs[rid]

    fail_count = 0

    with _serialized_models(members) as taxonomy:
        from micom import Community

        with _silence_micom():
            community = Community(taxonomy, progress=False, solver=solver)

            # Open the community medium for all pool + unlimited exchanges at a
            # permissive cap — the real constraint on pool uptake is enforced by
            # per-taxon exchange bounds (set each step below), not the medium.
            medium_caps: dict[str, float] = {rid: float(unlimited_uptake) for rid in pool_ids}
            for rid in medium.unlimited_components:
                medium_caps[rid] = float(unlimited_uptake)
            community.medium = _translate_to_community_ids(medium_caps, community)

            # Pre-cache per-taxon exchange reaction objects. Missing ones
            # (the taxon's model doesn't have that pool exchange) become None.
            taxon_pool_rxns: list[dict[str, object]] = []
            for m in members:
                cache: dict[str, object] = {}
                for rid in pool_ids:
                    taxon_rxn_id = f"{rid}__{m.name}"
                    try:
                        cache[rid] = community.reactions.get_by_id(taxon_rxn_id)
                    except KeyError:
                        cache[rid] = None
                taxon_pool_rxns.append(cache)

            step_iter = progress_bar(
                range(n_steps),
                enabled=progress,
                desc="fusion_dmicom",
                total=n_steps,
                unit="step",
            )
            for i in step_iter:
                # 1. Per-taxon exchange bounds from current pool + per-member MM.
                for k, m in enumerate(members):
                    mm = (
                        mm_params_by_member.get(m.name, {})
                        if mm_params_by_member is not None
                        else {}
                    )
                    for rid in pool_ids:
                        rxn = taxon_pool_rxns[k][rid]
                        if rxn is None:
                            continue
                        c = max(pool_hist[i, pool_idx[rid]], 0.0)
                        if rid in mm:
                            p = mm[rid]
                            bound = -float(michaelis_menten(c, p.vmax, p.km))
                        else:
                            bound = -float(default_uptake)
                        rxn.lower_bound = bound

                # 2. Update abundances from current biomass ratios.
                total_biomass = float(biomass_hist[i].sum())
                if total_biomass > 0:
                    community.set_abundance(biomass_hist[i] / total_biomass)

                # 3. Solve cooperative tradeoff.
                try:
                    solution = community.cooperative_tradeoff(
                        fraction=fraction,
                        fluxes=True,
                        pfba=pfba,
                    )
                except Exception:
                    solution = None

                if solution is None or solution.status != "optimal":
                    biomass_hist[i + 1] = biomass_hist[i]
                    pool_hist[i + 1] = pool_hist[i]
                    fail_count += 1
                    continue

                # 4. Per-member growth rates.
                mu_step = np.array(
                    [float(solution.members.loc[name, "growth_rate"]) for name in names],
                    dtype=float,
                )
                growth_hist[i] = mu_step
                community_mu_hist[i] = float(solution.growth_rate)

                # 5. Biomass update + x_avg.
                new_biomass = biomass_hist[i].copy()
                x_avg = np.zeros(n_members, dtype=float)
                for k in range(n_members):
                    if mu_step[k] > 0:
                        new_biomass[k] = biomass_hist[i, k] * np.exp(mu_step[k] * dt)
                        x_avg[k] = (new_biomass[k] - biomass_hist[i, k]) / (mu_step[k] * dt)
                biomass_hist[i + 1] = new_biomass

                # 6. Pool update from per-taxon fluxes (per gDW of that taxon).
                fluxes = solution.fluxes
                delta = np.zeros(len(pool_ids), dtype=float)
                for pj, rid in enumerate(pool_ids):
                    if rid not in fluxes.columns:
                        continue
                    col = fluxes[rid]
                    for k, name in enumerate(names):
                        v = col.get(name, np.nan)
                        if pd.isna(v):
                            continue
                        v_f = float(v)
                        delta[pj] += v_f * x_avg[k]
                        if flux_hist is not None:
                            flux_hist[i, k, pj] = v_f
                pool_hist[i + 1] = np.maximum(pool_hist[i] + delta * dt, 0.0)

    if n_steps > 0:
        growth_hist[-1] = growth_hist[-2]
        community_mu_hist[-1] = community_mu_hist[-2]
        if flux_hist is not None:
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

    return FusionResult(
        time_h=time_h,
        biomass=biomass_df,
        pool=pool_df,
        growth_rate=growth_df,
        community_growth_rate=community_mu_hist,
        fraction=float(fraction),
        fail_count=fail_count,
        exchange_fluxes=exchange_fluxes_df,
    )


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_FRACTION",
    "FusionResult",
    "simulate_fusion_dmicom",
]

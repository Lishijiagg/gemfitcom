"""Per-cell dFBA: apply MM bounds, optimise, extract growth + flux.

Sign convention:
    - cobra exchange flux > 0 ⇒ secretion (medium gains)
    - cobra exchange flux < 0 ⇒ uptake (medium loses)
    The sign is passed straight through into ``flux``; the caller scales by
    biomass and dt to get a mmol/L delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinetics import ExchangeKinetics


@dataclass(frozen=True, slots=True)
class SingleCellResult:
    """Result of one species × one grid cell FBA solve."""

    mu: float
    flux: np.ndarray
    infeasible: bool


def build_exchange_index(
    model,
    kinetics: ExchangeKinetics,
    metabolite_ids: tuple[str, ...],
) -> dict[int, int]:
    """Cache met_idx → kinetics entry idx for the hot loop.

    Validates that every kinetics entry references a real exchange in the
    model (raises KeyError otherwise — fail-fast at build time, not in the
    optimisation hot path).

    Args:
        model: cobra Model.
        kinetics: this species's ExchangeKinetics.
        metabolite_ids: ordered metabolite ids tracked on the grid. Exchange
            reactions are looked up as ``f"EX_{met_id}"``.

    Returns:
        Mapping from position in ``metabolite_ids`` to position in
        ``kinetics.entries``. Metabolites without a kinetics entry are
        omitted from the result.
    """
    model_rxn_ids = {r.id for r in model.reactions}
    kin_idx_by_exch = {e.exchange_id: k for k, e in enumerate(kinetics.entries)}

    for exch_id in kin_idx_by_exch:
        if exch_id not in model_rxn_ids:
            raise KeyError(
                f"Kinetics for {kinetics.species!r} references {exch_id} "
                f"which is not in the model"
            )

    out: dict[int, int] = {}
    for met_idx, met_id in enumerate(metabolite_ids):
        exch_id = f"EX_{met_id}"
        if exch_id in kin_idx_by_exch:
            out[met_idx] = kin_idx_by_exch[exch_id]
    return out


def solve_cell(
    *,
    model,
    kinetics: ExchangeKinetics,
    exchange_index: dict[int, int],
    C_local: np.ndarray,
) -> SingleCellResult:
    """Solve one species × one grid cell FBA at local concentrations.

    Mutates ``model.reactions[exchange_id].lower_bound`` (and ``upper_bound``
    for bidirectional). Caller owns model isolation if needed.

    Returns mu=0, flux=zeros, infeasible=True on solver failure — never
    raises from the optimisation itself.
    """
    n_metabolites = C_local.shape[0]
    flux = np.zeros(n_metabolites, dtype=float)

    # Build the kinetics-ordered concentration vector for MM evaluation.
    n_exch = kinetics.n_exchanges
    C_for_kinetics = np.zeros(n_exch, dtype=float)
    for met_idx, kin_idx in exchange_index.items():
        C_for_kinetics[kin_idx] = C_local[met_idx]

    mm_bounds = kinetics.mm_upper_bound(C_for_kinetics)

    # Write into cobra exchange bounds.
    for k, entry in enumerate(kinetics.entries):
        rxn = model.reactions.get_by_id(entry.exchange_id)
        rxn.lower_bound = -mm_bounds[k]
        if entry.mode == "bidirectional":
            rxn.upper_bound = mm_bounds[k]

    try:
        sol = model.optimize()
        infeasible = sol.status != "optimal"
    except Exception:
        infeasible = True
        sol = None

    if infeasible or sol is None:
        return SingleCellResult(mu=0.0, flux=flux, infeasible=True)

    mu = float(sol.objective_value) if sol.objective_value is not None else 0.0
    for met_idx, kin_idx in exchange_index.items():
        exch_id = kinetics.entries[kin_idx].exchange_id
        flux[met_idx] = float(sol.fluxes[exch_id])

    return SingleCellResult(mu=mu, flux=flux, infeasible=False)

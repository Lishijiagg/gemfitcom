"""Atomically apply a KB entry to a :class:`cobra.Model`.

Adds missing metabolites and reactions (by ID). If post-apply
verification fails — the model still cannot secrete the entry's target
exchange — every object just added is removed so the model is left in
its pre-call state, and :class:`ApplyError` is raised.

Existing metabolites and reactions (matched by ID) are never
overwritten. This keeps the gap-fill conservative: a dangling or
misconfigured reaction already in the model remains the caller's
responsibility to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cobra import Metabolite, Reaction

from gemfitcom.gapfill.detect import DEFAULT_TOL, can_secrete
from gemfitcom.gapfill.knowledge import GapfillKBEntry

if TYPE_CHECKING:
    from cobra import Model


class ApplyError(RuntimeError):
    """Raised when applying a KB entry does not enable secretion of its target."""


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of :func:`apply_entry`.

    Attributes:
        entry_exchange_id: The target exchange ID (``entry.exchange_id``).
        added_metabolites: Metabolite IDs newly inserted into the model.
        skipped_metabolites: Metabolite IDs already present (left
            untouched).
        added_reactions: Reaction IDs newly inserted.
        skipped_reactions: Reaction IDs already present.
        verified: True iff, after apply, :func:`can_secrete` returned
            True for ``entry_exchange_id`` (or verification was
            disabled).
    """

    entry_exchange_id: str
    added_metabolites: tuple[str, ...]
    skipped_metabolites: tuple[str, ...]
    added_reactions: tuple[str, ...]
    skipped_reactions: tuple[str, ...]
    verified: bool


def apply_entry(
    model: Model,
    entry: GapfillKBEntry,
    *,
    verify: bool = True,
    tol: float = DEFAULT_TOL,
) -> ApplyResult:
    """Apply a :class:`GapfillKBEntry` to ``model`` in place.

    Args:
        model: A cobrapy model (mutated on success, left unchanged on
            failure).
        entry: The KB entry to apply.
        verify: If True, run :func:`can_secrete` against the entry's
            target exchange after insertion and roll back + raise on
            failure.
        tol: Secretion flux tolerance forwarded to :func:`can_secrete`.

    Returns:
        An :class:`ApplyResult` describing what was added or skipped.

    Raises:
        ApplyError: when ``verify=True`` and the model still cannot
            secrete ``entry.exchange_id`` after insertion.
    """
    existing_met_ids = {m.id for m in model.metabolites}
    new_mets: list[Metabolite] = []
    added_met_ids: list[str] = []
    skipped_met_ids: list[str] = []
    for spec in entry.metabolites:
        if spec.id in existing_met_ids:
            skipped_met_ids.append(spec.id)
            continue
        met = Metabolite(
            id=spec.id,
            name=spec.name or spec.id,
            compartment=spec.compartment,
            formula=spec.formula,
            charge=spec.charge,
        )
        new_mets.append(met)
        added_met_ids.append(spec.id)

    existing_rxn_ids = {r.id for r in model.reactions}
    new_rxns: list[Reaction] = []
    added_rxn_ids: list[str] = []
    skipped_rxn_ids: list[str] = []

    try:
        if new_mets:
            model.add_metabolites(new_mets)

        for spec in entry.reactions:
            if spec.id in existing_rxn_ids:
                skipped_rxn_ids.append(spec.id)
                continue
            lb, ub = spec.bounds
            rxn = Reaction(
                id=spec.id,
                name=spec.name or spec.id,
                lower_bound=lb,
                upper_bound=ub,
            )
            stoich = {
                model.metabolites.get_by_id(met_id): coef
                for met_id, coef in spec.stoichiometry.items()
            }
            rxn.add_metabolites(stoich)
            new_rxns.append(rxn)
            added_rxn_ids.append(spec.id)

        if new_rxns:
            model.add_reactions(new_rxns)

        verified = True
        if verify:
            verified = can_secrete(model, entry.exchange_id, tol=tol)
            if not verified:
                _rollback(model, new_mets, new_rxns)
                raise ApplyError(
                    f"apply_entry({entry.exchange_id!r}) did not enable secretion "
                    f"(FBA objective <= tol={tol}); rolled back."
                )
    except ApplyError:
        raise
    except Exception:
        _rollback(model, new_mets, new_rxns)
        raise

    return ApplyResult(
        entry_exchange_id=entry.exchange_id,
        added_metabolites=tuple(added_met_ids),
        skipped_metabolites=tuple(skipped_met_ids),
        added_reactions=tuple(added_rxn_ids),
        skipped_reactions=tuple(skipped_rxn_ids),
        verified=verified,
    )


def _rollback(model: Model, mets: list[Metabolite], rxns: list[Reaction]) -> None:
    """Remove reactions and metabolites that were freshly added; no-op if absent."""
    if rxns:
        present = [r for r in rxns if r in model.reactions]
        if present:
            model.remove_reactions(present)
    if mets:
        present = [m for m in mets if m in model.metabolites]
        if present:
            model.remove_metabolites(present)


__all__ = [
    "ApplyError",
    "ApplyResult",
    "apply_entry",
]

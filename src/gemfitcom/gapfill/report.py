"""Structured report returned by :func:`gemfitcom.gapfill.run_gapfill`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ProductStatus = Literal["added", "already_present", "no_kb_entry", "failed"]


@dataclass(frozen=True, slots=True)
class ProductOutcome:
    """Per-product outcome of a gap-fill run.

    Attributes:
        exchange_id: The HPLC-observed product's exchange reaction ID.
        status: One of
            ``"added"``            — KB entry applied, verification passed;
            ``"already_present"``  — model could already secrete the product;
            ``"no_kb_entry"``      — KB had no recipe for this exchange;
            ``"failed"``           — apply attempted but did not enable
            secretion (only possible with ``strict=False``; on strict mode
            such a case raises instead of being recorded).
        added_metabolites: Metabolite IDs newly inserted (empty unless
            ``status == "added"``).
        added_reactions: Reaction IDs newly inserted (empty unless
            ``status == "added"``).
        skipped_metabolites: Metabolite IDs already present in the model.
        skipped_reactions: Reaction IDs already present in the model.
        message: Free-form detail; populated for ``no_kb_entry`` and
            ``failed``.
    """

    exchange_id: str
    status: ProductStatus
    added_metabolites: tuple[str, ...] = ()
    added_reactions: tuple[str, ...] = ()
    skipped_metabolites: tuple[str, ...] = ()
    skipped_reactions: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class GapfillReport:
    """Audit record of a :func:`run_gapfill` call.

    Attributes:
        strain_id: ``model.id`` at call time.
        source: Model provenance (``curated`` / ``agora2`` / ``carveme``).
        kb_name: Name of the KB consulted, or ``""`` when skipped.
        skipped: True when the dispatch decided not to run gap-fill
            (currently: ``source == "curated"``).
        outcomes: Per-product results, in the order observed products
            were processed.
        warnings: Free-form warnings (e.g. unknown products, fallbacks).
    """

    strain_id: str
    source: str
    kb_name: str
    skipped: bool
    outcomes: tuple[ProductOutcome, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def added_reaction_ids(self) -> tuple[str, ...]:
        """Flat tuple of every reaction ID added across all outcomes."""
        return tuple(rid for o in self.outcomes for rid in o.added_reactions)

    @property
    def added_metabolite_ids(self) -> tuple[str, ...]:
        """Flat tuple of every metabolite ID added across all outcomes."""
        return tuple(mid for o in self.outcomes for mid in o.added_metabolites)

    @property
    def products_added(self) -> tuple[str, ...]:
        return tuple(o.exchange_id for o in self.outcomes if o.status == "added")

    @property
    def products_already_present(self) -> tuple[str, ...]:
        return tuple(o.exchange_id for o in self.outcomes if o.status == "already_present")

    @property
    def products_missing_kb(self) -> tuple[str, ...]:
        return tuple(o.exchange_id for o in self.outcomes if o.status == "no_kb_entry")

    @property
    def products_failed(self) -> tuple[str, ...]:
        return tuple(o.exchange_id for o in self.outcomes if o.status == "failed")


__all__ = [
    "GapfillReport",
    "ProductOutcome",
    "ProductStatus",
]

"""Top-level gap-fill orchestration.

Dispatches on ``model.annotation['source']`` (or an override passed at
the call site):

* ``curated``  — skip gap-fill entirely; the model is returned as-is
  and the report carries ``skipped=True``.
* ``agora2`` / ``carveme`` — the (a) + (c) hybrid strategy:
    (a) detect which observed products the model cannot secrete,
    (c) consult the KB for pre-curated "transport + exchange" recipes
        and apply them.

The generic cobrapy :func:`gapfill` algorithm is intentionally NOT used
(see project decision log): auto gap-filling on an unreviewed universal
database tends to introduce physiologically implausible shortcuts.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from gemfitcom.gapfill.apply import ApplyError, apply_entry
from gemfitcom.gapfill.detect import DEFAULT_TOL, can_secrete
from gemfitcom.gapfill.knowledge import GapfillKB, load_kb
from gemfitcom.gapfill.report import GapfillReport, ProductOutcome

if TYPE_CHECKING:
    from cobra import Model

_VALID_SOURCES: tuple[str, ...] = ("curated", "agora2", "carveme")
_DEFAULT_KB_NAME: str = "scfa"


def run_gapfill(
    model: Model,
    observed_products: set[str] | list[str] | tuple[str, ...],
    kb: GapfillKB | str | None = None,
    *,
    source: str | None = None,
    strict: bool = True,
    tol: float = DEFAULT_TOL,
) -> GapfillReport:
    """Gap-fill a model's secretion capacity for observed HPLC products.

    Args:
        model: A cobrapy model. Mutated in place when
            ``source in {"agora2", "carveme"}``; left untouched when
            ``source == "curated"`` or when every product is already
            secretable. On failure with ``strict=True`` the model is
            rolled back to its pre-call state.
        observed_products: Exchange reaction IDs observed in HPLC data
            (e.g. ``{"EX_but_e", "EX_ac_e"}``). Thresholding — deciding
            whether a peak is "detected" — is the caller's
            responsibility.
        kb: Knowledge base to consult. Accepts
            * ``None`` — load the built-in ``"scfa"`` KB,
            * a string — resolved via :func:`load_kb`,
            * a :class:`GapfillKB` — used directly.
        source: Model provenance. If ``None``, read from
            ``model.annotation["source"]``. Must be one of
            ``curated`` / ``agora2`` / ``carveme``.
        strict: If True, an apply that fails verification raises
            :class:`ApplyError` and aborts the run. If False, the
            failure is recorded as a ``"failed"`` outcome and the loop
            continues.
        tol: Secretion flux tolerance passed to :func:`can_secrete` /
            :func:`apply_entry`.

    Returns:
        A :class:`GapfillReport`.

    Raises:
        ValueError: if ``source`` is missing / invalid.
        ApplyError: with ``strict=True`` when an entry's apply cannot
            enable secretion.
    """
    resolved_source = _resolve_source(model, source)
    strain_id = str(getattr(model, "id", "") or "")

    if resolved_source == "curated":
        return GapfillReport(
            strain_id=strain_id,
            source=resolved_source,
            kb_name="",
            skipped=True,
            outcomes=(),
            warnings=(),
        )

    kb_obj = _resolve_kb(kb)
    products = sorted(set(observed_products))

    outcomes: list[ProductOutcome] = []
    warn_msgs: list[str] = []

    for eid in products:
        entry = kb_obj.get(eid)
        if entry is None:
            msg = (
                f"{eid!r} observed but KB {kb_obj.name!r} has no recipe; "
                "skipped (consider extending the KB)."
            )
            warnings.warn(msg, stacklevel=2)
            warn_msgs.append(msg)
            outcomes.append(ProductOutcome(exchange_id=eid, status="no_kb_entry", message=msg))
            continue

        if can_secrete(model, eid, tol=tol):
            outcomes.append(ProductOutcome(exchange_id=eid, status="already_present"))
            continue

        try:
            result = apply_entry(model, entry, verify=True, tol=tol)
        except ApplyError as exc:
            if strict:
                raise
            msg = str(exc)
            warnings.warn(f"gap-fill for {eid!r} failed: {msg}", stacklevel=2)
            warn_msgs.append(msg)
            outcomes.append(ProductOutcome(exchange_id=eid, status="failed", message=msg))
            continue

        outcomes.append(
            ProductOutcome(
                exchange_id=eid,
                status="added",
                added_metabolites=result.added_metabolites,
                added_reactions=result.added_reactions,
                skipped_metabolites=result.skipped_metabolites,
                skipped_reactions=result.skipped_reactions,
            )
        )

    return GapfillReport(
        strain_id=strain_id,
        source=resolved_source,
        kb_name=kb_obj.name,
        skipped=False,
        outcomes=tuple(outcomes),
        warnings=tuple(warn_msgs),
    )


def _resolve_source(model: Model, source: str | None) -> str:
    if source is None:
        annot = getattr(model, "annotation", None) or {}
        source = annot.get("source") if isinstance(annot, dict) else None
    if source is None:
        raise ValueError(
            "model source is unknown; pass source='curated'/'agora2'/'carveme' "
            "explicitly or set model.annotation['source'] "
            "(e.g. via io.models.load_model(..., source=...))"
        )
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
    return source


def _resolve_kb(kb: GapfillKB | str | None) -> GapfillKB:
    if kb is None:
        return load_kb(_DEFAULT_KB_NAME)
    if isinstance(kb, GapfillKB):
        return kb
    return load_kb(kb)


__all__ = ["run_gapfill"]

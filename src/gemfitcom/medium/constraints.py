"""Apply a :class:`Medium` to a COBRA model's exchange reactions.

This only sets the STRUCTURE of what can be taken up: pool components get a
placeholder uptake bound (the kinetics module overwrites it later with the
fitted Vmax/Km-derived value), and unlimited trace components get a large
negative bound. Secretion upper bounds are left untouched.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from gemfitcom.medium.medium import Medium

if TYPE_CHECKING:
    from cobra import Model

OnMissing = Literal["warn", "error", "ignore"]

DEFAULT_POOL_BOUND: float = -10.0
DEFAULT_UNLIMITED_BOUND: float = -1000.0


@dataclass(frozen=True, slots=True)
class MediumApplicationReport:
    """Summary of :func:`apply_medium` on a specific model.

    Attributes:
        medium_name: Name of the applied medium.
        applied_pool: Exchange IDs whose lb was set from ``pool_components``.
        applied_unlimited: Exchange IDs whose lb was set to the unlimited bound.
        missing_pool: ``pool_components`` IDs not present in the model.
        missing_unlimited: ``unlimited_components`` IDs not present in the model.
        closed: Exchange IDs whose uptake was closed (lb set to 0) because
            ``close_others=True`` and they are not listed in the medium.
    """

    medium_name: str
    applied_pool: tuple[str, ...]
    applied_unlimited: tuple[str, ...]
    missing_pool: tuple[str, ...]
    missing_unlimited: tuple[str, ...]
    closed: tuple[str, ...] = field(default_factory=tuple)


def apply_medium(
    model: Model,
    medium: Medium,
    *,
    close_others: bool = True,
    default_pool_bound: float = DEFAULT_POOL_BOUND,
    unlimited_bound: float = DEFAULT_UNLIMITED_BOUND,
    on_missing: OnMissing = "warn",
) -> MediumApplicationReport:
    """Apply a :class:`Medium` to ``model`` in place.

    Args:
        model: A cobrapy :class:`cobra.Model`.
        medium: The medium definition.
        close_others: If True, every exchange reaction NOT mentioned in
            ``medium`` has its lower bound set to 0 (uptake blocked; secretion
            left unchanged). If False, unlisted exchanges are left at whatever
            bounds they already carried.
        default_pool_bound: Lower bound assigned to pool components. Negative
            values permit uptake. This is a placeholder; the kinetics module
            overrides it per substrate after fitting Vmax/Km.
        unlimited_bound: Lower bound assigned to unlimited components.
        on_missing: How to handle medium exchange IDs that are not in the
            model's reaction list — ``"warn"`` (default) emits a warning,
            ``"error"`` raises :class:`KeyError`, ``"ignore"`` is silent.

    Returns:
        A :class:`MediumApplicationReport` describing what was applied.

    Raises:
        ValueError: on invalid numeric arguments.
        KeyError: if ``on_missing="error"`` and any medium exchange is absent.
    """
    if default_pool_bound > 0:
        raise ValueError(
            f"default_pool_bound must be <= 0 (negative = uptake), got {default_pool_bound}"
        )
    if unlimited_bound > 0:
        raise ValueError(f"unlimited_bound must be <= 0 (negative = uptake), got {unlimited_bound}")
    if on_missing not in ("warn", "error", "ignore"):
        raise ValueError(f"on_missing must be one of 'warn', 'error', 'ignore'; got {on_missing!r}")

    model_rxn_ids = {r.id for r in model.reactions}

    applied_pool: list[str] = []
    missing_pool: list[str] = []
    for rxn_id in medium.pool_components:
        if rxn_id in model_rxn_ids:
            model.reactions.get_by_id(rxn_id).lower_bound = default_pool_bound
            applied_pool.append(rxn_id)
        else:
            missing_pool.append(rxn_id)

    applied_unlimited: list[str] = []
    missing_unlimited: list[str] = []
    for rxn_id in sorted(medium.unlimited_components):
        if rxn_id in model_rxn_ids:
            model.reactions.get_by_id(rxn_id).lower_bound = unlimited_bound
            applied_unlimited.append(rxn_id)
        else:
            missing_unlimited.append(rxn_id)

    missing = [*missing_pool, *missing_unlimited]
    if missing:
        msg = (
            f"Medium {medium.name!r} references exchanges not present in model "
            f"{getattr(model, 'id', '<unnamed>')!r}: {sorted(missing)}"
        )
        if on_missing == "error":
            raise KeyError(msg)
        if on_missing == "warn":
            warnings.warn(msg, stacklevel=2)

    closed: list[str] = []
    if close_others:
        medium_ids = medium.exchange_ids
        for rxn in model.exchanges:
            if rxn.id not in medium_ids and rxn.lower_bound < 0:
                rxn.lower_bound = 0.0
                closed.append(rxn.id)

    return MediumApplicationReport(
        medium_name=medium.name,
        applied_pool=tuple(applied_pool),
        applied_unlimited=tuple(applied_unlimited),
        missing_pool=tuple(missing_pool),
        missing_unlimited=tuple(missing_unlimited),
        closed=tuple(closed),
    )

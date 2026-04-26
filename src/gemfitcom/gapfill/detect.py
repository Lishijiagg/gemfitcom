"""Detect whether a COBRA model can secrete a given product.

"Can secrete" is decided structurally AND numerically:

1. The exchange reaction ``exchange_id`` exists in the model.
2. Its ``upper_bound`` is strictly positive — positive flux on an
   exchange denotes secretion by cobrapy convention.
3. An FBA that maximizes this exchange as the sole objective terminates
   with status ``optimal`` and an objective value above ``tol``.

The third criterion filters out "structurally open but biologically
unreachable" exchanges — for example a dangling exchange whose
extracellular metabolite has no producing reaction.

The caller's model objective and reaction bounds are restored on return
(cobrapy context manager).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobra import Model

DEFAULT_TOL: float = 1e-6


def can_secrete(
    model: Model,
    exchange_id: str,
    *,
    tol: float = DEFAULT_TOL,
) -> bool:
    """Return True if the model can carry a positive flux through ``exchange_id``.

    Args:
        model: A cobrapy model (state is restored on return).
        exchange_id: ID of the target exchange reaction.
        tol: Minimum secretion flux to count as "can secrete".

    Returns:
        True iff the exchange reaction exists, has positive upper bound,
        and an FBA finds an optimum strictly greater than ``tol``.
    """
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    try:
        rxn = model.reactions.get_by_id(exchange_id)
    except KeyError:
        return False
    if rxn.upper_bound <= 0:
        return False

    with model:
        model.objective = rxn
        model.objective_direction = "max"
        try:
            value = model.slim_optimize(error_value=math.nan)
        except Exception:
            return False

    if value is None or math.isnan(value):
        return False
    return value > tol


def missing_products(
    model: Model,
    observed: set[str] | list[str] | tuple[str, ...],
    *,
    tol: float = DEFAULT_TOL,
) -> set[str]:
    """Return the subset of ``observed`` exchange IDs the model cannot secrete.

    Args:
        model: A cobrapy model.
        observed: Exchange IDs observed in HPLC (e.g. ``{"EX_but_e"}``).
        tol: See :func:`can_secrete`.

    Returns:
        The subset of ``observed`` for which :func:`can_secrete` is
        False.
    """
    return {eid for eid in observed if not can_secrete(model, eid, tol=tol)}


__all__ = [
    "DEFAULT_TOL",
    "can_secrete",
    "missing_products",
]

"""LP solver auto-detection for cobra / MICOM.

Resolution order: CPLEX -> Gurobi -> GLPK (the default free solver bundled with
cobra). ``get_best_solver`` returns the first solver whose Python bindings are
importable in the current environment.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)

SOLVER_PREFERENCE: tuple[str, ...] = ("cplex", "gurobi", "glpk")


def available_solvers() -> list[str]:
    """Return the list of LP solvers cobra can currently use.

    Result is taken from ``cobra.util.solver.solvers``, which only contains
    solvers whose Python interface imported successfully.
    """
    from cobra.util.solver import solvers

    return list(solvers.keys())


def get_best_solver(preference: Iterable[str] = SOLVER_PREFERENCE) -> str:
    """Pick the first solver from ``preference`` that is available.

    Falls back to an arbitrary available solver if none of the preferred ones
    are installed. Raises ``RuntimeError`` if no LP solver is available at all.
    """
    avail = available_solvers()
    if not avail:
        raise RuntimeError("No LP solver available to cobra. Install one of: cplex, gurobi, glpk.")
    for name in preference:
        if name in avail:
            logger.info("Using LP solver: %s", name)
            return name
    fallback = avail[0]
    logger.warning(
        "None of the preferred solvers %s are available; falling back to %r.",
        tuple(preference),
        fallback,
    )
    return fallback

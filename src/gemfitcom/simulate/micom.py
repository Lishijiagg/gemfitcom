"""MICOM cooperative-tradeoff steady-state community FBA.

Thin wrapper over :class:`micom.Community`. The community-level QP in MICOM
first maximizes total community growth, then for a user-chosen fraction of
that optimum redistributes growth across taxa by minimizing ``||μ||_2``
(cooperative tradeoff). The result is a single steady-state snapshot — no
dynamics, no pool evolution — so MM kinetics here simply map a *fixed*
pool concentration into an uptake cap for the snapshot.

Usage pattern:

1. Build :class:`CommunityMember` entries with a COBRA model, a relative
   abundance, and a strain name.
2. Call :func:`simulate_micom` with the shared :class:`Medium` and either:
   - explicit ``uptake`` (reaction_id → mmol/gDW/h positive flux bound), or
   - ``mm_params`` + ``pool_init`` to derive bounds via
     ``michaelis_menten(conc, vmax, km)``.
   Pool components with neither specification fall back to
   ``default_uptake``. Unlimited components use ``unlimited_uptake``.
3. Inspect the returned :class:`MICOMResult` — community growth rate,
   per-member growth rates, and the MICOM fluxes DataFrame (rows = taxa +
   ``medium``, columns = reaction IDs).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from gemfitcom.kinetics.mm import MMParams, michaelis_menten
from gemfitcom.medium.medium import Medium

if TYPE_CHECKING:
    from cobra import Model
    from micom.solution import CommunitySolution

DEFAULT_FRACTION: float = 0.5
DEFAULT_UPTAKE: float = 10.0
DEFAULT_UNLIMITED_UPTAKE: float = 1000.0


@dataclass(frozen=True, slots=True)
class CommunityMember:
    """One taxon in a MICOM community.

    Attributes:
        name: Unique member identifier used as the MICOM taxon id.
        model: COBRA model. Serialized to disk during the call (the
            original object is not mutated).
        abundance: Relative abundance. Must be > 0; MICOM normalizes the
            vector internally so absolute scale is arbitrary.
    """

    name: str
    model: Model
    abundance: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"CommunityMember.name must be a non-empty string, got {self.name!r}")
        if self.abundance <= 0:
            raise ValueError(
                f"CommunityMember({self.name!r}).abundance must be > 0, got {self.abundance}"
            )


@dataclass(frozen=True, slots=True)
class MICOMResult:
    """MICOM cooperative-tradeoff snapshot.

    Attributes:
        community_growth_rate: Community μ (1/h), normalized to 1 gDW of
            total community biomass.
        member_growth_rate: Per-member μ (1/h). Index is member name.
        fluxes: MICOM flux frame. Rows are taxa (by name) plus one row
            ``"medium"`` for external exchange fluxes; columns are
            community-wide reaction IDs.
        fraction: The tradeoff fraction actually used.
        status: Solver status string from MICOM.
        solution: Underlying ``micom.CommunitySolution`` for inspection.
    """

    community_growth_rate: float
    member_growth_rate: pd.Series
    fluxes: pd.DataFrame
    fraction: float
    status: str
    solution: CommunitySolution


def simulate_micom(
    members: list[CommunityMember],
    medium: Medium,
    *,
    fraction: float = DEFAULT_FRACTION,
    uptake: dict[str, float] | None = None,
    mm_params: dict[str, MMParams] | None = None,
    pool_init: dict[str, float] | None = None,
    default_uptake: float = DEFAULT_UPTAKE,
    unlimited_uptake: float = DEFAULT_UNLIMITED_UPTAKE,
    pfba: bool = True,
    solver: str | None = None,
) -> MICOMResult:
    """Run MICOM cooperative tradeoff on a community.

    Args:
        members: Non-empty list of :class:`CommunityMember` with unique
            names.
        medium: Defines which exchanges are bounded as a pool (uptake set
            from MM or ``uptake``) vs treated as unlimited.
        fraction: Cooperative tradeoff fraction in ``(0, 1]`` — the
            minimum share of maximal community growth each member must
            sustain. ``1.0`` collapses to pure community max-growth.
        uptake: Explicit ``{exchange_id: max_uptake_mmol_per_gDW_h}`` for
            pool components. Takes precedence over MM-derived bounds.
            Positive values.
        mm_params: ``{exchange_id: MMParams}`` to derive uptake from
            ``pool_init`` (or ``medium.pool_components`` defaults) via
            ``michaelis_menten(conc, vmax, km)``. Ignored for any exchange
            already in ``uptake``.
        pool_init: Fixed pool concentrations (mM) for MM evaluation.
            Overrides ``medium.pool_components`` per key.
        default_uptake: Cap for pool components with no explicit / MM
            specification. Must be > 0.
        unlimited_uptake: Cap assigned to ``medium.unlimited_components``.
            Must be > 0.
        pfba: If True, MICOM resolves fluxes by parsimonious FBA. Strongly
            recommended.
        solver: Override optlang solver (``"cplex"``, ``"glpk"``, ...). If
            None, MICOM picks (typically CPLEX → GLPK).

    Returns:
        A :class:`MICOMResult`.

    Raises:
        ValueError: on invalid numeric arguments or empty/duplicate member
            lists.
        KeyError: if an ``uptake`` / ``mm_params`` / ``pool_init`` key is
            not a pool component of ``medium``.
    """
    if not members:
        raise ValueError("members must be a non-empty list")
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"member names must be unique; duplicates: {dupes}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must satisfy 0 < fraction <= 1, got {fraction}")
    if default_uptake <= 0:
        raise ValueError(f"default_uptake must be > 0, got {default_uptake}")
    if unlimited_uptake <= 0:
        raise ValueError(f"unlimited_uptake must be > 0, got {unlimited_uptake}")

    pool_id_set = set(medium.pool_components)
    for kind, dct in (("uptake", uptake), ("mm_params", mm_params), ("pool_init", pool_init)):
        if dct is None:
            continue
        for k in dct:
            if k not in pool_id_set:
                raise KeyError(
                    f"{kind} key {k!r} is not a pool component of medium {medium.name!r}"
                )
    if uptake is not None:
        for k, v in uptake.items():
            if v < 0:
                raise KeyError(f"uptake[{k!r}] = {v}; uptake caps must be >= 0")

    concs: dict[str, float] = dict(medium.pool_components)
    if pool_init is not None:
        concs.update(pool_init)

    flux_bounds: dict[str, float] = {}
    for rid in medium.pool_components:
        if uptake is not None and rid in uptake:
            flux_bounds[rid] = float(uptake[rid])
        elif mm_params is not None and rid in mm_params:
            p = mm_params[rid]
            flux_bounds[rid] = float(michaelis_menten(max(concs[rid], 0.0), p.vmax, p.km))
        else:
            flux_bounds[rid] = float(default_uptake)
    for rid in medium.unlimited_components:
        flux_bounds[rid] = float(unlimited_uptake)

    with _serialized_models(members) as taxonomy:
        from micom import Community

        with _silence_micom():
            community = Community(taxonomy, progress=False, solver=solver)
            community.medium = _translate_to_community_ids(flux_bounds, community)
            solution = community.cooperative_tradeoff(
                fraction=fraction,
                fluxes=True,
                pfba=pfba,
            )

    member_growth = pd.Series(
        {name: float(solution.members.loc[name, "growth_rate"]) for name in names},
        name="growth_rate",
    )
    fluxes_df = solution.fluxes.copy() if solution.fluxes is not None else pd.DataFrame()

    return MICOMResult(
        community_growth_rate=float(solution.growth_rate),
        member_growth_rate=member_growth,
        fluxes=fluxes_df,
        fraction=float(fraction),
        status=str(solution.status),
        solution=solution,
    )


@contextmanager
def _serialized_models(members: list[CommunityMember]) -> Iterator[pd.DataFrame]:
    """Write each member's cobra model to a JSON file and yield a MICOM taxonomy frame."""
    from cobra.io import save_json_model

    with tempfile.TemporaryDirectory(prefix="gemfitcom_micom_") as tmpdir:
        rows = []
        for m in members:
            path = Path(tmpdir) / f"{m.name}.json"
            save_json_model(m.model, str(path))
            rows.append({"id": m.name, "file": str(path), "abundance": float(m.abundance)})
        yield pd.DataFrame(rows)


def _translate_to_community_ids(
    flux_bounds: dict[str, float], community: object
) -> dict[str, float]:
    """Map our ``EX_*_e`` medium keys to MICOM's community exchange IDs.

    MICOM duplicates each taxon's external exchange into a shuttle reaction
    and a community-level exchange whose compartment is ``m`` (medium).
    Concretely, ``EX_<met>_e`` in a member model becomes ``EX_<met>_m`` at
    the community boundary. We try the ``_e → _m`` rewrite first and fall
    back to the original ID so user-supplied IDs that already match
    community naming still work.
    """
    exchange_ids = {r.id for r in community.exchanges}  # type: ignore[attr-defined]
    translated: dict[str, float] = {}
    for rid, v in flux_bounds.items():
        if rid in exchange_ids:
            translated[rid] = v
            continue
        if rid.endswith("_e"):
            candidate = rid[:-2] + "_m"
            if candidate in exchange_ids:
                translated[candidate] = v
                continue
        # No match — keep original; MICOM's setter will log it as not-found.
        translated[rid] = v
    return translated


@contextmanager
def _silence_micom() -> Iterator[None]:
    """Mute MICOM's chatty INFO log lines for the duration of a call."""
    logger = logging.getLogger("micom")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


__all__ = [
    "DEFAULT_FRACTION",
    "DEFAULT_UNLIMITED_UPTAKE",
    "DEFAULT_UPTAKE",
    "CommunityMember",
    "MICOMResult",
    "simulate_micom",
]

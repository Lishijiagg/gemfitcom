"""Gap-fill knowledge base: data classes, YAML loader, and registry.

A :class:`GapfillKB` maps observed product exchange IDs (e.g.
``EX_but_e``) to the minimal set of metabolites and reactions needed to
let a model SECRETE that product. KB entries are authored as YAML (see
``gemfitcom/data/gapfill_kb/scfa.yaml``) and validated on load: each
reaction's equation is parsed and checked for mass balance (per element)
and charge balance. Exchange reactions — one stoichiometric side empty —
are exempt because they are sources/sinks by construction.

v0.1 limitation: every metabolite referenced by a reaction in an entry
must itself be listed in the entry's ``metabolites`` block. The loader
cannot consult the host model's metabolite library to resolve externally
supplied species (e.g. ``h_c`` for a proton-symport transport). This
restricts v0.1 KB content to simple-diffusion transport + exchange
pairs; full synthesis reactions must wait for a future "external
metabolite reference" mechanism.

This module avoids importing cobra so KB parsing is fast and
cobra-independent; cobra objects are only built in :mod:`apply`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from gemfitcom.medium.medium import EXCHANGE_ID_PATTERN

_BUILTIN_PACKAGE = "gemfitcom.data.gapfill_kb"
_CUSTOM_REGISTRY: dict[str, GapfillKB] = {}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_ARROW_PATTERNS: tuple[tuple[str, bool], ...] = (
    # (separator, reversible?)
    ("<=>", True),
    ("-->", False),
    ("<--", False),  # reversed irreversible; swap sides on parse
)


class KBError(ValueError):
    """Raised when a gap-fill knowledge base YAML or dict fails validation."""


@dataclass(frozen=True, slots=True)
class MetaboliteSpec:
    """A metabolite declaration within a KB entry.

    Attributes:
        id: Unique identifier (BiGG-style; e.g. ``but_c``, ``but_e``).
        compartment: Compartment suffix — ``"c"`` (cytosol) or ``"e"``
            (extracellular) for v0.1.
        formula: Chemical formula (e.g. ``"C4H7O2"``). Parsed for mass
            balance; an empty string disables balance checks against
            this metabolite (not recommended).
        charge: Net charge (integer).
        name: Free-form name used when adding the metabolite to a model.
    """

    id: str
    compartment: str
    formula: str
    charge: int
    name: str = ""


@dataclass(frozen=True, slots=True)
class ReactionSpec:
    """A reaction declaration within a KB entry.

    ``stoichiometry`` is derived from :attr:`equation` at construction and
    is keyed by metabolite ID with stoichiometric coefficients signed as
    negative for reactants and positive for products.

    Attributes:
        id: Unique reaction identifier (BiGG-style when possible).
        equation: Human-readable equation string using ``-->``, ``<--``,
            or ``<=>`` and ``+`` separated terms (optional integer/float
            coefficients, e.g. ``"2 atp_c + h2o_c --> 2 adp_c + h_c"``).
        bounds: ``(lower_bound, upper_bound)`` applied on model insertion.
        stoichiometry: Parsed coefficients.
        reversible: True if the equation used ``<=>``.
        is_exchange: True if one side of the equation is empty — the
            reaction is treated as a source/sink and skipped by balance
            checks.
        name: Optional long-form name.
    """

    id: str
    equation: str
    bounds: tuple[float, float]
    stoichiometry: dict[str, float]
    reversible: bool
    is_exchange: bool
    name: str = ""


@dataclass(frozen=True, slots=True)
class GapfillKBEntry:
    """A product-level gap-fill recipe.

    Attributes:
        exchange_id: Target exchange reaction ID (must match
            :data:`gemfitcom.medium.medium.EXCHANGE_ID_PATTERN`).
        display_name: Human label shown in reports (e.g. ``"butyrate"``).
        metabolites: Metabolites to add to the host model if absent.
        reactions: Reactions to add to the host model if absent. Order is
            preserved; apply typically follows the order transport →
            exchange for readability.
        references: Free-form citations (BiGG / KEGG IDs, DOIs).
    """

    exchange_id: str
    display_name: str
    metabolites: tuple[MetaboliteSpec, ...]
    reactions: tuple[ReactionSpec, ...]
    references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GapfillKB:
    """A named collection of :class:`GapfillKBEntry` keyed by exchange ID.

    Attributes:
        name: Registration key (e.g. ``"scfa"``).
        entries: Map of exchange ID → entry.
        version: KB schema / data version (free form).
        description: Free-form description.
        metadata: Any additional top-level YAML keys.
    """

    name: str
    entries: dict[str, GapfillKBEntry]
    version: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exchange_ids(self) -> frozenset[str]:
        """All exchange IDs covered by this KB."""
        return frozenset(self.entries)

    def __contains__(self, exchange_id: object) -> bool:
        return exchange_id in self.entries

    def get(self, exchange_id: str) -> GapfillKBEntry | None:
        """Return the entry for ``exchange_id`` or ``None`` if absent."""
        return self.entries.get(exchange_id)


# ---------------------------------------------------------------------------
# YAML / dict parsing
# ---------------------------------------------------------------------------


def kb_from_dict(data: dict[str, Any], *, source: str | Path | None = None) -> GapfillKB:
    """Build a :class:`GapfillKB` from a parsed YAML / dict payload.

    Validates per-entry metabolite declarations and performs mass +
    charge balance on every non-exchange reaction.

    Raises:
        KBError: on schema or balance violations.
    """
    loc = f" (in {source})" if source is not None else ""

    if not isinstance(data, dict):
        raise KBError(f"KB definition must be a mapping{loc}, got {type(data).__name__}")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise KBError(f"KB 'name' must be a non-empty string{loc}")

    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise KBError(f"KB 'entries' must be a non-empty list{loc}")

    entries: dict[str, GapfillKBEntry] = {}
    for i, raw in enumerate(raw_entries):
        entry = _parse_entry(raw, where=f"{loc}, entries[{i}]")
        if entry.exchange_id in entries:
            raise KBError(f"duplicate entry exchange_id {entry.exchange_id!r}{loc} at entries[{i}]")
        entries[entry.exchange_id] = entry

    version = str(data.get("version", "") or "")
    description = str(data.get("description", "") or "")
    extra = data.get("metadata", {}) or {}
    if not isinstance(extra, dict):
        raise KBError(f"'metadata' must be a mapping{loc}")
    reserved = {"name", "entries", "version", "description", "metadata"}
    leftover = {k: v for k, v in data.items() if k not in reserved}
    merged_metadata = {**extra, **leftover}

    return GapfillKB(
        name=name,
        entries=entries,
        version=version,
        description=description,
        metadata=merged_metadata,
    )


def kb_from_yaml(path: str | Path) -> GapfillKB:
    """Load and validate a KB YAML file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"gap-fill KB YAML not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise KBError(f"failed to parse YAML {p}: {exc}") from exc
    return kb_from_dict(data, source=p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_kb(name_or_path: str | Path) -> GapfillKB:
    """Resolve a KB by registered name or filesystem path.

    Search order: custom registrations → built-in YAML files (bundled
    under ``gemfitcom.data.gapfill_kb``) → direct path if the argument
    looks like one.

    Raises:
        FileNotFoundError: when a path is given that does not exist.
        KeyError: when a bare name is neither registered nor built in.
    """
    if isinstance(name_or_path, Path) or _looks_like_path(str(name_or_path)):
        return kb_from_yaml(name_or_path)

    name = str(name_or_path)
    if name in _CUSTOM_REGISTRY:
        return _CUSTOM_REGISTRY[name]

    builtin = _load_builtin(name)
    if builtin is not None:
        return builtin

    available = sorted({*_CUSTOM_REGISTRY, *_list_builtin()})
    raise KeyError(
        f"gap-fill KB {name!r} is not registered. Available: {available}. "
        "Pass a file path to load a custom KB."
    )


def list_kbs() -> list[str]:
    """Return every KB name resolvable by :func:`load_kb` (sorted)."""
    return sorted({*_CUSTOM_REGISTRY, *_list_builtin()})


def register_kb(name: str, source: str | Path | GapfillKB | dict) -> GapfillKB:
    """Register a KB under ``name`` for later :func:`load_kb` calls.

    Args:
        name: Registration key; overwrites any existing custom entry.
        source: A :class:`GapfillKB`, a parsed dict, or a path to a YAML
            file.

    Returns:
        The registered :class:`GapfillKB`.
    """
    if isinstance(source, GapfillKB):
        kb = source
    elif isinstance(source, dict):
        kb = kb_from_dict(source)
    else:
        kb = kb_from_yaml(source)
    _CUSTOM_REGISTRY[name] = kb
    return kb


def unregister_kb(name: str) -> None:
    """Remove a previously :func:`register_kb`-ed entry. No-op if absent."""
    _CUSTOM_REGISTRY.pop(name, None)


def clear_custom_kb_registry() -> None:
    """Drop all custom registrations (does not affect built-ins)."""
    _CUSTOM_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _looks_like_path(s: str) -> bool:
    return any(sep in s for sep in ("/", "\\")) or s.endswith((".yaml", ".yml"))


def _load_builtin(name: str) -> GapfillKB | None:
    for suffix in (".yaml", ".yml"):
        filename = f"{name}{suffix}"
        try:
            ref = resources.files(_BUILTIN_PACKAGE).joinpath(filename)
        except (ModuleNotFoundError, FileNotFoundError):
            return None
        if ref.is_file():
            with resources.as_file(ref) as path:
                return kb_from_yaml(path)
    return None


def _list_builtin() -> list[str]:
    try:
        entries = list(resources.files(_BUILTIN_PACKAGE).iterdir())
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    names: list[str] = []
    for entry in entries:
        entry_name = entry.name
        if entry_name.endswith((".yaml", ".yml")):
            names.append(entry_name.rsplit(".", 1)[0])
    return names


def _parse_entry(raw: Any, *, where: str) -> GapfillKBEntry:
    if not isinstance(raw, dict):
        raise KBError(f"entry must be a mapping{where}, got {type(raw).__name__}")

    exchange_id = raw.get("exchange_id")
    if not isinstance(exchange_id, str) or not EXCHANGE_ID_PATTERN.match(exchange_id):
        raise KBError(
            f"entry exchange_id {exchange_id!r}{where} must match {EXCHANGE_ID_PATTERN.pattern!r}"
        )
    display_name = str(raw.get("display_name", "") or "")

    metabolites: list[MetaboliteSpec] = []
    seen_met_ids: set[str] = set()
    for j, m in enumerate(raw.get("metabolites", []) or []):
        met = _parse_metabolite(m, where=f"{where}, metabolites[{j}]")
        if met.id in seen_met_ids:
            raise KBError(f"duplicate metabolite id {met.id!r}{where}, metabolites[{j}]")
        seen_met_ids.add(met.id)
        metabolites.append(met)

    met_by_id = {m.id: m for m in metabolites}

    reactions: list[ReactionSpec] = []
    seen_rxn_ids: set[str] = set()
    for j, r in enumerate(raw.get("reactions", []) or []):
        rxn = _parse_reaction(r, met_by_id=met_by_id, where=f"{where}, reactions[{j}]")
        if rxn.id in seen_rxn_ids:
            raise KBError(f"duplicate reaction id {rxn.id!r}{where}, reactions[{j}]")
        seen_rxn_ids.add(rxn.id)
        reactions.append(rxn)

    if not reactions:
        raise KBError(f"entry {exchange_id!r}{where} must declare at least one reaction")
    if exchange_id not in seen_rxn_ids:
        raise KBError(
            f"entry {exchange_id!r}{where} does not declare its own exchange reaction "
            f"(expected a reaction with id {exchange_id!r})"
        )

    references = tuple(str(x) for x in raw.get("references", []) or [])

    return GapfillKBEntry(
        exchange_id=exchange_id,
        display_name=display_name,
        metabolites=tuple(metabolites),
        reactions=tuple(reactions),
        references=references,
    )


def _parse_metabolite(raw: Any, *, where: str) -> MetaboliteSpec:
    if not isinstance(raw, dict):
        raise KBError(f"metabolite must be a mapping{where}, got {type(raw).__name__}")
    met_id = raw.get("id")
    if not isinstance(met_id, str) or not met_id:
        raise KBError(f"metabolite 'id' must be a non-empty string{where}")
    compartment = raw.get("compartment")
    if not isinstance(compartment, str) or not compartment:
        raise KBError(f"metabolite {met_id!r}{where} requires a 'compartment' string")
    formula = str(raw.get("formula", "") or "")
    if not formula:
        raise KBError(f"metabolite {met_id!r}{where} requires a non-empty 'formula'")
    _parse_formula(formula, where=f"{where} (metabolite {met_id!r})")  # validates
    charge_raw = raw.get("charge")
    if (
        charge_raw is None
        or not isinstance(charge_raw, int | float)
        or isinstance(charge_raw, bool)
    ):
        raise KBError(f"metabolite {met_id!r}{where} requires an integer 'charge'")
    charge = int(charge_raw)
    if charge != charge_raw:
        raise KBError(f"metabolite {met_id!r}{where} 'charge' must be integer, got {charge_raw!r}")
    name = str(raw.get("name", "") or "")
    return MetaboliteSpec(
        id=met_id,
        compartment=compartment,
        formula=formula,
        charge=charge,
        name=name,
    )


def _parse_reaction(
    raw: Any,
    *,
    met_by_id: dict[str, MetaboliteSpec],
    where: str,
) -> ReactionSpec:
    if not isinstance(raw, dict):
        raise KBError(f"reaction must be a mapping{where}, got {type(raw).__name__}")
    rxn_id = raw.get("id")
    if not isinstance(rxn_id, str) or not rxn_id:
        raise KBError(f"reaction 'id' must be a non-empty string{where}")

    equation = raw.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        raise KBError(f"reaction {rxn_id!r}{where} requires an 'equation' string")

    bounds_raw = raw.get("bounds")
    bounds = _parse_bounds(bounds_raw, where=f"{where} (reaction {rxn_id!r})")

    stoich, reversible, is_exchange = _parse_equation(
        equation, where=f"{where} (reaction {rxn_id!r})"
    )

    # Every metabolite referenced must be declared in the entry.
    for met_id in stoich:
        if met_id not in met_by_id:
            raise KBError(
                f"reaction {rxn_id!r}{where} references metabolite {met_id!r} "
                "not declared in this entry's 'metabolites' list"
            )

    # Mass & charge balance for non-exchange reactions.
    if not is_exchange:
        _check_balance(
            rxn_id=rxn_id,
            stoich=stoich,
            met_by_id=met_by_id,
            where=where,
        )

    name = str(raw.get("name", "") or "")
    return ReactionSpec(
        id=rxn_id,
        equation=equation,
        bounds=bounds,
        stoichiometry=stoich,
        reversible=reversible,
        is_exchange=is_exchange,
        name=name,
    )


def _parse_bounds(raw: Any, *, where: str) -> tuple[float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise KBError(f"'bounds'{where} must be a length-2 sequence of numbers, got {raw!r}")
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise KBError(f"'bounds'{where} must be numeric, got {raw!r}") from exc
    if lo > hi:
        raise KBError(f"'bounds'{where} must satisfy lower <= upper, got ({lo}, {hi})")
    return (lo, hi)


def _parse_equation(equation: str, *, where: str) -> tuple[dict[str, float], bool, bool]:
    """Parse a reaction equation into stoichiometry, reversibility, exchange flag."""
    separator: str | None = None
    reversible = False
    reverse_sides = False
    for sep, rev in _ARROW_PATTERNS:
        if sep in equation:
            separator = sep
            reversible = rev
            reverse_sides = sep == "<--"
            break
    if separator is None:
        raise KBError(f"equation{where} must contain one of <=> / --> / <--: {equation!r}")

    lhs_s, rhs_s = equation.split(separator, 1)
    if reverse_sides:
        lhs_s, rhs_s = rhs_s, lhs_s  # normalise: lhs = reactants, rhs = products

    lhs_terms = _parse_terms(lhs_s, where=where)
    rhs_terms = _parse_terms(rhs_s, where=where)
    is_exchange = not lhs_terms or not rhs_terms

    stoich: dict[str, float] = {}
    for met_id, coef in lhs_terms:
        stoich[met_id] = stoich.get(met_id, 0.0) - coef
    for met_id, coef in rhs_terms:
        stoich[met_id] = stoich.get(met_id, 0.0) + coef
    # Drop zero entries produced by cancellation (e.g. same met on both sides).
    stoich = {k: v for k, v in stoich.items() if v != 0}

    if not stoich:
        raise KBError(f"equation{where} has no net stoichiometry: {equation!r}")

    return stoich, reversible, is_exchange


def _parse_terms(side: str, *, where: str) -> list[tuple[str, float]]:
    side = side.strip()
    if not side:
        return []
    terms: list[tuple[str, float]] = []
    for chunk in side.split(" + "):
        token = chunk.strip()
        if not token:
            raise KBError(f"empty term in equation{where}: {side!r}")
        parts = token.split()
        if len(parts) == 1:
            coef, met_id = 1.0, parts[0]
        elif len(parts) == 2:
            try:
                coef = float(parts[0])
            except ValueError as exc:
                raise KBError(f"bad coefficient in equation{where}: {token!r}") from exc
            met_id = parts[1]
        else:
            raise KBError(f"bad term in equation{where}: {token!r}")
        if not met_id:
            raise KBError(f"empty metabolite id in equation{where}: {token!r}")
        terms.append((met_id, coef))
    return terms


def _check_balance(
    *,
    rxn_id: str,
    stoich: dict[str, float],
    met_by_id: dict[str, MetaboliteSpec],
    where: str,
) -> None:
    element_delta: dict[str, float] = {}
    charge_delta = 0.0
    for met_id, coef in stoich.items():
        spec = met_by_id[met_id]
        atoms = _parse_formula(spec.formula, where=f"{where} (metabolite {met_id!r})")
        for element, count in atoms.items():
            element_delta[element] = element_delta.get(element, 0.0) + coef * count
        charge_delta += coef * spec.charge

    off_elements = {e: d for e, d in element_delta.items() if abs(d) > 1e-9}
    if off_elements:
        raise KBError(
            f"reaction {rxn_id!r}{where} is not mass-balanced: element deltas {off_elements}"
        )
    if abs(charge_delta) > 1e-9:
        raise KBError(
            f"reaction {rxn_id!r}{where} is not charge-balanced: net charge delta = {charge_delta}"
        )


def _parse_formula(formula: str, *, where: str) -> dict[str, int]:
    atoms: dict[str, int] = {}
    pos = 0
    while pos < len(formula):
        m = _FORMULA_TOKEN.match(formula, pos)
        if m is None or m.end() == pos:
            raise KBError(f"invalid chemical formula{where}: {formula!r}")
        element, count_s = m.group(1), m.group(2)
        count = int(count_s) if count_s else 1
        atoms[element] = atoms.get(element, 0) + count
        pos = m.end()
    if not atoms:
        raise KBError(f"chemical formula{where} is empty: {formula!r}")
    return atoms


__all__ = [
    "GapfillKB",
    "GapfillKBEntry",
    "KBError",
    "MetaboliteSpec",
    "ReactionSpec",
    "clear_custom_kb_registry",
    "kb_from_dict",
    "kb_from_yaml",
    "list_kbs",
    "load_kb",
    "register_kb",
    "unregister_kb",
]

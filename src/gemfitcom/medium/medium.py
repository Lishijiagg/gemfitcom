"""Medium schema and YAML parsing.

A :class:`Medium` captures the minimum information needed to constrain a
COBRA model's exchange reactions for single-strain or community dFBA:

* ``pool_components`` — exchange-reaction IDs mapped to their *initial*
  concentrations (mM). These are the metabolites tracked dynamically in
  the ODE pool; their uptake bounds are set from MM kinetics by the
  ``kinetics`` module, so the value recorded here is the starting pool
  size, not the flux bound.
* ``unlimited_components`` — exchange-reaction IDs assumed to be in excess
  (bound set to ``-1000`` by :func:`apply_medium`).

The YAML representation mirrors this structure 1:1; see
``src/gemfitcom/data/media/YCFA.yaml`` for a reference example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXCHANGE_ID_PATTERN: re.Pattern[str] = re.compile(r"^EX_[A-Za-z0-9_]+_e$")


class MediumError(ValueError):
    """Raised when a medium YAML or dict fails schema validation."""


@dataclass(frozen=True, slots=True)
class Medium:
    """A culture medium definition.

    Attributes:
        name: Canonical name (e.g. ``"YCFA"``).
        pool_components: Exchange ID → initial concentration in mM.
        unlimited_components: Exchange IDs with unlimited supply (set to -1000).
        description: Free-form description.
        version: Schema / data version string (free-form).
        metadata: Arbitrary extra fields from the YAML (source, reference, ...).
    """

    name: str
    pool_components: dict[str, float]
    unlimited_components: frozenset[str]
    description: str = ""
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.pool_components).intersection(self.unlimited_components)
        if overlap:
            raise MediumError(
                f"pool_components and unlimited_components overlap on {sorted(overlap)}; "
                "each exchange must appear in exactly one list"
            )
        for rxn_id, conc in self.pool_components.items():
            _validate_exchange_id(rxn_id)
            if conc < 0:
                raise MediumError(
                    f"pool_components[{rxn_id!r}] = {conc}; concentrations must be >= 0"
                )
        for rxn_id in self.unlimited_components:
            _validate_exchange_id(rxn_id)

    @property
    def exchange_ids(self) -> frozenset[str]:
        """All exchange IDs mentioned by the medium (pool + unlimited)."""
        return frozenset(self.pool_components) | self.unlimited_components


def medium_from_dict(data: dict[str, Any], *, source: str | Path | None = None) -> Medium:
    """Build a :class:`Medium` from a parsed YAML / dict payload.

    Args:
        data: Mapping with at least ``name`` and ``pool_components``.
        source: Optional path string used in error messages.

    Raises:
        MediumError: on schema violations.
    """
    loc = f" (in {source})" if source is not None else ""

    if not isinstance(data, dict):
        raise MediumError(f"medium definition must be a mapping{loc}, got {type(data).__name__}")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise MediumError(f"medium 'name' must be a non-empty string{loc}")

    pool_raw = data.get("pool_components", {})
    if not isinstance(pool_raw, dict):
        raise MediumError(f"'pool_components' must be a mapping{loc}")
    pool_components: dict[str, float] = {}
    for k, v in pool_raw.items():
        try:
            pool_components[str(k)] = float(v)
        except (TypeError, ValueError) as exc:
            raise MediumError(f"pool_components[{k!r}] = {v!r} is not a number{loc}") from exc

    unlimited_raw = data.get("unlimited_components", [])
    if not isinstance(unlimited_raw, list):
        raise MediumError(f"'unlimited_components' must be a list{loc}")
    unlimited_components = frozenset(str(x) for x in unlimited_raw)

    description = str(data.get("description", "") or "")
    version = str(data.get("version", "") or "")
    reserved = {
        "name",
        "pool_components",
        "unlimited_components",
        "description",
        "version",
        "metadata",
    }
    extra_metadata = data.get("metadata", {}) or {}
    if not isinstance(extra_metadata, dict):
        raise MediumError(f"'metadata' must be a mapping{loc}")
    leftover = {k: v for k, v in data.items() if k not in reserved}
    merged_metadata = {**extra_metadata, **leftover}

    return Medium(
        name=name,
        pool_components=pool_components,
        unlimited_components=unlimited_components,
        description=description,
        version=version,
        metadata=merged_metadata,
    )


def medium_from_yaml(path: str | Path) -> Medium:
    """Load and validate a medium YAML file.

    Args:
        path: Path to a ``*.yaml`` / ``*.yml`` file.

    Raises:
        FileNotFoundError: if the file does not exist.
        MediumError: on YAML parse errors or schema violations.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"medium YAML not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MediumError(f"failed to parse YAML {p}: {exc}") from exc
    return medium_from_dict(data, source=p)


def _validate_exchange_id(rxn_id: str) -> None:
    if not EXCHANGE_ID_PATTERN.match(rxn_id):
        raise MediumError(
            f"{rxn_id!r} does not look like a BiGG-style exchange reaction ID "
            "(expected pattern ^EX_<name>_e$)"
        )

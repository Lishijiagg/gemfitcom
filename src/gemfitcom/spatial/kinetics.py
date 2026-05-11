"""Per-species Michaelis–Menten kinetics for spatial dFBA.

Wraps the lower-level :mod:`gemfitcom.kinetics.mm` MM formula into a
per-species container ``ExchangeKinetics`` that produces the upper bound on
substrate uptake at a given local concentration vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import cobra
import numpy as np
import yaml

from gemfitcom.io.models import load_model as load_sbml_model
from gemfitcom.kinetics.mm import michaelis_menten

ExchangeMode = Literal["uptake_only", "bidirectional"]


@dataclass(frozen=True, slots=True)
class ExchangeEntry:
    """Single substrate's MM parameters for one species.

    Attributes:
        exchange_id: Cobra exchange reaction id (e.g. ``"EX_glc__D_e"``).
        vmax: Maximum uptake rate (mmol / gDW / h). Must be > 0.
        km: Half-saturation concentration (mM). Must be > 0.
        mode: Either ``"uptake_only"`` (default) or ``"bidirectional"``. This field
            tags how the per-cell FBA bound applier should treat the exchange — it is
            NOT consumed by :meth:`mm_upper_bound` itself (which just returns the
            MM-derived magnitude). The reaction engine (Task 7+) reads this to decide
            whether to write only ``lower_bound`` (uptake) or both ``lower_bound`` and
            ``upper_bound`` (bidirectional).
    """

    exchange_id: str
    vmax: float
    km: float
    mode: ExchangeMode = "uptake_only"

    def __post_init__(self) -> None:
        if self.vmax <= 0:
            raise ValueError(f"{self.exchange_id}: vmax must be > 0, got {self.vmax}")
        if self.km <= 0:
            raise ValueError(f"{self.exchange_id}: km must be > 0, got {self.km}")
        valid_modes = get_args(ExchangeMode)
        if self.mode not in valid_modes:
            raise ValueError(
                f"{self.exchange_id}: mode must be one of {valid_modes}, got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ExchangeKinetics:
    """All exchange kinetics for a single species.

    Attributes:
        species: Logical species name (e.g. ``"ecoli"``); informational only.
        entries: Ordered tuple of :class:`ExchangeEntry`, one per substrate
            the species transports through an exchange reaction. The order
            is load-bearing — index ``k`` in ``entries`` matches index ``k``
            in the ``C_local`` array passed to :meth:`mm_upper_bound`, and
            matches the order in which bounds are written into cobra by the
            per-cell FBA solver downstream.
    """

    species: str
    entries: tuple[ExchangeEntry, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for e in self.entries:
            if e.exchange_id in seen:
                raise ValueError(
                    f"{self.species}: duplicate exchange_id {e.exchange_id!r} in entries"
                )
            seen.add(e.exchange_id)

    @property
    def n_exchanges(self) -> int:
        return len(self.entries)

    @property
    def exchange_ids(self) -> tuple[str, ...]:
        return tuple(e.exchange_id for e in self.entries)

    def mm_upper_bound(self, C_local: np.ndarray) -> np.ndarray:
        """Compute MM-derived uptake upper bound at a single grid cell."""
        C_local = np.asarray(C_local, dtype=float)
        if C_local.shape != (self.n_exchanges,):
            raise ValueError(
                f"{self.species}: C_local length {C_local.shape} does not match "
                f"n_exchanges={self.n_exchanges}"
            )
        out = np.empty(self.n_exchanges, dtype=float)
        for k, entry in enumerate(self.entries):
            out[k] = float(michaelis_menten(C_local[k], entry.vmax, entry.km))
        return out


def load_kinetics_yaml(path: str | Path) -> ExchangeKinetics:
    """Load a per-species kinetics YAML into :class:`ExchangeKinetics`.

    Expected schema (design doc §5.4)::

        species: ecoli
        exchanges:
          EX_glc__D_e: {v_max: 10.0, K_m: 0.5}
          EX_o2_e:     {v_max: 15.0, K_m: 0.005}
          EX_ac_e:     {v_max: 5.0,  K_m: 0.1, mode: bidirectional}

    YAML keys are ``v_max`` / ``K_m`` (scientific convention); the Python
    dataclass uses lowercase ``vmax`` / ``km``.

    Args:
        path: Path to the kinetics YAML.

    Returns:
        :class:`ExchangeKinetics`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if a required field is missing.
        ValueError: if a value is out of range (re-raised from ExchangeEntry).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Kinetics YAML not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    if "species" not in data:
        raise KeyError(f"{path}: missing required field 'species'")
    if "exchanges" not in data:
        raise KeyError(f"{path}: missing required field 'exchanges'")

    entries: list[ExchangeEntry] = []
    for exch_id, params in data["exchanges"].items():
        if not isinstance(exch_id, str):
            raise TypeError(
                f"{path}: exchange id must be a string (got {type(exch_id).__name__}: {exch_id!r}); "
                "wrap numeric-looking ids in quotes in the YAML"
            )
        if not isinstance(params, dict):
            raise ValueError(f"{path}: exchange {exch_id} must map to a dict")
        if "v_max" not in params:
            raise KeyError(f"{path}: exchange {exch_id} missing 'v_max'")
        if "K_m" not in params:
            raise KeyError(f"{path}: exchange {exch_id} missing 'K_m'")
        entries.append(
            ExchangeEntry(
                exchange_id=exch_id,
                vmax=float(params["v_max"]),
                km=float(params["K_m"]),
                mode=params.get("mode", "uptake_only"),
            )
        )
    return ExchangeKinetics(species=data["species"], entries=tuple(entries))


_COBRA_URI_PREFIX = "cobra://"


def resolve_gem(uri: str) -> cobra.Model:
    """Resolve a GEM identifier to a loaded :class:`cobra.Model`.

    Supported forms:
        - ``cobra://<name>``: built-in cobra-shipped models, e.g.
          ``cobra://textbook`` (E. coli core).
        - Filesystem path: delegates to :func:`gemfitcom.io.models.load_model`.

    Args:
        uri: GEM identifier.

    Returns:
        Loaded :class:`cobra.Model`.

    Raises:
        ValueError: unknown scheme, or cobra:// name not recognised.
        FileNotFoundError: filesystem path does not exist.
    """
    if uri.startswith(_COBRA_URI_PREFIX):
        name = uri[len(_COBRA_URI_PREFIX) :].strip()
        if not name:
            raise ValueError(f"Empty cobra model name in URI {uri!r}")
        try:
            return cobra.io.load_model(name)
        except Exception as exc:
            raise ValueError(f"Unknown cobra built-in model {name!r} (URI={uri!r}): {exc}") from exc
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"Unknown GEM URI scheme {scheme!r} in {uri!r}; expected 'cobra://' or a path"
        )
    return load_sbml_model(uri)

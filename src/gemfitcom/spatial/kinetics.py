"""Per-species Michaelis–Menten kinetics for spatial dFBA.

Wraps the lower-level :mod:`gemfitcom.kinetics.mm` MM formula into a
per-species container ``ExchangeKinetics`` that produces the upper bound on
substrate uptake at a given local concentration vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gemfitcom.kinetics.mm import michaelis_menten

ExchangeMode = Literal["uptake_only", "bidirectional"]
_VALID_MODES: tuple[ExchangeMode, ...] = ("uptake_only", "bidirectional")


@dataclass(frozen=True, slots=True)
class ExchangeEntry:
    """Single substrate's MM parameters for one species.

    Attributes:
        exchange_id: Cobra exchange reaction id (e.g. ``"EX_glc__D_e"``).
        vmax: Maximum uptake rate (mmol / gDW / h). Must be > 0.
        km: Half-saturation concentration (mM). Must be > 0.
        mode: ``"uptake_only"`` writes only the lower bound; ``"bidirectional"``
            also caps the upper bound (secretion) with the same MM expression.
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
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"{self.exchange_id}: mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ExchangeKinetics:
    """All exchange kinetics for a single species."""

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

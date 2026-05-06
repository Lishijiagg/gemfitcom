"""SnapshotRecorder: periodic .npz dumps of SpatialState during a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .state import SpatialState

Precision = Literal["float32", "float64"]


@dataclass
class SnapshotRecorder:
    """Save SpatialState snapshots to disk at a configurable cadence.

    Snapshots are written as compressed .npz files named 'snapshot_t={t:.4f}.npz'.
    """

    output_dir: Path
    every: float
    precision: Precision = "float32"
    _last_save: float = field(default=-float("inf"), init=False, repr=False)
    _saved_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.every <= 0:
            raise ValueError(f"every must be > 0; got {self.every}")

    def save(self, state: SpatialState) -> Path:
        """Force-save the current state. Returns the file path written."""
        path = self.output_dir / f"snapshot_t={state.t:.4f}.npz"
        dtype = np.dtype(self.precision)
        np.savez_compressed(
            path,
            t=np.float64(state.t),
            metabolites=state.metabolites.astype(dtype),
            biomass=state.biomass.astype(dtype),
        )
        self._last_save = state.t
        self._saved_count += 1
        return path

    def maybe_save(self, state: SpatialState) -> Path | None:
        """Save only if `every` hours have elapsed since the last save.

        First call always saves (last_save starts at -inf).
        """
        # Tiny tolerance to handle float accumulation: 1e-12 is well below any
        # realistic dt and avoids missing a save by a billionth of an hour.
        if state.t - self._last_save >= self.every - 1e-12:
            return self.save(state)
        return None

    @staticmethod
    def load(path: Path) -> SpatialState:
        """Load a snapshot back into a SpatialState (always float64)."""
        data = np.load(path)
        return SpatialState(
            metabolites=data["metabolites"].astype(np.float64),
            biomass=data["biomass"].astype(np.float64),
            t=float(data["t"]),
        )

"""Michaelis-Menten kinetics for substrate exchange bounds.

The MM rate law ``v = vmax * [S] / (Km + [S])`` gives the maximum per-biomass
uptake rate (mmol/gDW/h) a cell can achieve at the given substrate
concentration [S] (mM). The dFBA loop uses it to set the lower bound on the
corresponding exchange reaction at each time step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class MMParams:
    """Michaelis-Menten kinetic parameters for one substrate.

    Attributes:
        vmax: Maximum uptake rate in mmol / gDW / h. Must be > 0.
        km: Half-saturation concentration in mM. Must be > 0.
    """

    vmax: float
    km: float

    def __post_init__(self) -> None:
        if self.vmax <= 0:
            raise ValueError(f"vmax must be > 0, got {self.vmax}")
        if self.km <= 0:
            raise ValueError(f"km must be > 0, got {self.km}")


def michaelis_menten(
    conc: ArrayLike,
    vmax: float,
    km: float,
) -> np.ndarray:
    """Evaluate ``vmax * conc / (km + conc)`` element-wise.

    Negative concentrations are clipped to 0 before the division so numerical
    drift below zero in the dFBA pool cannot produce a negative uptake rate.

    Args:
        conc: Substrate concentration(s) in mM (scalar or array).
        vmax: Maximum uptake rate (mmol/gDW/h). Must be > 0.
        km: Half-saturation (mM). Must be > 0.

    Returns:
        Uptake rate(s), same shape as ``conc``.
    """
    if vmax <= 0:
        raise ValueError(f"vmax must be > 0, got {vmax}")
    if km <= 0:
        raise ValueError(f"km must be > 0, got {km}")
    c = np.asarray(conc, dtype=float)
    c = np.maximum(c, 0.0)
    return vmax * c / (km + c)

"""Extract per-strain biomass trajectories in long form.

Complement to :func:`exchange_panel`: interactions analyses weight per-strain
fluxes by biomass to get absolute metabolite exchange (``flux * biomass``,
mmol / L / h) rather than per-gDW rates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from gemfitcom.simulate.fusion import FusionResult
from gemfitcom.simulate.micom import MICOMResult
from gemfitcom.simulate.sequential_dfba import SequentialDFBAResult


def biomass_panel(
    result: SequentialDFBAResult | FusionResult | MICOMResult,
) -> pd.DataFrame:
    """Per-strain biomass in long form: columns ``[time_h, strain, biomass]``.

    For :class:`MICOMResult` the biomass column is a placeholder
    (``1.0`` per member) because MICOM is a normalized steady-state
    snapshot; a :class:`UserWarning` is emitted so the caller knows
    cross-feeding amounts are unweighted.

    Raises:
        TypeError: on unsupported result types.
    """
    if isinstance(result, SequentialDFBAResult | FusionResult):
        wide = result.biomass
        long = wide.melt(id_vars="time_h", var_name="strain", value_name="biomass")
        return long.reset_index(drop=True)

    if isinstance(result, MICOMResult):
        warnings.warn(
            "biomass_panel(MICOMResult): MICOM is a normalized steady state; "
            "biomass is filled with 1.0 per member (cross-feeding amounts "
            "below are not biomass-weighted).",
            stacklevel=2,
        )
        strains = list(result.member_growth_rate.index)
        return pd.DataFrame(
            {
                "time_h": np.zeros(len(strains), dtype=float),
                "strain": strains,
                "biomass": np.ones(len(strains), dtype=float),
            }
        )

    raise TypeError(
        f"unsupported result type {type(result).__name__}; "
        "expected SequentialDFBAResult, FusionResult, or MICOMResult."
    )


__all__ = ["biomass_panel"]

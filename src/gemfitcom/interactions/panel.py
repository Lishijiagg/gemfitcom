"""Standardise simulate outputs into one long-form exchange-flux table.

Every interaction analysis (cross-feeding, competition, community
network) consumes the same schema:

    columns = [time_h, strain, exchange_id, flux]
    flux > 0  secretion
    flux < 0  uptake
    flux == 0 idle / not in the model

For dynamic simulations (``sequential_dfba``, ``fusion``) the table is
a time series; for :class:`MICOMResult` it is a single steady-state
row with ``time_h = 0.0``.
"""

from __future__ import annotations

import pandas as pd

from gemfitcom.simulate.fusion import FusionResult
from gemfitcom.simulate.micom import MICOMResult
from gemfitcom.simulate.sequential_dfba import SequentialDFBAResult

PANEL_COLUMNS: tuple[str, ...] = ("time_h", "strain", "exchange_id", "flux")


def exchange_panel(
    result: SequentialDFBAResult | FusionResult | MICOMResult,
) -> pd.DataFrame:
    """Return a standardised per-strain per-exchange flux long-form table.

    Args:
        result: Output of a :mod:`gemfitcom.simulate` call. Dynamic
            result types must have been produced with ``save_fluxes=True``.

    Returns:
        A :class:`pandas.DataFrame` with columns ``(time_h, strain,
        exchange_id, flux)``. Flux sign convention: positive =
        secretion, negative = uptake.

    Raises:
        ValueError: if a dynamic result carries ``exchange_fluxes=None``
            (simulate was not called with ``save_fluxes=True``).
        TypeError: if ``result`` is not a supported simulate result type.
    """
    if isinstance(result, SequentialDFBAResult | FusionResult):
        ef = result.exchange_fluxes
        if ef is None:
            raise ValueError(
                f"{type(result).__name__}.exchange_fluxes is None; "
                "re-run the simulation with save_fluxes=True."
            )
        return ef.copy()

    if isinstance(result, MICOMResult):
        return _micom_to_panel(result)

    raise TypeError(
        f"unsupported result type {type(result).__name__}; "
        "expected SequentialDFBAResult, FusionResult, or MICOMResult."
    )


def _micom_to_panel(result: MICOMResult) -> pd.DataFrame:
    """Melt a MICOM steady-state flux frame into the long-form panel."""
    fluxes = result.fluxes
    if "medium" in fluxes.index:
        fluxes = fluxes.drop(index="medium")
    exchange_cols = [c for c in fluxes.columns if c.startswith("EX_")]
    wide = fluxes[exchange_cols].copy()
    wide.index.name = "strain"
    long = wide.reset_index().melt(
        id_vars="strain",
        var_name="exchange_id",
        value_name="flux",
    )
    long.insert(0, "time_h", 0.0)
    return long[list(PANEL_COLUMNS)].reset_index(drop=True)


__all__ = ["PANEL_COLUMNS", "exchange_panel"]

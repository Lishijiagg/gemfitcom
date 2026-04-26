"""Tests for interactions.panel and interactions.biomass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from micom.solution import CommunitySolution

from gemfitcom.interactions import PANEL_COLUMNS, biomass_panel, exchange_panel
from gemfitcom.simulate.fusion import FusionResult
from gemfitcom.simulate.micom import MICOMResult
from gemfitcom.simulate.sequential_dfba import SequentialDFBAResult

# ---------- helpers to build synthetic result objects ----------


def _seq_result(exchange_fluxes: pd.DataFrame | None) -> SequentialDFBAResult:
    time_h = np.array([0.0, 1.0])
    biomass = pd.DataFrame({"time_h": time_h, "A": [0.01, 0.02], "B": [0.01, 0.015]})
    pool = pd.DataFrame({"time_h": time_h, "EX_glc__D_e": [10.0, 8.0]})
    growth = pd.DataFrame({"time_h": time_h, "A": [0.5, 0.5], "B": [0.3, 0.3]})
    return SequentialDFBAResult(
        time_h=time_h,
        biomass=biomass,
        pool=pool,
        growth_rate=growth,
        exchange_fluxes=exchange_fluxes,
    )


def _fusion_result(exchange_fluxes: pd.DataFrame | None) -> FusionResult:
    time_h = np.array([0.0, 1.0])
    biomass = pd.DataFrame({"time_h": time_h, "A": [0.01, 0.02]})
    pool = pd.DataFrame({"time_h": time_h, "EX_glc__D_e": [10.0, 8.0]})
    growth = pd.DataFrame({"time_h": time_h, "A": [0.5, 0.5]})
    return FusionResult(
        time_h=time_h,
        biomass=biomass,
        pool=pool,
        growth_rate=growth,
        community_growth_rate=np.array([0.5, 0.5]),
        fraction=0.5,
        fail_count=0,
        exchange_fluxes=exchange_fluxes,
    )


def _micom_result() -> MICOMResult:
    fluxes = pd.DataFrame(
        {
            "EX_glc__D_e": [-5.0, 0.0, 5.0],
            "EX_ac_e": [2.0, -1.5, 0.0],
            "BIOMASS": [0.3, 0.2, 0.0],
        },
        index=["A", "B", "medium"],
    )
    return MICOMResult(
        community_growth_rate=0.5,
        member_growth_rate=pd.Series({"A": 0.5, "B": 0.4}),
        fluxes=fluxes,
        fraction=0.5,
        status="optimal",
        solution=CommunitySolution.__new__(CommunitySolution),
    )


# ---------- exchange_panel: dynamic results ----------


def test_exchange_panel_returns_saved_long_form_for_sequential() -> None:
    ef = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 1.0, 1.0],
            "strain": ["A", "B", "A", "B"],
            "exchange_id": ["EX_ac_e", "EX_ac_e", "EX_ac_e", "EX_ac_e"],
            "flux": [2.0, -1.0, 1.5, -0.8],
        }
    )
    res = _seq_result(ef)
    out = exchange_panel(res)
    assert list(out.columns) == list(PANEL_COLUMNS)
    pd.testing.assert_frame_equal(out, ef)
    # returned frame is a copy — mutating it should not affect the source
    out.loc[0, "flux"] = 999.0
    assert res.exchange_fluxes is not None
    assert res.exchange_fluxes.loc[0, "flux"] == 2.0


def test_exchange_panel_returns_saved_long_form_for_fusion() -> None:
    ef = pd.DataFrame(
        {
            "time_h": [0.0, 1.0],
            "strain": ["A", "A"],
            "exchange_id": ["EX_glc__D_e", "EX_glc__D_e"],
            "flux": [-3.0, -2.5],
        }
    )
    res = _fusion_result(ef)
    out = exchange_panel(res)
    pd.testing.assert_frame_equal(out, ef)


def test_exchange_panel_raises_when_fluxes_not_saved() -> None:
    res = _seq_result(exchange_fluxes=None)
    with pytest.raises(ValueError, match="save_fluxes=True"):
        exchange_panel(res)


def test_exchange_panel_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported result type"):
        exchange_panel("not a result")  # type: ignore[arg-type]


# ---------- exchange_panel: MICOM ----------


def test_exchange_panel_micom_drops_medium_and_keeps_only_ex() -> None:
    out = exchange_panel(_micom_result())
    assert list(out.columns) == list(PANEL_COLUMNS)
    assert (out["time_h"] == 0.0).all()
    # BIOMASS (not an EX_) should be filtered out; medium row is dropped.
    assert set(out["exchange_id"]) == {"EX_glc__D_e", "EX_ac_e"}
    assert set(out["strain"]) == {"A", "B"}
    # flux values copied verbatim
    flux_a_glc = out.query("strain == 'A' and exchange_id == 'EX_glc__D_e'")["flux"]
    assert flux_a_glc.iloc[0] == -5.0


# ---------- biomass_panel ----------


def test_biomass_panel_dynamic_is_long_form() -> None:
    res = _seq_result(exchange_fluxes=None)
    out = biomass_panel(res)
    assert list(out.columns) == ["time_h", "strain", "biomass"]
    assert set(out["strain"]) == {"A", "B"}
    row_a_t1 = out.query("strain == 'A' and time_h == 1.0")["biomass"].iloc[0]
    assert row_a_t1 == pytest.approx(0.02)


def test_biomass_panel_micom_fills_ones_with_warning() -> None:
    res = _micom_result()
    with pytest.warns(UserWarning, match="MICOM"):
        out = biomass_panel(res)
    assert (out["biomass"] == 1.0).all()
    assert (out["time_h"] == 0.0).all()
    assert set(out["strain"]) == {"A", "B"}


def test_biomass_panel_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported result type"):
        biomass_panel(42)  # type: ignore[arg-type]

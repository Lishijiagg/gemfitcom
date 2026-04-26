"""Tests for kinetics.mono_dfba against toy cobra models."""

from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.kinetics.mono_dfba import simulate_mono_dfba
from gemfitcom.medium import Medium, apply_medium

YIELD_GDW_PER_MMOL: float = 0.1


def _toy_glucose_model() -> Model:
    """Minimal model: 10 mmol glucose → 1 gDW biomass."""
    m = Model("toy")
    glc = Metabolite("glc__D_e", compartment="e")
    bio = Metabolite("biomass_c", compartment="c")
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc: -1})
    biomass_rxn = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    biomass_rxn.add_metabolites({glc: -1.0 / YIELD_GDW_PER_MMOL, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex_glc, biomass_rxn, sink])
    m.objective = "BIOMASS"
    return m


def _single_carbon_medium(initial_mM: float = 10.0) -> Medium:
    return Medium(
        name="toy",
        pool_components={"EX_glc__D_e": initial_mM},
        unlimited_components=frozenset(),
    )


def test_simulate_mono_dfba_shapes_and_initial_condition() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium()
    apply_medium(model, medium, close_others=False)

    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=4.0,
        dt=0.5,
    )

    assert res.time_h.shape == (9,)
    assert res.time_h[0] == 0.0
    assert res.time_h[-1] == 4.0
    assert res.biomass.shape == (9,)
    assert res.biomass[0] == pytest.approx(0.01)
    assert res.pool.shape == (9, 2)
    assert "EX_glc__D_e" in res.pool.columns
    assert res.pool["EX_glc__D_e"].iloc[0] == pytest.approx(10.0)


def test_simulate_mono_dfba_biomass_grows_and_substrate_depletes() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium()
    apply_medium(model, medium, close_others=False)

    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=10.0,
        dt=0.25,
    )
    assert res.biomass[-1] > res.biomass[0]
    assert res.pool["EX_glc__D_e"].iloc[-1] < 10.0
    assert (res.pool["EX_glc__D_e"] >= 0).all()


def test_simulate_mono_dfba_mass_balance_approx() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium()
    apply_medium(model, medium, close_others=False)

    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=5.0,
        dt=0.1,
    )
    delta_biomass = res.biomass[-1] - res.biomass[0]
    delta_glc = res.pool["EX_glc__D_e"].iloc[0] - res.pool["EX_glc__D_e"].iloc[-1]
    assert delta_glc == pytest.approx(delta_biomass / YIELD_GDW_PER_MMOL, rel=0.05)


def test_simulate_mono_dfba_growth_rate_matches_mm() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(initial_mM=10.0)
    apply_medium(model, medium, close_others=False)

    vmax, km = 5.0, 1.0
    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=vmax, km=km)},
        initial_biomass=0.01,
        t_total=2.0,
        dt=0.5,
    )
    expected_mu0 = YIELD_GDW_PER_MMOL * vmax * 10.0 / (km + 10.0)
    assert res.growth_rate[0] == pytest.approx(expected_mu0, rel=1e-3)


def test_simulate_mono_dfba_rejects_bad_timing() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium()
    apply_medium(model, medium, close_others=False)
    mm = {"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)}

    with pytest.raises(ValueError, match="initial_biomass"):
        simulate_mono_dfba(model, medium, mm, initial_biomass=0.0, t_total=1.0, dt=0.5)
    with pytest.raises(ValueError, match="t_total"):
        simulate_mono_dfba(model, medium, mm, initial_biomass=0.01, t_total=0.0, dt=0.5)
    with pytest.raises(ValueError, match="dt"):
        simulate_mono_dfba(model, medium, mm, initial_biomass=0.01, t_total=1.0, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        simulate_mono_dfba(model, medium, mm, initial_biomass=0.01, t_total=1.0, dt=2.0)


def test_simulate_mono_dfba_rejects_mm_param_not_in_medium() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium()
    apply_medium(model, medium, close_others=False)

    with pytest.raises(KeyError, match="not a pool component"):
        simulate_mono_dfba(
            model,
            medium,
            mm_params={"EX_xyl__D_e": MMParams(vmax=1.0, km=1.0)},
            initial_biomass=0.01,
            t_total=1.0,
            dt=0.5,
        )


def test_simulate_mono_dfba_rejects_pool_missing_from_model() -> None:
    model = _toy_glucose_model()
    medium = Medium(
        name="toy",
        pool_components={"EX_glc__D_e": 10.0, "EX_not_in_model_e": 1.0},
        unlimited_components=frozenset(),
    )
    apply_medium(model, medium, close_others=False, on_missing="ignore")

    with pytest.raises(KeyError, match="not present as reactions"):
        simulate_mono_dfba(
            model,
            medium,
            mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
            initial_biomass=0.01,
            t_total=1.0,
            dt=0.5,
        )


def test_simulate_mono_dfba_pool_init_override() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(initial_mM=10.0)
    apply_medium(model, medium, close_others=False)

    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=1.0,
        dt=0.5,
        pool_init={"EX_glc__D_e": 2.0},
    )
    assert res.pool["EX_glc__D_e"].iloc[0] == pytest.approx(2.0)


def test_simulate_mono_dfba_zero_substrate_no_growth() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(initial_mM=0.0)
    apply_medium(model, medium, close_others=False)

    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        initial_biomass=0.01,
        t_total=2.0,
        dt=0.5,
    )
    assert np.all(res.biomass == pytest.approx(0.01))
    assert np.all(res.growth_rate == pytest.approx(0.0))

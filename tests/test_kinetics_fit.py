"""Tests for kinetics.fit (DE + grid Vmax/Km fitting)."""

from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.fit import fit_kinetics
from gemfitcom.kinetics.mm import MMParams
from gemfitcom.kinetics.mono_dfba import simulate_mono_dfba
from gemfitcom.medium import Medium, apply_medium

YIELD_GDW_PER_MMOL: float = 0.1


def _toy_glucose_model() -> Model:
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


def _generate_synthetic_biomass(
    vmax_true: float,
    km_true: float,
    *,
    initial_biomass: float = 0.01,
    initial_glc: float = 10.0,
    t_total: float = 8.0,
    dt: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(initial_glc)
    apply_medium(model, medium, close_others=False)
    res = simulate_mono_dfba(
        model,
        medium,
        mm_params={"EX_glc__D_e": MMParams(vmax=vmax_true, km=km_true)},
        initial_biomass=initial_biomass,
        t_total=t_total,
        dt=dt,
    )
    return res.time_h.copy(), res.biomass.copy()


def test_fit_recovers_parameters_from_clean_synthetic_data() -> None:
    vmax_true, km_true = 4.0, 2.0
    # t_total must be long enough for the substrate to actually deplete:
    # while glucose is fully saturating, the biomass curve only constrains
    # vmax (km is unidentifiable). With initial_biomass=0.01 and yield=0.1,
    # consuming ~80% of 10 mM glucose takes ~14h.
    t_obs, biomass_obs = _generate_synthetic_biomass(vmax_true, km_true, t_total=14.0, dt=0.5)

    model = _toy_glucose_model()
    medium = _single_carbon_medium(10.0)
    apply_medium(model, medium, close_others=False)

    fit = fit_kinetics(
        model,
        medium,
        carbon_exchange="EX_glc__D_e",
        t_obs=t_obs,
        biomass_obs=biomass_obs,
        vmax_bounds=(0.5, 10.0),
        km_bounds=(0.1, 10.0),
        de_maxiter=10,
        de_popsize=8,
        grid_points=7,
        grid_span=0.5,
        dt=0.5,
        seed=0,
    )

    assert fit.r_squared > 0.98
    assert fit.params.vmax == pytest.approx(vmax_true, rel=0.3)
    assert fit.params.km == pytest.approx(km_true, rel=0.5)


def test_fit_grid_has_correct_shape_and_axes() -> None:
    vmax_true, km_true = 3.0, 1.5
    t_obs, biomass_obs = _generate_synthetic_biomass(vmax_true, km_true, t_total=4.0, dt=0.5)

    model = _toy_glucose_model()
    medium = _single_carbon_medium(10.0)
    apply_medium(model, medium, close_others=False)

    fit = fit_kinetics(
        model,
        medium,
        carbon_exchange="EX_glc__D_e",
        t_obs=t_obs,
        biomass_obs=biomass_obs,
        vmax_bounds=(0.5, 10.0),
        km_bounds=(0.1, 10.0),
        de_maxiter=5,
        de_popsize=5,
        grid_points=5,
        dt=0.5,
        seed=0,
    )

    assert fit.grid_r_squared.shape == (5, 5)
    assert fit.grid_vmax_axis.shape == (5,)
    assert fit.grid_km_axis.shape == (5,)
    assert fit.grid_vmax_axis[0] < fit.grid_vmax_axis[-1]
    assert fit.grid_km_axis[0] < fit.grid_km_axis[-1]
    assert np.isfinite(fit.grid_r_squared).any()


def test_fit_validates_inputs() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(10.0)
    apply_medium(model, medium, close_others=False)
    t = np.array([0.0, 1.0, 2.0])
    b = np.array([0.01, 0.02, 0.04])

    with pytest.raises(KeyError, match="pool component"):
        fit_kinetics(model, medium, "EX_ac_e", t, b)

    with pytest.raises(ValueError, match="t_obs and biomass_obs"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, np.array([0.01, 0.02]))

    with pytest.raises(ValueError, match="t_obs\\[0\\] must be 0"):
        fit_kinetics(model, medium, "EX_glc__D_e", np.array([1.0, 2.0, 3.0]), b)

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_kinetics(model, medium, "EX_glc__D_e", np.array([0.0, 2.0, 1.0]), b)

    with pytest.raises(ValueError, match="at least 3"):
        fit_kinetics(model, medium, "EX_glc__D_e", np.array([0.0, 1.0]), np.array([0.01, 0.02]))

    with pytest.raises(ValueError, match="vmax_bounds"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, b, vmax_bounds=(0.0, 10.0))

    with pytest.raises(ValueError, match="km_bounds"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, b, km_bounds=(5.0, 1.0))

    with pytest.raises(ValueError, match="grid_points"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, b, grid_points=2)

    with pytest.raises(ValueError, match="grid_span"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, b, grid_span=0.0)


def test_fit_rejects_constant_biomass_obs() -> None:
    model = _toy_glucose_model()
    medium = _single_carbon_medium(10.0)
    apply_medium(model, medium, close_others=False)

    t = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.full_like(t, 0.01)
    with pytest.raises(ValueError, match="constant"):
        fit_kinetics(model, medium, "EX_glc__D_e", t, b)

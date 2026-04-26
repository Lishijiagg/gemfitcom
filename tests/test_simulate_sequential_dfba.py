"""Tests for simulate.sequential_dfba against toy multi-strain cobra models."""

from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.medium import Medium, apply_medium
from gemfitcom.simulate import SequentialDFBAResult, StrainSpec, simulate_sequential_dfba

Y_GLC: float = 0.1  # gDW per mmol glucose
Y_AC: float = 0.05  # gDW per mmol acetate
ALPHA_AC: float = 2.0  # mmol acetate produced per gDW biomass (strain A)


def _strain_a_glc_to_ac() -> Model:
    """Strain A: 10 mmol glucose → 1 gDW biomass + 20 mmol acetate (secreted)."""
    m = Model("strainA")
    glc = Metabolite("glc__D_e", compartment="e")
    ac = Metabolite("ac_e", compartment="e")
    bio = Metabolite("biomass_c", compartment="c")

    ex_glc = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc: -1})
    ex_ac = Reaction("EX_ac_e", lower_bound=0.0, upper_bound=1000.0)
    ex_ac.add_metabolites({ac: -1})

    biomass_rxn = Reaction("BIOMASS_A", lower_bound=0.0, upper_bound=1000.0)
    biomass_rxn.add_metabolites({glc: -1.0 / Y_GLC, ac: ALPHA_AC, bio: 1.0})

    sink = Reaction("EX_biomass_A", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})

    m.add_reactions([ex_glc, ex_ac, biomass_rxn, sink])
    m.objective = "BIOMASS_A"
    return m


def _strain_b_ac_to_biomass() -> Model:
    """Strain B: 20 mmol acetate → 1 gDW biomass (no glucose machinery)."""
    m = Model("strainB")
    ac = Metabolite("ac_e", compartment="e")
    bio = Metabolite("biomass_B_c", compartment="c")

    ex_ac = Reaction("EX_ac_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_ac.add_metabolites({ac: -1})

    biomass_rxn = Reaction("BIOMASS_B", lower_bound=0.0, upper_bound=1000.0)
    biomass_rxn.add_metabolites({ac: -1.0 / Y_AC, bio: 1.0})

    sink = Reaction("EX_biomass_B", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})

    m.add_reactions([ex_ac, biomass_rxn, sink])
    m.objective = "BIOMASS_B"
    return m


def _glc_only_medium(initial_glc: float = 10.0) -> Medium:
    return Medium(
        name="toyCF",
        pool_components={"EX_glc__D_e": initial_glc, "EX_ac_e": 0.0},
        unlimited_components=frozenset(),
    )


def _two_strain_community(
    glc_mm: MMParams | None = None,
    ac_mm: MMParams | None = None,
    xa: float = 0.01,
    xb: float = 0.01,
) -> tuple[list[StrainSpec], Medium]:
    medium = _glc_only_medium()
    a = _strain_a_glc_to_ac()
    b = _strain_b_ac_to_biomass()
    apply_medium(a, medium, close_others=False)
    apply_medium(b, medium, close_others=False, on_missing="ignore")
    strains = [
        StrainSpec(
            name="A",
            model=a,
            mm_params={"EX_glc__D_e": glc_mm or MMParams(vmax=5.0, km=1.0)},
            initial_biomass=xa,
        ),
        StrainSpec(
            name="B",
            model=b,
            mm_params={"EX_ac_e": ac_mm or MMParams(vmax=3.0, km=0.5)},
            initial_biomass=xb,
        ),
    ]
    return strains, medium


# ---------- shape / initial-condition tests ----------


def test_shapes_and_initial_condition() -> None:
    strains, medium = _two_strain_community()
    res = simulate_sequential_dfba(strains, medium, t_total=4.0, dt=0.5)

    assert isinstance(res, SequentialDFBAResult)
    assert res.time_h.shape == (9,)
    assert res.time_h[0] == 0.0
    assert res.time_h[-1] == 4.0

    assert list(res.biomass.columns) == ["time_h", "A", "B"]
    assert res.biomass["A"].iloc[0] == pytest.approx(0.01)
    assert res.biomass["B"].iloc[0] == pytest.approx(0.01)

    assert list(res.pool.columns) == ["time_h", "EX_glc__D_e", "EX_ac_e"]
    assert res.pool["EX_glc__D_e"].iloc[0] == pytest.approx(10.0)
    assert res.pool["EX_ac_e"].iloc[0] == pytest.approx(0.0)

    assert list(res.growth_rate.columns) == ["time_h", "A", "B"]


# ---------- mechanistic tests ----------


def test_cross_feeding_b_grows_on_a_secreted_acetate() -> None:
    strains, medium = _two_strain_community(xa=0.05, xb=0.01)
    res = simulate_sequential_dfba(strains, medium, t_total=8.0, dt=0.25)

    # A grows on glucose.
    assert res.biomass["A"].iloc[-1] > res.biomass["A"].iloc[0]
    # Glucose depletes.
    assert res.pool["EX_glc__D_e"].iloc[-1] < 10.0
    # Acetate accumulates (secreted by A) even though it started at 0.
    assert res.pool["EX_ac_e"].max() > 0.0
    # B grows despite starting with zero acetate — only possible if A fed it.
    assert res.biomass["B"].iloc[-1] > res.biomass["B"].iloc[0]


def test_strain_a_initial_growth_rate_matches_mm() -> None:
    strains, medium = _two_strain_community(glc_mm=MMParams(vmax=5.0, km=1.0))
    res = simulate_sequential_dfba(strains, medium, t_total=2.0, dt=0.5)
    expected_mu_a = Y_GLC * 5.0 * 10.0 / (1.0 + 10.0)
    assert res.growth_rate["A"].iloc[0] == pytest.approx(expected_mu_a, rel=1e-3)
    # B sees zero acetate initially → zero growth.
    assert res.growth_rate["B"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_mass_balance_glucose_matches_a_biomass() -> None:
    strains, medium = _two_strain_community(xa=0.05, xb=0.0001)
    res = simulate_sequential_dfba(strains, medium, t_total=5.0, dt=0.1)
    # Glucose consumed should equal A's biomass growth divided by its yield
    # (B doesn't touch glucose).
    delta_a = res.biomass["A"].iloc[-1] - res.biomass["A"].iloc[0]
    delta_glc = res.pool["EX_glc__D_e"].iloc[0] - res.pool["EX_glc__D_e"].iloc[-1]
    assert delta_glc == pytest.approx(delta_a / Y_GLC, rel=0.05)


def test_pool_concentrations_nonnegative() -> None:
    strains, medium = _two_strain_community()
    res = simulate_sequential_dfba(strains, medium, t_total=20.0, dt=0.25)
    assert (res.pool["EX_glc__D_e"] >= 0).all()
    assert (res.pool["EX_ac_e"] >= 0).all()


# ---------- strain-without-pool-exchange case ----------


def test_strain_missing_all_pool_exchanges_is_inert() -> None:
    """A strain whose model has no pool exchanges contributes nothing."""
    inert_model = Model("inert")
    bio = Metabolite("bio_c", compartment="c")
    biomass_rxn = Reaction("BIOMASS_I", lower_bound=0.0, upper_bound=0.0)
    biomass_rxn.add_metabolites({bio: 1.0})
    sink = Reaction("EX_bio_i", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    inert_model.add_reactions([biomass_rxn, sink])
    inert_model.objective = "BIOMASS_I"

    medium = _glc_only_medium()
    a = _strain_a_glc_to_ac()
    apply_medium(a, medium, close_others=False)
    strains = [
        StrainSpec(
            name="A",
            model=a,
            mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
            initial_biomass=0.01,
        ),
        StrainSpec(
            name="Inert",
            model=inert_model,
            mm_params={},
            initial_biomass=0.01,
        ),
    ]
    res = simulate_sequential_dfba(strains, medium, t_total=3.0, dt=0.5)
    # Inert strain never grows.
    assert np.allclose(res.biomass["Inert"].to_numpy(), 0.01)
    # A still behaves normally.
    assert res.biomass["A"].iloc[-1] > res.biomass["A"].iloc[0]


# ---------- validation tests ----------


def test_rejects_empty_strain_list() -> None:
    medium = _glc_only_medium()
    with pytest.raises(ValueError, match="non-empty"):
        simulate_sequential_dfba([], medium, t_total=1.0, dt=0.5)


def test_rejects_duplicate_strain_names() -> None:
    medium = _glc_only_medium()
    a = _strain_a_glc_to_ac()
    apply_medium(a, medium, close_others=False)
    a2 = _strain_a_glc_to_ac()
    apply_medium(a2, medium, close_others=False)
    strains = [
        StrainSpec(name="A", model=a, mm_params={}, initial_biomass=0.01),
        StrainSpec(name="A", model=a2, mm_params={}, initial_biomass=0.01),
    ]
    with pytest.raises(ValueError, match="unique"):
        simulate_sequential_dfba(strains, medium, t_total=1.0, dt=0.5)


def test_rejects_bad_timing() -> None:
    strains, medium = _two_strain_community()
    with pytest.raises(ValueError, match="t_total"):
        simulate_sequential_dfba(strains, medium, t_total=0.0, dt=0.5)
    with pytest.raises(ValueError, match="dt"):
        simulate_sequential_dfba(strains, medium, t_total=1.0, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        simulate_sequential_dfba(strains, medium, t_total=1.0, dt=2.0)


def test_rejects_mm_param_not_in_medium() -> None:
    medium = _glc_only_medium()
    a = _strain_a_glc_to_ac()
    apply_medium(a, medium, close_others=False)
    strains = [
        StrainSpec(
            name="A",
            model=a,
            mm_params={"EX_xyl__D_e": MMParams(vmax=1.0, km=1.0)},
            initial_biomass=0.01,
        ),
    ]
    with pytest.raises(KeyError, match="not a pool component"):
        simulate_sequential_dfba(strains, medium, t_total=1.0, dt=0.5)


def test_rejects_pool_init_key_not_in_medium() -> None:
    strains, medium = _two_strain_community()
    with pytest.raises(KeyError, match="not a pool component"):
        simulate_sequential_dfba(
            strains,
            medium,
            t_total=1.0,
            dt=0.5,
            pool_init={"EX_not_here_e": 1.0},
        )


def test_pool_init_override() -> None:
    strains, medium = _two_strain_community()
    res = simulate_sequential_dfba(
        strains,
        medium,
        t_total=1.0,
        dt=0.5,
        pool_init={"EX_ac_e": 3.0},
    )
    assert res.pool["EX_ac_e"].iloc[0] == pytest.approx(3.0)


def test_strainspec_rejects_bad_initial_biomass() -> None:
    a = _strain_a_glc_to_ac()
    with pytest.raises(ValueError, match="initial_biomass"):
        StrainSpec(
            name="A",
            model=a,
            mm_params={},
            initial_biomass=0.0,
        )


def test_strainspec_rejects_empty_name() -> None:
    a = _strain_a_glc_to_ac()
    with pytest.raises(ValueError, match="name"):
        StrainSpec(name="", model=a, mm_params={}, initial_biomass=0.01)


# ---------- save_fluxes tests ----------


def test_save_fluxes_default_is_none() -> None:
    strains, medium = _two_strain_community()
    res = simulate_sequential_dfba(strains, medium, t_total=2.0, dt=0.5)
    assert res.exchange_fluxes is None


def test_save_fluxes_true_populates_long_frame() -> None:
    strains, medium = _two_strain_community(xa=0.05, xb=0.01)
    res = simulate_sequential_dfba(strains, medium, t_total=4.0, dt=0.5, save_fluxes=True)

    assert res.exchange_fluxes is not None
    ef = res.exchange_fluxes
    assert list(ef.columns) == ["time_h", "strain", "exchange_id", "flux"]
    # 9 time points * 2 strains * 2 pool metabolites = 36 rows.
    n_points = len(res.time_h)
    assert len(ef) == n_points * 2 * 2
    assert set(ef["strain"].unique()) == {"A", "B"}
    assert set(ef["exchange_id"].unique()) == {"EX_glc__D_e", "EX_ac_e"}


def test_save_fluxes_signs_and_cross_feeding() -> None:
    """A secretes acetate (+flux on EX_ac_e); B uptakes it (-flux)."""
    strains, medium = _two_strain_community(xa=0.05, xb=0.01)
    res = simulate_sequential_dfba(strains, medium, t_total=8.0, dt=0.25, save_fluxes=True)
    ef = res.exchange_fluxes
    assert ef is not None

    # Exclude initial (t=0) row where growth hasn't happened yet.
    a_ac = ef[(ef["strain"] == "A") & (ef["exchange_id"] == "EX_ac_e") & (ef["time_h"] > 0)]
    b_ac = ef[(ef["strain"] == "B") & (ef["exchange_id"] == "EX_ac_e") & (ef["time_h"] > 0)]
    # A consumes glucose, never acetate — its EX_ac_e flux must be >= 0.
    assert (a_ac["flux"] >= -1e-9).all()
    assert (a_ac["flux"] > 1e-6).any()
    # B consumes acetate — negative flux at least once during the trajectory.
    assert (b_ac["flux"] < -1e-6).any()

    # A never touches glucose uptake as positive.
    a_glc = ef[(ef["strain"] == "A") & (ef["exchange_id"] == "EX_glc__D_e") & (ef["time_h"] > 0)]
    assert (a_glc["flux"] < 0).any()


def test_save_fluxes_final_row_duplicates_previous() -> None:
    strains, medium = _two_strain_community()
    res = simulate_sequential_dfba(strains, medium, t_total=2.0, dt=0.5, save_fluxes=True)
    ef = res.exchange_fluxes
    assert ef is not None
    t_last = res.time_h[-1]
    t_prev = res.time_h[-2]
    last = ef[ef["time_h"] == t_last].sort_values(["strain", "exchange_id"])["flux"].to_numpy()
    prev = ef[ef["time_h"] == t_prev].sort_values(["strain", "exchange_id"])["flux"].to_numpy()
    np.testing.assert_allclose(last, prev)

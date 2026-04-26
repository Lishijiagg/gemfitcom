"""Tests for simulate.fusion — dynamic MICOM (dMICOM)."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.medium import Medium
from gemfitcom.simulate import CommunityMember, FusionResult, simulate_fusion_dmicom

Y_GLC: float = 0.1  # gDW / mmol glucose
Y_AC: float = 0.05  # gDW / mmol acetate


def _strain_on_glucose(name: str) -> Model:
    m = Model(name)
    glc = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex.add_metabolites({glc: -1})
    br = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    br.add_metabolites({glc: -1.0 / Y_GLC, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, br, sink])
    m.objective = "BIOMASS"
    return m


def _strain_on_acetate(name: str) -> Model:
    m = Model(name)
    ac = Metabolite("ac_e", compartment="e", formula="C2H3O2")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_ac_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex.add_metabolites({ac: -1})
    br = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    br.add_metabolites({ac: -1.0 / Y_AC, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, br, sink])
    m.objective = "BIOMASS"
    return m


def _glc_medium() -> Medium:
    return Medium(
        name="toy",
        pool_components={"EX_glc__D_e": 10.0},
        unlimited_components=frozenset(),
    )


# ---------- shapes / initial conditions ----------


def test_shapes_and_initial_condition() -> None:
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 0.01),
        CommunityMember("B", _strain_on_glucose("B"), 0.01),
    ]
    mm = {n: {"EX_glc__D_e": MMParams(5.0, 1.0)} for n in ("A", "B")}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=2.0,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pfba=False,
    )
    assert isinstance(res, FusionResult)
    assert res.time_h.shape == (5,)
    assert res.time_h[0] == 0.0
    assert res.time_h[-1] == 2.0
    assert list(res.biomass.columns) == ["time_h", "A", "B"]
    assert res.biomass["A"].iloc[0] == pytest.approx(0.01)
    assert res.biomass["B"].iloc[0] == pytest.approx(0.01)
    assert res.pool["EX_glc__D_e"].iloc[0] == pytest.approx(10.0)
    assert res.fail_count == 0
    assert res.community_growth_rate.shape == (5,)


# ---------- mass balance + symmetry ----------


def test_symmetric_community_grows_equally_and_balances_glucose() -> None:
    """Two identical members should have indistinguishable trajectories, and
    Δglucose should equal Δbiomass_total / yield."""
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 0.01),
        CommunityMember("B", _strain_on_glucose("B"), 0.01),
    ]
    mm = {n: {"EX_glc__D_e": MMParams(5.0, 1.0)} for n in ("A", "B")}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=5.0,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pfba=False,
    )
    # Symmetry.
    assert res.biomass["A"].iloc[-1] == pytest.approx(res.biomass["B"].iloc[-1], rel=1e-6)
    # Mass balance.
    delta_biomass = (
        res.biomass["A"].iloc[-1]
        + res.biomass["B"].iloc[-1]
        - res.biomass["A"].iloc[0]
        - res.biomass["B"].iloc[0]
    )
    delta_glc = res.pool["EX_glc__D_e"].iloc[0] - res.pool["EX_glc__D_e"].iloc[-1]
    assert delta_glc == pytest.approx(delta_biomass / Y_GLC, rel=0.02)


# ---------- cross-feeding emerges dynamically ----------


def test_cross_feeding_consumer_grows_on_producer_secreted_acetate() -> None:
    """Producer P ferments glucose → biomass + 2 acetate. Consumer C has
    no glucose machinery. Initial acetate pool is 0. Under dMICOM with
    fraction<1, C should gain biomass because P secretes acetate into the
    shared internal community pool."""
    prod = Model("P")
    glc = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    ac = Metabolite("ac_e", compartment="e", formula="C2H3O2")
    bio = Metabolite("biomass_c", compartment="c")
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-1000, upper_bound=1000)
    ex_glc.add_metabolites({glc: -1})
    ex_ac = Reaction("EX_ac_e", lower_bound=0, upper_bound=1000)
    ex_ac.add_metabolites({ac: -1})
    br = Reaction("BIOMASS", lower_bound=0, upper_bound=1000)
    br.add_metabolites({glc: -1.0 / Y_GLC, ac: 2.0, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0, upper_bound=1000)
    sink.add_metabolites({bio: -1})
    prod.add_reactions([ex_glc, ex_ac, br, sink])
    prod.objective = "BIOMASS"

    cons = _strain_on_acetate("C")

    medium = Medium(
        name="cf",
        pool_components={"EX_glc__D_e": 10.0, "EX_ac_e": 0.0},
        unlimited_components=frozenset(),
    )
    mm = {
        "P": {"EX_glc__D_e": MMParams(5.0, 1.0)},
        "C": {"EX_ac_e": MMParams(3.0, 0.5)},
    }

    res = simulate_fusion_dmicom(
        [CommunityMember("P", prod, 0.02), CommunityMember("C", cons, 0.005)],
        medium,
        t_total=8.0,
        dt=0.5,
        fraction=0.5,
        mm_params_by_member=mm,
        pfba=False,
    )
    assert res.fail_count == 0
    # P grows on glucose.
    assert res.biomass["P"].iloc[-1] > res.biomass["P"].iloc[0]
    # Glucose depletes.
    assert res.pool["EX_glc__D_e"].iloc[-1] < 10.0
    # Acetate accumulates (P secretes) even though it started at 0.
    assert res.pool["EX_ac_e"].max() > 0.0
    # C grows despite starting with zero acetate — only possible via cross-feeding.
    assert res.biomass["C"].iloc[-1] > res.biomass["C"].iloc[0]


# ---------- fraction parameter ----------


def test_fraction_between_zero_and_one_runs_cleanly() -> None:
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 0.01),
        CommunityMember("B", _strain_on_glucose("B"), 0.01),
    ]
    mm = {n: {"EX_glc__D_e": MMParams(5.0, 1.0)} for n in ("A", "B")}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=1.0,
        dt=0.5,
        fraction=0.5,
        mm_params_by_member=mm,
        pfba=False,
    )
    assert res.fail_count == 0
    assert res.biomass["A"].iloc[-1] > 0.01
    assert res.biomass["B"].iloc[-1] > 0.01


# ---------- progress ----------


def test_progress_true_does_not_break() -> None:
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 0.01),
        CommunityMember("B", _strain_on_glucose("B"), 0.01),
    ]
    mm = {n: {"EX_glc__D_e": MMParams(5.0, 1.0)} for n in ("A", "B")}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=1.0,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pfba=False,
        progress=True,
    )
    assert res.fail_count == 0


# ---------- validation ----------


def test_rejects_empty_members() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        simulate_fusion_dmicom([], _glc_medium(), t_total=1.0, dt=0.5)


def test_rejects_duplicate_names() -> None:
    a1 = CommunityMember("A", _strain_on_glucose("A"), 0.01)
    a2 = CommunityMember("A", _strain_on_glucose("A"), 0.01)
    with pytest.raises(ValueError, match="unique"):
        simulate_fusion_dmicom([a1, a2], _glc_medium(), t_total=1.0, dt=0.5)


def test_rejects_bad_timing() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    with pytest.raises(ValueError, match="t_total"):
        simulate_fusion_dmicom(members, _glc_medium(), t_total=0.0, dt=0.5)
    with pytest.raises(ValueError, match="dt"):
        simulate_fusion_dmicom(members, _glc_medium(), t_total=1.0, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        simulate_fusion_dmicom(members, _glc_medium(), t_total=1.0, dt=2.0)


def test_rejects_bad_fraction() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    with pytest.raises(ValueError, match="fraction"):
        simulate_fusion_dmicom(members, _glc_medium(), t_total=1.0, dt=0.5, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        simulate_fusion_dmicom(members, _glc_medium(), t_total=1.0, dt=0.5, fraction=1.1)


def test_rejects_mm_params_unknown_member() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    with pytest.raises(KeyError, match="not in members"):
        simulate_fusion_dmicom(
            members,
            _glc_medium(),
            t_total=1.0,
            dt=0.5,
            mm_params_by_member={"Nonexistent": {"EX_glc__D_e": MMParams(1.0, 1.0)}},
        )


def test_rejects_mm_params_unknown_exchange() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    with pytest.raises(KeyError, match="not a pool component"):
        simulate_fusion_dmicom(
            members,
            _glc_medium(),
            t_total=1.0,
            dt=0.5,
            mm_params_by_member={"A": {"EX_xyl__D_e": MMParams(1.0, 1.0)}},
        )


def test_pool_init_override() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    mm = {"A": {"EX_glc__D_e": MMParams(5.0, 1.0)}}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=0.5,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pool_init={"EX_glc__D_e": 2.0},
        pfba=False,
    )
    assert res.pool["EX_glc__D_e"].iloc[0] == pytest.approx(2.0)


# ---------- save_fluxes tests ----------


def test_save_fluxes_default_is_none() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 0.01)]
    mm = {"A": {"EX_glc__D_e": MMParams(5.0, 1.0)}}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=1.0,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pfba=False,
    )
    assert res.exchange_fluxes is None


def test_save_fluxes_true_populates_long_frame() -> None:
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 0.01),
        CommunityMember("B", _strain_on_glucose("B"), 0.01),
    ]
    mm = {n: {"EX_glc__D_e": MMParams(5.0, 1.0)} for n in ("A", "B")}
    res = simulate_fusion_dmicom(
        members,
        _glc_medium(),
        t_total=2.0,
        dt=0.5,
        fraction=1.0,
        mm_params_by_member=mm,
        pfba=False,
        save_fluxes=True,
    )
    ef = res.exchange_fluxes
    assert ef is not None
    assert list(ef.columns) == ["time_h", "strain", "exchange_id", "flux"]
    n_points = len(res.time_h)
    # 2 members * 1 pool metabolite.
    assert len(ef) == n_points * 2 * 1
    assert set(ef["strain"].unique()) == {"A", "B"}
    assert set(ef["exchange_id"].unique()) == {"EX_glc__D_e"}
    # Glucose uptake flux should be negative once growth starts.
    g_a = ef[(ef["strain"] == "A") & (ef["time_h"] > 0)]["flux"]
    assert (g_a < -1e-6).any()

"""Tests for simulate.micom (cooperative-tradeoff steady-state FBA wrapper)."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.kinetics.mm import MMParams
from gemfitcom.medium import Medium
from gemfitcom.simulate import (
    CommunityMember,
    MICOMResult,
    simulate_micom,
)

Y_GLC: float = 0.1


def _strain_on_glucose(name: str) -> Model:
    """Minimal model: 10 mmol glucose → 1 gDW biomass."""
    m = Model(name)
    glc = Metabolite("glc__D_e", compartment="e", formula="C6H12O6")
    bio = Metabolite("biomass_c", compartment="c")
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc: -1})
    biomass_rxn = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    biomass_rxn.add_metabolites({glc: -1.0 / Y_GLC, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex_glc, biomass_rxn, sink])
    m.objective = "BIOMASS"
    return m


def _strain_on_acetate(name: str, ac_yield: float = 0.05) -> Model:
    """Consumes acetate → biomass; no glucose machinery."""
    m = Model(name)
    ac = Metabolite("ac_e", compartment="e", formula="C2H3O2")
    bio = Metabolite("biomass_c", compartment="c")
    ex = Reaction("EX_ac_e", lower_bound=-1000.0, upper_bound=1000.0)
    ex.add_metabolites({ac: -1})
    br = Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    br.add_metabolites({ac: -1.0 / ac_yield, bio: 1.0})
    sink = Reaction("EX_biomass", lower_bound=0.0, upper_bound=1000.0)
    sink.add_metabolites({bio: -1})
    m.add_reactions([ex, br, sink])
    m.objective = "BIOMASS"
    return m


def _glc_medium() -> Medium:
    return Medium(
        name="mini",
        pool_components={"EX_glc__D_e": 10.0},
        unlimited_components=frozenset(),
    )


# ---------- basic run / result structure ----------


def test_returns_micom_result_with_expected_fields() -> None:
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 1.0),
        CommunityMember("B", _strain_on_glucose("B"), 1.0),
    ]
    res = simulate_micom(
        members,
        _glc_medium(),
        fraction=0.5,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        pfba=False,
    )
    assert isinstance(res, MICOMResult)
    assert res.status == "optimal"
    assert set(res.member_growth_rate.index) == {"A", "B"}
    assert res.community_growth_rate > 0
    # Fluxes frame rows include members + 'medium'.
    assert set(res.fluxes.index) == {"A", "B", "medium"}


def test_community_growth_matches_mm_analytical() -> None:
    """With two identical members sharing glucose, community μ equals the
    single-organism MM rate of glucose uptake times yield (each takes half)."""
    members = [
        CommunityMember("A", _strain_on_glucose("A"), 1.0),
        CommunityMember("B", _strain_on_glucose("B"), 1.0),
    ]
    vmax, km, conc = 5.0, 1.0, 10.0
    res = simulate_micom(
        members,
        _glc_medium(),
        fraction=1.0,  # force community max-growth
        mm_params={"EX_glc__D_e": MMParams(vmax=vmax, km=km)},
        pool_init={"EX_glc__D_e": conc},
        pfba=False,
    )
    # Total glucose uptake capped at MM rate for community of 1 gDW;
    # community μ = yield * mm / 1 gDW = yield * vmax*S/(Km+S).
    expected = Y_GLC * vmax * conc / (km + conc)
    assert res.community_growth_rate == pytest.approx(expected, rel=1e-3)


def test_fraction_below_one_spreads_growth_across_members() -> None:
    """At fraction=0.5 with unequal strains, both members should be growing
    (no strain parked at zero) because the L2-minimization redistributes."""
    strong = _strain_on_glucose("Strong")
    weak = _strain_on_glucose("Weak")
    # Cripple 'Weak' by halving its biomass yield.
    weak.reactions.BIOMASS.add_metabolites({weak.metabolites.glc__D_e: -1.0 / Y_GLC}, combine=True)
    members = [
        CommunityMember("Strong", strong, 1.0),
        CommunityMember("Weak", weak, 1.0),
    ]
    res = simulate_micom(
        members,
        _glc_medium(),
        fraction=0.5,
        mm_params={"EX_glc__D_e": MMParams(vmax=5.0, km=1.0)},
        pfba=False,
    )
    assert res.member_growth_rate["Strong"] > 0
    assert res.member_growth_rate["Weak"] > 0


# ---------- uptake vs mm vs default precedence ----------


def test_explicit_uptake_overrides_mm_params() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    medium = _glc_medium()
    res_tight = simulate_micom(
        members,
        medium,
        fraction=1.0,
        uptake={"EX_glc__D_e": 0.5},  # tight cap
        mm_params={"EX_glc__D_e": MMParams(vmax=100.0, km=1.0)},  # would be ~99
        pfba=False,
    )
    # Community growth rate = yield * 0.5 = 0.05 (uptake wins).
    assert res_tight.community_growth_rate == pytest.approx(Y_GLC * 0.5, rel=1e-3)


def test_default_uptake_used_when_no_spec_given() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    res = simulate_micom(
        members,
        _glc_medium(),
        fraction=1.0,
        default_uptake=2.0,
        pfba=False,
    )
    # Community growth rate = yield * default_uptake.
    assert res.community_growth_rate == pytest.approx(Y_GLC * 2.0, rel=1e-3)


# ---------- cross-feeding (structural) ----------


def test_cross_feeding_partner_lifted_by_producer_secretion() -> None:
    """Strain P ferments glucose into acetate (product); strain C lives on
    acetate. With fraction<1, MICOM's cooperative tradeoff should give both
    members a positive growth rate."""
    # Producer: glucose -> biomass + 2 acetate (stoichiometrically coupled).
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

    cons = _strain_on_acetate("C", ac_yield=0.05)

    medium = Medium(
        name="cf",
        pool_components={"EX_glc__D_e": 10.0, "EX_ac_e": 0.0},
        unlimited_components=frozenset(),
    )

    # No acetate in the medium — C can only grow on P's secreted acetate.
    res = simulate_micom(
        [CommunityMember("P", prod, 1.0), CommunityMember("C", cons, 1.0)],
        medium,
        fraction=0.5,
        uptake={"EX_glc__D_e": 5.0, "EX_ac_e": 0.0},
        pfba=False,
    )
    assert res.status == "optimal"
    assert res.member_growth_rate["P"] > 0
    assert res.member_growth_rate["C"] > 0


# ---------- validation ----------


def test_rejects_empty_members() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        simulate_micom([], _glc_medium())


def test_rejects_duplicate_member_names() -> None:
    a1 = CommunityMember("A", _strain_on_glucose("A"), 1.0)
    a2 = CommunityMember("A", _strain_on_glucose("A"), 1.0)
    with pytest.raises(ValueError, match="unique"):
        simulate_micom([a1, a2], _glc_medium())


def test_rejects_bad_fraction() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    with pytest.raises(ValueError, match="fraction"):
        simulate_micom(members, _glc_medium(), fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        simulate_micom(members, _glc_medium(), fraction=1.5)


def test_rejects_uptake_key_not_in_medium() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    with pytest.raises(KeyError, match="not a pool component"):
        simulate_micom(members, _glc_medium(), uptake={"EX_xyl__D_e": 1.0})


def test_rejects_negative_uptake_value() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    with pytest.raises(KeyError, match=">= 0"):
        simulate_micom(members, _glc_medium(), uptake={"EX_glc__D_e": -1.0})


def test_rejects_bad_default_uptake() -> None:
    members = [CommunityMember("A", _strain_on_glucose("A"), 1.0)]
    with pytest.raises(ValueError, match="default_uptake"):
        simulate_micom(members, _glc_medium(), default_uptake=0.0)


def test_community_member_rejects_bad_abundance() -> None:
    with pytest.raises(ValueError, match="abundance"):
        CommunityMember("A", _strain_on_glucose("A"), 0.0)


def test_community_member_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        CommunityMember("", _strain_on_glucose("A"), 1.0)

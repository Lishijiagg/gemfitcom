"""Tests for gapfill.detect: can_secrete & missing_products on toy cobra models."""

from __future__ import annotations

from cobra import Metabolite, Model, Reaction

from gemfitcom.gapfill import can_secrete, missing_products


def _model_that_secretes_butyrate() -> Model:
    """Toy model: glc_e uptake → glc_c → but_c → but_e → EX_but_e."""
    m = Model("secretes_but")
    glc_e = Metabolite("glc__D_e", compartment="e")
    glc_c = Metabolite("glc__D_c", compartment="c")
    but_c = Metabolite("but_c", compartment="c", formula="C4H7O2", charge=-1)
    but_e = Metabolite("but_e", compartment="e", formula="C4H7O2", charge=-1)

    ex_glc = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})
    glct = Reaction("GLCt", lower_bound=-1000.0, upper_bound=1000.0)
    glct.add_metabolites({glc_e: -1, glc_c: 1})
    conv = Reaction("GLC_TO_BUT", lower_bound=0.0, upper_bound=1000.0)
    conv.add_metabolites({glc_c: -1, but_c: 1})
    butt = Reaction("BUTtex", lower_bound=0.0, upper_bound=1000.0)
    butt.add_metabolites({but_c: -1, but_e: 1})
    ex_but = Reaction("EX_but_e", lower_bound=0.0, upper_bound=1000.0)
    ex_but.add_metabolites({but_e: -1})

    m.add_reactions([ex_glc, glct, conv, butt, ex_but])
    return m


def _model_without_butyrate_transport() -> Model:
    """Toy model: can produce but_c internally but lacks transport + exchange."""
    m = Model("no_transport")
    glc_e = Metabolite("glc__D_e", compartment="e")
    glc_c = Metabolite("glc__D_c", compartment="c")
    but_c = Metabolite("but_c", compartment="c", formula="C4H7O2", charge=-1)

    ex_glc = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})
    glct = Reaction("GLCt", lower_bound=-1000.0, upper_bound=1000.0)
    glct.add_metabolites({glc_e: -1, glc_c: 1})
    conv = Reaction("GLC_TO_BUT", lower_bound=0.0, upper_bound=1000.0)
    conv.add_metabolites({glc_c: -1, but_c: 1})

    m.add_reactions([ex_glc, glct, conv])
    return m


def test_can_secrete_true_when_pathway_exists() -> None:
    m = _model_that_secretes_butyrate()
    assert can_secrete(m, "EX_but_e") is True


def test_can_secrete_false_when_exchange_missing() -> None:
    m = _model_without_butyrate_transport()
    assert can_secrete(m, "EX_but_e") is False


def test_can_secrete_false_when_upper_bound_zero() -> None:
    m = _model_that_secretes_butyrate()
    m.reactions.get_by_id("EX_but_e").upper_bound = 0.0
    assert can_secrete(m, "EX_but_e") is False


def test_can_secrete_false_for_dangling_exchange() -> None:
    """Exchange + extracellular metabolite, but nothing produces it."""
    m = Model("dangling")
    ac_e = Metabolite("ac_e", compartment="e", formula="C2H3O2", charge=-1)
    ex = Reaction("EX_ac_e", lower_bound=0.0, upper_bound=1000.0)
    ex.add_metabolites({ac_e: -1})
    m.add_reactions([ex])
    assert can_secrete(m, "EX_ac_e") is False


def test_can_secrete_restores_objective_and_bounds() -> None:
    m = _model_that_secretes_butyrate()
    # Set an objective that is NOT the exchange we'll probe.
    m.objective = m.reactions.get_by_id("GLC_TO_BUT")
    original_expr = str(m.objective.expression)
    lb_before = m.reactions.get_by_id("EX_but_e").lower_bound
    ub_before = m.reactions.get_by_id("EX_but_e").upper_bound

    can_secrete(m, "EX_but_e")

    assert str(m.objective.expression) == original_expr
    assert m.reactions.get_by_id("EX_but_e").lower_bound == lb_before
    assert m.reactions.get_by_id("EX_but_e").upper_bound == ub_before


def test_missing_products_returns_only_unsecretable() -> None:
    m = _model_that_secretes_butyrate()
    # EX_but_e is secretable; EX_ac_e is not in the model at all; EX_ppa_e same.
    missing = missing_products(m, {"EX_but_e", "EX_ac_e", "EX_ppa_e"})
    assert missing == {"EX_ac_e", "EX_ppa_e"}


def test_missing_products_accepts_list_and_tuple() -> None:
    m = _model_that_secretes_butyrate()
    assert missing_products(m, ["EX_but_e"]) == set()
    assert missing_products(m, ("EX_ac_e",)) == {"EX_ac_e"}

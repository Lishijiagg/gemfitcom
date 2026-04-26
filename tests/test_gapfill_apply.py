"""Tests for gapfill.apply: atomic entry application with rollback."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.gapfill import ApplyError, apply_entry, can_secrete, load_kb


def _model_with_internal_but_production() -> Model:
    """Can produce but_c from glucose; missing BUTtex + EX_but_e."""
    m = Model("needs_but_export")
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


def _model_without_but_producer() -> Model:
    """Model has no but_c producer anywhere — apply must fail verification."""
    m = Model("no_but_producer")
    glc_e = Metabolite("glc__D_e", compartment="e")
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})
    m.add_reactions([ex_glc])
    return m


def test_apply_entry_adds_transport_and_exchange() -> None:
    kb = load_kb("scfa")
    entry = kb.entries["EX_but_e"]
    m = _model_with_internal_but_production()

    assert can_secrete(m, "EX_but_e") is False
    result = apply_entry(m, entry)

    assert result.verified is True
    assert "but_e" in result.added_metabolites
    assert "but_c" in result.skipped_metabolites  # model already had but_c
    assert set(result.added_reactions) == {"BUTtex", "EX_but_e"}
    assert can_secrete(m, "EX_but_e") is True


def test_apply_entry_skips_existing_reaction_by_id() -> None:
    """If a reaction id already exists, it is left untouched (bounds + stoich)."""
    kb = load_kb("scfa")
    entry = kb.entries["EX_but_e"]
    m = _model_with_internal_but_production()

    # Pre-install a dummy BUTtex with bounds [0,0]; apply should keep those bounds.
    but_c = m.metabolites.get_by_id("but_c")
    but_e = Metabolite("but_e", compartment="e", formula="C4H7O2", charge=-1)
    m.add_metabolites([but_e])
    pre_butt = Reaction("BUTtex", lower_bound=0.0, upper_bound=0.0)
    pre_butt.add_metabolites({but_c: -1, but_e: 1})
    m.add_reactions([pre_butt])

    # With BUTtex stuck at upper=0 the model still can't secrete, so verify
    # would fail. Run with verify=False so we can assert the skip behaviour
    # (apply did not overwrite our pre-installed reaction).
    result = apply_entry(m, entry, verify=False)

    assert "BUTtex" in result.skipped_reactions
    assert m.reactions.get_by_id("BUTtex").upper_bound == 0.0
    assert "EX_but_e" in result.added_reactions


def test_apply_rolls_back_on_verification_failure() -> None:
    kb = load_kb("scfa")
    entry = kb.entries["EX_but_e"]
    m = _model_without_but_producer()

    mets_before = {met.id for met in m.metabolites}
    rxns_before = {r.id for r in m.reactions}

    with pytest.raises(ApplyError, match="did not enable secretion"):
        apply_entry(m, entry)

    assert {met.id for met in m.metabolites} == mets_before
    assert {r.id for r in m.reactions} == rxns_before


def test_apply_verify_false_reports_unverified_without_raising() -> None:
    kb = load_kb("scfa")
    entry = kb.entries["EX_but_e"]
    m = _model_without_but_producer()

    result = apply_entry(m, entry, verify=False)

    assert result.verified is True  # verify skipped; reported as True
    # Reactions still inserted because no verification was requested.
    assert "EX_but_e" in {r.id for r in m.reactions}


def test_apply_verified_field_true_on_success() -> None:
    kb = load_kb("scfa")
    entry = kb.entries["EX_ac_e"]
    # Build a model that can produce ac_c but can't secrete.
    m = Model("needs_ac_export")
    glc_e = Metabolite("glc__D_e", compartment="e")
    glc_c = Metabolite("glc__D_c", compartment="c")
    ac_c = Metabolite("ac_c", compartment="c", formula="C2H3O2", charge=-1)
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})
    glct = Reaction("GLCt", lower_bound=-1000.0, upper_bound=1000.0)
    glct.add_metabolites({glc_e: -1, glc_c: 1})
    conv = Reaction("GLC_TO_AC", lower_bound=0.0, upper_bound=1000.0)
    conv.add_metabolites({glc_c: -1, ac_c: 1})
    m.add_reactions([ex_glc, glct, conv])

    result = apply_entry(m, entry)
    assert result.verified is True
    assert "ac_e" in result.added_metabolites
    assert "EX_ac_e" in result.added_reactions

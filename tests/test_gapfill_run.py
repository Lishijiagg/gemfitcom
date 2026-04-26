"""Tests for gapfill.run: top-level orchestration across model sources."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.gapfill import ApplyError, run_gapfill


def _model_needs_butyrate_export(source: str | None = "agora2") -> Model:
    """Can produce but_c from glucose; missing BUTtex + EX_but_e."""
    m = Model("toy_strain")
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
    if source is not None:
        m.annotation["source"] = source
    return m


def _model_already_secretes_butyrate(source: str = "agora2") -> Model:
    m = _model_needs_butyrate_export(source=source)
    but_c = m.metabolites.get_by_id("but_c")
    but_e = Metabolite("but_e", compartment="e", formula="C4H7O2", charge=-1)
    m.add_metabolites([but_e])
    butt = Reaction("BUTtex", lower_bound=0.0, upper_bound=1000.0)
    butt.add_metabolites({but_c: -1, but_e: 1})
    ex = Reaction("EX_but_e", lower_bound=0.0, upper_bound=1000.0)
    ex.add_metabolites({but_e: -1})
    m.add_reactions([butt, ex])
    return m


def _model_without_producer(source: str = "agora2") -> Model:
    m = Model("no_producer")
    glc_e = Metabolite("glc__D_e", compartment="e")
    ex_glc = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glc.add_metabolites({glc_e: -1})
    m.add_reactions([ex_glc])
    m.annotation["source"] = source
    return m


# --- source dispatch --------------------------------------------------------


def test_curated_is_skipped_without_mutation() -> None:
    m = _model_needs_butyrate_export(source="curated")
    rxns_before = {r.id for r in m.reactions}
    mets_before = {met.id for met in m.metabolites}

    report = run_gapfill(m, {"EX_but_e"})

    assert report.skipped is True
    assert report.source == "curated"
    assert report.kb_name == ""
    assert report.outcomes == ()
    assert {r.id for r in m.reactions} == rxns_before
    assert {met.id for met in m.metabolites} == mets_before


def test_source_explicit_overrides_annotation() -> None:
    m = _model_needs_butyrate_export(source="curated")
    report = run_gapfill(m, set(), source="agora2")
    assert report.source == "agora2"
    assert report.skipped is False


def test_missing_source_raises() -> None:
    m = _model_needs_butyrate_export(source=None)
    with pytest.raises(ValueError, match="source is unknown"):
        run_gapfill(m, {"EX_but_e"})


def test_invalid_source_raises() -> None:
    m = _model_needs_butyrate_export(source=None)
    with pytest.raises(ValueError, match="source must be one of"):
        run_gapfill(m, set(), source="gibberish")


# --- per-product outcomes ---------------------------------------------------


def test_agora2_adds_missing_exchange() -> None:
    m = _model_needs_butyrate_export("agora2")

    report = run_gapfill(m, {"EX_but_e"})

    assert report.skipped is False
    assert report.source == "agora2"
    assert report.kb_name == "scfa"
    assert report.products_added == ("EX_but_e",)
    assert "EX_but_e" in report.added_reaction_ids
    assert "BUTtex" in report.added_reaction_ids
    assert "but_e" in report.added_metabolite_ids
    # Model now has the exchange.
    assert "EX_but_e" in {r.id for r in m.reactions}


def test_already_present_is_recorded_and_not_reapplied() -> None:
    m = _model_already_secretes_butyrate("carveme")
    rxns_before = {r.id for r in m.reactions}

    report = run_gapfill(m, {"EX_but_e"})

    assert report.products_already_present == ("EX_but_e",)
    assert report.products_added == ()
    assert {r.id for r in m.reactions} == rxns_before


def test_no_kb_entry_is_warned_and_recorded() -> None:
    m = _model_needs_butyrate_export("agora2")

    with pytest.warns(UserWarning, match="has no recipe"):
        report = run_gapfill(m, {"EX_no_such_product_e"})

    assert report.products_missing_kb == ("EX_no_such_product_e",)
    assert report.products_added == ()


def test_strict_true_raises_on_failed_apply() -> None:
    m = _model_without_producer("agora2")
    with pytest.raises(ApplyError, match="did not enable secretion"):
        run_gapfill(m, {"EX_but_e"}, strict=True)


def test_strict_false_records_failed_outcome() -> None:
    m = _model_without_producer("agora2")
    rxns_before = {r.id for r in m.reactions}

    with pytest.warns(UserWarning, match="failed"):
        report = run_gapfill(m, {"EX_but_e"}, strict=False)

    assert report.products_failed == ("EX_but_e",)
    # Rollback leaves model untouched.
    assert {r.id for r in m.reactions} == rxns_before


def test_multiple_products_mixed_outcomes() -> None:
    m = _model_needs_butyrate_export("agora2")

    with pytest.warns(UserWarning, match="no recipe"):
        report = run_gapfill(m, {"EX_but_e", "EX_no_such_product_e"})

    assert set(report.products_added) == {"EX_but_e"}
    assert set(report.products_missing_kb) == {"EX_no_such_product_e"}


def test_empty_observed_yields_empty_outcomes() -> None:
    m = _model_needs_butyrate_export("agora2")
    report = run_gapfill(m, set())
    assert report.outcomes == ()
    assert report.skipped is False

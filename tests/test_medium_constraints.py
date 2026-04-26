"""Tests for apply_medium against mini COBRA models."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from gemfitcom.medium import (
    DEFAULT_POOL_BOUND,
    DEFAULT_UNLIMITED_BOUND,
    Medium,
    apply_medium,
)


def _mini_model(exchange_ids: list[str]) -> Model:
    model = Model("mini")
    for ex_id in exchange_ids:
        met_id = ex_id[len("EX_") :]
        met = Metabolite(met_id, compartment="e")
        rxn = Reaction(ex_id, lower_bound=-1.0, upper_bound=1000.0)
        rxn.add_metabolites({met: -1})
        model.add_reactions([rxn])
    return model


def _ycfa_like_medium() -> Medium:
    return Medium(
        name="Mini",
        pool_components={"EX_glc__D_e": 5.0, "EX_ac_e": 0.0},
        unlimited_components=frozenset({"EX_h2o_e", "EX_nh4_e"}),
    )


def test_apply_medium_sets_pool_and_unlimited_bounds() -> None:
    model = _mini_model(["EX_glc__D_e", "EX_ac_e", "EX_h2o_e", "EX_nh4_e"])
    medium = _ycfa_like_medium()

    report = apply_medium(model, medium)

    assert model.reactions.get_by_id("EX_glc__D_e").lower_bound == DEFAULT_POOL_BOUND
    assert model.reactions.get_by_id("EX_ac_e").lower_bound == DEFAULT_POOL_BOUND
    assert model.reactions.get_by_id("EX_h2o_e").lower_bound == DEFAULT_UNLIMITED_BOUND
    assert model.reactions.get_by_id("EX_nh4_e").lower_bound == DEFAULT_UNLIMITED_BOUND

    assert set(report.applied_pool) == {"EX_glc__D_e", "EX_ac_e"}
    assert set(report.applied_unlimited) == {"EX_h2o_e", "EX_nh4_e"}
    assert report.missing_pool == ()
    assert report.missing_unlimited == ()


def test_apply_medium_closes_unlisted_exchanges_by_default() -> None:
    model = _mini_model(["EX_glc__D_e", "EX_h2o_e", "EX_xyl__D_e"])
    medium = Medium(
        name="Mini",
        pool_components={"EX_glc__D_e": 5.0},
        unlimited_components=frozenset({"EX_h2o_e"}),
    )

    report = apply_medium(model, medium)

    assert model.reactions.get_by_id("EX_xyl__D_e").lower_bound == 0.0
    assert model.reactions.get_by_id("EX_xyl__D_e").upper_bound == 1000.0
    assert "EX_xyl__D_e" in report.closed


def test_apply_medium_close_others_false_leaves_unlisted_bounds() -> None:
    model = _mini_model(["EX_glc__D_e", "EX_h2o_e", "EX_xyl__D_e"])
    medium = Medium(
        name="Mini",
        pool_components={"EX_glc__D_e": 5.0},
        unlimited_components=frozenset({"EX_h2o_e"}),
    )

    apply_medium(model, medium, close_others=False)
    assert model.reactions.get_by_id("EX_xyl__D_e").lower_bound == -1.0


def test_apply_medium_missing_exchange_warns() -> None:
    model = _mini_model(["EX_glc__D_e"])
    medium = _ycfa_like_medium()

    with pytest.warns(UserWarning, match="not present in model"):
        report = apply_medium(model, medium)

    assert "EX_ac_e" in report.missing_pool
    assert "EX_h2o_e" in report.missing_unlimited


def test_apply_medium_missing_exchange_error_mode() -> None:
    model = _mini_model(["EX_glc__D_e"])
    medium = _ycfa_like_medium()

    with pytest.raises(KeyError, match="not present in model"):
        apply_medium(model, medium, on_missing="error")


def test_apply_medium_missing_exchange_ignore_mode() -> None:
    model = _mini_model(["EX_glc__D_e"])
    medium = _ycfa_like_medium()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = apply_medium(model, medium, on_missing="ignore")

    assert "EX_ac_e" in report.missing_pool


def test_apply_medium_custom_bounds() -> None:
    model = _mini_model(["EX_glc__D_e", "EX_h2o_e"])
    medium = Medium(
        name="Mini",
        pool_components={"EX_glc__D_e": 1.0},
        unlimited_components=frozenset({"EX_h2o_e"}),
    )

    apply_medium(model, medium, default_pool_bound=-5.0, unlimited_bound=-500.0)

    assert model.reactions.get_by_id("EX_glc__D_e").lower_bound == -5.0
    assert model.reactions.get_by_id("EX_h2o_e").lower_bound == -500.0


def test_apply_medium_rejects_positive_bounds() -> None:
    model = _mini_model(["EX_glc__D_e"])
    medium = Medium(
        name="Mini",
        pool_components={"EX_glc__D_e": 1.0},
        unlimited_components=frozenset(),
    )
    with pytest.raises(ValueError, match="default_pool_bound"):
        apply_medium(model, medium, default_pool_bound=1.0)
    with pytest.raises(ValueError, match="unlimited_bound"):
        apply_medium(model, medium, unlimited_bound=1.0)


def test_apply_medium_invalid_on_missing() -> None:
    model = _mini_model(["EX_glc__D_e"])
    medium = _ycfa_like_medium()
    with pytest.raises(ValueError, match="on_missing"):
        apply_medium(model, medium, on_missing="bogus")  # type: ignore[arg-type]


def test_apply_medium_preserves_secretion_upper_bound() -> None:
    model = _mini_model(["EX_glc__D_e", "EX_ac_e"])
    medium = _ycfa_like_medium()
    medium_no_unlim = Medium(
        name="Mini",
        pool_components=medium.pool_components,
        unlimited_components=frozenset(),
    )

    apply_medium(model, medium_no_unlim)

    assert model.reactions.get_by_id("EX_glc__D_e").upper_bound == 1000.0
    assert model.reactions.get_by_id("EX_ac_e").upper_bound == 1000.0

"""Tests for the gap-fill knowledge base: parsing, validation, registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gemfitcom.gapfill import (
    GapfillKB,
    KBError,
    clear_custom_kb_registry,
    kb_from_dict,
    list_kbs,
    load_kb,
    register_kb,
    unregister_kb,
)


@pytest.fixture(autouse=True)
def _clean_kb_registry() -> Iterator[None]:
    clear_custom_kb_registry()
    yield
    clear_custom_kb_registry()


# --- built-in + registry ----------------------------------------------------


def test_builtin_scfa_loads() -> None:
    kb = load_kb("scfa")
    assert isinstance(kb, GapfillKB)
    assert kb.name == "scfa"
    expected = {
        "EX_ac_e",
        "EX_ppa_e",
        "EX_but_e",
        "EX_lac__D_e",
        "EX_lac__L_e",
        "EX_succ_e",
        "EX_for_e",
    }
    assert kb.exchange_ids == expected


def test_list_kbs_includes_builtin() -> None:
    assert "scfa" in list_kbs()


def test_load_kb_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        load_kb("nosuch_kb_123")


def test_register_and_load_custom_kb_from_dict() -> None:
    data = _minimal_kb_dict("custom")
    register_kb("custom", data)
    kb = load_kb("custom")
    assert kb.name == "custom"
    assert "custom" in list_kbs()
    unregister_kb("custom")
    assert "custom" not in list_kbs()


def test_register_kb_accepts_kb_instance() -> None:
    kb = kb_from_dict(_minimal_kb_dict("inmem"))
    register_kb("inmem", kb)
    assert load_kb("inmem") is kb


def test_load_kb_from_path(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(_minimal_kb_dict("fromfile")), encoding="utf-8")
    kb = load_kb(p)
    assert kb.name == "fromfile"


def test_load_kb_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_kb(tmp_path / "missing.yaml")


def test_custom_registration_shadows_builtin() -> None:
    data = _minimal_kb_dict("scfa")
    register_kb("scfa", data)
    kb = load_kb("scfa")
    # The minimal replacement has only one entry.
    assert len(kb.entries) == 1


# --- top-level schema errors ------------------------------------------------


def test_rejects_non_mapping() -> None:
    with pytest.raises(KBError, match="must be a mapping"):
        kb_from_dict(["nope"])  # type: ignore[arg-type]


def test_rejects_missing_name() -> None:
    with pytest.raises(KBError, match="'name'"):
        kb_from_dict({"entries": []})


def test_rejects_empty_entries() -> None:
    with pytest.raises(KBError, match="non-empty list"):
        kb_from_dict({"name": "k", "entries": []})


def test_rejects_duplicate_exchange_id() -> None:
    data = _minimal_kb_dict("dup")
    data["entries"].append(data["entries"][0])
    with pytest.raises(KBError, match="duplicate entry"):
        kb_from_dict(data)


# --- entry-level validation -------------------------------------------------


def test_entry_requires_valid_exchange_id() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["exchange_id"] = "bogus_format"
    with pytest.raises(KBError, match="exchange_id"):
        kb_from_dict(data)


def test_entry_must_declare_exchange_reaction() -> None:
    data = _minimal_kb_dict("k")
    # Drop the exchange reaction, leaving only transport.
    data["entries"][0]["reactions"] = [data["entries"][0]["reactions"][0]]
    with pytest.raises(KBError, match="does not declare its own exchange"):
        kb_from_dict(data)


def test_entry_requires_at_least_one_reaction() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["reactions"] = []
    with pytest.raises(KBError, match="at least one reaction"):
        kb_from_dict(data)


# --- metabolite validation --------------------------------------------------


def test_metabolite_requires_formula() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["metabolites"][0]["formula"] = ""
    with pytest.raises(KBError, match="formula"):
        kb_from_dict(data)


def test_metabolite_requires_integer_charge() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["metabolites"][0]["charge"] = 0.5
    with pytest.raises(KBError, match="charge"):
        kb_from_dict(data)


def test_metabolite_invalid_formula_rejected() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["metabolites"][0]["formula"] = "42invalid"
    with pytest.raises(KBError, match="formula"):
        kb_from_dict(data)


# --- reaction validation ----------------------------------------------------


def test_reaction_references_unknown_metabolite() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["reactions"][0]["equation"] = "absent_c --> ac_e"
    with pytest.raises(KBError, match="not declared"):
        kb_from_dict(data)


def test_reaction_rejects_mass_unbalanced() -> None:
    data = _minimal_kb_dict("k")
    # Break mass balance by changing the extracellular metabolite's formula.
    data["entries"][0]["metabolites"][1]["formula"] = "C3H5O2"
    with pytest.raises(KBError, match="mass-balanced"):
        kb_from_dict(data)


def test_reaction_rejects_charge_unbalanced() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["metabolites"][1]["charge"] = 0  # was -1
    with pytest.raises(KBError, match="charge-balanced"):
        kb_from_dict(data)


def test_reaction_exchange_bypasses_balance_check() -> None:
    # An exchange reaction (empty RHS) never triggers balance checks —
    # this is relied upon by every KB entry's EX_ reaction.
    kb = kb_from_dict(_minimal_kb_dict("k"))
    entry = kb.entries["EX_ac_e"]
    ex_rxn = next(r for r in entry.reactions if r.id == "EX_ac_e")
    assert ex_rxn.is_exchange is True
    assert ex_rxn.reversible is True


def test_reaction_bounds_must_be_ordered() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["reactions"][0]["bounds"] = [10, 0]
    with pytest.raises(KBError, match="lower <= upper"):
        kb_from_dict(data)


def test_equation_coefficient_parsed() -> None:
    # "2 a + b --> 2 c + ..." — coefficients are read as floats; balance
    # must still hold. Verify parse by constructing a balanced reaction.
    data: dict = {
        "name": "coef",
        "entries": [
            {
                "exchange_id": "EX_ac_e",
                "metabolites": [
                    {"id": "ac_c", "compartment": "c", "formula": "C2H3O2", "charge": -1},
                    {"id": "ac_e", "compartment": "e", "formula": "C2H3O2", "charge": -1},
                    {"id": "dimer_c", "compartment": "c", "formula": "C4H6O4", "charge": -2},
                ],
                "reactions": [
                    {
                        "id": "DIMER_SPLIT",
                        "equation": "dimer_c --> 2 ac_c",
                        "bounds": [0, 1000],
                    },
                    {
                        "id": "ACtex",
                        "equation": "ac_c --> ac_e",
                        "bounds": [0, 1000],
                    },
                    {
                        "id": "EX_ac_e",
                        "equation": "ac_e <=>",
                        "bounds": [0, 1000],
                    },
                ],
            }
        ],
    }
    kb = kb_from_dict(data)
    split = kb.entries["EX_ac_e"].reactions[0]
    assert split.stoichiometry == {"dimer_c": -1.0, "ac_c": 2.0}


def test_missing_arrow_rejected() -> None:
    data = _minimal_kb_dict("k")
    data["entries"][0]["reactions"][0]["equation"] = "ac_c ac_e"
    with pytest.raises(KBError, match="<=>"):
        kb_from_dict(data)


# --- helpers ----------------------------------------------------------------


def _minimal_kb_dict(name: str) -> dict:
    """A balanced KB dict with a single acetate entry — useful as a test fixture."""
    return {
        "name": name,
        "version": "test",
        "entries": [
            {
                "exchange_id": "EX_ac_e",
                "display_name": "acetate",
                "metabolites": [
                    {"id": "ac_c", "compartment": "c", "formula": "C2H3O2", "charge": -1},
                    {"id": "ac_e", "compartment": "e", "formula": "C2H3O2", "charge": -1},
                ],
                "reactions": [
                    {
                        "id": "ACtex",
                        "equation": "ac_c --> ac_e",
                        "bounds": [0, 1000],
                    },
                    {
                        "id": "EX_ac_e",
                        "equation": "ac_e <=>",
                        "bounds": [0, 1000],
                    },
                ],
                "references": ["BiGG:ac_c"],
            },
        ],
    }

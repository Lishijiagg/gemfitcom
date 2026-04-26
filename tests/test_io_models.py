"""Tests for the SBML model loader."""

from __future__ import annotations

from pathlib import Path

import cobra
import pytest

from gemfitcom.io.models import load_model


def _minimal_model() -> cobra.Model:
    """Build a tiny cobra.Model with enough structure to survive SBML round-trip."""
    m = cobra.Model("tiny")
    glc = cobra.Metabolite("glc_c", compartment="c")
    pyr = cobra.Metabolite("pyr_c", compartment="c")
    biomass = cobra.Metabolite("biomass_c", compartment="c")
    m.add_metabolites([glc, pyr, biomass])
    r_glyc = cobra.Reaction("GLYC")
    r_glyc.add_metabolites({glc: -1, pyr: 2})
    r_glyc.bounds = (0, 1000)
    r_bio = cobra.Reaction("BIOMASS")
    r_bio.add_metabolites({pyr: -1, biomass: 1})
    r_bio.bounds = (0, 1000)
    ex_glc = cobra.Reaction("EX_glc_e")
    ex_glc.add_metabolites({glc: -1})
    ex_glc.bounds = (-10, 1000)
    m.add_reactions([r_glyc, r_bio, ex_glc])
    m.objective = "BIOMASS"
    return m


def test_load_model_round_trip(tmp_path: Path) -> None:
    original = _minimal_model()
    sbml_path = tmp_path / "tiny.xml"
    cobra.io.write_sbml_model(original, str(sbml_path))

    loaded = load_model(sbml_path)
    assert isinstance(loaded, cobra.Model)
    assert {r.id for r in loaded.reactions} == {r.id for r in original.reactions}
    assert {met.id for met in loaded.metabolites} == {met.id for met in original.metabolites}


def test_load_model_attaches_metadata(tmp_path: Path) -> None:
    sbml_path = tmp_path / "tiny.xml"
    cobra.io.write_sbml_model(_minimal_model(), str(sbml_path))

    loaded = load_model(sbml_path, strain_name="TestStrain", source="agora2")
    assert loaded.annotation["strain_name"] == "TestStrain"
    assert loaded.annotation["source"] == "agora2"


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.xml"
    with pytest.raises(FileNotFoundError):
        load_model(missing)


def test_load_model_accepts_str_path(tmp_path: Path) -> None:
    sbml_path = tmp_path / "tiny.xml"
    cobra.io.write_sbml_model(_minimal_model(), str(sbml_path))
    loaded = load_model(str(sbml_path))
    assert isinstance(loaded, cobra.Model)

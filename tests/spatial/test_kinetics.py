"""Tests for spatial/kinetics.py — ExchangeEntry + ExchangeKinetics + YAML loader + GEM resolver."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial.kinetics import ExchangeEntry, ExchangeKinetics


class TestExchangeKineticsBasics:
    def test_construct_with_two_substrates(self):
        ek = ExchangeKinetics(
            species="ecoli",
            entries=(
                ExchangeEntry(exchange_id="EX_glc__D_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_o2_e", vmax=15.0, km=0.005, mode="uptake_only"),
            ),
        )
        assert ek.species == "ecoli"
        assert ek.n_exchanges == 2
        assert ek.exchange_ids == ("EX_glc__D_e", "EX_o2_e")

    def test_mm_upper_bound_at_saturation(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([1000.0]))
        assert bounds.shape == (1,)
        assert np.isclose(bounds[0], 10.0, rtol=1e-3)

    def test_mm_upper_bound_at_zero(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([0.0]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_monotonic(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        C = np.array([0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
        bounds = np.array([ek.mm_upper_bound(np.array([c]))[0] for c in C])
        assert np.all(np.diff(bounds) > 0)

    def test_mm_upper_bound_negative_input_clipped(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),),
        )
        bounds = ek.mm_upper_bound(np.array([-1e-9]))
        assert bounds[0] == 0.0

    def test_mm_upper_bound_shape_mismatch_raises(self):
        ek = ExchangeKinetics(
            species="x",
            entries=(
                ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                ExchangeEntry(exchange_id="EX_b_e", vmax=20.0, km=1.0, mode="uptake_only"),
            ),
        )
        with pytest.raises(ValueError, match="does not match n_exchanges"):
            ek.mm_upper_bound(np.array([0.5]))

    def test_invalid_vmax_rejected(self):
        with pytest.raises(ValueError, match="vmax"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=0.0, km=0.5, mode="uptake_only")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="weird_mode")

    def test_bidirectional_mode_stored(self):
        e = ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="bidirectional")
        assert e.mode == "bidirectional"

        ek = ExchangeKinetics(species="x", entries=(e,))
        assert ek.entries[0].mode == "bidirectional"
        # mm_upper_bound does NOT consume mode; just returns MM magnitude
        bounds = ek.mm_upper_bound(np.array([1000.0]))
        assert np.isclose(bounds[0], 10.0, rtol=1e-3)

    def test_duplicate_exchange_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ExchangeKinetics(
                species="x",
                entries=(
                    ExchangeEntry(exchange_id="EX_a_e", vmax=10.0, km=0.5, mode="uptake_only"),
                    ExchangeEntry(exchange_id="EX_a_e", vmax=20.0, km=1.0, mode="uptake_only"),
                ),
            )


class TestLoadKineticsYaml:
    def test_load_minimal(self, tmp_path):
        yaml_path = tmp_path / "ecoli.yaml"
        yaml_path.write_text(
            "species: ecoli\n"
            "exchanges:\n"
            "  EX_glc__D_e: {v_max: 10.0, K_m: 0.5}\n"
            "  EX_o2_e:     {v_max: 15.0, K_m: 0.005}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        ek = load_kinetics_yaml(yaml_path)
        assert ek.species == "ecoli"
        assert ek.exchange_ids == ("EX_glc__D_e", "EX_o2_e")
        assert ek.entries[0].vmax == 10.0
        assert ek.entries[1].km == 0.005
        assert all(e.mode == "uptake_only" for e in ek.entries)

    def test_load_with_bidirectional_mode(self, tmp_path):
        yaml_path = tmp_path / "k.yaml"
        yaml_path.write_text(
            "species: x\nexchanges:\n  EX_ac_e: {v_max: 5.0, K_m: 0.1, mode: bidirectional}\n"
        )
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        ek = load_kinetics_yaml(yaml_path)
        assert ek.entries[0].mode == "bidirectional"

    def test_missing_file_raises(self, tmp_path):
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(FileNotFoundError):
            load_kinetics_yaml(tmp_path / "nope.yaml")

    def test_missing_required_field_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("species: x\nexchanges:\n  EX_a_e: {v_max: 1.0}\n")
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(KeyError, match="K_m"):
            load_kinetics_yaml(yaml_path)

    def test_top_level_species_required(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("exchanges:\n  EX_a_e: {v_max: 1.0, K_m: 0.1}\n")
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(KeyError, match="species"):
            load_kinetics_yaml(yaml_path)

    def test_non_string_exchange_id_rejected(self, tmp_path):
        """Numeric YAML keys (typo'd or unquoted) raise TypeError, not silent coerce."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("species: x\nexchanges:\n  123: {v_max: 1.0, K_m: 0.1}\n")
        from gemfitcom.spatial.kinetics import load_kinetics_yaml

        with pytest.raises(TypeError, match="exchange id must be a string"):
            load_kinetics_yaml(yaml_path)


class TestResolveGem:
    def test_resolve_cobra_textbook(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        m = resolve_gem("cobra://textbook")
        assert hasattr(m, "reactions")
        assert hasattr(m, "metabolites")
        assert any(r.id.startswith("EX_") for r in m.reactions)

    def test_resolve_filesystem_path(self, tmp_path):
        import cobra

        from gemfitcom.spatial.kinetics import resolve_gem

        m = cobra.io.load_model("textbook")
        sbml = tmp_path / "core.xml"
        cobra.io.write_sbml_model(m, str(sbml))
        loaded = resolve_gem(str(sbml))
        assert len(loaded.reactions) == len(m.reactions)

    def test_resolve_unknown_cobra_name_raises(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(ValueError, match="cobra"):
            resolve_gem("cobra://this_model_does_not_exist_42")

    def test_resolve_missing_path_raises(self, tmp_path):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(FileNotFoundError):
            resolve_gem(str(tmp_path / "nope.xml"))

    def test_resolve_unknown_scheme_raises(self):
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(ValueError, match="scheme"):
            resolve_gem("http://example.com/model.xml")

    def test_resolve_empty_cobra_name_raises(self):
        """`cobra://` with no name (or whitespace) fails fast, not deep in cobra internals."""
        from gemfitcom.spatial.kinetics import resolve_gem

        with pytest.raises(ValueError, match="Empty cobra model name"):
            resolve_gem("cobra://")
        with pytest.raises(ValueError, match="Empty cobra model name"):
            resolve_gem("cobra://   ")

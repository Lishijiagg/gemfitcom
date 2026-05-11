"""Tests for spatial/_init_fields.py — concentration/biomass initialisation."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.spatial._init_fields import build_field_1d
from gemfitcom.spatial.config import InitConfig, SpeciesInitConfig


class TestBuildField1D:
    def test_uniform(self):
        cfg = InitConfig(mode="uniform", value=2.5)
        field = build_field_1d(cfg, n_grid=10)
        assert field.shape == (10,)
        assert np.all(field == 2.5)

    def test_uniform_zero(self):
        cfg = InitConfig(mode="uniform", value=0.0)
        field = build_field_1d(cfg, n_grid=5)
        assert np.all(field == 0.0)

    def test_gaussian_peak_position(self):
        cfg = SpeciesInitConfig(mode="gaussian", center=0.5, sigma=0.1, peak=1.0)
        field = build_field_1d(cfg, n_grid=11)
        assert np.argmax(field) == 5
        assert np.isclose(field[5], 1.0, rtol=1e-3)

    def test_gaussian_non_negative(self):
        cfg = SpeciesInitConfig(mode="gaussian", center=0.7, sigma=0.05, peak=1.0e-3)
        field = build_field_1d(cfg, n_grid=50)
        assert np.isclose(field.max(), 1.0e-3, rtol=1e-2)
        assert np.all(field >= 0)

    def test_step(self):
        cfg = SpeciesInitConfig(mode="step", center=0.5, peak=1.0)
        field = build_field_1d(cfg, n_grid=10)
        x = np.linspace(0.0, 1.0, 10)
        assert np.all(field[x < 0.5] == 0.0)
        assert np.all(field[x >= 0.5] == 1.0)

    def test_from_array(self, tmp_path):
        arr_path = tmp_path / "init.npy"
        np.save(arr_path, np.linspace(0, 1, 8))
        cfg = SpeciesInitConfig(mode="from_array", path=arr_path)
        field = build_field_1d(cfg, n_grid=8)
        assert np.allclose(field, np.linspace(0, 1, 8))

    def test_from_array_shape_mismatch(self, tmp_path):
        arr_path = tmp_path / "init.npy"
        np.save(arr_path, np.zeros(5))
        cfg = SpeciesInitConfig(mode="from_array", path=arr_path)
        with pytest.raises(ValueError, match="shape"):
            build_field_1d(cfg, n_grid=8)

    def test_missing_required_field_raises(self):
        cfg = SpeciesInitConfig(mode="uniform")
        with pytest.raises(ValueError, match="value"):
            build_field_1d(cfg, n_grid=5)

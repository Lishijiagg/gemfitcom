"""Tests for kinetics.mm (MMParams + michaelis_menten)."""

from __future__ import annotations

import numpy as np
import pytest

from gemfitcom.kinetics.mm import MMParams, michaelis_menten


def test_mmparams_valid() -> None:
    p = MMParams(vmax=5.0, km=1.0)
    assert p.vmax == 5.0
    assert p.km == 1.0


def test_mmparams_rejects_nonpositive_vmax() -> None:
    with pytest.raises(ValueError, match="vmax"):
        MMParams(vmax=0.0, km=1.0)
    with pytest.raises(ValueError, match="vmax"):
        MMParams(vmax=-1.0, km=1.0)


def test_mmparams_rejects_nonpositive_km() -> None:
    with pytest.raises(ValueError, match="km"):
        MMParams(vmax=1.0, km=0.0)
    with pytest.raises(ValueError, match="km"):
        MMParams(vmax=1.0, km=-0.5)


def test_mmparams_is_frozen() -> None:
    p = MMParams(vmax=1.0, km=1.0)
    with pytest.raises((AttributeError, TypeError)):
        p.vmax = 2.0  # type: ignore[misc]


def test_michaelis_menten_scalar_formula() -> None:
    v = michaelis_menten(10.0, vmax=5.0, km=1.0)
    assert v == pytest.approx(5.0 * 10.0 / (1.0 + 10.0))


def test_michaelis_menten_at_km_is_half_vmax() -> None:
    assert michaelis_menten(2.0, vmax=4.0, km=2.0) == pytest.approx(2.0)


def test_michaelis_menten_vectorized() -> None:
    conc = np.array([0.0, 1.0, 5.0, 1e6])
    out = michaelis_menten(conc, vmax=2.0, km=1.0)
    assert out.shape == conc.shape
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(2.0 * 5.0 / 6.0)
    assert out[3] == pytest.approx(2.0, rel=1e-4)  # approaches vmax


def test_michaelis_menten_clips_negative_conc_to_zero() -> None:
    assert michaelis_menten(-1.0, vmax=5.0, km=1.0) == pytest.approx(0.0)
    out = michaelis_menten(np.array([-1.0, -0.5, 0.0, 1.0]), vmax=4.0, km=1.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(2.0)


def test_michaelis_menten_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        michaelis_menten(1.0, vmax=0.0, km=1.0)
    with pytest.raises(ValueError):
        michaelis_menten(1.0, vmax=1.0, km=0.0)

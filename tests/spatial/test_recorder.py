"""Tests for SnapshotRecorder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gemfitcom.spatial.recorder import SnapshotRecorder
from gemfitcom.spatial.state import SpatialState


def _make_state(t: float = 0.0) -> SpatialState:
    return SpatialState(
        metabolites=np.array([[1.0, 2.0, 3.0]]),
        biomass=np.array([[0.1, 0.2, 0.3]]),
        t=t,
    )


def test_recorder_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "snapshots"
    SnapshotRecorder(output_dir=out, every=1.0)
    assert out.is_dir()


def test_save_writes_npz_with_state_data(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    state = _make_state(t=2.5)
    path = rec.save(state)
    assert path.exists()
    assert "t=2.5000" in path.name
    data = np.load(path)
    assert float(data["t"]) == pytest.approx(2.5)
    np.testing.assert_allclose(data["metabolites"], state.metabolites, rtol=1e-6)
    np.testing.assert_allclose(data["biomass"], state.biomass, rtol=1e-6)


def test_load_roundtrip(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0, precision="float64")
    original = _make_state(t=1.25)
    path = rec.save(original)
    restored = SnapshotRecorder.load(path)
    np.testing.assert_array_equal(restored.metabolites, original.metabolites)
    np.testing.assert_array_equal(restored.biomass, original.biomass)
    assert restored.t == original.t


def test_maybe_save_first_call_always_saves(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    path = rec.maybe_save(_make_state(t=0.0))
    assert path is not None and path.exists()


def test_maybe_save_skips_when_too_soon(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    rec.maybe_save(_make_state(t=0.0))
    skipped = rec.maybe_save(_make_state(t=0.5))
    assert skipped is None


def test_maybe_save_resumes_after_interval(tmp_path: Path) -> None:
    rec = SnapshotRecorder(output_dir=tmp_path, every=1.0)
    rec.maybe_save(_make_state(t=0.0))
    rec.maybe_save(_make_state(t=0.5))
    saved = rec.maybe_save(_make_state(t=1.0))
    assert saved is not None

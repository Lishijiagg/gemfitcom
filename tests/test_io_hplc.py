"""Tests for HPLC loader and wide-to-long conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gemfitcom.io.hplc import HPLC_COLUMNS, hplc_wide_to_long, load_hplc


def _write_csv(path: Path, df: pd.DataFrame, **kwargs: object) -> Path:
    df.to_csv(path, index=False, **kwargs)
    return path


def test_load_hplc_long_canonical(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "carbon_source": ["GMC", "GMC", "GMC"],
            "metabolite": ["acetate", "butyrate", "lactate"],
            "value_mM": [1.2, 0.3, 0.0],
            "replicate": [1, 1, 1],
        }
    )
    path = _write_csv(tmp_path / "hplc.csv", df)
    loaded = load_hplc(path)
    assert list(loaded.columns) == list(HPLC_COLUMNS)
    assert len(loaded) == 3


def test_load_hplc_clips_negatives_by_default(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "carbon_source": ["GMC", "GMC"],
            "metabolite": ["acetate", "butyrate"],
            "value_mM": [1.2, -0.05],
        }
    )
    path = _write_csv(tmp_path / "hplc.csv", df)
    loaded = load_hplc(path)
    assert loaded["value_mM"].min() == 0.0


def test_load_hplc_keeps_negatives_when_disabled(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "carbon_source": ["GMC"],
            "metabolite": ["butyrate"],
            "value_mM": [-0.05],
        }
    )
    path = _write_csv(tmp_path / "hplc.csv", df)
    loaded = load_hplc(path, clip_negatives=False)
    assert loaded["value_mM"].iloc[0] == -0.05


def test_load_hplc_replicate_defaults_to_one(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "carbon_source": ["GMC"],
            "metabolite": ["butyrate"],
            "value_mM": [0.3],
        }
    )
    path = _write_csv(tmp_path / "hplc.csv", df)
    loaded = load_hplc(path)
    assert loaded["replicate"].tolist() == [1]


def test_load_hplc_missing_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"carbon_source": ["GMC"], "metabolite": ["butyrate"]})
    path = _write_csv(tmp_path / "hplc.csv", df)
    with pytest.raises(ValueError, match="missing required columns"):
        load_hplc(path)


def test_load_hplc_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hplc(tmp_path / "nope.csv")


def test_hplc_wide_to_long_from_index() -> None:
    wide = pd.DataFrame(
        {
            "acetate": [1.2, 2.0],
            "butyrate": [0.3, -0.1],
            "lactate": [0.0, 0.5],
        },
        index=pd.Index(["GMC", "2FL"], name="carbon_source"),
    )
    long_df = hplc_wide_to_long(wide)
    assert list(long_df.columns) == list(HPLC_COLUMNS)
    assert set(long_df["carbon_source"].unique()) == {"GMC", "2FL"}
    assert set(long_df["metabolite"].unique()) == {"acetate", "butyrate", "lactate"}
    assert long_df["value_mM"].min() == 0.0  # -0.1 clipped


def test_hplc_wide_to_long_from_index_column() -> None:
    wide = pd.DataFrame(
        {
            "sample": ["GMC", "2FL"],
            "acetate": [1.2, 2.0],
            "butyrate": [0.3, 0.1],
        }
    )
    long_df = hplc_wide_to_long(wide, index_column="sample")
    assert len(long_df) == 4
    assert set(long_df["carbon_source"].unique()) == {"GMC", "2FL"}


def test_hplc_wide_to_long_keeps_negatives_when_disabled() -> None:
    wide = pd.DataFrame({"acetate": [-0.1]}, index=["GMC"])
    long_df = hplc_wide_to_long(wide, clip_negatives=False)
    assert long_df["value_mM"].iloc[0] == -0.1

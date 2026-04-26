"""Tests for OD loader and wide-to-long conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gemfitcom.io.od import OD_COLUMNS, load_od, od_wide_to_long


def _write_csv(path: Path, df: pd.DataFrame, **kwargs: object) -> Path:
    df.to_csv(path, index=False, **kwargs)
    return path


def test_load_od_long_canonical(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "carbon_source": ["GMC"] * 6,
            "replicate": [1, 1, 1, 2, 2, 2],
            "od": [0.1, 0.3, 0.6, 0.1, 0.28, 0.55],
        }
    )
    path = _write_csv(tmp_path / "od.csv", df)
    loaded = load_od(path)
    assert list(loaded.columns) == list(OD_COLUMNS)
    assert len(loaded) == 6
    assert loaded["replicate"].dtype.kind == "i"
    assert loaded["od"].dtype.kind == "f"


def test_load_od_missing_required_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"time_h": [0.0, 1.0], "od": [0.1, 0.2]})
    path = _write_csv(tmp_path / "od.csv", df)
    with pytest.raises(ValueError, match="missing required columns"):
        load_od(path)


def test_load_od_time_unit_conversion(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time_h": [0.0, 60.0, 120.0],  # minutes in the file
            "carbon_source": ["GMC"] * 3,
            "replicate": [1, 1, 1],
            "od": [0.1, 0.3, 0.6],
        }
    )
    path = _write_csv(tmp_path / "od.csv", df)
    loaded = load_od(path, time_unit="min")
    assert loaded["time_h"].tolist() == [0.0, 1.0, 2.0]


def test_load_od_decimal_comma(tmp_path: Path) -> None:
    raw = "time_h;carbon_source;replicate;od\n0,0;GMC;1;0,10\n1,0;GMC;1;0,30\n2,0;GMC;1;0,60\n"
    path = tmp_path / "od.csv"
    path.write_text(raw, encoding="utf-8")
    loaded = load_od(path, decimal=",", sep=";")
    assert loaded["od"].tolist() == [0.10, 0.30, 0.60]
    assert loaded["time_h"].tolist() == [0.0, 1.0, 2.0]


def test_load_od_custom_time_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "carbon_source": ["GMC", "GMC"],
            "replicate": [1, 1],
            "od": [0.1, 0.3],
        }
    )
    path = _write_csv(tmp_path / "od.csv", df)
    loaded = load_od(path, time_column="time")
    assert "time_h" in loaded.columns


def test_load_od_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_od(tmp_path / "nope.csv")


def test_od_wide_to_long_basic() -> None:
    wide = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "GMC_1": [0.1, 0.3, 0.6],
            "GMC_2": [0.11, 0.29, 0.58],
            "FL_1": [0.1, 0.2, 0.4],
        }
    )
    long_df = od_wide_to_long(wide, time_column="time")
    assert list(long_df.columns) == list(OD_COLUMNS)
    assert set(long_df["carbon_source"].unique()) == {"GMC", "FL"}
    assert set(long_df["replicate"].unique()) == {1, 2}
    assert len(long_df) == 9


def test_od_wide_to_long_no_replicate_suffix_defaults_to_one() -> None:
    wide = pd.DataFrame({"time": [0.0, 1.0], "GMC": [0.1, 0.3]})
    long_df = od_wide_to_long(wide, time_column="time")
    assert long_df["replicate"].tolist() == [1, 1]


def test_od_wide_to_long_unparseable_column_raises() -> None:
    wide = pd.DataFrame({"time": [0.0, 1.0], "GMC_1": [0.1, 0.3]})
    # Pattern that requires an explicit trailing .rN suffix.
    strict_pat = r"^(?P<carbon_source>.+?)\.r(?P<replicate>\d+)$"
    with pytest.raises(ValueError, match="does not match pattern"):
        od_wide_to_long(wide, time_column="time", column_pattern=strict_pat)


def test_od_wide_to_long_time_unit_min() -> None:
    wide = pd.DataFrame({"time": [0.0, 30.0, 60.0], "GMC_1": [0.1, 0.2, 0.3]})
    long_df = od_wide_to_long(wide, time_column="time", time_unit="min")
    assert long_df["time_h"].tolist() == [0.0, 0.5, 1.0]


def test_od_wide_to_long_missing_time_column_raises() -> None:
    wide = pd.DataFrame({"GMC_1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="time column"):
        od_wide_to_long(wide, time_column="time")

"""OD growth-curve loader.

Canonical long format:
    time_h : float  (hours)
    carbon_source : str
    replicate : int
    od : float

Use :func:`od_wide_to_long` to convert wide layouts (one column per
carbon-source/replicate) into the canonical form before calling
:func:`load_od`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

TimeUnit = Literal["h", "min", "s"]
_TIME_UNIT_TO_H: dict[str, float] = {"h": 1.0, "min": 1.0 / 60.0, "s": 1.0 / 3600.0}

OD_COLUMNS: tuple[str, ...] = ("time_h", "carbon_source", "replicate", "od")


def load_od(
    path: str | Path,
    *,
    decimal: str = ".",
    sep: str | None = None,
    time_column: str = "time_h",
    time_unit: TimeUnit = "h",
) -> pd.DataFrame:
    """Load a long-format OD CSV.

    Args:
        path: path to the CSV file.
        decimal: decimal separator used in the file (``","`` for European).
        sep: field separator; if ``None``, pandas auto-detects.
        time_column: name of the time column in the file; renamed to
            ``time_h`` after unit conversion.
        time_unit: unit of the time column; values are scaled to hours.

    Returns:
        DataFrame with columns ``time_h, carbon_source, replicate, od``
        (strict schema, other columns are dropped).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if required columns are missing or ``time_unit`` unknown.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OD file not found: {path}")
    if time_unit not in _TIME_UNIT_TO_H:
        raise ValueError(f"time_unit must be one of {list(_TIME_UNIT_TO_H)}, got {time_unit!r}")

    df = pd.read_csv(path, decimal=decimal, sep=sep, engine="python")
    if time_column != "time_h" and time_column in df.columns:
        df = df.rename(columns={time_column: "time_h"})

    missing = [c for c in OD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"OD file {path} missing required columns {missing}. "
            f"Expected long format {OD_COLUMNS}; use od_wide_to_long() first "
            "if your file is wide."
        )

    df = df.loc[:, list(OD_COLUMNS)].copy()
    df["time_h"] = df["time_h"].astype(float) * _TIME_UNIT_TO_H[time_unit]
    df["carbon_source"] = df["carbon_source"].astype(str)
    df["replicate"] = df["replicate"].astype(int)
    df["od"] = df["od"].astype(float)
    return df


def od_wide_to_long(
    df: pd.DataFrame,
    *,
    time_column: str = "time",
    column_pattern: str = r"^(?P<carbon_source>.+?)(?:[._-]r?(?P<replicate>\d+))?$",
    time_unit: TimeUnit = "h",
) -> pd.DataFrame:
    """Pivot a wide OD DataFrame to the canonical long format.

    Each non-time column encodes a ``(carbon_source, replicate)`` pair via
    ``column_pattern``. Columns without a trailing replicate index are assumed
    to be replicate 1.

    Args:
        df: wide DataFrame; one column per ``(carbon_source, replicate)``.
        time_column: name of the time column in ``df``.
        column_pattern: regex with named groups ``carbon_source`` and
            ``replicate`` (optional). Default matches e.g. ``GMC_1``,
            ``GMC.r2``, ``GMC``.
        time_unit: unit of the time column in ``df``; values are converted to
            hours on output.

    Returns:
        Long DataFrame with columns ``time_h, carbon_source, replicate, od``.

    Raises:
        ValueError: on missing time column, unparseable data column, or
            unknown ``time_unit``.
    """
    if time_column not in df.columns:
        raise ValueError(f"time column {time_column!r} not in DataFrame")
    if time_unit not in _TIME_UNIT_TO_H:
        raise ValueError(f"time_unit must be one of {list(_TIME_UNIT_TO_H)}, got {time_unit!r}")

    regex = re.compile(column_pattern)
    scale = _TIME_UNIT_TO_H[time_unit]
    records: list[dict[str, object]] = []
    for col in df.columns:
        if col == time_column:
            continue
        m = regex.match(str(col))
        if m is None:
            raise ValueError(f"column {col!r} does not match pattern {column_pattern!r}")
        carbon_source = m.group("carbon_source")
        rep = m.group("replicate")
        replicate = int(rep) if rep is not None else 1
        for t, od in zip(df[time_column], df[col], strict=True):
            records.append(
                {
                    "time_h": float(t) * scale,
                    "carbon_source": str(carbon_source),
                    "replicate": replicate,
                    "od": float(od),
                }
            )
    return pd.DataFrame.from_records(records, columns=list(OD_COLUMNS))

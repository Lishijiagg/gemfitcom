"""HPLC metabolite measurement loader.

Canonical long format:
    carbon_source : str
    metabolite : str
    value_mM : float
    replicate : int   (defaults to 1 when absent)

Use :func:`hplc_wide_to_long` to convert wide tables (rows = carbon sources,
columns = metabolites) into the canonical form before calling
:func:`load_hplc`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HPLC_COLUMNS: tuple[str, ...] = ("carbon_source", "metabolite", "value_mM", "replicate")
_REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("carbon_source", "metabolite", "value_mM")


def load_hplc(
    path: str | Path,
    *,
    decimal: str = ".",
    sep: str | None = None,
    clip_negatives: bool = True,
) -> pd.DataFrame:
    """Load a long-format HPLC CSV.

    Args:
        path: path to the CSV file.
        decimal: decimal separator used in the file.
        sep: field separator; if ``None``, pandas auto-detects.
        clip_negatives: replace negative ``value_mM`` entries with 0. HPLC
            baseline noise can produce slightly negative readings that are
            not physically meaningful.

    Returns:
        DataFrame with columns ``carbon_source, metabolite, value_mM, replicate``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HPLC file not found: {path}")

    df = pd.read_csv(path, decimal=decimal, sep=sep, engine="python")
    missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"HPLC file {path} missing required columns {missing}. "
            f"Expected long format {HPLC_COLUMNS}; use hplc_wide_to_long() "
            "first if your file is wide."
        )

    if "replicate" not in df.columns:
        df = df.assign(replicate=1)

    df = df.loc[:, list(HPLC_COLUMNS)].copy()
    df["carbon_source"] = df["carbon_source"].astype(str)
    df["metabolite"] = df["metabolite"].astype(str)
    df["value_mM"] = df["value_mM"].astype(float)
    df["replicate"] = df["replicate"].astype(int)

    if clip_negatives:
        df.loc[df["value_mM"] < 0, "value_mM"] = 0.0

    return df


def hplc_wide_to_long(
    df: pd.DataFrame,
    *,
    index_column: str | None = None,
    clip_negatives: bool = True,
) -> pd.DataFrame:
    """Pivot a wide HPLC table (rows=carbon sources, columns=metabolites) to long format.

    Args:
        df: wide DataFrame. If ``index_column`` is given, that column is used
            as the carbon-source index; otherwise the DataFrame's index is
            used. All remaining columns are treated as metabolite measurements.
        index_column: optional name of a column to use as carbon-source index.
        clip_negatives: replace negative values with 0.

    Returns:
        Long DataFrame with columns ``carbon_source, metabolite, value_mM, replicate``.
    """
    if index_column is not None:
        df = df.set_index(index_column)
    records: list[dict[str, object]] = []
    for carbon_source, row in df.iterrows():
        for metabolite, value in row.items():
            val = float(value)
            if clip_negatives and val < 0:
                val = 0.0
            records.append(
                {
                    "carbon_source": str(carbon_source),
                    "metabolite": str(metabolite),
                    "value_mM": val,
                    "replicate": 1,
                }
            )
    return pd.DataFrame.from_records(records, columns=list(HPLC_COLUMNS))

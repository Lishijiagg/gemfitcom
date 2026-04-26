"""HPLC endpoint-measurement preprocessing.

Operates on the canonical long format produced by :mod:`gemfitcom.io.hplc`::

    carbon_source : str
    metabolite : str
    value_mM : float
    replicate : int

HPLC values in this pipeline are endpoint snapshots (single time), so the
preprocessing is limited to replicate aggregation and layout conversion for
downstream kinetics fitting.
"""

from __future__ import annotations

import pandas as pd


def average_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicates to mean/sd/n per ``(carbon_source, metabolite)``.

    Args:
        df: Long-format HPLC DataFrame.

    Returns:
        DataFrame with columns
        ``carbon_source, metabolite, mean_mM, sd_mM, n_replicates``.
        ``sd_mM`` is NaN for single-replicate groups.
    """
    _require_columns(df, ("carbon_source", "metabolite", "value_mM"))
    grouped = df.groupby(["carbon_source", "metabolite"], sort=True, as_index=False)
    return grouped["value_mM"].agg(mean_mM="mean", sd_mM="std", n_replicates="count")


def hplc_long_to_wide(
    df: pd.DataFrame,
    *,
    value_column: str = "value_mM",
    aggregate: bool = True,
) -> pd.DataFrame:
    """Pivot long HPLC to a ``(carbon_source, metabolite)`` wide table.

    Args:
        df: Long-format HPLC DataFrame. Must contain ``carbon_source``,
            ``metabolite``, and ``value_column``. If ``aggregate`` is False,
            ``replicate`` must also be present and uniquely identify rows.
        value_column: Column holding concentrations (default ``"value_mM"``).
        aggregate: If True, average replicates before pivoting (common case
            for downstream kinetics fitting). If False, require one row per
            ``(carbon_source, metabolite, replicate)`` and raise on conflict.

    Returns:
        Wide DataFrame indexed by ``carbon_source`` with one column per
        metabolite.
    """
    _require_columns(df, ("carbon_source", "metabolite", value_column))
    if aggregate:
        avg = df.groupby(["carbon_source", "metabolite"], sort=True, as_index=False)[
            value_column
        ].mean()
        wide = avg.pivot(index="carbon_source", columns="metabolite", values=value_column)
    else:
        _require_columns(df, ("replicate",))
        dup = df.duplicated(subset=["carbon_source", "metabolite", "replicate"])
        if dup.any():
            raise ValueError(
                "Duplicate (carbon_source, metabolite, replicate) rows — "
                "pass aggregate=True to average them explicitly."
            )
        wide = df.pivot(
            index=["carbon_source", "replicate"],
            columns="metabolite",
            values=value_column,
        )
    wide.columns.name = None
    return wide


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns {missing}")

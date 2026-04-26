"""Internal helpers shared across simulate modes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flux_tensor_to_long(
    flux_hist: np.ndarray,
    time_h: np.ndarray,
    strain_names: list[str],
    pool_ids: list[str],
) -> pd.DataFrame:
    """Melt an ``(n_points, n_strains, n_pool)`` tensor into long form.

    Returns a DataFrame with columns ``[time_h, strain, exchange_id,
    flux]``. Ordering is (time, strain, exchange) with exchange varying
    fastest — matches ``flux_hist.reshape(-1)`` layout.
    """
    n_points, n_strains, n_pool = flux_hist.shape
    if n_points != len(time_h):
        raise ValueError(
            f"time_h length {len(time_h)} does not match flux_hist first dim {n_points}"
        )
    if n_strains != len(strain_names):
        raise ValueError(
            f"strain_names length {len(strain_names)} does not match flux_hist second dim {n_strains}"
        )
    if n_pool != len(pool_ids):
        raise ValueError(
            f"pool_ids length {len(pool_ids)} does not match flux_hist third dim {n_pool}"
        )

    time_col = np.repeat(time_h, n_strains * n_pool)
    strain_col = np.tile(np.repeat(strain_names, n_pool), n_points)
    exch_col = np.tile(pool_ids, n_points * n_strains)
    flux_col = flux_hist.reshape(-1)
    return pd.DataFrame(
        {
            "time_h": time_col,
            "strain": strain_col,
            "exchange_id": exch_col,
            "flux": flux_col,
        }
    )

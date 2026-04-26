"""Summarise cross-feeding and competition into a single graph."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from gemfitcom.interactions.competition import competition_edges
from gemfitcom.interactions.cross_feeding import cross_feeding_edges


def summary_graph(
    panel: pd.DataFrame,
    biomass: pd.DataFrame | None = None,
    *,
    include_competition: bool = True,
    threshold: float = 0.0,
    dt: float | None = None,
    tol: float = 1e-9,
) -> nx.MultiDiGraph:
    """Build a MultiDiGraph summarising strain-strain interactions.

    The graph has one node per strain (any strain that appears in the
    panel) and two families of edges:

    * **Cross-feeding**: directed edges ``donor -> recipient`` with
      ``type="cross_feeding"``, ``exchange_id``, and
      ``weight = cumulative_flow``.
    * **Competition** (when ``include_competition=True``): a pair of
      directed edges ``a <-> b`` with ``type="competition"``,
      ``exchange_id``, and ``weight = competition_intensity``. The same
      weight is placed on both directions so downstream code can treat
      them as undirected without extra bookkeeping.

    Args:
        panel: Long-form exchange-flux table (see :func:`exchange_panel`).
        biomass: Optional biomass weighting (see :func:`biomass_panel`).
        include_competition: If ``False``, only cross-feeding edges are
            added.
        threshold: Edges whose weight is ``<= threshold`` are dropped.
        dt: Forwarded to the underlying edge computations.
        tol: Forwarded to the underlying edge computations.

    Returns:
        A :class:`networkx.MultiDiGraph`.
    """
    g = nx.MultiDiGraph()

    strains = pd.unique(panel["strain"]) if "strain" in panel.columns else []
    for s in strains:
        g.add_node(str(s))

    xfeed = cross_feeding_edges(panel, biomass=biomass, dt=dt, tol=tol)
    for row in xfeed.itertuples(index=False):
        w = float(row.cumulative_flow)
        if w <= threshold:
            continue
        g.add_edge(
            str(row.donor),
            str(row.recipient),
            type="cross_feeding",
            exchange_id=str(row.exchange_id),
            weight=w,
        )

    if include_competition:
        comp = competition_edges(panel, biomass=biomass, dt=dt, tol=tol)
        for row in comp.itertuples(index=False):
            w = float(row.competition_intensity)
            if w <= threshold:
                continue
            a, b = str(row.strain_a), str(row.strain_b)
            exch = str(row.exchange_id)
            g.add_edge(a, b, type="competition", exchange_id=exch, weight=w)
            g.add_edge(b, a, type="competition", exchange_id=exch, weight=w)

    return g


__all__ = ["summary_graph"]

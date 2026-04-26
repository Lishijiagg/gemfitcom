"""Tests for interactions.network.summary_graph."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from gemfitcom.interactions import summary_graph


def _panel(rows: list[tuple[float, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["time_h", "strain", "exchange_id", "flux"])


def test_returns_multi_di_graph_with_strain_nodes() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_ac_e", 2.0),
            (0.0, "B", "EX_ac_e", -1.0),
            (0.0, "C", "EX_glc__D_e", -1.0),
            (1.0, "A", "EX_ac_e", 0.0),
            (1.0, "B", "EX_ac_e", 0.0),
            (1.0, "C", "EX_glc__D_e", 0.0),
        ]
    )
    g = summary_graph(panel)
    assert isinstance(g, nx.MultiDiGraph)
    assert set(g.nodes) == {"A", "B", "C"}


def test_cross_feeding_edges_added_as_directed() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_ac_e", 2.0),
            (0.0, "B", "EX_ac_e", -1.0),
            (1.0, "A", "EX_ac_e", 0.0),
            (1.0, "B", "EX_ac_e", 0.0),
        ]
    )
    g = summary_graph(panel, include_competition=False)
    xfeed = [(u, v, d) for u, v, d in g.edges(data=True) if d["type"] == "cross_feeding"]
    assert len(xfeed) == 1
    u, v, d = xfeed[0]
    assert u == "A" and v == "B"
    assert d["exchange_id"] == "EX_ac_e"
    assert d["weight"] > 0


def test_competition_adds_bidirectional_edges() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
        ]
    )
    g = summary_graph(panel, include_competition=True)
    comp = [(u, v, d) for u, v, d in g.edges(data=True) if d["type"] == "competition"]
    pairs = {(u, v) for u, v, _ in comp}
    assert pairs == {("A", "B"), ("B", "A")}
    for _, _, d in comp:
        assert d["weight"] > 0
        assert d["exchange_id"] == "EX_x"


def test_include_competition_false_omits_competition_edges() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", -1.0),
            (0.0, "B", "EX_x", -2.0),
            (0.0, "C", "EX_x", 1.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
            (1.0, "C", "EX_x", 0.0),
        ]
    )
    g = summary_graph(panel, include_competition=False)
    types = {d["type"] for _, _, d in g.edges(data=True)}
    assert "competition" not in types


def test_threshold_drops_small_edges() -> None:
    # Cross-feeding flow = min(0.5, 0.3) * 1h = 0.3
    # Threshold 0.5 → drop
    panel = _panel(
        [
            (0.0, "A", "EX_x", 0.5),
            (0.0, "B", "EX_x", -0.3),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
        ]
    )
    g_no_thresh = summary_graph(panel, include_competition=False, threshold=0.0)
    g_thresh = summary_graph(panel, include_competition=False, threshold=0.5)
    assert g_no_thresh.number_of_edges() == 1
    assert g_thresh.number_of_edges() == 0
    # Nodes are still present even when their edges are dropped.
    assert set(g_thresh.nodes) == {"A", "B"}


def test_biomass_flows_through_to_weights() -> None:
    panel = _panel(
        [
            (0.0, "A", "EX_x", 1.0),
            (0.0, "B", "EX_x", -1.0),
            (1.0, "A", "EX_x", 0.0),
            (1.0, "B", "EX_x", 0.0),
        ]
    )
    bio = pd.DataFrame(
        {
            "time_h": [0.0, 0.0, 1.0, 1.0],
            "strain": ["A", "B", "A", "B"],
            "biomass": [4.0, 2.0, 4.0, 2.0],
        }
    )
    g = summary_graph(panel, biomass=bio, include_competition=False)
    weights = [d["weight"] for _, _, d in g.edges(data=True)]
    # min(4, 2) * 1h = 2
    assert weights == [2.0]

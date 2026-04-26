"""Interaction-network plot from a :func:`summary_graph` result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

CROSS_FEEDING_COLOR: str = "tab:blue"
COMPETITION_COLOR: str = "tab:red"


def plot_interaction_network(
    graph: nx.MultiDiGraph,
    *,
    ax: Axes | None = None,
    layout: str = "spring",
    seed: int = 0,
    node_size: float = 800.0,
    edge_width_scale: float = 3.0,
    show_competition: bool = True,
    label_edges: bool = False,
    title: str | None = None,
) -> Figure:
    """Draw a strain-strain interaction network.

    Nodes are strains. Edges are coloured by ``edge["type"]``:

    * ``"cross_feeding"`` → blue, drawn as a single arrow donor → recipient.
    * ``"competition"`` → red, drawn as an undirected (no-arrow) line.
      :func:`summary_graph` emits two directed competition edges per pair so
      each pair appears once after de-duplication.

    Edge widths scale with ``edge["weight"]`` per edge type
    (max weight per type maps to ``edge_width_scale`` points).

    Args:
        graph: A :class:`networkx.MultiDiGraph` from :func:`summary_graph`.
        ax: Optional axes to draw on; a new figure is created when omitted.
        layout: ``"spring"``, ``"circular"``, ``"kamada_kawai"``, or
            ``"shell"``. Spring uses ``seed`` for determinism.
        seed: Spring-layout seed.
        node_size: Node size in matplotlib points² (forwarded to networkx).
        edge_width_scale: Maximum edge linewidth (per edge type).
        show_competition: If ``False``, competition edges are hidden.
        label_edges: If ``True``, write the ``exchange_id`` next to each edge.
        title: Optional axes title.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
    else:
        fig = ax.figure

    pos = _layout(graph, layout=layout, seed=seed)

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="lightgrey",
        edgecolors="black",
        node_size=node_size,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10)

    xfeed = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get("type") == "cross_feeding"]
    comp_pairs = _dedup_competition_edges(graph) if show_competition else []

    _draw_typed_edges(
        ax,
        graph,
        pos,
        xfeed,
        color=CROSS_FEEDING_COLOR,
        width_scale=edge_width_scale,
        directed=True,
        node_size=node_size,
    )
    _draw_typed_edges(
        ax,
        graph,
        pos,
        comp_pairs,
        color=COMPETITION_COLOR,
        width_scale=edge_width_scale,
        directed=False,
        node_size=node_size,
    )

    if label_edges:
        labels = {(u, v): d.get("exchange_id", "") for u, v, d in xfeed}
        labels.update({(u, v): d.get("exchange_id", "") for u, v, d in comp_pairs})
        if labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax, font_size=8)

    legend_handles = []
    if xfeed:
        legend_handles.append(
            plt.Line2D([], [], color=CROSS_FEEDING_COLOR, linewidth=2.0, label="Cross-feeding")
        )
    if comp_pairs:
        legend_handles.append(
            plt.Line2D([], [], color=COMPETITION_COLOR, linewidth=2.0, label="Competition")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=False)

    ax.set_axis_off()
    ax.set_title(title or "Strain interaction network")
    fig.tight_layout()
    return fig


def _layout(graph: nx.Graph, *, layout: str, seed: int) -> dict:
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed)
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "shell":
        return nx.shell_layout(graph)
    raise ValueError(f"unknown layout {layout!r}")


def _dedup_competition_edges(graph: nx.MultiDiGraph) -> list:
    """Pick one of the two mirrored competition edges per (pair, exchange).

    :func:`summary_graph` adds both ``a -> b`` and ``b -> a`` with identical
    weights for every competition pair. We keep one direction (lexicographic
    min) so the plot draws a single line per pair.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list = []
    for u, v, d in graph.edges(data=True):
        if d.get("type") != "competition":
            continue
        a, b = sorted((str(u), str(v)))
        key = (a, b, str(d.get("exchange_id", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, d))
    return out


def _draw_typed_edges(
    ax: Axes,
    graph: nx.MultiDiGraph,
    pos: dict,
    edges: list,
    *,
    color: str,
    width_scale: float,
    directed: bool,
    node_size: float,
) -> None:
    if not edges:
        return
    weights = [float(d.get("weight", 0.0)) for _, _, d in edges]
    w_max = max(weights) if weights else 0.0
    if w_max <= 0:
        widths = [1.0 for _ in edges]
    else:
        widths = [max(0.5, width_scale * (w / w_max)) for w in weights]

    edge_list = [(u, v) for u, v, _ in edges]
    kwargs: dict = {
        "edgelist": edge_list,
        "ax": ax,
        "edge_color": color,
        "width": widths,
        "node_size": node_size,
    }
    if directed:
        kwargs["arrows"] = True
        kwargs["arrowstyle"] = "-|>"
        kwargs["connectionstyle"] = "arc3,rad=0.08"
    else:
        kwargs["arrows"] = False
    nx.draw_networkx_edges(graph, pos, **kwargs)


__all__ = ["plot_interaction_network"]

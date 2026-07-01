"""Static figure generator for 09-GraphRAG.

Imports the SAME canonical functions the page and notebook use (graph_rag.py) so every plotted number
is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to the shared chapter image
dir (../../images/) with the per-chapter prefix `rag09_`.

    python make_figures_09.py

Figures produced:
  rag09_knowledge_graph.png   -- the knowledge graph: entities (nodes) + relations (labelled edges),
                                 coloured by the REAL detected community, with modularity Q.
  rag09_multihop_vs_flat.png  -- flat vector RAG retrieves disconnected endpoints (no chunk links both
                                 ends) vs the graph traversal path that connects them (the real path).
  rag09_communities.png       -- the detected communities as coloured clusters + the modularity Q vs
                                 the trivial one-blob partition (real networkx numbers).
  rag09_local_vs_global.png   -- LOCAL search (entity link + neighbourhood traversal) vs GLOBAL search
                                 (community detection -> summaries -> map-reduce): when to use each.
  rag09_global_mapreduce.png  -- map-reduce over community summaries: each community -> a partial theme
                                 (map), aggregated into the final themes answer (reduce).

Verified on Python 3.12 / matplotlib 3.x / networkx 3.x / numpy 2.x / sentence-transformers (CPU).
Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import networkx as nx

from graph_rag import (
    TOP_K,
    DenseRetriever,
    build_graph,
    build_multihop_probe,
    detect_communities,
    flat_rag_can_answer_multihop,
    full_corpus,
    global_search,
    modularity_score,
    shortest_path,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / entity
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # hit / traversal / community A
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / community B
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines
# distinct muted community colours (blue / amber / green, then extras)
COMMUNITY_COLORS = [BLUE, AMBER, GREEN, PURPLE, SLATE, RED]

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110
SEED = 7  # fixed layout seed so the graph drawing is reproducible


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _community_of(node: str, communities) -> int:
    """Index of the community containing `node`."""
    for i, comm in enumerate(communities):
        if node in comm:
            return i
    return -1


# Short display labels so entity names fit inside the drawn nodes (the full names are in the prose).
_SHORT_LABEL: dict[str, str] = {
    "Helios-7": "Helios-7",
    "Dr. Amara Okoye": "Dr. Okoye",
    "Nairobi office": "Nairobi\noffice",
    "Kourou spaceport": "Kourou",
    "Ariane 6": "Ariane 6",
    "hyperspectral imager": "imager",
    "sun-synchronous orbit": "orbit",
    "Error E-4011": "E-4011",
    "telemetry stream": "telemetry",
    "ground team": "ground\nteam",
}


def _labels_for(graph: nx.Graph) -> dict[str, str]:
    """Node -> short display label (falls back to the name if unmapped)."""
    return {n: _SHORT_LABEL.get(n, n) for n in graph.nodes()}


def fig_knowledge_graph(graph: nx.Graph, communities) -> None:
    """The knowledge graph: entities as nodes, relations as labelled edges, coloured by community.

    A spring layout (fixed seed) of the REAL graph; node colour is the REAL detected community; edge
    labels are the real relation types. Shows the connected map that flat RAG never builds.
    """
    q = modularity_score(graph, communities)
    pos = nx.spring_layout(graph, seed=SEED, k=1.1)
    node_colors = [COMMUNITY_COLORS[_community_of(n, communities) % len(COMMUNITY_COLORS)] for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.axis("off")
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=SLATE, width=1.4, alpha=0.55)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=2900,
                           edgecolors=INK, linewidths=1.2, alpha=0.92)
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax, font_size=7.6,
                            font_color="white", font_weight="bold")
    edge_labels = {(a, b): graph.edges[a, b]["relation"] for a, b in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=7.0,
                                 font_color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                 edgecolor="none", alpha=0.7))
    ax.set_title(f"The Helios-7 knowledge graph — {graph.number_of_nodes()} entities, "
                 f"{graph.number_of_edges()} relations, {len(communities)} communities (Q={q:.3f})\n"
                 "node colour = detected community; edge labels = extracted relations",
                 fontsize=11.5, color=INK, pad=14)
    # honesty caption
    ax.text(0.5, -0.02, "graph structure, traversal & community detection are REAL (networkx); "
            "the entity/relation extraction is an illustrative LLM stand-in",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color=SLATE)
    _save(fig, "rag09_knowledge_graph.png")


def fig_multihop_vs_flat(graph: nx.Graph) -> None:
    """Flat vector RAG retrieves disconnected endpoints; the graph path connects them (real path).

    Left: the flat-RAG top-k chunks as separate cards -- the endpoints are there but nothing links
    them (no chunk mentions both ends). Right: the real graph traversal path, hop by hop, arriving at
    the answer entity. The contrast is the whole multi-hop argument.
    """
    corpus = full_corpus()
    retriever = DenseRetriever(corpus)
    probe = build_multihop_probe()
    flat_hits = retriever.search(probe.query, k=TOP_K).indices
    flat_ok = flat_rag_can_answer_multihop(retriever, probe, corpus)
    path = shortest_path(graph, probe.start, probe.end)

    fig, (ax_flat, ax_graph) = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # --- left: flat RAG's disconnected chunks ---
    ax_flat.axis("off")
    ax_flat.set_xlim(0, 1)
    ax_flat.set_ylim(0, 1)
    ax_flat.text(0.5, 0.97, "Flat vector RAG: top-k similar chunks", ha="center", fontsize=11,
                 fontweight="bold", color=RED)
    ax_flat.text(0.5, 0.90, f"query: “{probe.query}”", ha="center", fontsize=8.2, style="italic", color=INK, wrap=True)

    def short(text, n=54):
        return text if len(text) <= n else text[: n - 1] + "…"

    for i, idx in enumerate(flat_hits):
        y = 0.74 - i * 0.17
        ax_flat.text(0.5, y, f"chunk[{idx}]: {short(corpus[idx])}", ha="center", fontsize=8.2, color=INK,
                     bbox=dict(boxstyle="round,pad=0.35", facecolor=BLUE, alpha=0.14, edgecolor=BLUE))
    ax_flat.text(0.5, 0.16, "endpoints retrieved, but NO chunk links\nthe imager to the office → cannot connect them",
                 ha="center", fontsize=9, color=RED, fontweight="bold")
    ax_flat.text(0.5, 0.04, f"any single chunk links both ends? {flat_ok}", ha="center", fontsize=8.8,
                 color=RED, style="italic")

    # --- right: the graph traversal path ---
    ax_graph.axis("off")
    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.text(0.5, 0.97, f"GraphRAG local search: traverse the graph ({len(path) - 1} hops)",
                  ha="center", fontsize=11, fontweight="bold", color=GREEN)
    n_nodes = len(path)
    ys = [0.82 - i * (0.68 / max(n_nodes - 1, 1)) for i in range(n_nodes)]
    for i, (node, y) in enumerate(zip(path, ys)):
        is_end = i == n_nodes - 1
        face = GREEN if is_end else BLUE
        ax_graph.text(0.5, y, node, ha="center", fontsize=9.5, color="white", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor=face, edgecolor=INK, alpha=0.92))
        if i < n_nodes - 1:
            rel = graph.edges[path[i], path[i + 1]]["relation"]
            ymid = (y + ys[i + 1]) / 2
            ax_graph.annotate("", xy=(0.5, ys[i + 1] + 0.035), xytext=(0.5, y - 0.035),
                              arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
            ax_graph.text(0.56, ymid, f"-[{rel}]->", ha="left", va="center", fontsize=8.2,
                          color=GREEN, fontweight="bold")
    ax_graph.text(0.5, 0.04, f"answer entity: {path[-1]}  ✓", ha="center", fontsize=9.5,
                  color=GREEN, fontweight="bold")
    fig.suptitle("Multi-hop: flat RAG retrieves endpoints; the graph connects them",
                 fontsize=12.5, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag09_multihop_vs_flat.png")


def fig_communities(graph: nx.Graph, communities) -> None:
    """The detected communities as coloured clusters + modularity Q vs the trivial one-blob partition.

    Left: the graph laid out, nodes coloured by REAL detected community. Right: modularity bars --
    the detected partition vs lumping everything in one community -- the number that says the split
    captured real structure. All values are the chapter's own networkx computation.
    """
    q = modularity_score(graph, communities)
    q_blob = modularity_score(graph, [frozenset(graph.nodes())])
    pos = nx.spring_layout(graph, seed=SEED, k=1.1)
    node_colors = [COMMUNITY_COLORS[_community_of(n, communities) % len(COMMUNITY_COLORS)] for n in graph.nodes()]

    fig, (ax_g, ax_q) = plt.subplots(1, 2, figsize=(12.2, 5.6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax_g.axis("off")
    nx.draw_networkx_edges(graph, pos, ax=ax_g, edge_color=SLATE, width=1.2, alpha=0.5)
    nx.draw_networkx_nodes(graph, pos, ax=ax_g, node_color=node_colors, node_size=2400,
                           edgecolors=INK, linewidths=1.1, alpha=0.92)
    nx.draw_networkx_labels(graph, pos, labels=_labels_for(graph), ax=ax_g, font_size=6.8,
                            font_color="white", font_weight="bold")
    ax_g.set_title(f"{len(communities)} communities by modularity optimization (real)", fontsize=11, pad=10)
    # legend of community themes
    for i, comm in enumerate(communities):
        ax_g.scatter([], [], color=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
                     label=f"community {i}: {len(comm)} entities", s=90, edgecolors=INK)
    ax_g.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=len(communities), fontsize=8,
                framealpha=0.95)

    # modularity bars
    ax_q.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax_q.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_q.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_q.spines[spine].set_color(GRID)
    bars = ax_q.bar(["detected\npartition", "trivial\n(one blob)"], [q, q_blob],
                    color=[GREEN, SLATE], edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, val in zip(bars, [q, q_blob]):
        ax_q.annotate(f"Q = {val:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      fontsize=10, color=INK, ha="center", va="bottom", xytext=(0, 3),
                      textcoords="offset points", fontweight="bold")
    ax_q.set_ylabel("modularity Q  (higher = stronger community structure)")
    ax_q.set_ylim(min(0, q_blob) - 0.02, q * 1.4)
    ax_q.tick_params(colors=INK, labelsize=9)
    ax_q.set_title("Q measures real structure, not noise", fontsize=10.5, pad=8)
    fig.suptitle(f"Community detection: {len(communities)} thematic clusters, modularity Q = {q:.4f} "
                 f"(vs {q_blob:.3f} for one blob)", fontsize=12, y=1.0, color=INK, fontweight="bold")
    _save(fig, "rag09_communities.png")


def fig_local_vs_global() -> None:
    """LOCAL search (entity link + traversal) vs GLOBAL search (communities -> summaries -> map-reduce).

    A schematic contrasting the two GraphRAG query paths and when to use each. Not a measurement -- the
    mechanism map; the measured numbers live in the multi-hop and community figures.
    """
    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax in (ax_l, ax_g):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def box(ax, x, y, w, h, text, color, fs=8.4):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fs, fontweight="bold")

    # LOCAL
    ax_l.text(0.5, 0.96, "LOCAL search — specific-entity / multi-hop", ha="center", fontsize=11,
              fontweight="bold", color=GREEN)
    box(ax_l, 0.22, 0.80, 0.56, 0.10, "query mentions specific entities", SLATE)
    box(ax_l, 0.22, 0.62, 0.56, 0.10, "link query → graph entities", BLUE)
    box(ax_l, 0.22, 0.44, 0.56, 0.10, "traverse neighbourhood / shortest path", GREEN)
    box(ax_l, 0.22, 0.26, 0.56, 0.10, "gather connected facts → answer", GREEN)
    for y0, y1 in ((0.80, 0.72), (0.62, 0.54), (0.44, 0.36)):
        ax_l.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax_l.text(0.5, 0.12, "good for: “which office runs the satellite\ncarrying the imager?” (connect facts)",
              ha="center", fontsize=8.4, style="italic", color=INK)

    # GLOBAL
    ax_g.text(0.5, 0.96, "GLOBAL search — whole-corpus / thematic", ha="center", fontsize=11,
              fontweight="bold", color=AMBER)
    box(ax_g, 0.22, 0.80, 0.56, 0.10, "query is about the whole dataset", SLATE)
    box(ax_g, 0.22, 0.62, 0.56, 0.10, "detect communities (modularity)", AMBER)
    box(ax_g, 0.22, 0.44, 0.56, 0.10, "summarize each community (LLM)", PURPLE)
    box(ax_g, 0.22, 0.26, 0.56, 0.10, "map-reduce summaries → answer", GREEN)
    for y0, y1 in ((0.80, 0.72), (0.62, 0.54), (0.44, 0.36)):
        ax_g.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax_g.text(0.5, 0.12, "good for: “what are the main themes\nacross the corpus?” (aggregate)",
              ha="center", fontsize=8.4, style="italic", color=INK)
    fig.suptitle("Two GraphRAG query paths: LOCAL (traverse) vs GLOBAL (summarize communities)",
                 fontsize=12.5, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag09_local_vs_global.png")


def fig_global_mapreduce(communities) -> None:
    """Map-reduce over community summaries: each community -> a partial theme (map) -> final (reduce).

    Uses the chapter's REAL detected communities and their (illustrative) summaries. Shows the
    GraphRAG global-search flow that answers a whole-corpus theme question no single chunk could.
    """
    result = global_search(communities)

    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "Global search: map-reduce over community summaries", ha="center",
            fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.90, "query: “What are the main themes across the corpus?”", ha="center",
            fontsize=9, style="italic", color=INK)

    def wrap(text, n=52):
        import textwrap
        return "\n".join(textwrap.wrap(text, n))

    n = len(result.partial_points)
    xs = [(i + 0.5) / n for i in range(n)]
    for i, (x, point) in enumerate(zip(xs, result.partial_points)):
        # community box
        ax.add_patch(plt.Rectangle((x - 0.15, 0.60), 0.30, 0.20, facecolor=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
                     alpha=0.16, edgecolor=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)], linewidth=1.3))
        ax.text(x, 0.77, f"community {i}", ha="center", fontsize=8.6, fontweight="bold",
                color=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)])
        ax.text(x, 0.68, wrap(point.split(":")[0] + ":" + point.split(":")[1][:40] + "…", 26),
                ha="center", va="center", fontsize=6.8, color=INK)
        ax.text(x, 0.55, "MAP → partial theme", ha="center", fontsize=7.4, style="italic", color=SLATE)
        # arrow down to reduce
        ax.annotate("", xy=(0.5, 0.40), xytext=(x, 0.52), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2, alpha=0.6))

    # reduce box
    ax.add_patch(plt.Rectangle((0.12, 0.16), 0.76, 0.22, facecolor=GREEN, alpha=0.14, edgecolor=GREEN, linewidth=1.5))
    ax.text(0.5, 0.345, "REDUCE → final themes answer", ha="center", fontsize=9.5, fontweight="bold", color=GREEN)
    ax.text(0.5, 0.24, wrap(result.final_answer.replace("Main themes across the corpus: ", ""), 96),
            ha="center", va="center", fontsize=6.8, color=INK)
    ax.text(0.5, 0.03, "communities are REAL (networkx modularity); the summaries/answer text is an "
            "illustrative LLM stand-in", ha="center", fontsize=7.6, style="italic", color=SLATE)
    _save(fig, "rag09_global_mapreduce.png")


def main() -> None:
    graph = build_graph()
    communities = detect_communities(graph)
    print(f"graph: {graph.number_of_nodes()} nodes | communities: {len(communities)}")
    fig_knowledge_graph(graph, communities)
    fig_multihop_vs_flat(graph)
    fig_communities(graph, communities)
    fig_local_vs_global()
    fig_global_mapreduce(communities)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

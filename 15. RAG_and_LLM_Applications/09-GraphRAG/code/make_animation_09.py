"""Animated (GIF) intuition figure for 09-GraphRAG.

Companion to the static PNGs. It brings the core GraphRAG idea to life -- MULTI-HOP TRAVERSAL: a
query lights up a start entity, and a highlight HOPS across the knowledge graph edge by edge,
following the real relationships to connect facts that live in different documents, until it reaches
the answer entity. You watch the graph do what flat top-k retrieval cannot: chain
imager -> Helios-7 -> Dr. Okoye -> Nairobi office.

    python make_animation_09.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The graph,
the traversal path, and each edge's relation are the chapter's OWN -- a real networkx graph and the
real shortest path from graph_rag.py.

Produced:
  rag09_traversal_hop.gif -- the multi-hop traversal hopping across the graph, edge by edge, to the
                             answer entity.

Verified on Python 3.12 / matplotlib 3.x / networkx 3.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

from graph_rag import build_graph, build_multihop_probe, shortest_path

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 100
FPS = 8
SEED = 7  # match the static figures' layout

_SHORT_LABEL = {
    "Helios-7": "Helios-7", "Dr. Amara Okoye": "Dr. Okoye", "Nairobi office": "Nairobi\noffice",
    "Kourou spaceport": "Kourou", "Ariane 6": "Ariane 6", "hyperspectral imager": "imager",
    "sun-synchronous orbit": "orbit", "Error E-4011": "E-4011", "telemetry stream": "telemetry",
    "ground team": "ground\nteam",
}


def build_animation() -> None:
    graph = build_graph()
    probe = build_multihop_probe()
    path = shortest_path(graph, probe.start, probe.end)
    path_edges = list(zip(path, path[1:]))
    pos = nx.spring_layout(graph, seed=SEED, k=1.1)
    labels = {n: _SHORT_LABEL.get(n, n) for n in graph.nodes()}

    # timeline: hold on the start, then reveal one hop per few frames, then hold on the full path
    hold_start = 4
    frames_per_hop = 4
    hold_end = 10
    total = hold_start + len(path_edges) * frames_per_hop + hold_end

    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.06)

    def hops_revealed(frame: int) -> int:
        if frame < hold_start:
            return 0
        if frame >= hold_start + len(path_edges) * frames_per_hop:
            return len(path_edges)
        return (frame - hold_start) // frames_per_hop + 1

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        revealed = hops_revealed(frame)
        nodes_lit = set(path[: revealed + 1])
        edges_lit = set(tuple(sorted(e)) for e in path_edges[:revealed])

        # base graph (dim)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=GRID, width=1.2, alpha=0.7)
        base_nodes = [n for n in graph.nodes() if n not in nodes_lit]
        nx.draw_networkx_nodes(graph, pos, nodelist=base_nodes, ax=ax, node_color=SLATE,
                               node_size=2200, edgecolors=INK, linewidths=0.8, alpha=0.35)

        # lit path edges (green, thick)
        if edges_lit:
            nx.draw_networkx_edges(graph, pos, edgelist=[tuple(e) for e in edges_lit], ax=ax,
                                   edge_color=GREEN, width=3.2, alpha=0.95)

        # lit nodes: start (amber), intermediate (blue), end (green once reached)
        for i, node in enumerate(path[: revealed + 1]):
            if i == 0:
                color = AMBER
            elif node == path[-1] and revealed == len(path_edges):
                color = GREEN
            else:
                color = BLUE
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], ax=ax, node_color=color,
                                   node_size=2600, edgecolors=INK, linewidths=1.6, alpha=0.95)

        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7.2,
                                font_color="white", font_weight="bold")
        # relation label on the most-recently-lit edge
        if 0 < revealed <= len(path_edges):
            a, b = path_edges[revealed - 1]
            rel = graph.edges[a, b]["relation"]
            mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, f"-[{rel}]->", fontsize=8.5, color=GREEN, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

        # header
        if revealed == 0:
            title, tcol = "Multi-hop query: start at the entity the query mentions", AMBER
        elif revealed < len(path_edges):
            title, tcol = f"HOP {revealed}: follow the edge to connect the next fact", BLUE
        else:
            title, tcol = "Arrived: the graph connected facts across documents", GREEN
        fig.suptitle(title, fontsize=13, color=tcol, y=0.955, fontweight="bold")
        ax.set_title(f"“{probe.query}”", fontsize=9, color=INK, style="italic", pad=8)

        # progress caption
        so_far = "  ".join(f"{_SHORT_LABEL.get(n, n).replace(chr(10), ' ')}" for n in path[: revealed + 1])
        ax.text(0.5, -0.02, f"path so far:  {so_far}", transform=ax.transAxes, ha="center",
                fontsize=8.5, color=GREEN if revealed == len(path_edges) else INK, fontweight="bold")
        ax.text(0.5, -0.06, "real networkx graph + real shortest path; entity/relation extraction is an "
                "illustrative LLM stand-in", transform=ax.transAxes, ha="center", fontsize=7.2,
                style="italic", color=SLATE)
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag09_traversal_hop.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()

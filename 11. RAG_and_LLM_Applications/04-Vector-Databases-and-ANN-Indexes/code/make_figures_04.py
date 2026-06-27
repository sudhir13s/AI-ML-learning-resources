"""Static figure generator for 04-Vector-Databases-and-ANN-Indexes.

Imports the SAME canonical functions the page and notebook use (vector_indexes.py) so every plotted
number from our own indexes is the chapter's own — no hand-typed values. Writes muted-palette PNGs
to the shared chapter image dir (../../images/) with the per-chapter prefix `rag04_`.

    python make_figures_04.py

Figures produced:
  rag04_voronoi_cells.png     -- the corpus partitioned into Voronoi cells (2D view); the query, its
                                true neighbours, and the nprobe probed cells highlighted.
  rag04_recall_cliff.png      -- recall@10 vs nprobe (the cliff) alongside the % of corpus scanned;
                                the recall/speed knob, measured on our own IVF.
  rag04_bruteforce_growth.png -- brute-force multiply-adds per query as N grows (linear), vs the
                                near-flat cost of probing a fixed handful of IVF cells.
  rag04_hnsw_layers.png       -- SCHEMATIC of HNSW's layered small-world graph: greedy descent from a
                                sparse top layer down to the dense base layer (illustrative).
  rag04_greedy_descent.png    -- the simplified single-layer greedy search on OUR data: hops walking
                                downhill (distance to query falling) toward a local optimum.
  rag04_pq_memory.png         -- product-quantization memory: a raw float32 vector vs its PQ code,
                                and the compression ratio.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x. Headless (Agg); no display needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from vector_indexes import (
    DIM,
    NPROBE_SWEEP,
    PQ_M,
    PQ_NBITS,
    brute_force_topk,
    build_ivf,
    build_knn_graph,
    evaluate_ivf_sweep,
    make_dataset,
    pq_compression_ratio,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / corpus
PURPLE = "#5D4A8A"  # process / cells
GREEN = "#2E7A5A"  # retrieved / good
RED = "#8B3B4A"  # miss / cost
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # query / highlight
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def fig_voronoi_cells() -> None:
    """The corpus partitioned into Voronoi cells (2D); query, true neighbours, probed cells shown.

    Builds a small 2D dataset (so cells are visible), runs the canonical IVF build, and highlights
    the nprobe nearest cells the query actually scans — the routing that lets IVF skip most vectors.
    """
    rng = np.random.default_rng(0)
    n_cells_2d = 12
    centres = rng.normal(0, 6, (6, 2))
    labels = rng.integers(0, 6, 1500)
    corpus = (centres[labels] + rng.normal(0, 1.4, (1500, 2))).astype(np.float32)
    query = np.array([1.5, 1.0], dtype=np.float32)
    index = build_ivf(corpus, n_cells=n_cells_2d, seed=0)
    nprobe = 3
    cell_dist = ((index.centroids - query) ** 2).sum(axis=1)
    probed = set(int(c) for c in np.argsort(cell_dist)[:nprobe])
    true_nn = brute_force_topk(query, corpus, k=10)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _style_axis(ax)
    # colour each point by its cell; probed cells saturated, others faded
    cmap = plt.cm.tab20(np.linspace(0, 1, n_cells_2d))
    for cell, ids in index.cells.items():
        if len(ids) == 0:
            continue
        is_probed = cell in probed
        ax.scatter(corpus[ids, 0], corpus[ids, 1], s=14,
                   color=cmap[cell], alpha=0.85 if is_probed else 0.18,
                   edgecolors="none", zorder=2 if is_probed else 1)
    # cell centroids
    ax.scatter(index.centroids[:, 0], index.centroids[:, 1], s=90, marker="P", color=INK,
               edgecolors="white", linewidths=1.0, zorder=4, label="cell centroids")
    # true neighbours and query
    ax.scatter(corpus[true_nn, 0], corpus[true_nn, 1], s=70, facecolors="none", edgecolors=GREEN,
               linewidths=2.0, zorder=5, label="true top-10 neighbours")
    ax.scatter(*query, s=380, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=6,
               label="query")
    ax.set_title(f"IVF probes only the {nprobe} nearest cells (saturated) — skipping the rest (faded)",
                 fontsize=11.5, pad=12)
    ax.set_xlabel("dim 1 (2D toy for visualization)")
    ax.set_ylabel("dim 2")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag04_voronoi_cells.png")


def fig_recall_cliff() -> None:
    """recall@10 vs nprobe (the cliff) and the % of corpus scanned — measured on our own IVF."""
    corpus, queries = make_dataset()
    ground_truth = np.array([brute_force_topk(q, corpus) for q in queries])
    index = build_ivf(corpus)
    sweep = evaluate_ivf_sweep(index, queries, ground_truth)
    nprobes = list(NPROBE_SWEEP)
    recalls = [sweep[np][0] for np in nprobes]
    fractions = [sweep[np][1] * 100 for np in nprobes]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    ax.plot(nprobes, recalls, marker="o", color=GREEN, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label="recall@10 (want high)")
    for np_, r in zip(nprobes, recalls):
        ax.annotate(f"{r:.2f}", (np_, r), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nprobes)
    ax.set_xticklabels([str(n) for n in nprobes])
    ax.set_xlabel("nprobe (cells scanned, of 64) — log scale")
    ax.set_ylabel("recall@10", color=GREEN)
    ax.set_ylim(0, 1.08)
    ax2 = ax.twinx()
    ax2.plot(nprobes, fractions, marker="s", color=RED, linewidth=2.0, markersize=6,
             markeredgecolor=INK, linestyle="--", label="% corpus scanned (cost)")
    ax2.set_ylabel("% of corpus scanned (cost)", color=RED)
    ax2.tick_params(colors=INK, labelsize=9)
    ax2.spines["top"].set_visible(False)
    # mark the sweet spot (first nprobe reaching >=0.95 recall)
    sweet = next(n for n in nprobes if sweep[n][0] >= 0.95)
    ax.axvline(sweet, color=AMBER, linewidth=1.6, linestyle=":", alpha=0.85)
    ax.annotate(f"sweet spot\nnprobe={sweet}\n(recall {sweep[sweet][0]:.2f}, "
                f"{sweep[sweet][1] * 100:.0f}% scanned)", (sweet, 0.45), color=INK, fontsize=8.5,
                ha="left", va="center", xytext=(8, 0), textcoords="offset points")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.95, fontsize=8.5)
    ax.set_title("The recall cliff: low nprobe is fast but misses neighbours", fontsize=12, pad=12)
    _save(fig, "rag04_recall_cliff.png")


def fig_bruteforce_growth() -> None:
    """Brute-force multiply-adds per query as N grows (linear) vs IVF's near-flat probed cost."""
    n_values = np.array([1e3, 1e4, 1e5, 1e6, 1e7], dtype=float)
    dim = 768  # a realistic production embedding dim for the cost illustration
    brute = n_values * dim  # O(N*d): every vector scanned
    # IVF with nlist = sqrt(N) cells (a common rule) and nprobe scanning ~ a handful of cells:
    nlist = np.sqrt(n_values)
    nprobe = 8
    ivf = nlist * dim + (n_values / nlist) * nprobe * dim  # centroid scan + probed-cell scan

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _style_axis(ax)
    ax.plot(n_values, brute, marker="o", color=RED, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label="brute force  O(N·d)")
    ax.plot(n_values, ivf, marker="s", color=GREEN, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label=f"IVF  O(√N·d + (N/√N)·{nprobe}·d)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size N (log scale)")
    ax.set_ylabel("multiply-adds per query (log scale)")
    ax.annotate("at 10M×768:\n~7.7B per query", (1e7, 1e7 * dim), color=RED, fontsize=8.5,
                ha="right", va="bottom", xytext=(-6, 8), textcoords="offset points")
    ax.set_title("Brute force grows linearly with N; IVF stays far below (illustrative asymptotic shapes)",
                 fontsize=11.5, pad=12)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    _save(fig, "rag04_bruteforce_growth.png")


def fig_hnsw_layers() -> None:
    """SCHEMATIC of HNSW's layered small-world graph: greedy descent top → base (illustrative)."""
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axis("off")
    layers = [(2, 0.82, "Layer 2 — sparse, long links"),
              (5, 0.5, "Layer 1 — medium density"),
              (14, 0.16, "Layer 0 — dense base (all points)")]
    # consistent x positions for points shared across layers
    base_x = np.sort(rng.uniform(0.08, 0.92, 14))
    query_x = 0.74
    prev_pick, prev_y = None, 0.0  # track the previous layer's chosen node for the descent arrow
    for n_pts, y, label in layers:
        xs = base_x[:: max(1, len(base_x) // n_pts)][:n_pts] if n_pts < len(base_x) else base_x
        ax.scatter(xs, [y] * len(xs), s=130, color=BLUE, alpha=0.5, edgecolors=INK, zorder=3)
        # link neighbours within the layer
        for i in range(len(xs) - 1):
            ax.plot([xs[i], xs[i + 1]], [y, y], color=SLATE, linewidth=0.8, alpha=0.4, zorder=2)
        ax.text(0.02, y, label, fontsize=9, color=INK, va="center", ha="left")
        # greedy pick: the point on this layer closest to the query x
        pick = xs[int(np.argmin(np.abs(xs - query_x)))]
        ax.scatter(pick, y, s=200, color=GREEN, edgecolors=INK, linewidths=1.4, zorder=4)
        if prev_pick is not None:
            ax.annotate("", (pick, y + 0.02), (prev_pick, prev_y - 0.02),
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.0))
        prev_pick, prev_y = pick, y
    # the query column
    ax.axvline(query_x, color=AMBER, linewidth=1.2, linestyle=":", alpha=0.7)
    ax.text(query_x, 0.9, "query", color=AMBER, fontsize=10, ha="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.02)
    ax.set_title("HNSW: greedy descent from a sparse top layer to the dense base "
                 "(schematic, illustrative)", fontsize=11.5, color=INK)
    ax.text(0.5, 0.02, "green = node greedily chosen at each layer · amber arrows = the descent path",
            ha="center", fontsize=8.5, color=INK, style="italic")
    _save(fig, "rag04_hnsw_layers.png")


def fig_greedy_descent() -> None:
    """The simplified single-layer greedy search on OUR data: distance-to-query falls per hop."""
    corpus, queries = make_dataset()
    sub = corpus[:2000]
    graph = build_knn_graph(sub)
    # trace one representative query's hop path, recording distance to the query at each node
    query = queries[3]
    current, path_dist = 0, []
    current_dist = float(((sub[current] - query) ** 2).sum())
    path_dist.append(current_dist)
    for _ in range(50):  # safety bound; greedy stops at a local optimum well before this
        neighbors = graph[current]
        nd = ((sub[neighbors] - query) ** 2).sum(axis=1)
        best = int(np.argmin(nd))
        if nd[best] >= current_dist:
            break
        current, current_dist = neighbors[best], float(nd[best])
        path_dist.append(current_dist)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _style_axis(ax)
    hops = list(range(len(path_dist)))
    ax.plot(hops, path_dist, marker="o", color=PURPLE, linewidth=2.4, markersize=8,
            markeredgecolor=INK, markerfacecolor=GREEN)
    for h, d in zip(hops, path_dist):
        ax.annotate(f"{d:.0f}", (h, d), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_title("Greedy graph search walks downhill: distance to the query falls each hop",
                 fontsize=11.5, pad=12)
    ax.set_xlabel("hop number")
    ax.set_ylabel("squared distance to the query")
    ax.set_xticks(hops)
    _save(fig, "rag04_greedy_descent.png")


def fig_pq_memory() -> None:
    """Product-quantization memory: a raw float32 vector vs its PQ code, and the compression ratio."""
    raw_bits, pq_bits, ratio = pq_compression_ratio(dim=DIM, m=PQ_M, nbits=PQ_NBITS)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _style_axis(ax)
    labels = [f"raw float32\n({DIM} dims × 32 bits)", f"PQ code\n({PQ_M} subq × {PQ_NBITS} bits)"]
    values = [raw_bits, pq_bits]
    bars = ax.bar(labels, values, color=[RED, GREEN], edgecolor=INK, linewidth=0.8, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:,} bits\n({v // 8} bytes)", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=9, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylabel("bits per vector (log scale)")
    ax.set_ylim(10, raw_bits * 3)
    ax.set_title(f"Product Quantization: {ratio:.0f}× smaller per vector", fontsize=12.5, pad=12)
    _save(fig, "rag04_pq_memory.png")


def main() -> None:
    fig_voronoi_cells()
    fig_recall_cliff()
    fig_bruteforce_growth()
    fig_hnsw_layers()
    fig_greedy_descent()
    fig_pq_memory()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

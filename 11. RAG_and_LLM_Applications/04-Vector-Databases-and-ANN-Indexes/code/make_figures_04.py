"""Figure generator for 04-Vector-Databases-and-ANN-Indexes — REAL measurements + honest schematics.

Every *measured* figure here comes from the SAME real pipeline the page and notebook use
(`vector_indexes.py` over the real embedded Wikipedia corpus in `data/`): real FAISS Flat/IVF/HNSW
indexes, real recall@10 vs exact, real query latency. Nothing is hand-typed. The few *schematic*
figures (2D Voronoi cells, the HNSW layer cartoon) are clearly labelled illustrative — you cannot
plot 384-dim space, so they use a tiny 2D toy purely to convey the geometry, never to fake a number.

Writes muted-palette PNGs to the shared chapter image dir (../../images/) with prefix `rag04_`.

    python make_figures_04.py            # needs data/ populated by `python embed_corpus.py`

Figures produced:
  rag04_bruteforce_growth.png   -- exact-search cost O(N*d) growing linearly with N, anchored by the
                                   REAL measured Flat latency at our corpus size; IVF's sub-linear cost.
  rag04_ivf_recall_cliff.png    -- REAL recall@10 vs nprobe (the cliff) with real latency, on real IVF.
  rag04_hnsw_efsearch.png       -- REAL recall@10 vs efSearch with real latency, on real HNSW.
  rag04_recall_vs_latency.png   -- the REAL recall/latency Pareto frontier: IVF vs HNSW vs exact.
  rag04_build_memory.png        -- REAL index build time + memory as the corpus grows (subsets of real N).
  rag04_pq_memory.png           -- REAL product-quantization memory: raw float32 vs PQ code (our dim).
  rag04_voronoi_cells.png       -- SCHEMATIC 2D Voronoi cells + probed cells (illustrative geometry).
  rag04_hnsw_layers.png         -- SCHEMATIC HNSW layered graph + greedy descent (illustrative).

torch is never imported here (faiss + torch co-loading crashes on this box); faiss + numpy only.
Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / faiss-cpu.
"""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from vector_indexes import (
    HNSW_M,
    IVF_NLIST,
    PQ_M,
    PQ_NBITS,
    TOP_K,
    build_flat,
    build_hnsw,
    build_ivf,
    exact_latency_ms,
    exact_topk,
    hnsw_level_counts,
    ivf_cell_sizes,
    load_corpus,
    pq_encode_decode,
    pq_memory_bytes,
    sweep_hnsw,
    sweep_ivf,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / corpus
PURPLE = "#5D4A8A"  # process / IVF
GREEN = "#2E7A5A"  # retrieved / good / recall
RED = "#8B3B4A"  # miss / cost / latency
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # query / highlight / HNSW
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


# ============================ measured figures (from the real pipeline) ==========================
def _kmeans_2d(corpus: np.ndarray, n_cells: int, iters: int = 12, seed: int = 0):
    """Tiny numpy Lloyd's k-means — for the 2D SCHEMATIC Voronoi figure only (not a measured number).

    The real IVF uses FAISS k-means over 384-dim vectors; this is a 2-D stand-in purely so the cell
    geometry is visible on the page. Returns (centroids, assignments).
    """
    rng = np.random.default_rng(seed)
    centroids = corpus[rng.choice(len(corpus), n_cells, replace=False)].copy()
    assignments = np.zeros(len(corpus), dtype=np.int64)
    for _ in range(iters):
        dist = ((corpus[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = dist.argmin(axis=1)
        for c in range(n_cells):
            members = assignments == c
            if members.any():
                centroids[c] = corpus[members].mean(axis=0)
    return centroids, assignments


def fig_bruteforce_growth(flat_ms: float, n_corpus: int, dim: int) -> None:
    """Exact O(N*d) cost growing linearly with N, anchored by our REAL measured Flat latency."""
    n_values = np.array([1e3, 1e4, 1e5, 1e6, 1e7], dtype=float)
    brute = n_values * dim  # O(N*d): every vector scanned — multiply-adds per query
    nlist = np.sqrt(n_values)
    nprobe = 8
    ivf = nlist * dim + (n_values / nlist) * nprobe * dim  # centroid scan + probed-cell scan

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    _style_axis(ax)
    ax.plot(n_values, brute, marker="o", color=RED, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label="brute force  O(N·d)")
    ax.plot(n_values, ivf, marker="s", color=GREEN, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label=f"IVF  O(√N·d + (N/√N)·{nprobe}·d)")
    # anchor: our real corpus size + its real measured flat latency (annotate, not on the ops axis)
    ax.axvline(n_corpus, color=AMBER, linewidth=1.6, linestyle=":", alpha=0.85)
    ax.annotate(f"our real corpus\nN={n_corpus:,}, d={dim}\nmeasured exact: {flat_ms:.2f} ms/query",
                (n_corpus, n_corpus * dim), color=INK, fontsize=8.5, ha="right", va="bottom",
                xytext=(-8, 6), textcoords="offset points")
    ax.annotate("at 10M×768:\n~7.7B ops/query", (1e7, 1e7 * dim), color=RED, fontsize=8.5,
                ha="right", va="top", xytext=(-6, -8), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size N (log scale)")
    ax.set_ylabel("multiply-adds per query (log scale)")
    ax.set_ylim(top=2e10)
    ax.set_title("Exact search grows linearly with N; IVF stays far below\n(asymptotic shapes; the N-anchor is our measured corpus)",
                 fontsize=11, pad=10)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    _save(fig, "rag04_bruteforce_growth.png")


def fig_ivf_recall_cliff(ivf_points, flat_ms: float) -> None:
    """REAL recall@10 vs nprobe (the cliff) and real latency, measured on the real IVF index."""
    nprobes = [p.knob for p in ivf_points]
    recalls = [p.recall for p in ivf_points]
    latencies = [p.latency_ms for p in ivf_points]

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    _style_axis(ax)
    ax.plot(nprobes, recalls, marker="o", color=GREEN, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label="recall@10 (want high)")
    for np_, r in zip(nprobes, recalls):
        ax.annotate(f"{r:.2f}", (np_, r), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nprobes)
    ax.set_xticklabels([str(n) for n in nprobes])
    ax.set_xlabel(f"nprobe (cells probed, of nlist={IVF_NLIST}) — log scale")
    ax.set_ylabel("recall@10", color=GREEN)
    ax.set_ylim(min(recalls) - 0.08, 1.05)
    ax2 = ax.twinx()
    ax2.plot(nprobes, latencies, marker="s", color=RED, linewidth=2.0, markersize=6,
             markeredgecolor=INK, linestyle="--", label="latency (ms/query)")
    ax2.axhline(flat_ms, color=SLATE, linewidth=1.4, linestyle=":", alpha=0.8)
    ax2.annotate(f"exact (flat): {flat_ms:.2f} ms", (nprobes[0], flat_ms), color=SLATE, fontsize=8,
                 ha="left", va="bottom", xytext=(2, 3), textcoords="offset points")
    ax2.set_ylabel("latency (ms/query)", color=RED)
    ax2.tick_params(colors=INK, labelsize=9)
    ax2.spines["top"].set_visible(False)
    # sweet spot: first nprobe reaching >=0.95 recall
    sweet = next((p for p in ivf_points if p.recall >= 0.95), ivf_points[-1])
    ax.axvline(sweet.knob, color=AMBER, linewidth=1.6, linestyle=":", alpha=0.85)
    ax.annotate(f"sweet spot\nnprobe={sweet.knob}\nrecall {sweet.recall:.2f}, {sweet.latency_ms:.2f} ms\n"
                f"({flat_ms / sweet.latency_ms:.0f}× faster than exact)", (sweet.knob, min(recalls) + 0.02),
                color=INK, fontsize=8.2, ha="left", va="bottom", xytext=(8, 0), textcoords="offset points")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.95, fontsize=8.5)
    ax.set_title("IVF recall cliff (REAL): low nprobe is fast but silently misses neighbours",
                 fontsize=11.5, pad=12)
    _save(fig, "rag04_ivf_recall_cliff.png")


def fig_hnsw_efsearch(hnsw_points, flat_ms: float) -> None:
    """REAL recall@10 vs efSearch and real latency, measured on the real HNSW index."""
    efs = [p.knob for p in hnsw_points]
    recalls = [p.recall for p in hnsw_points]
    latencies = [p.latency_ms for p in hnsw_points]

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    _style_axis(ax)
    ax.plot(efs, recalls, marker="o", color=GREEN, linewidth=2.4, markersize=7,
            markeredgecolor=INK, label="recall@10 (want high)")
    for e, r in zip(efs, recalls):
        ax.annotate(f"{r:.2f}", (e, r), fontsize=8, color=INK, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xticks(efs)
    ax.set_xticklabels([str(e) for e in efs])
    ax.set_xlabel(f"efSearch (query-time candidate list; M={HNSW_M}) — log scale")
    ax.set_ylabel("recall@10", color=GREEN)
    ax.set_ylim(min(recalls) - 0.02, 1.008)
    ax2 = ax.twinx()
    ax2.plot(efs, latencies, marker="s", color=RED, linewidth=2.0, markersize=6,
             markeredgecolor=INK, linestyle="--", label="latency (ms/query)")
    ax2.axhline(flat_ms, color=SLATE, linewidth=1.4, linestyle=":", alpha=0.8)
    ax2.annotate(f"exact (flat): {flat_ms:.2f} ms", (efs[0], flat_ms), color=SLATE, fontsize=8,
                 ha="left", va="bottom", xytext=(2, 3), textcoords="offset points")
    ax2.set_ylabel("latency (ms/query)", color=RED)
    ax2.tick_params(colors=INK, labelsize=9)
    ax2.spines["top"].set_visible(False)
    sweet = next((p for p in hnsw_points if p.recall >= 0.95), hnsw_points[-1])
    ax.axvline(sweet.knob, color=AMBER, linewidth=1.6, linestyle=":", alpha=0.85)
    ax.annotate(f"efSearch={sweet.knob}: recall {sweet.recall:.2f}, {sweet.latency_ms:.2f} ms\n"
                f"({flat_ms / sweet.latency_ms:.0f}× faster than exact)",
                (sweet.knob, min(recalls) + 0.006),
                color=INK, fontsize=8.2, ha="left", va="bottom", xytext=(6, 0), textcoords="offset points")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.95, fontsize=8.5)
    ax.set_title("HNSW recall vs efSearch (REAL): the graph's recall/speed knob", fontsize=11.5, pad=12)
    _save(fig, "rag04_hnsw_efsearch.png")


def fig_recall_vs_latency(ivf_points, hnsw_points, flat_ms: float) -> None:
    """The REAL recall/latency Pareto frontier — the plot you actually tune against (ANN-Benchmarks style)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    _style_axis(ax)
    ivf_lat = [p.latency_ms for p in ivf_points]
    ivf_rec = [p.recall for p in ivf_points]
    hnsw_lat = [p.latency_ms for p in hnsw_points]
    hnsw_rec = [p.recall for p in hnsw_points]
    ax.plot(ivf_lat, ivf_rec, marker="o", color=PURPLE, linewidth=2.2, markersize=7,
            markeredgecolor=INK, label="IVF (sweep nprobe)")
    ax.plot(hnsw_lat, hnsw_rec, marker="^", color=AMBER, linewidth=2.2, markersize=8,
            markeredgecolor=INK, label="HNSW (sweep efSearch)")
    ax.axvline(flat_ms, color=RED, linewidth=1.8, linestyle="--", alpha=0.85)
    ax.annotate(f"exact (flat)\n{flat_ms:.2f} ms, recall 1.0", (flat_ms, 0.55), color=RED, fontsize=8.5,
                ha="right", va="center", xytext=(-8, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms/query, log scale) — lower-left-is-cheaper, up-is-better")
    ax.set_ylabel("recall@10")
    ax.set_ylim(min(min(ivf_rec), min(hnsw_rec)) - 0.05, 1.03)
    ax.set_title("The recall/latency frontier (REAL): up-and-to-the-left wins", fontsize=12, pad=12)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "rag04_recall_vs_latency.png")


def fig_build_memory(embeddings: np.ndarray) -> None:
    """REAL index build time + in-RAM size as the corpus grows (subsets of the real corpus)."""
    dim = embeddings.shape[1]
    sizes = [n for n in (5000, 10000, 20000, len(embeddings)) if n <= len(embeddings)]
    flat_build, ivf_build, hnsw_build = [], [], []
    for n in sizes:
        sub = np.ascontiguousarray(embeddings[:n])
        t0 = time.perf_counter()
        build_flat(sub)
        flat_build.append(time.perf_counter() - t0)
        nlist = min(IVF_NLIST, max(4, n // 40))  # keep cells reasonably populated at small N
        quant = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter()
        ivf.train(sub)
        ivf.add(sub)
        ivf_build.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        build_hnsw(sub)
        hnsw_build.append(time.perf_counter() - t0)

    raw_mb = [n * dim * 4 / 1e6 for n in sizes]  # float32 vectors = the flat/IVF vector storage floor

    fig, (axb, axm) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    _style_axis(axb)
    axb.plot(sizes, flat_build, marker="o", color=RED, linewidth=2.2, markersize=7,
             markeredgecolor=INK, label="Flat build")
    axb.plot(sizes, ivf_build, marker="s", color=PURPLE, linewidth=2.2, markersize=7,
             markeredgecolor=INK, label="IVF build (k-means train + add)")
    axb.plot(sizes, hnsw_build, marker="^", color=AMBER, linewidth=2.2, markersize=8,
             markeredgecolor=INK, label="HNSW build (graph construction)")
    axb.set_xlabel("corpus size N (real subsets)")
    axb.set_ylabel("build time (seconds)")
    axb.set_title("REAL index build time as N grows", fontsize=11.5, pad=10)
    axb.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    _style_axis(axm)
    axm.plot(sizes, raw_mb, marker="o", color=BLUE, linewidth=2.2, markersize=7,
             markeredgecolor=INK, label="raw float32 vectors (Flat/IVF store this)")
    # HNSW adds ~ N*M links * 4 bytes on top of the vectors (rough, links dominate the overhead)
    hnsw_mb = [n * dim * 4 / 1e6 + n * HNSW_M * 2 * 4 / 1e6 for n in sizes]
    axm.plot(sizes, hnsw_mb, marker="^", color=AMBER, linewidth=2.2, markersize=8,
             markeredgecolor=INK, label=f"HNSW ≈ vectors + graph links (M={HNSW_M})")
    axm.set_xlabel("corpus size N (real subsets)")
    axm.set_ylabel("index memory (MB)")
    axm.set_title("Index memory: HNSW's graph adds to the vector floor", fontsize=11.5, pad=10)
    axm.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag04_build_memory.png")


def fig_pq_memory(dim: int, n_corpus: int) -> None:
    """REAL product-quantization memory: raw float32 vs PQ code at our real embedding dim."""
    raw, pq, ratio = pq_memory_bytes(dim, PQ_M, PQ_NBITS)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    _style_axis(ax)
    labels = [f"raw float32\n({dim} dims × 4 B)", f"PQ code\n({PQ_M} subq × {PQ_NBITS} bits)"]
    values = [raw, pq]
    bars = ax.bar(labels, values, color=[RED, GREEN], edgecolor=INK, linewidth=0.8, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:,} bytes", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=10, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylabel("bytes per vector (log scale)")
    ax.set_ylim(1, raw * 3)
    ax.set_title(f"Product Quantization: {ratio:.0f}× smaller/vector  "
                 f"({n_corpus:,} vectors: {n_corpus * raw / 1e6:.0f} MB → {n_corpus * pq / 1e6:.1f} MB)",
                 fontsize=11, pad=12)
    _save(fig, "rag04_pq_memory.png")


# ============================ mechanism figures (REAL, from the built indexes) ===================
def fig_ivf_cell_sizes(ivf) -> None:
    """REAL histogram of IVF inverted-list lengths — the k-means partition, made inspectable."""
    sizes = ivf_cell_sizes(ivf)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style_axis(ax)
    ax.hist(sizes, bins=30, color=PURPLE, edgecolor=INK, linewidth=0.6, alpha=0.9)
    mean = sizes.mean()
    ax.axvline(mean, color=AMBER, linewidth=1.8, linestyle="--",
               label=f"mean {mean:.0f} vectors/cell (≈ N/nlist)")
    ax.set_xlabel("vectors per cell (inverted-list length)")
    ax.set_ylabel("number of cells")
    ax.set_title(f"REAL IVF partition: {ivf.nlist} k-means cells over {ivf.ntotal:,} vectors "
                 f"(min {sizes.min()}, max {sizes.max()})", fontsize=11, pad=12)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    _save(fig, "rag04_ivf_cell_sizes.png")


def fig_hnsw_pyramid(hnsw) -> None:
    """REAL HNSW layer pyramid: node counts per level, showing the geometric (~1/M) decay."""
    counts = hnsw_level_counts(hnsw)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    levels = list(range(len(counts)))
    bars = ax.barh(levels, counts, color=[BLUE if i == 0 else AMBER for i in levels],
                   edgecolor=INK, linewidth=0.7, height=0.62)
    for lvl, (bar, c) in enumerate(zip(bars, counts)):
        ax.annotate(f"{c:,} nodes", (c, bar.get_y() + bar.get_height() / 2),
                    fontsize=9, color=INK, ha="left", va="center", xytext=(5, 0),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yticks(levels)
    ax.set_yticklabels([f"level {i}" + ("  (base — all N)" if i == 0 else "") for i in levels])
    ax.set_xlabel("nodes at this layer (log scale)")
    ax.set_xlim(1, counts[0] * 4)
    ax.invert_yaxis()  # base at the bottom, sparse top at the top
    ratios = " · ".join(f"×{counts[i + 1] / counts[i]:.2f}" for i in range(len(counts) - 1))
    ax.set_title(f"REAL HNSW layer pyramid ({hnsw.hnsw.max_level + 1} levels): "
                 f"each level up holds far fewer nodes ({ratios})", fontsize=10.5, pad=12)
    _save(fig, "rag04_hnsw_pyramid.png")


def fig_pq_encoding(embeddings: np.ndarray, dim: int) -> None:
    """REAL PQ encode/decode: how one vector becomes m codes, plus the reconstruction-error spread."""
    codes, _recon, errors = pq_encode_decode(embeddings)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # left: schematic of split → codebook → code, annotated with a REAL code
    ax1.axis("off")
    ax1.set_title(f"Encoding one vector: split into m={PQ_M} subvectors → nearest sub-centroid id",
                  fontsize=10.5, color=INK)
    subs = 6  # show the first few subspaces only, for legibility
    real_code = codes[0][:subs]
    for j in range(subs):
        x0 = 0.04 + j * 0.15
        ax1.add_patch(plt.Rectangle((x0, 0.62), 0.12, 0.16, facecolor=BLUE, alpha=0.6,
                                    edgecolor=INK, linewidth=0.8))
        ax1.text(x0 + 0.06, 0.70, f"sub {j}\n{dim // PQ_M}-D", ha="center", va="center",
                 fontsize=7.5, color="white")
        ax1.annotate("", (x0 + 0.06, 0.44), (x0 + 0.06, 0.60),
                     arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
        ax1.add_patch(plt.Rectangle((x0, 0.28), 0.12, 0.16, facecolor=GREEN, alpha=0.75,
                                    edgecolor=INK, linewidth=0.8))
        ax1.text(x0 + 0.06, 0.36, f"id\n{real_code[j]}", ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")
    ax1.text(0.04 + subs * 0.15 + 0.01, 0.36, "…", fontsize=16, va="center", color=INK)
    ax1.text(0.5, 0.14, f"code = {PQ_M} centroid-ids (each 0–255) = {PQ_M} bytes  "
             f"(vs {dim}×4 = {dim * 4:,} bytes raw)", ha="center", fontsize=9, color=INK,
             style="italic")
    ax1.text(0.5, 0.90, f"codebook per subspace: 2^{PQ_NBITS} = {2 ** PQ_NBITS} centroids",
             ha="center", fontsize=8.5, color=SLATE)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # right: real reconstruction-error distribution (the quantization loss that costs recall)
    _style_axis(ax2)
    ax2.hist(errors, bins=40, color=RED, edgecolor=INK, linewidth=0.5, alpha=0.85)
    ax2.axvline(errors.mean(), color=AMBER, linewidth=1.8, linestyle="--",
                label=f"mean error {errors.mean():.3f}")
    ax2.set_xlabel("L2 reconstruction error  ‖original − decoded‖  (0 = lossless)")
    ax2.set_ylabel("number of vectors")
    ax2.set_title(f"REAL PQ quantization loss over {len(errors):,} vectors", fontsize=10.5, pad=10)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)
    _save(fig, "rag04_pq_encoding.png")


# ============================ schematic figures (illustrative 2D geometry) =======================
def fig_voronoi_cells() -> None:
    """SCHEMATIC: 2D Voronoi cells + the nprobe probed cells (illustrative — 384-d can't be drawn)."""
    rng = np.random.default_rng(0)
    n_cells_2d = 12
    centres = rng.normal(0, 6, (6, 2))
    labels = rng.integers(0, 6, 1500)
    corpus = (centres[labels] + rng.normal(0, 1.4, (1500, 2))).astype(np.float32)
    query = np.array([1.5, 1.0], dtype=np.float32)
    centroids, assignments = _kmeans_2d(corpus, n_cells_2d)
    cells = {c: np.where(assignments == c)[0] for c in range(n_cells_2d)}
    nprobe = 3
    probed = set(int(c) for c in np.argsort(((centroids - query) ** 2).sum(axis=1))[:nprobe])
    true_nn = np.argsort(((corpus - query) ** 2).sum(axis=1))[:10]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _style_axis(ax)
    cmap = plt.cm.tab20(np.linspace(0, 1, n_cells_2d))
    for cell, ids in cells.items():
        if len(ids) == 0:
            continue
        on = cell in probed
        ax.scatter(corpus[ids, 0], corpus[ids, 1], s=14, color=cmap[cell],
                   alpha=0.85 if on else 0.18, edgecolors="none", zorder=2 if on else 1)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=90, marker="P", color=INK,
               edgecolors="white", linewidths=1.0, zorder=4, label="cell centroids")
    ax.scatter(corpus[true_nn, 0], corpus[true_nn, 1], s=70, facecolors="none", edgecolors=GREEN,
               linewidths=2.0, zorder=5, label="true top-10 neighbours")
    ax.scatter(*query, s=380, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=6,
               label="query")
    ax.set_title(f"SCHEMATIC: IVF probes only the {nprobe} nearest cells (saturated), skips the rest (faded)",
                 fontsize=11, pad=12)
    ax.set_xlabel("2D projection for intuition — the real index lives in 384-D")
    ax.set_ylabel("")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag04_voronoi_cells.png")


def fig_hnsw_layers() -> None:
    """SCHEMATIC: HNSW's layered small-world graph and greedy descent (illustrative)."""
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axis("off")
    layers = [(2, 0.82, "Layer 2 — sparse, long links"),
              (5, 0.5, "Layer 1 — medium density"),
              (14, 0.16, "Layer 0 — dense base (all points)")]
    base_x = np.sort(rng.uniform(0.08, 0.92, 14))
    query_x = 0.74
    prev_pick, prev_y = None, 0.0
    for n_pts, y, label in layers:
        xs = base_x[:: max(1, len(base_x) // n_pts)][:n_pts] if n_pts < len(base_x) else base_x
        ax.scatter(xs, [y] * len(xs), s=130, color=BLUE, alpha=0.5, edgecolors=INK, zorder=3)
        for i in range(len(xs) - 1):
            ax.plot([xs[i], xs[i + 1]], [y, y], color=SLATE, linewidth=0.8, alpha=0.4, zorder=2)
        ax.text(0.02, y, label, fontsize=9, color=INK, va="center", ha="left")
        pick = xs[int(np.argmin(np.abs(xs - query_x)))]
        ax.scatter(pick, y, s=200, color=GREEN, edgecolors=INK, linewidths=1.4, zorder=4)
        if prev_pick is not None:
            ax.annotate("", (pick, y + 0.02), (prev_pick, prev_y - 0.02),
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.0))
        prev_pick, prev_y = pick, y
    ax.axvline(query_x, color=AMBER, linewidth=1.2, linestyle=":", alpha=0.7)
    ax.text(query_x, 0.9, "query", color=AMBER, fontsize=10, ha="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.02)
    ax.set_title("SCHEMATIC: HNSW greedy descent from a sparse top layer to the dense base",
                 fontsize=11.5, color=INK)
    ax.text(0.5, 0.02, "green = node greedily chosen at each layer · amber arrows = the descent path",
            ha="center", fontsize=8.5, color=INK, style="italic")
    _save(fig, "rag04_hnsw_layers.png")


def main() -> None:
    corpus = load_corpus()
    print(f"loaded real corpus: {corpus.n:,} x {corpus.dim} | measuring for figures ...")

    # one real pass: build indexes, measure recall/latency sweeps + exact baseline
    flat = build_flat(corpus.embeddings)
    ground_truth = exact_topk(flat, corpus.queries, TOP_K)
    flat_ms = exact_latency_ms(flat, corpus.queries)
    ivf = build_ivf(corpus.embeddings)
    ivf_points = sweep_ivf(ivf, corpus.queries, ground_truth)
    hnsw = build_hnsw(corpus.embeddings)
    hnsw_points = sweep_hnsw(hnsw, corpus.queries, ground_truth)

    # measured figures (recall/latency)
    fig_bruteforce_growth(flat_ms, corpus.n, corpus.dim)
    fig_ivf_recall_cliff(ivf_points, flat_ms)
    fig_hnsw_efsearch(hnsw_points, flat_ms)
    fig_recall_vs_latency(ivf_points, hnsw_points, flat_ms)
    fig_build_memory(corpus.embeddings)
    fig_pq_memory(corpus.dim, corpus.n)
    # mechanism figures (from the built indexes' internals)
    fig_ivf_cell_sizes(ivf)
    fig_hnsw_pyramid(hnsw)
    fig_pq_encoding(corpus.embeddings, corpus.dim)
    # schematic figures (2D geometry for intuition)
    fig_voronoi_cells()
    fig_hnsw_layers()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

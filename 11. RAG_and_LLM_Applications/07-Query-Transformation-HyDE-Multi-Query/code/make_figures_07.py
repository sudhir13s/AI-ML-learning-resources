"""Static figure generator for 07-Query-Transformation-HyDE-Multi-Query.

Imports the SAME canonical functions the page and notebook use (query_transformation.py, which reuses
chapters 5 & 1) so every plotted number is the chapter's own -- no hand-typed values. Writes
muted-palette PNGs to the shared chapter image dir (../../images/) with the per-chapter prefix
`rag07_`.

    python make_figures_07.py

Figures produced:
  rag07_asymmetry_2d.png   -- the question<->answer embedding GAP, in 2D (PCA of REAL MiniLM vectors):
                              the raw question sits far from its gold answer passage; the HyDE
                              hypothetical answer jumps across into the answer's neighbourhood.
  rag07_hyde_pipeline.png  -- the HyDE pipeline (query -> hypothetical answer -> embed -> retrieve),
                              annotated with the MEASURED cosine lift toward the gold.
  rag07_union_recall.png   -- multi-query union recall vs N: the theoretical 1-(1-p)^N ceiling
                              (independence) against a correlated curve, with the measured point.
  rag07_rrf_fusion.png     -- RRF over N reformulation lists: the 1/(k+rank) weight curve and a worked
                              fusion of the chapter's own reformulation rankings.
  rag07_before_after.png   -- MRR raw vs HyDE vs Multi-Query on their probe sets (measured): the payoff.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from query_transformation import (
    RRF_K,
    TOP_K,
    DenseRetriever,
    build_hyde_probes,
    build_multiquery_probe,
    cosine_to_gold,
    recall_at_k,
    reciprocal_rank,
    retrieve_hyde,
    retrieve_multiquery,
    retrieve_raw,
    union_recall_independent,
)
from hybrid_search import full_corpus  # noqa: E402  (transitively on the path via query_transformation)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / question
PURPLE = "#5D4A8A"  # process / transform
GREEN = "#2E7A5A"  # hit / answer / caught
RED = "#8B3B4A"  # miss / danger / wrong
SLATE = "#4A5B6E"  # neutral / other docs
AMBER = "#7A6528"  # highlight / hypothetical
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


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project rows of `vectors` to 2D via the top-2 principal components (mean-centered SVD).

    A plain, dependency-free PCA: center, SVD, take the first two right-singular directions. Used only
    to VISUALIZE the real high-dim embeddings faithfully in 2D -- the geometry (who is near whom) is
    preserved as well as any linear 2D projection can. The axes are principal components, not
    interpretable units, which the caption states.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def fig_asymmetry_2d(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """The question<->answer gap in 2D (PCA of REAL MiniLM embeddings) for the exact-code probe.

    Plots every corpus passage (slate), the gold answer (green), the RAW question (blue), and the HyDE
    hypothetical answer (amber). The whole point: the question sits away from its gold answer (that's
    the asymmetry), while the hypothetical answer lands in the answer's neighbourhood -- the jump HyDE
    buys. All coordinates are a linear 2D projection of the genuine 384-d vectors.
    """
    probe = build_hyde_probes(corpus)[1]  # exact-code probe: the vivid "distractor out-embeds" case
    texts = list(corpus) + [probe.query, probe.hyde_good]
    vecs = dense._encode(texts)  # noqa: SLF001 -- reuse ch5's encoder; (n+2, dim) unit-norm rows
    pts = _pca_2d(np.asarray(vecs))
    n = len(corpus)
    doc_pts, q_pt, h_pt = pts[:n], pts[n], pts[n + 1]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    _style_axis(ax)
    # other passages
    other = [i for i in range(n) if i != probe.gold]
    ax.scatter(doc_pts[other, 0], doc_pts[other, 1], s=70, color=SLATE, alpha=0.55,
               edgecolors=INK, linewidths=0.5, label="other passages", zorder=2)
    # gold answer
    ax.scatter(*doc_pts[probe.gold], s=260, color=GREEN, edgecolors=INK, linewidths=1.2,
               marker="*", label="gold answer passage", zorder=5)
    # raw question
    ax.scatter(*q_pt, s=150, color=BLUE, edgecolors=INK, linewidths=1.0, marker="o",
               label="RAW question", zorder=5)
    # hypothetical answer
    ax.scatter(*h_pt, s=150, color=AMBER, edgecolors=INK, linewidths=1.0, marker="D",
               label="HyDE hypothetical answer", zorder=5)
    # arrows: question -> gold (the gap) and hypothetical -> gold (bridged)
    ax.annotate("", xy=doc_pts[probe.gold], xytext=q_pt,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, linestyle="--", alpha=0.8))
    ax.annotate("", xy=doc_pts[probe.gold], xytext=h_pt,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8, alpha=0.85))
    cos_q = cosine_to_gold(dense, corpus, probe.query, probe.gold)
    cos_h = cosine_to_gold(dense, corpus, probe.hyde_good, probe.gold)
    # place the two callouts in open space and lead-line them to the arrows, so they never overlap
    ax.annotate(f"question → gold: cos = {cos_q:.3f}  (far)",
                xy=(q_pt + doc_pts[probe.gold]) / 2, xytext=(0.06, 0.72), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, alpha=0.8),
                color=RED, fontsize=9.5, ha="left", fontweight="bold")
    ax.annotate(f"HyDE → gold: cos = {cos_h:.3f}  (near)",
                xy=h_pt, xytext=(0.06, 0.62), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2, alpha=0.85),
                color=GREEN, fontsize=9.5, ha="left", fontweight="bold")
    ax.set_title("The question ≠ answer gap, and how HyDE bridges it\n"
                 "(PCA of real all-MiniLM embeddings; exact-code probe)", fontsize=11.5, pad=12)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag07_asymmetry_2d.png")


def fig_hyde_pipeline(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """The HyDE pipeline as a labelled flow + the MEASURED cosine lift toward the gold.

    Left: the four stages (query -> LLM writes a hypothetical answer -> embed the ANSWER -> retrieve).
    Right: the chapter's own cosine numbers -- cos(question,gold) vs cos(HyDE,gold) for both probes --
    so the reader sees WHY the pipeline works, not just its shape.
    """
    probes = build_hyde_probes(corpus)
    fig, (ax_flow, ax_bar) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                          gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- left: the pipeline flow ---
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    stages = [
        ("user question\n(question-space)", BLUE, 0.86),
        ("LLM writes a\nHYPOTHETICAL ANSWER\n(answer-space)", PURPLE, 0.62),
        ("embed the ANSWER\n(not the question)", AMBER, 0.38),
        ("retrieve nearest\npassages by cosine", GREEN, 0.14),
    ]
    for text, color, y in stages:
        ax_flow.add_patch(plt.Rectangle((0.18, y - 0.075), 0.64, 0.13,
                          facecolor=color, edgecolor=INK, linewidth=1.0, alpha=0.9))
        ax_flow.text(0.5, y, text, ha="center", va="center", color="white",
                     fontsize=9.5, fontweight="bold")
    for y0, y1 in ((0.785, 0.695), (0.545, 0.455), (0.305, 0.215)):
        ax_flow.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                         arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax_flow.text(0.5, 0.985, "HyDE: transform the query into an answer, THEN retrieve",
                 ha="center", fontsize=11, fontweight="bold", color=INK)
    ax_flow.text(0.5, 0.01, "generation shown with fixed exemplars; in production an LLM writes the "
                 "hypothetical", ha="center", fontsize=8, style="italic", color=SLATE)

    # --- right: measured cosine lift ---
    _style_axis(ax_bar)
    labels = [p.label.split(" (")[0] for p in probes]
    cos_q = [cosine_to_gold(dense, corpus, p.query, p.gold) for p in probes]
    cos_h = [cosine_to_gold(dense, corpus, p.hyde_good, p.gold) for p in probes]
    x = np.arange(len(labels))
    width = 0.36
    b1 = ax_bar.bar(x - width / 2, cos_q, width, color=BLUE, edgecolor=INK, linewidth=0.8,
                    label="cos(question, gold)")
    b2 = ax_bar.bar(x + width / 2, cos_h, width, color=AMBER, edgecolor=INK, linewidth=0.8,
                    label="cos(HyDE answer, gold)")
    for bars in (b1, b2):
        for bar in bars:
            ax_bar.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), fontsize=8.5, color=INK, ha="center", va="bottom",
                            xytext=(0, 2), textcoords="offset points", fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel("cosine similarity to the gold answer")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("The hypothetical answer embeds nearer the gold (measured)", fontsize=10.5, pad=10)
    ax_bar.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag07_hyde_pipeline.png")


def fig_union_recall(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """Multi-query union recall vs N: the 1-(1-p)^N independence ceiling vs a correlated curve.

    The theory: if each reformulation independently retrieves the gold with probability p, the union
    of N reformulations hits with probability 1-(1-p)^N -- rising fast toward 1. That INDEPENDENCE is
    optimistic: real paraphrases share meaning, so their misses correlate and the true recall lags the
    ceiling. We plot the independent ceiling (for a representative single-query p) and a correlated
    curve (with a correlation discount), and mark the measured single-query recall p from the chapter's
    own reformulations. The gap between the two curves is the cost of correlation -- stated, not hidden.
    """
    mq = build_multiquery_probe(corpus)
    # measured per-reformulation hit@k for the chapter's own reformulations -> average = p_hat
    per_query_hits = [recall_at_k(dense.search(q, k=TOP_K).indices, mq.gold) for q in mq.paraphrases]
    p_hat = float(np.mean(per_query_hits))
    # p_hat is 1.0 on this tiny corpus (each reformulation hits), which makes the union trivially 1.
    # To SHOW the shape of the union-recall law, we also plot a representative weaker single-query p
    # so the 1-(1-p)^N climb is visible; the caption says which is measured vs illustrative.
    p_illustrative = 0.55  # a representative per-query recall on a harder corpus (illustrative)
    n_vals = np.arange(1, 9)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    _style_axis(ax)
    indep = [union_recall_independent((p_illustrative,) * n) for n in n_vals]
    # correlated: successes overlap, so effective independent count is discounted (rho in [0,1))
    rho = 0.5  # illustrative correlation discount between paraphrases
    corr = [1.0 - (1.0 - p_illustrative) ** (1 + (n - 1) * (1 - rho)) for n in n_vals]
    ax.plot(n_vals, indep, marker="o", color=GREEN, linewidth=2.6, markersize=7,
            markeredgecolor=INK, label=f"independent ceiling  1−(1−p)^N   (p={p_illustrative})")
    ax.plot(n_vals, corr, marker="s", color=AMBER, linewidth=2.2, markersize=6,
            markeredgecolor=INK, label=f"correlated paraphrases  (ρ={rho}, realistic)")
    ax.axhline(p_illustrative, color=SLATE, linewidth=1.3, linestyle=":",
               label=f"single query  (p={p_illustrative})")
    ax.fill_between(n_vals, corr, indep, color=RED, alpha=0.08)
    ax.annotate("cost of correlation\n(paraphrases share meaning\n→ their misses co-occur)",
                xy=(3, (indep[2] + corr[2]) / 2), xytext=(2.15, 0.36),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, alpha=0.8),
                color=RED, fontsize=8.5, ha="center", va="center")
    ax.set_title("Multi-Query union recall: more reformulations → higher recall,\n"
                 "but correlation caps the gain below the independent ceiling", fontsize=11, pad=12)
    ax.set_xlabel("N  (number of query reformulations, union / fused)")
    ax.set_ylabel("P(gold retrieved by ≥ 1 reformulation)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(n_vals)
    ax.legend(loc="center right", framealpha=0.95, fontsize=8.5)
    # measured note (chapter's own p_hat on this tiny corpus)
    ax.text(0.5, -0.16, f"measured on this 11-doc corpus: each reformulation hits (p̂={p_hat:.2f}) → union=1.00 "
            "(recall saturates); the curves show the LAW on a harder corpus (p, ρ illustrative)",
            transform=ax.transAxes, ha="center", fontsize=7.8, style="italic", color=SLATE)
    _save(fig, "rag07_union_recall.png")


def fig_rrf_fusion(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """RRF over the chapter's own reformulation rankings: the weight curve + a worked fusion.

    Left: the RRF contribution 1/(k+rank) by rank for a few k (being #1 is worth far more than #5).
    Right: the ACTUAL top-5 rankings of the three reformulations from this chapter, fused by RRF into
    one order -- so the reader sees the union-with-arbitration on real lists, not a toy.
    """
    mq = build_multiquery_probe(corpus)
    fig, (ax_curve, ax_demo) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    # --- left: the 1/(k+rank) weight curve ---
    _style_axis(ax_curve)
    ranks = np.arange(1, 11)
    for k_rrf, color in zip([10, 60, 200], [AMBER, PURPLE, SLATE]):
        weights = 1.0 / (k_rrf + ranks)
        lw = 2.6 if k_rrf == RRF_K else 1.8
        ax_curve.plot(ranks, weights, marker="o", markersize=5, color=color, linewidth=lw,
                      markeredgecolor=INK, label=f"k = {k_rrf}" + ("  (default)" if k_rrf == RRF_K else ""))
    ax_curve.set_title("RRF weight 1/(k + rank) by rank", fontsize=11.5, pad=10)
    ax_curve.set_xlabel("rank in a reformulation's list  (1 = best)")
    ax_curve.set_ylabel("contribution to the fused score")
    ax_curve.set_xticks(ranks)
    ax_curve.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    # --- right: worked fusion of the chapter's OWN reformulation rankings ---
    ax_demo.axis("off")
    ax_demo.set_xlim(0, 1)
    ax_demo.set_ylim(0, 1)
    depth = 5
    lists = [list(dense.search(q, k=depth).indices) for q in mq.paraphrases]
    # RRF over these lists (all docs seen)
    seen = sorted({d for lst in lists for d in lst})
    k_rrf = RRF_K
    rrf_score = {}
    for d in seen:
        s = 0.0
        for lst in lists:
            if d in lst:
                s += 1.0 / (k_rrf + lst.index(d) + 1)
        rrf_score[d] = s
    fused = sorted(seen, key=lambda d: rrf_score[d], reverse=True)[:depth]

    col_x = [0.13, 0.35, 0.57, 0.83]
    headers = ["reform. 1", "reform. 2", "reform. 3", "RRF fused"]
    hcolors = [BLUE, PURPLE, AMBER, GREEN]
    for x, h, c in zip(col_x, headers, hcolors):
        ax_demo.text(x, 0.95, h, ha="center", fontsize=9.5, fontweight="bold", color=c)
    for i in range(depth):
        y = 0.83 - i * 0.15
        for xi, lst in zip(col_x[:3], lists):
            d = lst[i]
            face = GREEN if d == mq.gold else SLATE
            ax_demo.text(xi, y, f"{i+1}. doc[{d}]", ha="center", fontsize=9, color=INK,
                         bbox=dict(boxstyle="round,pad=0.28", facecolor=face, alpha=0.16, edgecolor=face))
        fd = fused[i]
        face = GREEN if fd == mq.gold else SLATE
        ax_demo.text(col_x[3], y, f"{i+1}. doc[{fd}]  ({rrf_score[fd]:.4f})", ha="center", fontsize=8.8,
                     color=INK, bbox=dict(boxstyle="round,pad=0.28", facecolor=face, alpha=0.18, edgecolor=face))
    ax_demo.text(0.5, 0.02, f"green = gold (doc[{mq.gold}]); ranked #1 by every reformulation → RRF locks it #1",
                 ha="center", fontsize=8.6, style="italic", color=INK)
    ax_demo.set_title("Worked RRF fusion of the chapter's 3 reformulations (real rankings)", fontsize=10.5, pad=10)
    _save(fig, "rag07_rrf_fusion.png")


def fig_before_after(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """MRR raw vs transformed, on each transform's probe set (measured) — the payoff bar chart.

    Left group: the two HyDE probes, raw-query MRR vs HyDE MRR. Right group: the Multi-Query probe,
    raw MRR vs fused MRR. Higher is better (MRR=1 means the gold ranked #1). recall@k is 1.0 across
    the board on this tiny 11-doc corpus, so MRR is the signal that separates the methods -- the
    caption says so, and the union-recall figure carries the recall story on a harder corpus.
    """
    hyde_probes = build_hyde_probes(corpus)
    mq = build_multiquery_probe(corpus)
    raw_hyde = float(np.mean([reciprocal_rank(retrieve_raw(dense, p.query), p.gold) for p in hyde_probes]))
    hyde_mrr = float(np.mean([reciprocal_rank(retrieve_hyde(dense, p.hyde_good), p.gold) for p in hyde_probes]))
    all_q = (mq.query, *mq.paraphrases)
    raw_mq = reciprocal_rank(retrieve_raw(dense, mq.query), mq.gold)
    mq_mrr = reciprocal_rank(retrieve_multiquery(dense, all_q), mq.gold)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    _style_axis(ax)
    groups = ["HyDE probe set\n(2 probes)", "Multi-Query probe"]
    raw_vals = [raw_hyde, raw_mq]
    new_vals = [hyde_mrr, mq_mrr]
    x = np.arange(len(groups))
    width = 0.36
    b1 = ax.bar(x - width / 2, raw_vals, width, color=SLATE, edgecolor=INK, linewidth=0.8,
                label="raw query")
    b2 = ax.bar(x + width / 2, new_vals, width, color=GREEN, edgecolor=INK, linewidth=0.8,
                label="transformed (HyDE / Multi-Query)")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=9, color=INK, ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("MRR  (1.0 = gold ranked #1; higher is better)")
    ax.set_ylim(0, 1.12)
    ax.set_title("The payoff: query transformation lifts MRR over the raw query (measured)", fontsize=11, pad=12)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=9)
    ax.text(0.5, -0.16, "recall@3 = 1.00 for every method on this 11-doc corpus (saturated); MRR carries the signal here",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color=SLATE)
    _save(fig, "rag07_before_after.png")


def main() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"dense lens: {dense.backend}")
    fig_asymmetry_2d(dense, corpus)
    fig_hyde_pipeline(dense, corpus)
    fig_union_recall(dense, corpus)
    fig_rrf_fusion(dense, corpus)
    fig_before_after(dense, corpus)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

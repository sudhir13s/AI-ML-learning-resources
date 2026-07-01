"""Static figure generator for 11-RAG-Evaluation.

Imports the SAME canonical functions the page and notebook use (rag_evaluation.py, which reuses ch5
& ch6) so every plotted number is the chapter's own -- no hand-typed values. Writes muted-palette
PNGs to the shared chapter image dir (../../images/) with the per-chapter prefix `rag11_`.

    python make_figures_11.py

Figures produced:
  rag11_eval_map.png          -- the two-stage evaluation map: RETRIEVAL metrics (context precision /
                                 recall) on the ranked context, GENERATION metrics (faithfulness,
                                 answer relevance) on the answer. Schematic mechanism diagram.
  rag11_faithful_vs_unfaithful.png -- the headline: a FAITHFUL answer (1.0) vs a fluent-but-
                                 UNFAITHFUL one (0.67), with the per-claim support that produced each.
  rag11_claim_support.png     -- claim decomposition: the unfaithful answer's claims as support bars
                                 vs the threshold; the hallucinated claim falls below it.
  rag11_precision_recall.png  -- context precision@k and recall on a GOOD vs a BAD (buried) ranking.
  rag11_ragas_triad.png       -- the RAGAS / TruLens triad: context relevance + faithfulness +
                                 answer relevance, each a separate lens, measured on the demo.
  rag11_faithful_but_irrelevant.png -- faithfulness vs answer relevance for an on-topic answer vs a
                                 grounded-but-off-topic one: why relevance is a SEPARATE axis.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from rag_evaluation import (
    FAITHFUL_ANSWER,
    IRRELEVANT_ANSWER,
    QUESTION,
    SUPPORT_THRESHOLD,
    UNFAITHFUL_ANSWER,
    DenseRetriever,
    answer_relevance,
    context_precision_at_k,
    context_recall,
    context_relevance,
    faithfulness,
    full_corpus,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # retrieval / data
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # supported / good / faithful
RED = "#8B3B4A"  # unsupported / bad / hallucination
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / threshold
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


def _box(ax, x, y, w, h, text, color, tcol="white", fs=8.6):
    """A filled rounded box with centred text -- the flow-diagram primitive."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcol, fontsize=fs, fontweight="bold")


def fig_eval_map() -> None:
    """The two-stage evaluation map: retrieval metrics on the context, generation metrics on the answer.

    Schematic mechanism diagram (labelled). Left: the RAG pipeline (question -> retrieve -> context ->
    generate -> answer). Below each stage, the metric that measures it: retrieval -> context precision
    / recall / context relevance; generation -> faithfulness + answer relevance. The point: you
    measure the two surfaces INDEPENDENTLY so you can localize a failure.
    """
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Evaluate RAG at BOTH stages: retrieval and generation, independently",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # pipeline row
    _box(ax, 0.02, 0.62, 0.15, 0.12, "question", SLATE, fs=9.0)
    _box(ax, 0.22, 0.62, 0.18, 0.12, "RETRIEVE\ntop-k chunks", BLUE, fs=8.6)
    _box(ax, 0.45, 0.62, 0.15, 0.12, "context", BLUE, fs=9.0)
    _box(ax, 0.65, 0.62, 0.16, 0.12, "GENERATE", PURPLE, fs=8.6)
    _box(ax, 0.86, 0.62, 0.12, 0.12, "answer", GREEN, fs=9.0)
    for x0, x1 in ((0.17, 0.22), (0.40, 0.45), (0.60, 0.65), (0.81, 0.86)):
        ax.annotate("", xy=(x1, 0.68), xytext=(x0, 0.68), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))

    # retrieval metrics box (under retrieve/context)
    ax.add_patch(plt.Rectangle((0.20, 0.10), 0.42, 0.36, facecolor=BLUE, alpha=0.09, edgecolor=BLUE, linewidth=1.4))
    ax.text(0.41, 0.42, "RETRIEVAL metrics", ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(0.41, 0.34, "context precision@k — are relevant chunks ranked HIGH?\n"
            "context recall — did we retrieve ALL relevant chunks?\n"
            "context relevance — is the context ON-TOPIC?\n"
            "(MRR / nDCG@k — reused from ch6)", ha="center", va="center", fontsize=8.4, color=INK)
    ax.annotate("", xy=(0.41, 0.46), xytext=(0.41, 0.62), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    # generation metrics box (under generate/answer)
    ax.add_patch(plt.Rectangle((0.64, 0.10), 0.34, 0.36, facecolor=PURPLE, alpha=0.09, edgecolor=PURPLE, linewidth=1.4))
    ax.text(0.81, 0.42, "GENERATION metrics", ha="center", fontsize=10.5, fontweight="bold", color=PURPLE)
    ax.text(0.81, 0.33, "faithfulness — is every claim\nSUPPORTED by the context?\n"
            "answer relevance — does it ANSWER\nthe question?", ha="center", va="center", fontsize=8.4, color=INK)
    ax.annotate("", xy=(0.81, 0.46), xytext=(0.81, 0.62), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5))

    ax.text(0.5, 0.03, "localize the failure: low retrieval metrics → fix the RETRIEVER;  "
            "high retrieval but low faithfulness → fix the GENERATOR / prompt",
            ha="center", fontsize=9.0, color=INK, style="italic")
    _save(fig, "rag11_eval_map.png")


def fig_faithful_vs_unfaithful(dense: DenseRetriever, context_text: str) -> None:
    """The headline: a FAITHFUL answer (1.0) vs a fluent-but-UNFAITHFUL one (0.67), per-claim.

    Runs the REAL faithfulness metric on both answers over the same retrieved context. Two panels of
    per-claim support bars (green supported / red unsupported) with the aggregate faithfulness in the
    title. The unfaithful answer's extra hallucinated claim is the red bar that drags its score below
    1.0. Vertical bars with claim labels below keep every label readable (no overlap).
    """
    faithful = faithfulness(dense, FAITHFUL_ANSWER, context_text)
    unfaithful = faithfulness(dense, UNFAITHFUL_ANSWER, context_text)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 6.0), gridspec_kw={"width_ratios": [2, 3]})

    def draw(ax, result, title, title_color):
        _style_axis(ax)
        x = np.arange(len(result.claims))
        colors = [GREEN if ok else RED for ok in result.supported]
        bars = ax.bar(x, result.supports, color=colors, edgecolor=INK, linewidth=0.9, width=0.6)
        for bar, s, ok in zip(bars, result.supports, result.supported):
            ax.annotate(f"{s:.3f}\n{'✓ supported' if ok else '✗ hallucinated'}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()), fontsize=8.2,
                        color=GREEN if ok else RED, ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontweight="bold")
        ax.axhline(SUPPORT_THRESHOLD, color=AMBER, linewidth=1.5, linestyle="--")
        ax.text(-0.45, SUPPORT_THRESHOLD + 0.02, f"threshold {SUPPORT_THRESHOLD}",
                color=AMBER, fontsize=7.8, ha="left", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([f"claim {i+1}\n{_short(c, 24)}" for i, c in enumerate(result.claims)], fontsize=7.6)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("claim support (max cosine to context)")
        ax.set_title(f"{title}\nfaithfulness = {result.score:.3f}  "
                     f"({sum(result.supported)}/{len(result.claims)} claims supported)",
                     fontsize=10.5, color=title_color, fontweight="bold", pad=10)

    draw(ax_l, faithful, "FAITHFUL answer", GREEN)
    draw(ax_r, unfaithful, "fluent but UNFAITHFUL answer", RED)
    fig.suptitle("Faithfulness catches a hallucination two fluent answers hide (measured)",
                 fontsize=12.5, y=1.0, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "rag11_faithful_vs_unfaithful.png")


def fig_claim_support(dense: DenseRetriever, context_text: str) -> None:
    """Claim decomposition: the unfaithful answer's claims as support bars vs the threshold.

    The DECOMPOSE -> check-each -> aggregate mechanism, made visible on the unfaithful answer: three
    claims, two clear the threshold (green, supported), one (the hallucinated 'solar panels') falls
    below it (red, unsupported) -- so faithfulness = 2/3. The claim-split is an illustrative stand-in
    for an LLM extractor (labelled); the support cosines are REAL.
    """
    result = faithfulness(dense, UNFAITHFUL_ANSWER, context_text)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    _style_axis(ax)
    x = np.arange(len(result.claims))
    colors = [GREEN if ok else RED for ok in result.supported]
    bars = ax.bar(x, result.supports, color=colors, edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, s, ok in zip(bars, result.supports, result.supported):
        ax.annotate(f"{s:.3f}\n{'SUPPORTED' if ok else 'HALLUCINATED'}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), fontsize=8.4,
                    color=GREEN if ok else RED, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontweight="bold")
    ax.axhline(SUPPORT_THRESHOLD, color=AMBER, linewidth=1.5, linestyle="--")
    ax.annotate(f"support threshold = {SUPPORT_THRESHOLD}", (len(result.claims) - 0.5, SUPPORT_THRESHOLD),
                color=AMBER, fontsize=8.6, ha="right", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([f"claim {i+1}\n{_short(c, 26)}" for i, c in enumerate(result.claims)], fontsize=7.8)
    ax.set_ylabel("claim support (max cosine to context)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Decompose → check each claim → aggregate:  faithfulness = "
                 f"{sum(result.supported)}/{len(result.claims)} = {result.score:.3f}",
                 fontsize=11, pad=12)
    ax.text(0.5, -0.24, "claim-splitting is an illustrative stand-in for an LLM extractor; the support cosines are REAL",
            transform=ax.transAxes, ha="center", fontsize=7.6, color=SLATE, style="italic")
    _save(fig, "rag11_claim_support.png")


def fig_precision_recall(dense: DenseRetriever) -> None:
    """Context precision@k and recall on a GOOD vs a BAD (buried) ranking -- measured.

    Grouped bars: precision@3 and recall@3 for the real retrieval (relevant chunks up top) vs a
    deliberately bad ranking (relevant chunks buried below the top-3). The good ranking scores 1.0/1.0;
    the buried ranking scores 0/0 -- localizing a failure to RETRIEVAL, upstream of generation.
    """
    relevant = frozenset({0, 1})
    k = 3
    good = dense.search(QUESTION, k=len(full_corpus())).indices
    distractors = tuple(i for i in range(len(full_corpus())) if i not in relevant)
    bad = distractors[:3] + tuple(sorted(relevant)) + distractors[3:]
    metrics = {
        "GOOD ranking\n(real retrieval)": (
            context_precision_at_k(good, relevant, k), context_recall(good, relevant, k), GREEN),
        "BAD ranking\n(relevant buried)": (
            context_precision_at_k(bad, relevant, k), context_recall(bad, relevant, k), RED),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    _style_axis(ax)
    x = np.arange(2)
    width = 0.36
    precs = [m[0] for m in metrics.values()]
    recs = [m[1] for m in metrics.values()]
    b1 = ax.bar(x - width / 2, precs, width, label=f"context precision@{k}", color=BLUE, edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + width / 2, recs, width, label=f"context recall@{k}", color=PURPLE, edgecolor=INK, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=9.5, color=INK, ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9.0)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.9)
    ax.set_title("Retrieval metrics localize the failure: good ranking 1.0, buried ranking 0.0 (measured)",
                 fontsize=10.5, pad=12)
    _save(fig, "rag11_precision_recall.png")


def fig_ragas_triad(dense: DenseRetriever, context_chunks: tuple[str, ...], context_text: str) -> None:
    """The RAGAS / TruLens triad: context relevance + faithfulness + answer relevance, measured.

    Three gauges, one per leg of the triad, each a real measured number on the demo: context relevance
    (is the context on-topic?), faithfulness (is the answer grounded?), answer relevance (does it
    answer the question?). All three high = trustworthy answer; any one low points at a different fix.
    """
    cr = context_relevance(dense, QUESTION, context_chunks)
    fa = faithfulness(dense, FAITHFUL_ANSWER, context_text).score
    ar = answer_relevance(dense, QUESTION, FAITHFUL_ANSWER)
    legs = [
        ("CONTEXT\nRELEVANCE", cr, BLUE, "is the retrieved\ncontext on-topic?"),
        ("FAITHFULNESS", fa, GREEN, "is every claim\ngrounded in context?"),
        ("ANSWER\nRELEVANCE", ar, PURPLE, "does it answer\nthe question?"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8))
    for ax, (name, val, color, sub) in zip(axes, legs):
        _style_axis(ax)
        ax.bar([0], [val], color=color, edgecolor=INK, linewidth=1.0, width=0.5)
        ax.bar([0], [1.0], color=color, alpha=0.10, edgecolor=color, linewidth=1.0, width=0.5)  # full-scale ghost
        ax.text(0, val + 0.03, f"{val:.3f}", ha="center", fontsize=15, color=color, fontweight="bold")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1.15)
        ax.set_xticks([])
        ax.set_title(name, fontsize=11, color=color, fontweight="bold", pad=8)
        ax.text(0, -0.13, sub, ha="center", fontsize=8.2, color=INK, transform=ax.get_xaxis_transform())
    fig.suptitle("The RAGAS / TruLens triad — three independent lenses on one RAG answer (measured)",
                 fontsize=12.5, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag11_ragas_triad.png")


def fig_faithful_but_irrelevant(dense: DenseRetriever) -> None:
    """Faithfulness vs answer relevance for an on-topic answer vs a grounded-but-off-topic one.

    The key intuition that relevance is a SEPARATE axis: the off-topic answer is FULLY faithful
    (faithfulness 1.0 -- every claim is a real corpus fact) yet scores low answer relevance (it
    answers the wrong question). Two grouped bars make the dissociation explicit -- both measured.
    """
    on_topic_faith = faithfulness(dense, FAITHFUL_ANSWER, " ".join(full_corpus())).score
    on_topic_rel = answer_relevance(dense, QUESTION, FAITHFUL_ANSWER)
    off_topic_faith = faithfulness(dense, IRRELEVANT_ANSWER, " ".join(full_corpus())).score
    off_topic_rel = answer_relevance(dense, QUESTION, IRRELEVANT_ANSWER)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    _style_axis(ax)
    x = np.arange(2)
    width = 0.36
    faiths = [on_topic_faith, off_topic_faith]
    rels = [on_topic_rel, off_topic_rel]
    b1 = ax.bar(x - width / 2, faiths, width, label="faithfulness", color=GREEN, edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + width / 2, rels, width, label="answer relevance", color=PURPLE, edgecolor=INK, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=9.0, color=INK, ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["on-topic answer\n(imager + launch)", "faithful-but-off-topic\n(project lead)"], fontsize=8.8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.32)
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.9)
    ax.annotate(f"grounded (1.0) but IRRELEVANT ({off_topic_rel:.2f}):\nrelevance is a separate axis",
                xy=(1 + width / 2, off_topic_rel), xytext=(0.5, 1.05), fontsize=8.8, color=RED,
                ha="center", fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, alpha=0.8))
    ax.set_title("A faithful answer to the WRONG question is still useless — measure relevance too",
                 fontsize=10.5, pad=12)
    _save(fig, "rag11_faithful_but_irrelevant.png")


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    ranked = dense.search(QUESTION, k=len(corpus)).indices
    context_chunks = tuple(corpus[i] for i in ranked[:3])
    context_text = " ".join(context_chunks)
    print(f"dense lens: {dense.backend}")
    fig_eval_map()
    fig_faithful_vs_unfaithful(dense, context_text)
    fig_claim_support(dense, context_text)
    fig_precision_recall(dense)
    fig_ragas_triad(dense, context_chunks, context_text)
    fig_faithful_but_irrelevant(dense)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

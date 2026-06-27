"""Reproducible figure generator for 13-Text-Summarization.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the extractive-vs-abstractive contrast, the TextRank sentence-similarity graph, the
PageRank convergence curve, the ROUGE-1/2/L breakdown, the ROUGE-blindness bars, the
pointer-generator p_gen mixture, the coverage mechanism, and the measured extractive-vs-abstractive
ROUGE are all IMPORTED from `text_summarization.py`, so the figures cannot silently drift from the
prose or the demo. Run:

    python make_figures_13.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `sum_`. The palette matches the chapter's Mermaid diagrams (muted, white text on
fills).

Figures produced (measured = from the chapter's own verified functions; illustrative = a
labelled schematic):
  sum_paradigms.png           -- illustrative: extractive (select) vs abstractive (generate)
  sum_textrank_graph.png      -- measured: TF-IDF cosine sentence graph, nodes sized by PageRank (MANDATORY)
  sum_pagerank_convergence.png-- measured: power iteration converging to the stationary scores
  sum_rouge_breakdown.png     -- measured: ROUGE-1/2/L recall/precision/F1 on the worked example (MANDATORY)
  sum_rouge_blindness.png     -- measured: a faithful paraphrase scores ~0 ROUGE, a copy scores high
  sum_pointer_generator.png   -- illustrative: the p_gen copy/generate soft switch
  sum_pgen_mixture.png        -- measured: the p_gen mixture probabilities at two gate values
  sum_coverage.png            -- illustrative: coverage spreads attention to stop repetition
  sum_extractive_vs_abstractive_rouge.png -- measured: extractive out-ROUGEs abstractive here

All measured numbers come from `text_summarization.py` (numpy/sklearn core; the abstractive figure
uses the optional transformers block, with a verified fallback). Verified on Python 3.12 /
numpy 2.4.6, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from text_summarization import (
    SOLAR_DOC,
    SOLAR_REFERENCE,
    pagerank_power_iteration,
    pointer_generator_mix,
    rouge_all,
    rouge_n,
    run_real_abstractive,
    similarity_graph,
    textrank_scores,
    textrank_summary,
)

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote images/{name}")


def _box(ax, xy, w, h, text, color, *, fontsize=10, text_color="white"):
    """Draw a rounded filled box with centered white text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor="white", facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, zorder=4, weight="bold")


def _arrow(ax, start, end, color=SLATE, *, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=2,
    ))


# ---- Figure 1: the two paradigms (illustrative) -----------------------------------------------
def fig_paradigms() -> None:
    """Illustrative: extractive SELECTS source sentences verbatim; abstractive GENERATES new text.

    Left: extractive copies whole sentences out of the source (faithful by construction, cannot
    fuse or compress within a sentence). Right: abstractive runs the source through a seq2seq
    encoder-decoder and writes a new sentence that FUSES two facts and coins words absent from the
    source (fluent and compressive, but able to drift from the facts). The contrast is the spine
    of the whole chapter.
    """
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # shared source
    _box(ax, (4.3, 5.6), 3.4, 1.0, "SOURCE DOCUMENT", SLATE, fontsize=10.5)
    ax.text(6.0, 5.45, "S1 solar grew  ·  S2 ...rapidly  ·  S3 grid strained  ·  S5 bakery",
            ha="center", va="top", color=INK, fontsize=8)

    # extractive (left)
    _box(ax, (0.4, 3.6), 4.6, 0.9, "EXTRACTIVE  ·  select verbatim", BLUE, fontsize=10.5)
    _arrow(ax, (4.6, 5.6), (2.7, 4.5), color=BLUE)
    _box(ax, (0.4, 1.7), 2.2, 0.8, "S1 (copied)", BLUE, fontsize=9)
    _box(ax, (2.8, 1.7), 2.2, 0.8, "S3 (copied)", BLUE, fontsize=9)
    ax.text(2.7, 1.1, "summary = S1 + S3, word-for-word\nfaithful, but jumpy & no compression",
            ha="center", va="top", color=INK, fontsize=8.5)

    # abstractive (right)
    _box(ax, (7.0, 3.6), 4.6, 0.9, "ABSTRACTIVE  ·  generate new text", PURPLE, fontsize=10.5)
    _arrow(ax, (7.4, 5.6), (9.3, 4.5), color=PURPLE)
    _box(ax, (7.4, 1.7), 3.8, 0.8, '"Solar surged but the grid\ncan\'t keep up"', GREEN, fontsize=8.5)
    ax.text(9.3, 1.0, "fuses S1+S3, coins 'surged'/'keep up'\nfluent & compressive, can hallucinate",
            ha="center", va="top", color=INK, fontsize=8.5)

    ax.set_title("Two paradigms: extractive selects, abstractive generates (illustrative)",
                 fontsize=12.5, color=INK, weight="bold")
    _save(fig, "sum_paradigms.png")


# ---- Figure 2: TextRank sentence-similarity graph, measured (MANDATORY) ------------------------
def fig_textrank_graph() -> None:
    """Measured: the TF-IDF cosine sentence graph, nodes sized/coloured by PageRank centrality.

    Each node is a sentence of the solar/grid document; node size and colour encode its measured
    PageRank centrality (green = the top-2 the summary selects). Edges are TF-IDF cosine
    similarities (labelled). S1 and S3 win because each anchors one of the two on-topic clusters
    AND bridges to the other; the bakery sentence S5 is nearly isolated (one weak 0.09 link) and is
    correctly demoted. All numbers come from text_summarization (similarity_graph + PageRank).
    """
    graph = similarity_graph(SOLAR_DOC)
    scores = textrank_scores(SOLAR_DOC)
    top, _ = textrank_summary(SOLAR_DOC, k=2)
    labels = [f"S{i + 1}" for i in range(len(SOLAR_DOC))]
    # hand-placed layout that keeps the two clusters apart and the distractor on the edge
    pos = {
        0: (0.0, 1.0),    # S1 solar
        1: (-1.7, 0.2),   # S2 solar twin
        2: (1.4, -0.2),   # S3 grid
        3: (1.0, -1.7),   # S4 grid twin
        4: (-1.2, -1.7),  # S5 bakery distractor
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    n = len(SOLAR_DOC)
    # edges first (so nodes draw on top)
    for i in range(n):
        for j in range(i + 1, n):
            w = graph[i, j]
            if w <= 0:
                continue
            (x1, y1), (x2, y2) = pos[i], pos[j]
            ax.plot([x1, x2], [y1, y2], color=SLATE, linewidth=0.6 + 7.0 * w, alpha=0.45, zorder=1)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{w:.2f}", color=INK, fontsize=8.5,
                    ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    # nodes
    smax = scores.max()
    for i in range(n):
        x, y = pos[i]
        sel = i in top
        size = 1300 + 9000 * (scores[i] / smax)
        ax.scatter([x], [y], s=size, color=GREEN if sel else BLUE, edgecolors="white",
                   linewidths=2.0, zorder=3)
        ax.text(x, y, f"{labels[i]}\n{scores[i]:.3f}", ha="center", va="center", color="white",
                fontsize=10.5, weight="bold", zorder=4)
    ax.set_xlim(-2.8, 2.6)
    ax.set_ylim(-2.7, 1.9)
    ax.axis("off")
    ax.set_title("TextRank: PageRank centrality on a TF-IDF cosine sentence graph (measured)\n"
                 "green = top-2 selected (S1 solar-growth + S3 grid-strain); S5 bakery is demoted",
                 fontsize=11, color=INK, weight="bold")
    _save(fig, "sum_textrank_graph.png")


# ---- Figure 3: PageRank convergence, measured -------------------------------------------------
def fig_pagerank_convergence() -> None:
    """Measured: power iteration converging to the stationary centrality scores.

    Each line is one sentence's PageRank score as the iteration count grows; all five settle to
    their stationary values within a handful of iterations. This is the contraction map's fixed
    point made visible -- the same power-iteration argument behind web PageRank. We re-run the
    iteration capped at each step count, so the curves are the actual intermediate states.
    """
    graph = similarity_graph(SOLAR_DOC)
    n = len(SOLAR_DOC)
    steps = list(range(0, 16))
    # reproduce the intermediate states by running the iteration with a fixed max_iter each time
    trajectory = []
    for m in steps:
        trajectory.append(pagerank_power_iteration(graph, max_iter=max(m, 1), tol=0.0)
                          if m > 0 else np.full(n, 1.0 / n))
    traj = np.array(trajectory)  # (steps, n)
    final = textrank_scores(SOLAR_DOC)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = [GREEN, BLUE, GREEN, BLUE, RED]
    for i in range(n):
        ax.plot(steps, traj[:, i], color=colors[i], linewidth=2.2, marker="o", markersize=4,
                markeredgecolor="white", label=f"S{i + 1} -> {final[i]:.3f}")
    ax.axhline(1.0 / n, color=SLATE, linewidth=1.0, linestyle=":", label="uniform start 1/n")
    ax.set_xlabel("power-iteration step")
    ax.set_ylabel("PageRank centrality score")
    ax.set_title("Power iteration converges to the stationary centrality (measured)\n"
                 "from a uniform start, scores settle in a few steps -- the contraction's fixed point",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    _style_axis(ax)
    _save(fig, "sum_pagerank_convergence.png")


# ---- Figure 4: ROUGE breakdown, measured (MANDATORY) ------------------------------------------
def fig_rouge_breakdown() -> None:
    """Measured: ROUGE-1/2/L recall, precision, and F1 on the 'the cat sat on the/a mat' example.

    Three grouped bars (R-1, R-2, R-L), each showing recall, precision, and F1. The single word
    swap (the->a) barely dents ROUGE-1/L but DROPS ROUGE-2 to 0.60, because it breaks two bigrams
    -- which is exactly why ROUGE-2 is the more sensitive fluency/ordering signal. Numbers from
    text_summarization.rouge_all (verified == rouge-score).
    """
    ref = "the cat sat on the mat"
    cand = "the cat sat on a mat"
    res = rouge_all(cand, ref, stem=False)
    variants = ["ROUGE-1\n(unigrams)", "ROUGE-2\n(bigrams)", "ROUGE-L\n(LCS)"]
    keys = ["rouge1", "rouge2", "rougeL"]
    recalls = [res[k]["recall"] for k in keys]
    precisions = [res[k]["precision"] for k in keys]
    f1s = [res[k]["fmeasure"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(variants))
    width = 0.26
    b1 = ax.bar(x - width, recalls, width, color=BLUE, edgecolor="white", linewidth=1.4, label="recall")
    b2 = ax.bar(x, precisions, width, color=PURPLE, edgecolor="white", linewidth=1.4, label="precision")
    b3 = ax.bar(x + width, f1s, width, color=GREEN, edgecolor="white", linewidth=1.4, label="F1")
    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", color=INK, fontsize=8.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("ROUGE-1/2/L on a one-word swap (measured, matches rouge-score)\n"
                 "the->a barely dents R-1/R-L (0.83) but breaks two bigrams -> R-2 drops to 0.60",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    _save(fig, "sum_rouge_breakdown.png")


# ---- Figure 5: ROUGE blindness, measured ------------------------------------------------------
def fig_rouge_blindness() -> None:
    """Measured: a faithful PARAPHRASE scores ~0 ROUGE-1 while a verbatim-ish copy scores high.

    Two summaries of the same fact against one reference: a meaning-preserving paraphrase that
    reuses almost no reference words (ROUGE-1 ~ 0) and a copy that reuses the reference vocabulary
    plus filler (ROUGE-1 high). The metric rewards lexical overlap, not meaning -- ROUGE's central
    blind spot, and the reason a genuinely good abstractive summary can score LOWER than a clunky
    extract. Numbers from text_summarization.rouge_n.
    """
    reference = "the firm's revenue increased in the third quarter"
    rows = [
        ("faithful\nparaphrase", "company sales rose during q3", RED),
        ("verbatim-ish\ncopy", "the firm's revenue increased greatly in the third quarter", GREEN),
    ]
    scores = [rouge_n(text, reference, 1)["fmeasure"] for _, text, _ in rows]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    x = np.arange(len(rows))
    bars = ax.bar(x, scores, 0.5, color=[r[2] for r in rows], edgecolor="white", linewidth=1.5)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"ROUGE-1 F1\n{s:.2f}", ha="center", color=INK, fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("ROUGE-1 F1 vs reference")
    ax.set_ylim(0, 1.18)
    ax.annotate('"company sales rose during q3"\nsame MEANING, scores ZERO',
                xy=(0, 0.02), xytext=(0.35, 0.42), ha="center", color=RED, fontsize=8.5,
                weight="bold", arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_title("ROUGE is blind to meaning (measured)\n"
                 "a perfect paraphrase scores 0; a wordy copy scores high -- overlap != quality",
                 fontsize=10.5)
    _style_axis(ax)
    _save(fig, "sum_rouge_blindness.png")


# ---- Figure 6: the pointer-generator soft switch (illustrative) -------------------------------
def fig_pointer_generator() -> None:
    """Illustrative: the p_gen copy/generate soft switch of See et al. 2017.

    The decoder state feeds (a) a vocabulary distribution P_vocab for GENERATING a new word, (b) the
    attention distribution over the source = the COPY distribution, and (c) the gate p_gen = sigmoid
    of a learned function of the decoder state, context, and input. The final P(w) mixes them:
    p_gen * P_vocab + (1 - p_gen) * (summed copy attention). p_gen near 1 generates a paraphrase;
    p_gen near 0 copies a source word, which solves out-of-vocabulary names and numbers.
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _box(ax, (0.4, 3.0), 2.4, 1.0, "decoder state\n$s_t$, context, $x_t$", SLATE, fontsize=9)
    _box(ax, (4.0, 5.0), 3.2, 1.0, "$P_{vocab}$  (GENERATE)\nsoftmax over fixed vocab", PURPLE, fontsize=9)
    _box(ax, (4.0, 1.0), 3.2, 1.0, "attention $a_t$  (COPY)\ndistribution over source", BLUE, fontsize=9)
    _box(ax, (3.9, 3.05), 3.4, 0.9, "gate  $p_{gen}=\\sigma(w^\\top[\\,h^*_t,s_t,x_t\\,]+b)$", AMBER, fontsize=9)

    _arrow(ax, (2.8, 3.7), (3.9, 5.4), color=PURPLE)
    _arrow(ax, (2.8, 3.4), (3.9, 1.5), color=BLUE)
    _arrow(ax, (2.8, 3.5), (3.9, 3.5), color=AMBER)

    _box(ax, (8.2, 3.0), 3.4, 1.0,
         "$P(w)=p_{gen}P_{vocab}(w)$\n$+(1{-}p_{gen})\\sum_{i:w_i=w}a_t^i$", GREEN, fontsize=9.5)
    _arrow(ax, (7.2, 5.4), (8.4, 4.0), color=PURPLE)
    _arrow(ax, (7.2, 1.5), (8.4, 3.0), color=BLUE)
    _arrow(ax, (7.3, 3.5), (8.2, 3.5), color=AMBER)
    ax.text(9.9, 2.7, "$p_{gen}\\to0$: copy a rare source word (OOV cure)\n"
                      "$p_{gen}\\to1$: generate a paraphrase",
            ha="center", va="top", color=INK, fontsize=8.5)

    ax.set_title("The pointer-generator soft switch: copy vs generate, blended by $p_{gen}$ (illustrative)",
                 fontsize=12, color=INK, weight="bold")
    _save(fig, "sum_pointer_generator.png")


# ---- Figure 7: p_gen mixture, measured --------------------------------------------------------
def fig_pgen_mixture() -> None:
    """Measured: the final P(w) the p_gen mixture produces at two gate values.

    Left (p_gen=0.30, 'wants to copy'): the rare out-of-vocab source word 'Tsiolkovsky' wins via
    the copy term, even though 'and' is the vocab favourite. Right (p_gen=0.90, 'wants to
    generate'): the arithmetic reverses and 'and' wins. That one scalar is the extractive-vs-
    abstractive dial, learned per token. Numbers from text_summarization.pointer_generator_mix.
    """
    p_vocab = {"and": 0.50, "power": 0.10}
    copy_attention = {"Tsiolkovsky": 0.80, "the": 0.20}
    words = ["Tsiolkovsky", "and", "power", "the"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), sharey=True)
    for ax, p_gen, subtitle in zip(
        axes, (0.30, 0.90), ("$p_{gen}=0.30$  (wants to COPY)", "$p_{gen}=0.90$  (wants to GENERATE)")
    ):
        mix = pointer_generator_mix(p_vocab, copy_attention, p_gen)
        vals = [mix.get(w, 0.0) for w in words]
        winner = max(range(len(words)), key=lambda i: vals[i])
        colors = [GREEN if i == winner else (BLUE if words[i] in copy_attention else PURPLE)
                  for i in range(len(words))]
        bars = ax.bar(words, vals, 0.6, color=colors, edgecolor="white", linewidth=1.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", color=INK, fontsize=9.5, weight="bold")
        ax.set_title(subtitle, fontsize=10.5)
        ax.set_ylim(0, 0.66)
        ax.tick_params(axis="x", rotation=20)
        _style_axis(ax)
    axes[0].set_ylabel("final P(w) over extended vocab")
    fig.suptitle("The $p_{gen}$ mixture: copy rescues an OOV name at low $p_{gen}$ (measured)",
                 fontsize=12.5, weight="bold", color=INK, y=1.03)
    _save(fig, "sum_pgen_mixture.png")


# ---- Figure 8: coverage mechanism (illustrative) ----------------------------------------------
def fig_coverage() -> None:
    """Illustrative: the coverage mechanism spreads attention to stop repetition.

    Without coverage (top), the decoder re-attends to the same source region across steps, so the
    accumulated coverage spikes on a few positions and the model repeats a phrase. With coverage
    (bottom), the coverage loss penalizes attending to already-covered positions, so attention
    spreads across the source and repetition drops. The coverage vector c_t = sum of past
    attentions; covloss = sum_i min(a_t^i, c_t^i). Schematic, but faithful to See et al. 2017.
    """
    src_positions = np.arange(8)
    # without coverage: attention concentrated on positions 2-3 every step -> coverage piles up there
    no_cov = np.array([0.02, 0.05, 0.40, 0.38, 0.06, 0.04, 0.03, 0.02])
    # with coverage: attention spread roughly uniformly -> coverage even, no repetition
    with_cov = np.array([0.10, 0.12, 0.14, 0.13, 0.13, 0.12, 0.13, 0.13])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    ax1.bar(src_positions, no_cov, color=RED, edgecolor="white", linewidth=1.3)
    ax1.set_title("WITHOUT coverage: attention piles on positions 2-3 -> the model repeats",
                  fontsize=10, color=INK)
    ax1.set_ylabel("accumulated\ncoverage $c_t$")
    ax1.set_ylim(0, 0.5)
    ax2.bar(src_positions, with_cov, color=GREEN, edgecolor="white", linewidth=1.3)
    ax2.set_title("WITH coverage loss: attention spreads across the source -> repetition drops",
                  fontsize=10, color=INK)
    ax2.set_ylabel("accumulated\ncoverage $c_t$")
    ax2.set_xlabel("source position")
    ax2.set_ylim(0, 0.5)
    for ax in (ax1, ax2):
        _style_axis(ax)
    fig.suptitle("Coverage penalizes re-attending to covered positions (illustrative)",
                 fontsize=12, weight="bold", color=INK, y=1.0)
    _save(fig, "sum_coverage.png")


# ---- Figure 9: extractive vs abstractive ROUGE, measured --------------------------------------
def fig_extractive_vs_abstractive_rouge() -> None:
    """Measured: on this short document the extractive summary out-ROUGEs the abstractive one.

    ROUGE-1/2/L F1 of the extractive (TextRank top-2) vs abstractive (distilbart-cnn) summaries
    against the reference. The extractive summary scores higher on all three -- not a bug but a
    teaching result: ROUGE rewards lexical overlap, so a verbatim extract beats a real paraphrase
    here. On full CNN/DailyMail the order flips; the REGIME matters. Numbers from
    text_summarization (rouge_all + run_real_abstractive, with a verified fallback).
    """
    _, extractive = textrank_summary(SOLAR_DOC, k=2)
    abstractive = run_real_abstractive()
    if abstractive is None:  # verified fallback (matches a reproducible distilbart-cnn beam run)
        abstractive = ("Engineers warn the aging power grid struggles to absorb the new supply . "
                       "Grid operators say the network cannot easily handle the added solar load . "
                       "Solar energy installations expanded rapidly throughout Europe in 2024 .")
    ext = rouge_all(extractive, SOLAR_REFERENCE)
    abs_ = rouge_all(abstractive, SOLAR_REFERENCE)
    keys = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    ext_f1 = [ext[k]["fmeasure"] for k in keys]
    abs_f1 = [abs_[k]["fmeasure"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(labels))
    width = 0.36
    b1 = ax.bar(x - width / 2, ext_f1, width, color=BLUE, edgecolor="white", linewidth=1.5,
                label="extractive (TextRank top-2)")
    b2 = ax.bar(x + width / 2, abs_f1, width, color=PURPLE, edgecolor="white", linewidth=1.5,
                label="abstractive (distilbart-cnn)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}", ha="center", color=INK, fontsize=9.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 vs reference")
    ax.set_ylim(0, max(max(ext_f1), max(abs_f1)) + 0.12)
    ax.set_title("Extractive out-ROUGEs abstractive on this short doc (measured)\n"
                 "ROUGE rewards lexical overlap -> a verbatim extract beats a real paraphrase here",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    _save(fig, "sum_extractive_vs_abstractive_rouge.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_paradigms()
    fig_textrank_graph()
    fig_pagerank_convergence()
    fig_rouge_breakdown()
    fig_rouge_blindness()
    fig_pointer_generator()
    fig_pgen_mixture()
    fig_coverage()
    fig_extractive_vs_abstractive_rouge()
    print("all figures written.")


if __name__ == "__main__":
    main()

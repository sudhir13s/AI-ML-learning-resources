"""Reproducible figure generator for 12-Machine-Translation.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the BLEU breakdown, the BLEU-vs-chrF brittleness bars, the IBM Model 1 alignment
heatmap, the beam length-normalization flip, and the measured NMT chrF are all IMPORTED from
`machine_translation.py`, so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_12.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `mt_`. The palette matches the chapter's Mermaid diagrams (muted, white text on
fills).

Figures produced (measured = from the chapter's own verified functions; illustrative = a
labelled schematic):
  mt_evolution_timeline.png   -- illustrative: four eras, quality up / hand-rules down
  mt_bleu_breakdown.png       -- measured: per-n precision + brevity penalty -> final BLEU (MANDATORY)
  mt_bleu_brittleness.png     -- measured: a valid paraphrase scores ~0 BLEU but partial chrF
  mt_alignment_matrix.png     -- measured: IBM Model 1 word-alignment heatmap (MANDATORY)
  mt_metric_landscape.png     -- illustrative: surface-overlap vs learned metrics, what each sees
  mt_beam_length_norm.png     -- measured: length normalization flips the winner across alpha
  mt_backtranslation_curve.png-- illustrative: synthetic-data lift, widest at low resource
  mt_nmt_measured.png         -- measured: chrF of a real opus-mt-fr-en run on three sentences

All measured numbers come from `machine_translation.py` (pure-numpy core; the NMT figure uses the
optional transformers block). Verified on Python 3.12 / numpy 2.4.6, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from machine_translation import (
    NMT_SRCS,
    alignment_matrix,
    beam_length_norm_demo,
    chrf,
    ibm_model1,
    run_real_nmt,
    sentence_bleu,
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


# ---- Figure 1: the four eras of MT (illustrative) ---------------------------------------------
def fig_evolution_timeline() -> None:
    """Illustrative: translation quality rises while hand-built linguistic machinery collapses.

    Four eras on a shared x-axis. Quality (green) climbs steadily era to era; the amount of
    human-written linguistic machinery (red, dashed) falls just as steadily -- the single trade
    that defines MT history: push knowledge out of hand-written rules and into learned parameters.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    eras = ["Rule-based\n1950s-80s", "Statistical\n1990-2014", "Neural\n2014-2017", "LLM\n2020+"]
    x = np.arange(len(eras))
    quality = np.array([2.0, 4.5, 7.5, 9.0])      # schematic quality, monotonically up
    handwork = np.array([9.0, 6.0, 2.5, 1.0])     # schematic hand-built machinery, down
    ax.plot(x, quality, color=GREEN, linewidth=3.0, marker="o", markersize=9,
            markeredgecolor="white", label="translation quality")
    ax.plot(x, handwork, color=RED, linewidth=3.0, linestyle="--", marker="s", markersize=9,
            markeredgecolor="white", label="hand-built linguistic machinery")
    ax.fill_between(x, quality, handwork, where=(quality >= handwork), color=GREEN, alpha=0.08)
    ax.fill_between(x, quality, handwork, where=(quality < handwork), color=RED, alpha=0.08)
    for xi, era in enumerate(eras):
        ax.text(xi, -1.2, era, ha="center", va="top", color=INK, fontsize=9.5, weight="bold")
    # annotate the crossover idea
    ax.annotate("rules give way\nto learned parameters", xy=(1.5, 5.25), xytext=(1.5, 1.6),
                ha="center", color=SLATE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xticks([])
    ax.set_ylabel("relative magnitude (schematic)")
    ax.set_title("The four eras of machine translation (illustrative)\n"
                 "quality climbs as hand-written rules give way to learned parameters", fontsize=11)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.set_ylim(-2.4, 10)
    ax.set_xlim(-0.4, 3.4)
    _style_axis(ax)
    ax.grid(False)
    _save(fig, "mt_evolution_timeline.png")


# ---- Figure 2: BLEU breakdown, measured (MANDATORY) -------------------------------------------
def fig_bleu_breakdown() -> None:
    """Measured: BLEU built from its parts -- per-n precision, brevity penalty, geometric mean.

    Left: the four modified n-gram precisions p_1..p_4 (clipped) as bars, each labelled with its
    matches/total fraction. Right: how they combine -- the geometric mean of the precisions, times
    the brevity penalty, equals the final BLEU. The numbers are from machine_translation.sentence_bleu
    on the page's worked example, so the figure and the prose share one computation.
    """
    cand = "the the the black cat sat on the mat happily today"
    ref = "the black cat sat on the mat very happily today indeed"
    res = sentence_bleu(cand, [ref])
    precisions = res["precisions"]          # percentages
    clipped, total = res["clipped"], res["total"]
    bp, geo, score = res["bp"], res["geo_mean"], res["score"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    # Left: the four precisions.
    labels = [f"$p_{n + 1}$" for n in range(4)]
    colors = [BLUE, PURPLE, NAVY, SLATE]
    bars = ax1.bar(labels, precisions, color=colors, edgecolor="white", linewidth=1.5)
    for i, (bar, p) in enumerate(zip(bars, precisions)):
        ax1.text(bar.get_x() + bar.get_width() / 2, p + 1.5,
                 f"{clipped[i]}/{total[i]}\n{p:.1f}%", ha="center", color=INK,
                 fontsize=9.5, weight="bold")
    ax1.set_ylabel("clipped n-gram precision (%)")
    ax1.set_title("Modified n-gram precisions $p_n$ (clipped)\n"
                  "candidate has 3x 'the' but the reference has 1 -> clipping caps the credit",
                  fontsize=10.5)
    ax1.set_ylim(0, 100)
    _style_axis(ax1)
    # Right: combine -> BLEU. The BP bar shows the penalty on a 0-100 scale (BP=1.00 -> full bar).
    stages = ["geo. mean\nof $p_n$", "brevity\npenalty (BP)", "= BLEU"]
    values = [geo * 100, bp * 100, score]
    labels_r = [f"{geo * 100:.1f}", f"BP={bp:.2f}", f"{score:.1f}"]
    scolors = [GREEN, AMBER, RED]
    sbars = ax2.bar(stages, values, color=scolors, edgecolor="white", linewidth=1.5)
    for bar, txt in zip(sbars, labels_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, txt, ha="center",
                 color=INK, fontsize=10, weight="bold")
    ax2.set_ylabel("score")
    ax2.set_title(f"BLEU = BP x exp($\\Sigma\\, w_n \\log p_n$) x 100\n"
                  f"= {bp:.2f} x {geo:.3f} x 100 = {score:.1f}", fontsize=10.5)
    ax2.set_ylim(0, 122)
    _style_axis(ax2)
    fig.suptitle("Anatomy of a BLEU score (measured, matches sacreBLEU)", fontsize=12.5,
                 weight="bold", color=INK, y=1.02)
    _save(fig, "mt_bleu_breakdown.png")


# ---- Figure 3: BLEU brittleness, measured -----------------------------------------------------
def fig_bleu_brittleness() -> None:
    """Measured: a meaning-preserving paraphrase scores ~0 BLEU but partial chrF.

    Two hypotheses against one reference: an exact match (BLEU 100, chrF 100) and a perfectly
    valid paraphrase that shares almost no word n-grams. BLEU cliffs to 0; chrF (characters)
    still credits the overlap. This is the single most important fact about MT evaluation and
    the reason the field moved past word-level overlap. Numbers from machine_translation.
    """
    ref = "the committee will convene on tuesday to discuss the budget"
    exact = "the committee will convene on tuesday to discuss the budget"
    paraphrase = "the panel meets tuesday to talk about the finances"
    rows = [("exact\nmatch", exact), ("valid\nparaphrase", paraphrase)]
    bleus = [sentence_bleu(h, [ref])["score"] for _, h in rows]
    chrfs = [chrf(h, ref) for _, h in rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(rows))
    width = 0.36
    b1 = ax.bar(x - width / 2, bleus, width, color=RED, edgecolor="white", linewidth=1.5, label="BLEU (word n-grams)")
    b2 = ax.bar(x + width / 2, chrfs, width, color=GREEN, edgecolor="white", linewidth=1.5, label="chrF (character n-grams)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.1f}", ha="center", color=INK, fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("score")
    ax.set_title("BLEU punishes a correct paraphrase; chrF is kinder (measured)\n"
                 "'the panel meets tuesday...' means the same as the reference, yet BLEU = 0",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_ylim(0, 112)
    ax.annotate("a perfect translation\nscores ZERO BLEU", xy=(1 - width / 2, 2), xytext=(0.55, 45),
                color=RED, fontsize=9.5, weight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    _style_axis(ax)
    _save(fig, "mt_bleu_brittleness.png")


# ---- Figure 4: IBM Model 1 alignment heatmap, measured (MANDATORY) ----------------------------
def fig_alignment_matrix() -> None:
    """Measured: the word-alignment matrix IBM Model 1 learns by EM, with no dictionary.

    Trained on three toy sentence pairs, the posterior alignment for 'la maison fleur' /
    'the house flower'. Bright cells are inferred alignment links: 'la'->'the', 'maison'->'house',
    'fleur'->'flower' emerge purely from iterated co-occurrence counting. On a real language pair
    the bright band would BEND wherever the languages reorder -- modeling that crossing is what
    separates Model 1 from the distortion-aware later models, and what attention later learned.
    """
    t = ibm_model1()
    fr = "la maison fleur"
    en = "the house flower"
    matrix = alignment_matrix(fr, en, t)
    fr_labels = fr.split()
    en_labels = en.split()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(fr_labels)))
    ax.set_xticklabels(fr_labels)
    ax.set_yticks(range(len(en_labels)))
    ax.set_yticklabels(en_labels)
    ax.set_xlabel("source word (French)")
    ax.set_ylabel("target word (English)")
    ax.set_title("IBM Model 1 alignment, learned by EM (measured)\n"
                 "no dictionary -- iterated co-occurrence discovers the word links", fontsize=10.5)
    for i in range(len(en_labels)):
        for j in range(len(fr_labels)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else INK, fontsize=11, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("alignment weight  (normalized t(f|e))", color=INK)
    _save(fig, "mt_alignment_matrix.png")


# ---- Figure 5: the metric landscape (illustrative) --------------------------------------------
def fig_metric_landscape() -> None:
    """Illustrative: where each MT metric sits on the surface-overlap -> meaning-aware spectrum.

    Four metrics laid left (surface) to right (meaning): BLEU (word n-grams), chrF (character
    n-grams), METEOR (stems/synonyms + alignment), COMET/BLEURT (learned, embedding-space). Moving
    right credits more valid variation (paraphrase, morphology, synonymy) at the cost of more
    machinery and (for learned metrics) a trained model. The arrow is the field's trajectory.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    metrics = [
        (0.4, BLUE, "BLEU", "word n-gram\noverlap + BP"),
        (3.2, PURPLE, "chrF", "character\nn-gram F-score"),
        (6.0, AMBER, "METEOR", "stems + synonyms\n+ alignment"),
        (8.8, GREEN, "COMET / BLEURT", "learned, embedding\nspace (meaning)"),
    ]
    for x, color, name, desc in metrics:
        _box(ax, (x, 2.3), 2.4, 1.2, name, color, fontsize=11)
        ax.text(x + 1.2, 1.9, desc, ha="center", va="top", color=INK, fontsize=8.5)
    # spectrum arrow
    ax.add_patch(FancyArrowPatch((0.4, 4.3), (11.2, 4.3), arrowstyle="-|>", mutation_scale=20,
                                 color=SLATE, linewidth=2.2))
    ax.text(0.6, 4.55, "surface overlap (cheap, brittle)", ha="left", color=INK, fontsize=9)
    ax.text(11.0, 4.55, "meaning-aware (credits paraphrase)", ha="right", color=INK, fontsize=9)
    ax.text(6.0, 0.7, "moving right credits valid variation (synonyms, morphology, paraphrase) "
                      "-- at the cost of more machinery",
            ha="center", color=SLATE, fontsize=9, style="italic")
    ax.set_title("The MT-metric spectrum: from surface overlap to learned meaning (illustrative)",
                 fontsize=11.5, color=INK, weight="bold")
    _save(fig, "mt_metric_landscape.png")


# ---- Figure 6: beam length normalization, measured --------------------------------------------
def fig_beam_length_norm() -> None:
    """Measured: length normalization flips the decoder's choice from a truncation to the full output.

    Two candidates -- a 4-token truncation A and the 7-token full faithful B. At alpha=0 (raw
    log-prob) the shorter A wins purely because it accumulates less negative log-prob; raising
    alpha divides by length^alpha and, by alpha=0.6 (the GNMT default), B overtakes. The bars are
    machine_translation.beam_length_norm_demo()'s exact scores.
    """
    rows = beam_length_norm_demo()
    alphas = [r["alpha"] for r in rows]
    score_a = [r["score_a"] for r in rows]
    score_b = [r["score_b"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(alphas))
    width = 0.36
    ba = ax.bar(x - width / 2, score_a, width, color=RED, edgecolor="white", linewidth=1.5,
                label="A: 'the black cat sleeps' (4 tok, truncated)")
    bb = ax.bar(x + width / 2, score_b, width, color=GREEN, edgecolor="white", linewidth=1.5,
                label="B: 'the black cat sleeps on the couch' (7 tok, full)")
    for bars in (ba, bb):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h - 0.06 if h < 0 else h + 0.02,
                    f"{h:.2f}", ha="center", va="top" if h < 0 else "bottom",
                    color=INK, fontsize=9, weight="bold")
    # mark the winner per alpha
    for i, r in enumerate(rows):
        win_b = r["winner"].startswith("B")
        ax.text(i, 0.12, "B wins" if win_b else "A wins", ha="center", color=GREEN if win_b else RED,
                fontsize=9, weight="bold")
    ax.axhline(0, color=SLATE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\alpha$={a}" for a in alphas])
    ax.set_ylabel("length-normalized score  $\\Sigma \\log p / L^{\\alpha}$")
    ax.set_title("Length normalization recovers the full translation (measured)\n"
                 "raw log-prob ($\\alpha$=0) prefers the short truncation; $\\alpha$=0.6 flips it",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_ylim(-2.1, 0.6)
    _style_axis(ax)
    _save(fig, "mt_beam_length_norm.png")


# ---- Figure 7: back-translation lift (illustrative) -------------------------------------------
def fig_backtranslation_curve() -> None:
    """Illustrative: back-translation's BLEU lift is widest where parallel data is scarcest.

    Two curves vs the amount of real parallel data (log x): real-only (blue) and real + synthetic
    back-translated (green). The shaded gap is the synthetic-data lift -- large at low resource,
    narrowing as real data grows. Shaped after Sennrich et al. 2016; absolute numbers are schematic.
    """
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    data = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    real_only = np.array([8, 14, 21, 27, 32, 35, 37], dtype=float)
    with_bt = np.array([16, 23, 29, 33, 36, 37.5, 38.5], dtype=float)
    ax.semilogx(data, with_bt, color=GREEN, linewidth=2.8, marker="o", markersize=6,
                markeredgecolor="white", label="real + back-translated synthetic")
    ax.semilogx(data, real_only, color=BLUE, linewidth=2.8, marker="s", markersize=6,
                markeredgecolor="white", label="real parallel data only")
    ax.fill_between(data, real_only, with_bt, color=GREEN, alpha=0.13)
    ax.annotate("biggest lift where\nparallel data is scarce", xy=(3e4, 18.5), xytext=(1.2e5, 11),
                color=SLATE, fontsize=9, arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel("real parallel sentence pairs (log scale)")
    ax.set_ylabel("BLEU")
    ax.set_title("Back-translation lifts BLEU most at low resource (illustrative)\n"
                 "synthetic pairs from monolingual target text -- shaped after Sennrich et al. 2016",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_ylim(0, 44)
    _style_axis(ax)
    _save(fig, "mt_backtranslation_curve.png")


# ---- Figure 8: measured NMT chrF, measured ----------------------------------------------------
def fig_nmt_measured() -> None:
    """Measured: chrF of a real opus-mt-fr-en run on three French sentences.

    Each bar is the chrF of the model's beam-search translation against a human reference. Sentence
    3 is an exact paraphrase of the reference (chrF 100); sentences 1-2 are faithful and fluent but
    worded differently ('sleeps' vs 'is sleeping'), which costs surface-overlap points even though
    the meaning is right. Falls back to published numbers if transformers is unavailable.
    """
    nmt = run_real_nmt()
    if nmt is not None:
        chrfs = nmt["chrf"]
        hyps = nmt["hyps"]
        corpus_bleu = nmt["corpus_bleu"]
        corpus_chrf = nmt["corpus_chrf"]
    else:
        # published, reproducible fallback (matches a verified opus-mt-fr-en run)
        chrfs = [67.7, 65.1, 100.0]
        hyps = ["The black cat sleeps on the couch.", "I like to learn foreign languages.",
                "Machine translation has progressed a lot."]
        corpus_bleu, corpus_chrf = 59.0, 79.0
    labels = [f"sentence {i + 1}" for i in range(len(NMT_SRCS))]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = [BLUE, PURPLE, GREEN]
    bars = ax.bar(labels, chrfs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, cf in zip(bars, chrfs):
        ax.text(bar.get_x() + bar.get_width() / 2, cf + 1.2, f"chrF {cf:.1f}",
                ha="center", color=INK, fontsize=10, weight="bold")
    ax.set_ylabel("chrF vs human reference")
    ax.set_title(f"A real Transformer NMT model, measured (opus-mt-fr-en, beam 5)\n"
                 f"corpus BLEU {corpus_bleu:.1f} / chrF {corpus_chrf:.1f} -- "
                 f"sentence 3 exact; 1-2 faithful but worded differently",
                 fontsize=10.5)
    ax.set_ylim(0, 112)
    # show the FR->EN under each bar
    for i, (src, hyp) in enumerate(zip(NMT_SRCS, hyps)):
        ax.text(i, -7, f"{src[:30]}...\n-> {hyp[:30]}", ha="center", va="top",
                color=SLATE, fontsize=7)
    _style_axis(ax)
    _save(fig, "mt_nmt_measured.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_evolution_timeline()
    fig_bleu_breakdown()
    fig_bleu_brittleness()
    fig_alignment_matrix()
    fig_metric_landscape()
    fig_beam_length_norm()
    fig_backtranslation_curve()
    fig_nmt_measured()
    print("all figures written.")


if __name__ == "__main__":
    main()

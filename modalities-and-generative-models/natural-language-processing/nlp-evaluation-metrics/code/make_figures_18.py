"""Reproducible figure generator for 18-NLP-Evaluation-Metrics.

Produces every embedded PNG for the chapter from the SAME functions used on the page and in the
notebook -- the metric-zoo disagreement bars, the meaning-blindness comparison, the brevity
penalty curve, the BLEU breakdown, the paired-bootstrap distribution, the metric-vs-human
correlation scatter, the correlation ladder, and the LLM-judge position-bias schematic are all
built from `nlp_evaluation.py`, so the figures cannot silently drift from the prose or the demo.

Run:
    python make_figures_18.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `nlpeval_`. The palette matches the chapter's Mermaid diagrams (muted, white
text on fills).

Figures produced (measured = from the chapter's own verified functions; illustrative = a
labelled schematic):
  nlpeval_taxonomy.png         -- illustrative: the four metric families on two axes (MANDATORY)
  nlpeval_metric_zoo.png       -- measured: five surface metrics disagree on the same pair (MANDATORY)
  nlpeval_bleu_vs_bertscore.png-- measured: surface vs semantic across exact/paraphrase/neg/unrelated
  nlpeval_brevity_penalty.png  -- measured: BLEU's brevity penalty vs candidate length
  nlpeval_bleu_breakdown.png   -- measured: per-order precision + BP -> final BLEU (Example B)
  nlpeval_bootstrap_ci.png     -- measured: paired-bootstrap distribution of (A - B), CI straddles 0
  nlpeval_correlation.png      -- measured: metric-vs-human scatter with Spearman/Pearson/Kendall
  nlpeval_judge_bias.png       -- illustrative: LLM-judge position bias + metric-correlation ladder

All measured numbers come from `nlp_evaluation.py` (pure-numpy core). Verified on
Python 3.12 / numpy 2.4.6, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from nlp_evaluation import (
    BLEU_CANDIDATE,
    BLEU_REFERENCE,
    ZOO_CANDIDATES,
    ZOO_REFERENCE,
    brevity_penalty,
    make_correlation_data,
    make_two_systems,
    metric_zoo,
    paired_bootstrap_diff,
    pearson,
    kendall_tau,
    semantic_overlap,
    sentence_bleu,
    spearman,
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
PAPER = "#FBFBFC"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===================================================================================================
# Figure 1 — the taxonomy (MANDATORY): four families on two axes.
# ===================================================================================================
def fig_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5, 6.7, "The NLP metric taxonomy", ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(
        5, 6.25,
        "left → right: cheaper & more reproducible  →  closer to true meaning & human judgement",
        ha="center", fontsize=9.5, color=SLATE, style="italic",
    )

    # Four family boxes left -> right, with the metrics each contains.
    families = [
        (BLUE, "n-gram / surface", ["BLEU", "ROUGE", "METEOR", "chrF", "EM", "token-F1"],
         "count shared words/chars\nmeaning-blind"),
        (PURPLE, "embedding-based", ["BERTScore", "MoverScore"],
         "compare vector reps\ncatches paraphrase"),
        (GREEN, "model / learned", ["BLEURT", "COMET", "perplexity"],
         "trained on human ratings\nhighest auto-correlation"),
        (RED, "LLM-as-judge", ["pairwise", "pointwise", "MT-Bench"],
         "a strong LLM scores/ranks\nreference-free, biased"),
    ]
    box_w, box_h = 2.15, 3.0
    xs = [0.25, 2.6, 4.95, 7.3]
    y0 = 2.2
    for (color, title, metrics, sub), x in zip(families, xs):
        box = FancyBboxPatch(
            (x, y0), box_w, box_h, boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=0, facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y0 + box_h - 0.32, title, ha="center", va="center",
                color="#fff", fontsize=10.5, fontweight="bold")
        ax.text(x + box_w / 2, y0 + box_h - 0.95, "\n".join(metrics), ha="center", va="top",
                color="#fff", fontsize=9.2, linespacing=1.35)
        ax.text(x + box_w / 2, y0 + 0.18, sub, ha="center", va="bottom",
                color="#E8ECF1", fontsize=8, style="italic", linespacing=1.2)

    # The "reference-based" bracket over the first two families.
    ax.annotate("", xy=(xs[0], 5.55), xytext=(xs[1] + box_w, 5.55),
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.2))
    ax.text((xs[0] + xs[1] + box_w) / 2, 5.72, "reference-based  (need a gold answer)",
            ha="center", fontsize=8.6, color=SLATE, fontweight="bold")
    ax.annotate("", xy=(xs[2], 5.55), xytext=(xs[3] + box_w, 5.55),
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.2))
    ax.text((xs[2] + xs[3] + box_w) / 2, 5.72, "often reference-free",
            ha="center", fontsize=8.6, color=SLATE, fontweight="bold")

    # Human eval as the foundation underneath everything.
    human_box = FancyBboxPatch(
        (0.25, 0.5), 9.2, 1.1, boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=0, facecolor=SLATE, zorder=2,
    )
    ax.add_patch(human_box)
    ax.text(4.85, 1.18, "HUMAN EVALUATION — the ground truth", ha="center", va="center",
            color="#fff", fontsize=11, fontweight="bold")
    ax.text(4.85, 0.78, "Likert / pairwise preference + inter-annotator agreement (κ). "
            "Every automatic metric is a proxy validated against this.",
            ha="center", va="center", color="#E8ECF1", fontsize=8.4, style="italic")
    for x in xs:
        ax.annotate("", xy=(x + box_w / 2, 1.6), xytext=(x + box_w / 2, 2.15),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.0))

    _save(fig, "nlpeval_taxonomy.png")


# ===================================================================================================
# Figure 2 — the metric zoo (MANDATORY): five surface metrics disagree on the same pair.
# ===================================================================================================
def fig_metric_zoo() -> None:
    # Use the paraphrase + negation candidates, which expose the sharpest disagreement.
    cases = ["paraphrase", "negation"]
    metric_names = ["Exact match", "Token-F1", "BLEU", "ROUGE-L", "chrF"]
    colors = [SLATE, BLUE, NAVY, PURPLE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, case in zip(axes, cases):
        z = metric_zoo(ZOO_CANDIDATES[case], ZOO_REFERENCE)
        sem = semantic_overlap(ZOO_CANDIDATES[case], ZOO_REFERENCE)
        values = [z[m] for m in metric_names] + [sem]
        labels = metric_names + ["semantic\n(embedding)"]
        bar_colors = colors + [RED]
        bars = ax.bar(range(len(values)), values, color=bar_colors, zorder=3, width=0.66)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold", color=INK)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5, rotation=0)
        ax.set_ylim(0, 112)
        _style_axis(ax)
        ref_short = "the movie was absolutely fantastic"
        cand = ZOO_CANDIDATES[case]
        ax.set_title(f"{case}: “{cand}”", fontsize=11)

    axes[0].set_ylabel("score (0–100)", fontsize=10.5)
    fig.suptitle(
        f"The metric zoo: five surface metrics + one semantic, same reference “{ZOO_REFERENCE}”",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "Paraphrase (left): surface metrics see “wrong” (BLEU 0, chrF 15) but the embedding metric "
        "sees “same meaning” (86).  Negation (right): surface metrics score high on shared words — "
        "and the embedding metric is FOOLED too (75), scoring an opposite-meaning sentence high.",
        ha="center", fontsize=8.8, color=SLATE, wrap=True,
    )
    fig.tight_layout()
    _save(fig, "nlpeval_metric_zoo.png")


# ===================================================================================================
# Figure 3 — BLEU/ROUGE vs semantic across all four cases (the meaning-blindness punchline).
# ===================================================================================================
def fig_bleu_vs_bertscore() -> None:
    cases = list(ZOO_CANDIDATES.keys())
    bleu_vals, rouge_vals, sem_vals = [], [], []
    for case in cases:
        z = metric_zoo(ZOO_CANDIDATES[case], ZOO_REFERENCE)
        bleu_vals.append(z["BLEU"])
        rouge_vals.append(z["ROUGE-L"])
        sem_vals.append(semantic_overlap(ZOO_CANDIDATES[case], ZOO_REFERENCE))

    x = np.arange(len(cases))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, bleu_vals, w, label="BLEU (surface)", color=BLUE, zorder=3)
    b2 = ax.bar(x, rouge_vals, w, label="ROUGE-L (surface)", color=PURPLE, zorder=3)
    b3 = ax.bar(x + w, sem_vals, w, label="semantic / embedding", color=RED, zorder=3)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8.3, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in cases], fontsize=10)
    ax.set_ylabel("score (0–100)", fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.set_title("Surface overlap vs meaning: where each family wins and where it is fooled",
                 fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style_axis(ax)

    # Annotate the two load-bearing cases.
    ax.annotate("paraphrase: surface punishes\na perfect rewrite; meaning wins",
                xy=(1 + w, sem_vals[1]), xytext=(1.05, 104),
                fontsize=8.2, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
    ax.annotate("negation: opposite meaning,\nyet semantic is fooled (high)",
                xy=(2 + w, sem_vals[2]), xytext=(2.6, 100),
                fontsize=8.2, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    fig.tight_layout()
    _save(fig, "nlpeval_bleu_vs_bertscore.png")


# ===================================================================================================
# Figure 4 — BLEU's brevity penalty vs candidate length.
# ===================================================================================================
def fig_brevity_penalty() -> None:
    ref_len = 20
    cand_lens = np.arange(4, 31)
    bp = np.array([brevity_penalty(int(c), ref_len) for c in cand_lens])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(cand_lens, bp, color=BLUE, lw=2.4, zorder=3)
    ax.axvline(ref_len, color=SLATE, ls="--", lw=1.2, zorder=2)
    ax.text(ref_len + 0.3, 0.2, "c = r\n(no penalty\nfor c ≥ r)", fontsize=8.5, color=SLATE)
    # Shade the penalized region (c < r).
    ax.fill_between(cand_lens, 0, bp, where=(cand_lens < ref_len), color=RED, alpha=0.12, zorder=1)
    # Mark a few worked points.
    for c in (8, 12, 16):
        ax.scatter([c], [brevity_penalty(c, ref_len)], color=RED, zorder=4, s=36)
        ax.text(c, brevity_penalty(c, ref_len) - 0.08, f"{brevity_penalty(c, ref_len):.2f}",
                ha="center", fontsize=8.5, color=RED, fontweight="bold")
    ax.set_xlabel("candidate length c  (reference length r = 20)", fontsize=10.5)
    ax.set_ylabel("brevity penalty  BP", fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("BLEU's brevity penalty: BP = exp(1 − r/c) punishes short output (the recall stand-in)",
                 fontsize=11.5)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "nlpeval_brevity_penalty.png")


# ===================================================================================================
# Figure 5 — BLEU breakdown (Example B): per-order precision + BP -> final BLEU.
# ===================================================================================================
def fig_bleu_breakdown() -> None:
    b = sentence_bleu(BLEU_CANDIDATE, BLEU_REFERENCE)
    precisions = b["precisions"]
    fig, ax = plt.subplots(figsize=(9, 4.8))

    labels = ["p₁", "p₂", "p₃", "p₄", "geo\nmean", "× BP", "BLEU\n(÷100)"]
    vals = precisions + [b["geo_mean"], b["bp"], b["bleu"] / 100]
    colors = [BLUE, BLUE, BLUE, BLUE, PURPLE, AMBER, GREEN]
    bars = ax.bar(range(len(vals)), vals, color=colors, zorder=3, width=0.66)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("value", fontsize=10.5)
    ax.set_title(f"BLEU built up (Example B): final BLEU = {b['bleu']:.1f}", fontsize=12.5)
    fig.text(0.5, -0.02,
             f"reference “{BLEU_REFERENCE}”  vs  candidate “{BLEU_CANDIDATE}”  —  "
             "the geometric mean (0.795) times the brevity penalty (0.847) gives 67.3; "
             "the one missing word “warm” costs ~12 points via BP.",
             ha="center", fontsize=8.6, color=SLATE)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "nlpeval_bleu_breakdown.png")


# ===================================================================================================
# Figure 6 — paired bootstrap distribution of (A - B): the CI straddles 0.
# ===================================================================================================
def fig_bootstrap_ci() -> None:
    sys_a, sys_b = make_two_systems()
    boot = paired_bootstrap_diff(sys_a, sys_b)
    diffs = boot["boot_diffs"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(diffs, bins=60, color=BLUE, alpha=0.75, zorder=3)
    ax.axvline(0, color=RED, ls="--", lw=1.8, zorder=4, label="zero (no difference)")
    ax.axvline(boot["observed_diff"], color=GREEN, lw=2.0, zorder=4,
               label=f"observed diff = {boot['observed_diff']:+.2f}")
    ax.axvspan(boot["ci_low"], boot["ci_high"], color=AMBER, alpha=0.18, zorder=1,
               label=f"95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]")
    ax.set_xlabel("bootstrap mean difference  (system A − system B)", fontsize=10.5)
    ax.set_ylabel("resample count", fontsize=10.5)
    ax.set_title("A “win” that isn’t: the 95% CI on (A − B) straddles zero — not significant",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    _style_axis(ax)
    fig.text(0.5, -0.02,
             "System A leads by +2.86 on average, but resampling the sentences shows that lead is "
             "indistinguishable from noise — reporting only the single number would ship noise as progress.",
             ha="center", fontsize=8.6, color=SLATE)
    fig.tight_layout()
    _save(fig, "nlpeval_bootstrap_ci.png")


# ===================================================================================================
# Figure 7 — metric-vs-human correlation scatter (Spearman / Pearson / Kendall).
# ===================================================================================================
def fig_correlation() -> None:
    human, metric = make_correlation_data()
    rho = spearman(metric, human)
    r = pearson(metric, human)
    tau = kendall_tau(metric, human)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(human, metric, color=BLUE, s=70, zorder=3, edgecolor="white", linewidth=1.2)
    # A guide line (perfect monotone agreement).
    lims = [min(human.min(), metric.min()) - 0.5, max(human.max(), metric.max()) + 0.5]
    ax.plot(lims, lims, color=SLATE, ls="--", lw=1.2, zorder=2, label="perfect agreement")
    ax.set_xlabel("human score (ground truth)", fontsize=10.5)
    ax.set_ylabel("automatic metric score", fontsize=10.5)
    ax.set_title("The meta-metric: how well does the metric correlate with humans?", fontsize=12)
    txt = f"Spearman ρ = {rho:.3f}\nPearson r = {r:.3f}\nKendall τ = {tau:.3f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.4", facecolor=PAPER, edgecolor=SLATE))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style_axis(ax)
    fig.text(0.5, -0.02,
             "ρ ≈ 0.94: high but < 1 — the metric ranks systems well (use it for iteration) yet is "
             "imperfect per single item (never accept/reject one output on it).",
             ha="center", fontsize=8.6, color=SLATE)
    fig.tight_layout()
    _save(fig, "nlpeval_correlation.png")


# ===================================================================================================
# Figure 8 — LLM-judge position bias (left) + metric-correlation ladder (right).
# ===================================================================================================
def fig_judge_bias() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    # --- LEFT: position bias schematic ---
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 10)
    axl.axis("off")
    axl.set_title("LLM-judge position bias", fontsize=12)
    # Order 1: (A, B) -> picks first
    box1 = FancyBboxPatch((0.5, 6.6), 9, 2.6, boxstyle="round,pad=0.08,rounding_size=0.15",
                          linewidth=0, facecolor=BLUE, alpha=0.9, zorder=2)
    axl.add_patch(box1)
    axl.text(5, 8.7, "Order 1:  shown  [Answer X]  then  [Answer Y]", ha="center",
             color="#fff", fontsize=10, fontweight="bold")
    axl.text(5, 7.9, "judge picks the FIRST → “X wins”", ha="center", color="#fff", fontsize=9.5)
    axl.text(5, 7.2, "✓ first", ha="center", color="#FFE08A", fontsize=9, fontweight="bold")
    # Order 2: (B, A) -> picks first again
    box2 = FancyBboxPatch((0.5, 3.4), 9, 2.6, boxstyle="round,pad=0.08,rounding_size=0.15",
                          linewidth=0, facecolor=PURPLE, alpha=0.9, zorder=2)
    axl.add_patch(box2)
    axl.text(5, 5.5, "Order 2:  shown  [Answer Y]  then  [Answer X]", ha="center",
             color="#fff", fontsize=10, fontweight="bold")
    axl.text(5, 4.7, "judge picks the FIRST → “Y wins”", ha="center", color="#fff", fontsize=9.5)
    axl.text(5, 4.0, "✓ first", ha="center", color="#FFE08A", fontsize=9, fontweight="bold")
    # Verdict box
    box3 = FancyBboxPatch((0.5, 0.4), 9, 2.4, boxstyle="round,pad=0.08,rounding_size=0.15",
                          linewidth=0, facecolor=RED, zorder=2)
    axl.add_patch(box3)
    axl.text(5, 1.95, "verdict FLIPS with order ⟹ position bias", ha="center",
             color="#fff", fontsize=10.5, fontweight="bold")
    axl.text(5, 1.1, "FIX: run both orders, require the SAME winner — else call it a tie",
             ha="center", color="#FFE0E4", fontsize=9, style="italic")

    # --- RIGHT: correlation ladder (illustrative) ---
    families = ["BLEU /\nROUGE-L", "METEOR /\nchrF", "BERTScore", "COMET\n(learned)", "LLM-as-\njudge"]
    corr = [0.30, 0.45, 0.60, 0.78, 0.85]  # illustrative typical correlations
    colors = [BLUE, BLUE, PURPLE, GREEN, RED]
    bars = axr.barh(range(len(families)), corr, color=colors, zorder=3, height=0.62)
    for b, c in zip(bars, corr):
        axr.text(c + 0.012, b.get_y() + b.get_height() / 2, f"{c:.2f}", va="center",
                 fontsize=9.5, fontweight="bold", color=INK)
    axr.set_yticks(range(len(families)))
    axr.set_yticklabels(families, fontsize=9)
    axr.set_xlim(0, 1.0)
    axr.set_xlabel("correlation with human judgement  (illustrative)", fontsize=10)
    axr.set_title("More semantic ⟶ better human correlation", fontsize=12)
    axr.invert_yaxis()
    _style_axis(axr)

    fig.text(0.5, -0.03,
             "Left: the same answer pair, two orders — the judge favours whichever is shown first, "
             "so the winner flips; the fix is swap-and-require-agreement.  Right: the robust trend "
             "(exact numbers vary by dataset) — surface < embedding < learned < LLM-judge.",
             ha="center", fontsize=8.7, color=SLATE)
    fig.tight_layout()
    _save(fig, "nlpeval_judge_bias.png")


def main() -> None:
    fig_taxonomy()
    fig_metric_zoo()
    fig_bleu_vs_bertscore()
    fig_brevity_penalty()
    fig_bleu_breakdown()
    fig_bootstrap_ci()
    fig_correlation()
    fig_judge_bias()
    print("all figures written.")


if __name__ == "__main__":
    main()

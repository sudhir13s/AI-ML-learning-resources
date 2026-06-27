"""Reproducible figure generator for 17-Chain-of-Thought-Reasoning.

Produces every embedded matplotlib PNG for the chapter from the SAME numbers used on the
page and in the teaching notebook -- the two measured figures import their data straight from
chain_of_thought.py (the single seeded source of truth), so they cannot silently drift from
the prose. The two illustrative figures (emergent-at-scale, test-time compute) are clearly
labelled "illustrative" and trace published curves (Wei et al. 2022; o1-style reasoning).

Run:
    python make_figures_17.py

Each figure is written to ../../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills).

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from chain_of_thought import (
    P_CORRECT,
    SC_K_VALUES,
    direct_vs_cot,
    self_consistency_sweep,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
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
    print(f"wrote {path.name}")


# =====================================================================================
# Figure 1 -- the felt problem: a multi-step word problem the direct answer gets wrong
# =====================================================================================
def fig_direct_vs_cot_prompt() -> None:
    """Two prompt panels side by side: answer-only (wrong) vs step-by-step (right).

    Illustrative of the PROMPT STRUCTURE difference -- the worked roller-skate problem from
    the page. Left: the model is asked for the answer and blurts a wrong number. Right: the
    same problem, but "Let's think step by step" gives it room to carry state and land right.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Left panel: direct prompt -> wrong
    ax_l.add_patch(plt.Rectangle((0.04, 0.06), 0.92, 0.88, facecolor="white", edgecolor=RED, linewidth=2))
    ax_l.text(0.5, 0.88, "DIRECT  (answer only)", ha="center", fontsize=13, fontweight="bold", color=RED)
    ax_l.text(0.08, 0.74,
              "Q: A jug holds 12 cups. You pour out\n"
              "    1/3, then add 5 cups. How many\n"
              "    cups are in the jug now?",
              ha="left", va="top", fontsize=10.5, color=INK)
    ax_l.text(0.08, 0.40, "A: 9", ha="left", va="top", fontsize=12, color=INK, fontfamily="monospace")
    ax_l.text(0.5, 0.18, "✗  wrong  (blurted)", ha="center", fontsize=12, fontweight="bold", color=RED)

    # Right panel: CoT prompt -> right
    ax_r.add_patch(plt.Rectangle((0.04, 0.06), 0.92, 0.88, facecolor="white", edgecolor=GREEN, linewidth=2))
    ax_r.text(0.5, 0.88, "CHAIN-OF-THOUGHT  (step by step)", ha="center", fontsize=13, fontweight="bold", color=GREEN)
    ax_r.text(0.08, 0.78,
              "Q: ... (same problem)\n"
              "A: Let's think step by step.",
              ha="left", va="top", fontsize=10.5, color=INK)
    ax_r.text(0.08, 0.56,
              "  Start: 12 cups.\n"
              "  Pour out 1/3 of 12 = 4, leaving 8.\n"
              "  Add 5: 8 + 5 = 13.",
              ha="left", va="top", fontsize=10, color=SLATE, fontfamily="monospace")
    ax_r.text(0.08, 0.24, "  Answer: 13", ha="left", va="top", fontsize=12, color=INK, fontfamily="monospace")
    ax_r.text(0.5, 0.11, "✓  right  (worked)", ha="center", fontsize=12, fontweight="bold", color=GREEN)

    fig.suptitle("Same problem, two prompt structures: room to work changes the answer",
                 fontsize=13, color=INK, y=1.01)
    _save(fig, "cot_direct_vs_cot_prompt.png")


# =====================================================================================
# Figure 2 -- direct vs CoT MEASURED accuracy (from chain_of_thought.direct_vs_cot)
# =====================================================================================
def fig_direct_vs_cot_accuracy() -> None:
    """Bar chart of the measured direct vs CoT accuracy on the 5-step modular task.

    The numbers come straight from chain_of_thought.direct_vs_cot() -- the same values the
    page quotes and the notebook prints. Showing intermediate state turns one hard joint
    guess into several easy local ones, so CoT roughly triples the accuracy here.
    """
    direct_acc, cot_acc = direct_vs_cot()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels = ["Direct\n(one-shot)", "CoT\n(emit each step)"]
    vals = [direct_acc, cot_acc]
    colors = [RED, GREEN]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5, zorder=3, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("final-answer accuracy")
    ax.set_title("Direct vs CoT on a 5-step chain (measured, mod 7)", fontsize=12)
    ax.annotate(f"+{cot_acc - direct_acc:.2f}", xy=(1, cot_acc), xytext=(0.5, 0.5),
                fontsize=12, color=GREEN, fontweight="bold", ha="center")
    _style_axis(ax)
    _save(fig, "cot_direct_vs_cot_accuracy.png")


# =====================================================================================
# Figure 3 -- self-consistency: MEASURED accuracy vs number of sampled chains K
# =====================================================================================
def fig_self_consistency_curve() -> None:
    """Majority-vote accuracy rising with K, from chain_of_thought.self_consistency_sweep().

    Single-chain accuracy is ~P_CORRECT; sampling K chains and voting concentrates on the
    correct answer, so accuracy climbs toward 1. Same numbers as the page table and notebook.
    """
    acc = self_consistency_sweep()
    ks = list(SC_K_VALUES)
    ys = [acc[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ks, ys, marker="o", markersize=7, color=PURPLE, linewidth=2.2, zorder=3,
            label="majority vote over K chains")
    ax.axhline(acc[1], color=RED, linestyle="--", linewidth=1.6, zorder=2,
               label=f"single chain ({acc[1]:.2f})")
    for k, y in zip(ks, ys):
        ax.text(k, y + 0.02, f"{y:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("number of sampled reasoning chains  K")
    ax.set_ylabel("answer accuracy")
    ax.set_ylim(0.35, 1.04)
    ax.set_xticks(ks)
    ax.set_title(f"Self-consistency: voting beats a single chain (P_correct={P_CORRECT})", fontsize=12)
    ax.legend(loc="lower right", frameon=False)
    _style_axis(ax)
    _save(fig, "cot_self_consistency_curve.png")


# =====================================================================================
# Figure 4 -- CoT is emergent at scale (illustrative, traces Wei et al. 2022 GSM8K)
# =====================================================================================
def fig_emergent_scale() -> None:
    """Standard vs CoT accuracy as model scale grows -- the emergent-ability shape.

    ILLUSTRATIVE: traces the qualitative GSM8K curve from Wei et al. (2022). Standard prompting
    stays near-flat and low as scale grows; CoT is flat-and-low (even slightly worse) for small
    models, then turns sharply upward past ~10-100B params. The anchor points use the published
    PaLM-540B GSM8K figures (standard ~18%, CoT ~57%); the small-model points are schematic.
    """
    # x = log10(params in billions); schematic small-model points + published 540B anchors
    sizes_b = np.array([0.4, 2.0, 8.0, 62.0, 540.0])
    log_sizes = np.log10(sizes_b)
    standard = np.array([0.05, 0.06, 0.07, 0.10, 0.18])  # near-flat, low; 540B ~18% is published
    cot = np.array([0.03, 0.05, 0.08, 0.28, 0.57])  # worse for tiny models, sharp rise; 540B ~57% published

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(log_sizes, standard, marker="s", markersize=7, color=SLATE, linewidth=2.2,
            label="standard prompting", zorder=3)
    ax.plot(log_sizes, cot, marker="o", markersize=7, color=GREEN, linewidth=2.2,
            label="chain-of-thought", zorder=3)
    ax.axvspan(np.log10(0.3), np.log10(10), color=RED, alpha=0.06, zorder=0)
    ax.text(np.log10(1.7), 0.50, "CoT can\nHURT small\nmodels", ha="center", fontsize=9, color=RED)
    ax.annotate("emerges\n~10-100B", xy=(np.log10(62), 0.28), xytext=(np.log10(20), 0.44),
                fontsize=9.5, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.set_xticks(log_sizes)
    ax.set_xticklabels([f"{s:g}B" for s in sizes_b])
    ax.set_xlabel("model scale (parameters, log)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_ylim(0, 0.65)
    ax.set_title("CoT is emergent at scale  (illustrative — traces Wei et al. 2022)", fontsize=12)
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax)
    _save(fig, "cot_emergent_scale.png")


# =====================================================================================
# Figure 5 -- test-time compute: accuracy vs reasoning tokens (illustrative, o1-style)
# =====================================================================================
def fig_test_time_compute() -> None:
    """Accuracy climbing with the number of reasoning ("thinking") tokens spent before answering.

    ILLUSTRATIVE: traces the qualitative "more test-time compute -> higher accuracy" curve that
    o1-style reasoning models report. The point is the SHAPE (diminishing-returns rise with a
    log-x reasoning budget), not exact values -- so the y-axis is unit-free "accuracy".
    """
    tokens = np.array([64, 256, 1024, 4096, 16384, 65536])
    acc = np.array([0.30, 0.42, 0.54, 0.65, 0.72, 0.76])  # diminishing-returns rise, schematic
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(tokens, acc, marker="o", markersize=7, color=NAVY, linewidth=2.4, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("reasoning tokens spent before answering (log)")
    ax.set_ylabel("task accuracy")
    ax.set_ylim(0.2, 0.85)
    ax.set_title("Test-time compute: more thinking, higher accuracy  (illustrative)", fontsize=12)
    ax.fill_between(tokens, acc, 0.2, color=NAVY, alpha=0.07, zorder=1)
    _style_axis(ax)
    _save(fig, "cot_test_time_compute.png")


def main() -> None:
    fig_direct_vs_cot_prompt()
    fig_direct_vs_cot_accuracy()
    fig_self_consistency_curve()
    fig_emergent_scale()
    fig_test_time_compute()
    print("\nall figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

"""Reproducible figure generator for 06. NLP / 17-Decoding-Strategies.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
teaching notebook -- the constants and functions are IMPORTED from decoding_strategies.py, so
the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_17.py

Each figure is written to ../../images/ (the shared NLP chapter image dir) at 150 dpi, prefixed
`decode_`. The palette matches the chapter's Mermaid diagrams (muted, white text on coloured
fills). VIEW each PNG before shipping.

Figures produced (8):
  decode_beam_tree.png            -- greedy-vs-beam two-step tree (AX 0.22 vs BP 0.4275)
  decode_beam_trace.png           -- beam B=2 log-space prune across 3 steps
  decode_temperature_softmax.png  -- logits [3,2,1,0] reshaped at T=0.5/1.0/2.0 (+ entropy)
  decode_entropy_vs_temperature.png -- entropy(T) sweep, with the three quoted T marked
  decode_topk_vs_topp.png         -- SAME 8-token dist: fixed top-k=2 vs adaptive top-p=0.9
  decode_nucleus_adapts.png       -- nucleus size vs entropy (peaked->flat), top-k flat line
  decode_degeneration.png         -- distinct-2: greedy loop vs low-T vs full-T sampling
  decode_quality_diversity.png    -- the quality-diversity plane with each strategy placed

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from decoding_strategies import (
    BEAM_TRACE_STEP1,
    BEAM_TRACE_STEP2_INC,
    BEAM_TRACE_WIDTH,
    EIGHT_TOKEN_DIST,
    FIRST_STEP,
    PEAKED_LOGITS,
    SECOND_STEP,
    TEMP_LOGITS,
    TEMPERATURES,
    TOP_K,
    TOP_P,
    distinct_rate,
    entropy_bits,
    generate_greedy,
    generate_sampled,
    nucleus_size,
    softmax,
    softmax_with_temperature,
    top_k_keep,
    top_p_keep,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------------
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
    print(f"wrote {path}")


def fig_beam_tree() -> None:
    """The two-step generation tree: greedy takes A->X (0.22); beam keeps B alive and finds BP."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axis("off")

    # Node coordinates (x = depth, y = vertical position).
    root = (0.0, 0.0)
    firsts = {"A": (1.0, 1.1), "B": (1.0, -1.1)}
    seconds = {
        "X": (2.0, 1.8), "Y": (2.0, 1.1), "Z": (2.0, 0.4),
        "P": (2.0, -0.6), "Q": (2.0, -1.6),
    }

    def draw_node(xy, label, color, r=0.16):
        circ = plt.Circle(xy, r, color=color, zorder=3)
        ax.add_patch(circ)
        ax.text(xy[0], xy[1], label, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=4)

    def draw_edge(a, b, weight, color, lw, label_off=0.0):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-", color=color, lw=lw, alpha=0.9), zorder=1)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.16 + label_off, weight, ha="center", va="center",
                fontsize=10, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color, lw=1.0), zorder=5)

    # Edges: first step (probabilities pulled from FIRST_STEP so labels can't drift).
    draw_edge(root, firsts["A"], f"{FIRST_STEP['A']:.2f}", BLUE, 3.2)
    draw_edge(root, firsts["B"], f"{FIRST_STEP['B']:.2f}", PURPLE, 2.4)
    # Second step.
    for tok, p in SECOND_STEP["A"].items():
        col = AMBER if tok == "X" else SLATE
        draw_edge(firsts["A"], seconds[tok], f"{p:.2f}", col, 2.6 if tok == "X" else 1.6)
    for tok, p in SECOND_STEP["B"].items():
        col = GREEN if tok == "P" else SLATE
        draw_edge(firsts["B"], seconds[tok], f"{p:.2f}", col, 3.0 if tok == "P" else 1.4)

    # Nodes.
    draw_node(root, "•", NAVY)
    draw_node(firsts["A"], "A", BLUE)
    draw_node(firsts["B"], "B", PURPLE)
    for tok, xy in seconds.items():
        col = AMBER if tok == "X" else GREEN if tok == "P" else SLATE
        draw_node(xy, tok, col)

    # Joint-probability annotations on the two terminal paths (values from FIRST_STEP/SECOND_STEP
    # so the figure can never disagree with the canonical numbers).
    ax_p = FIRST_STEP["A"] * SECOND_STEP["A"]["X"]   # greedy AX
    bp_p = FIRST_STEP["B"] * SECOND_STEP["B"]["P"]   # global-best BP
    ax.text(2.55, seconds["X"][1],
            f"greedy: A·X\n{FIRST_STEP['A']:.2f}×{SECOND_STEP['A']['X']:.2f} = {ax_p:.2f}",
            ha="left", va="center", fontsize=10.5, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F3E2E4", ec=RED, lw=1.2))
    ax.text(2.55, seconds["P"][1],
            f"GLOBAL BEST: B·P\n{FIRST_STEP['B']:.2f}×{SECOND_STEP['B']['P']:.2f} = {bp_p:.4f}",
            ha="left", va="center", fontsize=10.5, color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#DDEFE6", ec=GREEN, lw=1.2))

    ax.text(0.0, 2.55, "Greedy picks the locally-best first token A and lands at AX (0.22).\n"
            "Beam (width 2) keeps B alive and recovers BP (0.4275) — nearly double.",
            ha="center", va="center", fontsize=11.5, color=INK)
    ax.set_xlim(-0.5, 4.4)
    ax.set_ylim(-2.2, 3.0)
    _save(fig, "decode_beam_tree.png")


def fig_beam_trace() -> None:
    """Beam B=2 log-space scoring across 3 steps: bars = scores, kept vs pruned by colour.

    Uses the small tree of the page's log-space worked example (vocab {a,b,c}). Each panel
    shows the candidate scores at that step; the top-2 (kept) are GREEN, the pruned are SLATE.
    """
    # Step-1 log-probs and step-2 increments come from decoding_strategies.py (single source of
    # truth), so this figure cannot drift from the page's worked example or the asserted trace.
    step1 = BEAM_TRACE_STEP1
    step2_inc = BEAM_TRACE_STEP2_INC
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Panel 1: step-1 candidates, keep top BEAM_TRACE_WIDTH.
    ax = axes[0]
    cands1 = sorted(step1.items(), key=lambda kv: kv[1], reverse=True)
    kept1 = {k for k, _ in cands1[:BEAM_TRACE_WIDTH]}
    labels1 = [k for k, _ in cands1]
    scores1 = [v for _, v in cands1]
    colors1 = [GREEN if k in kept1 else SLATE for k in labels1]
    ax.bar(labels1, scores1, color=colors1, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (k, v) in enumerate(cands1):
        ax.text(i, v - 0.08, f"{v:.1f}", ha="center", va="top", fontsize=11, color="white",
                fontweight="bold")
    ax.set_title("Step 1: keep top B=2 (green); prune c (slate)", fontsize=12)
    ax.set_ylabel("score = Σ log p  (less negative = better)")
    ax.set_ylim(min(scores1) - 0.5, 0.1)
    _style_axis(ax)

    # Panel 2: step-2 candidates (expand the step-1 survivors), keep top BEAM_TRACE_WIDTH.
    ax = axes[1]
    survivors1 = [k for k, _ in cands1[:BEAM_TRACE_WIDTH]]
    cands2 = []
    for parent in survivors1:
        for tok, inc in step2_inc[parent].items():
            cands2.append((parent + tok, step1[parent] + inc))
    cands2.sort(key=lambda kv: kv[1], reverse=True)
    kept2 = {k for k, _ in cands2[:BEAM_TRACE_WIDTH]}
    labels2 = [k for k, _ in cands2]
    scores2 = [v for _, v in cands2]
    colors2 = [GREEN if k in kept2 else SLATE for k in labels2]
    ax.bar(labels2, scores2, color=colors2, edgecolor="white", linewidth=0.8, zorder=3)
    for i, (k, v) in enumerate(cands2):
        ax.text(i, v - 0.06, f"{v:.1f}", ha="center", va="top", fontsize=10, color="white",
                fontweight="bold")
    ax.set_title("Step 2: expand both, score globally, keep ab & ba", fontsize=12)
    ax.set_ylim(min(scores2) - 0.5, 0.1)
    _style_axis(ax)

    fig.suptitle("Beam search scores in log-space and prunes GLOBALLY: ba (−1.0) beats aa (−1.7) "
                 "though 'a' was the better first token", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "decode_beam_trace.png")


def fig_temperature_softmax() -> None:
    """The 4-logit vector [3,2,1,0] reshaped by temperature T = 0.5 / 1.0 / 2.0."""
    tokens = ["A", "B", "C", "D"]
    logits = np.array(TEMP_LOGITS)
    colors = {0.5: BLUE, 1.0: PURPLE, 2.0: AMBER}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    x = np.arange(len(tokens))
    for ax, temp in zip(axes, TEMPERATURES):
        probs = softmax_with_temperature(logits, temp)
        h = entropy_bits(probs)
        ax.bar(x, probs, color=colors[temp], edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0.25, color=SLATE, linestyle=":", linewidth=1.4, zorder=2)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(f"T = {temp}    (entropy {h:.2f} bits)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, fontsize=11)
        ax.set_ylim(0, 1.02)
        _style_axis(ax)
    axes[0].set_ylabel("probability")
    axes[2].text(3.05, 0.27, "uniform 0.25", fontsize=8, color=SLATE, va="bottom", ha="right")
    fig.suptitle("Temperature reshapes the softmax of logits [3,2,1,0]: "
                 "T<1 sharpens toward greedy, T>1 flattens toward uniform",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "decode_temperature_softmax.png")


def fig_entropy_vs_temperature() -> None:
    """Entropy as a function of temperature for the [3,2,1,0] vector, with the three quoted T."""
    logits = np.array(TEMP_LOGITS)
    temps = np.linspace(0.2, 4.0, 80)
    h = [entropy_bits(softmax_with_temperature(logits, float(t))) for t in temps]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(temps, h, color=PURPLE, linewidth=2.8, zorder=4)
    ax.axhline(np.log2(len(logits)), color=SLATE, linestyle=":", linewidth=1.8,
               label=f"max entropy log₂(4) = {np.log2(4):.2f} bits (uniform)", zorder=2)
    for t in TEMPERATURES:
        ht = entropy_bits(softmax_with_temperature(logits, t))
        ax.plot(t, ht, "o", color=AMBER, markersize=9, zorder=5)
        ax.annotate(f"T={t}\n{ht:.2f} bits", (t, ht), textcoords="offset points",
                    xytext=(8, -2), fontsize=9.5, color=INK)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("entropy (bits)")
    ax.set_title("Higher temperature → higher entropy → flatter sampling distribution")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_ylim(0, 2.1)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "decode_entropy_vs_temperature.png")


def fig_topk_vs_topp() -> None:
    """SAME 8-token distribution under top-k=2 (fixed) vs top-p=0.9 (adaptive)."""
    dist = np.array(EIGHT_TOKEN_DIST)
    tokens = [f"t{i+1}" for i in range(len(dist))]
    k_keep = top_k_keep(dist, TOP_K)
    p_keep = top_p_keep(dist, TOP_P)
    x = np.arange(len(dist))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)

    for ax, keep, title, keep_color in (
        (axes[0], k_keep, f"top-k (k={TOP_K}): {int(k_keep.sum())} tokens, mass {dist[k_keep].sum():.2f}", BLUE),
        (axes[1], p_keep, f"top-p (p={TOP_P}): {int(p_keep.sum())} tokens, mass {dist[p_keep].sum():.2f}", GREEN),
    ):
        for i in range(len(dist)):
            ax.bar(x[i], dist[i], color=keep_color if keep[i] else SLATE,
                   alpha=1.0 if keep[i] else 0.30, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, fontsize=10)
        _style_axis(ax)
    axes[0].set_ylabel("probability")
    fig.suptitle("Same distribution, two cutoffs: top-k keeps a FIXED 2; "
                 "top-p keeps the smallest set with cumulative mass ≥ 0.9 (here 5)",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "decode_topk_vs_topp.png")


def fig_nucleus_adapts() -> None:
    """Nucleus size vs distribution entropy (peaked → flat); top-k stays a flat line."""
    peaked = np.array(PEAKED_LOGITS)
    uniform = np.full_like(peaked, float(peaked.mean()))
    steps = np.linspace(0.0, 1.0, 11)  # 0 = peaked, 1 = uniform
    entropies, nuclei = [], []
    for s in steps:
        logits = (1 - s) * peaked + s * uniform
        probs = softmax(logits)
        entropies.append(entropy_bits(probs))
        nuclei.append(nucleus_size(probs, TOP_P))
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(entropies, nuclei, "-o", color=GREEN, linewidth=2.8, markersize=8,
            label=f"top-p (p={TOP_P}) nucleus — ADAPTS to shape", zorder=4)
    ax.axhline(TOP_K, color=BLUE, linestyle="--", linewidth=2.4,
               label=f"top-k (k={TOP_K}) — FIXED, ignores shape", zorder=3)
    ax.set_xlabel("distribution entropy (bits)  —  left = peaked, right = flat")
    ax.set_ylabel("tokens kept")
    ax.set_title("top-p widens as the model gets less certain; top-k is stuck at k")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "decode_nucleus_adapts.png")


def fig_degeneration() -> None:
    """Repetition (distinct-2) for greedy vs low-T vs full-T sampling on the toy loop model."""
    greedy_ids = generate_greedy()
    lowt_ids = generate_sampled(temperature=0.3)
    samp_ids = generate_sampled(temperature=1.0)
    labels = ["greedy\n(argmax)", "sampling\nT=0.3", "sampling\nT=1.0"]
    rates = [distinct_rate(greedy_ids), distinct_rate(lowt_ids), distinct_rate(samp_ids)]
    colors = [RED, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8, zorder=3, width=0.6)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.015, f"{r:.3f}",
                ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel("distinct-2  (fraction of unique bigrams)")
    ax.set_ylim(0, 0.62)
    ax.set_title("Neural text degeneration: greedy collapses into a loop;\n"
                 "sampling keeps the text varied — same model, decoding alone")
    ax.annotate("'I love pizza . I love pizza . …'\n(the repeating loop)", (0, rates[0]),
                textcoords="offset points", xytext=(0, 38), ha="center", fontsize=9.5,
                color=RED, bbox=dict(boxstyle="round,pad=0.3", fc="#F3E2E4", ec=RED, lw=1.0))
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "decode_degeneration.png")


def fig_quality_diversity() -> None:
    """The quality-diversity plane: each strategy placed as a point, with the human-like band."""
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    # Illustrative placements (axes are conceptual, not measured) — labelled as such in caption.
    points = [
        ("greedy", 0.15, 0.85, RED),
        ("beam search", 0.22, 0.92, PURPLE),
        ("low-T top-p", 0.55, 0.86, BLUE),
        ("nucleus p≈0.9", 0.70, 0.80, GREEN),
        ("pure sampling T=1\n(no truncation)", 0.95, 0.34, AMBER),
    ]
    for label, div, qual, color in points:
        ax.scatter(div, qual, s=320, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(label, (div, qual), textcoords="offset points", xytext=(0, -28),
                    ha="center", fontsize=10, color=INK)
    # Human-like band.
    ax.axvspan(0.5, 0.78, color=GREEN, alpha=0.10, zorder=1)
    ax.text(0.64, 0.18, "human-like band\n(diverse AND coherent)", ha="center", va="center",
            fontsize=10.5, color=GREEN, fontweight="bold")
    ax.set_xlabel("diversity  (variety · surprise · distinct-n)  →")
    ax.set_ylabel("quality  (coherence · factuality · on-topic)  →")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.1, 1.02)
    ax.set_title("The quality–diversity trade-off: every decoding knob picks a point on this plane\n"
                 "(illustrative placement — axes are conceptual, not measured)")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "decode_quality_diversity.png")


def main() -> None:
    fig_beam_tree()
    fig_beam_trace()
    fig_temperature_softmax()
    fig_entropy_vs_temperature()
    fig_topk_vs_topp()
    fig_nucleus_adapts()
    fig_degeneration()
    fig_quality_diversity()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

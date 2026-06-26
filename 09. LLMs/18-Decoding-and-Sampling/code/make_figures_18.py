"""Reproducible figure generator for 18-Decoding-and-Sampling.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the constants and filter functions are IMPORTED from decoding_sampling.py, so the
figures cannot silently drift from the prose or the demo. Run:

    python make_figures_18.py

Each figure is written to ../../images/ (the shared chapter image dir) at 150 dpi. The palette
matches the chapter's Mermaid diagrams (muted, white text on coloured fills).

Figures produced:
  dec_temperature_softmax.png  -- the peaked next-token dist reshaped by T=0.5/1.0/2.0 (+ entropy)
  dec_greedy_vs_sampling.png   -- greedy's single deterministic spike vs a sampled spread
  dec_topk_vs_topp.png         -- SAME peaked & flat dists: fixed-k cutoff vs adaptive nucleus
  dec_nucleus_adapts.png       -- nucleus size: small when peaked, large when flat (top-k flat line)
  dec_entropy_vs_temperature.png-- entropy(T) curve over a temperature sweep (peaked vs flat)

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from decoding_sampling import (
    FLAT_LOGITS,
    PEAKED_LOGITS,
    TEMPERATURES,
    TOP_K,
    TOP_P,
    VOCAB,
    entropy_bits,
    nucleus_size,
    softmax_with_temperature,
    top_k_filter,
    top_p_filter,
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

PEAKED = torch.tensor(PEAKED_LOGITS)
FLAT = torch.tensor(FLAT_LOGITS)


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


def fig_temperature_softmax() -> None:
    """The peaked next-token distribution reshaped by temperature T = 0.5 / 1.0 / 2.0."""
    colors = {0.5: BLUE, 1.0: PURPLE, 2.0: AMBER}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    x = np.arange(len(VOCAB))
    for ax, temp in zip(axes, TEMPERATURES):
        probs = softmax_with_temperature(PEAKED, temp).numpy()
        h = entropy_bits(softmax_with_temperature(PEAKED, temp))
        ax.bar(x, probs, color=colors[temp], edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_title(f"T = {temp}    (entropy {h:.2f} bits)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(VOCAB, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        _style_axis(ax)
    axes[0].set_ylabel("probability")
    fig.suptitle(
        "Temperature reshapes the next-token softmax: T<1 sharpens, T>1 flattens",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "dec_temperature_softmax.png")


def fig_greedy_vs_sampling() -> None:
    """Greedy picks ONE token every time (a spike); sampling draws a spread proportional to p.

    Uses the FLAT distribution so the contrast is vivid: greedy collapses a high-uncertainty
    distribution onto a single token, throwing away every alternative, while sampling keeps
    the genuine spread the model expressed.
    """
    probs = F.softmax(FLAT, dim=-1).numpy()
    greedy = np.zeros_like(probs)
    greedy[int(np.argmax(probs))] = 1.0  # all mass on the argmax -- the deterministic spike
    x = np.arange(len(VOCAB))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - width / 2, greedy, width, color=RED, edgecolor="white", linewidth=0.6,
           label="greedy (argmax) — always this one token", zorder=3)
    ax.bar(x + width / 2, probs, width, color=GREEN, edgecolor="white", linewidth=0.6,
           label="sampling — picks each token with prob p", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(VOCAB, rotation=45, ha="right")
    ax.set_ylabel("probability of being chosen")
    ax.set_title("On an uncertain (flat) dist: greedy collapses to one token; sampling keeps the spread")
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "dec_greedy_vs_sampling.png")


def _filtered_probs(logits: torch.Tensor, filtered: torch.Tensor) -> np.ndarray:
    """Renormalised probabilities after a filter (masked tokens -> exactly 0)."""
    return F.softmax(filtered, dim=-1).numpy()


def fig_topk_vs_topp() -> None:
    """SAME two distributions, two truncations: fixed-k cutoff vs adaptive nucleus.

    Row 1 = peaked dist; row 2 = flat dist. Left col = top-k (k fixed); right col = top-p
    (nucleus adapts). Kept tokens coloured, removed tokens greyed -- the visual proof that
    top-p widens on the flat dist while top-k does not.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=True)
    x = np.arange(len(VOCAB))
    rows = [("peaked", PEAKED), ("flat", FLAT)]
    for r, (name, logits) in enumerate(rows):
        base = F.softmax(logits, dim=-1).numpy()
        k_filt = top_k_filter(logits, TOP_K)
        p_filt = top_p_filter(logits, TOP_P)
        k_kept = torch.isfinite(k_filt).numpy()
        p_kept = torch.isfinite(p_filt).numpy()
        for c, (kept, label, keep_color) in enumerate(
            [(k_kept, f"top-k (k={TOP_K})", BLUE), (p_kept, f"top-p (p={TOP_P})", GREEN)]
        ):
            ax = axes[r][c]
            bar_colors = [keep_color if kept[i] else SLATE for i in range(len(VOCAB))]
            alphas = [1.0 if kept[i] else 0.30 for i in range(len(VOCAB))]
            for i in range(len(VOCAB)):
                ax.bar(x[i], base[i], color=bar_colors[i], alpha=alphas[i],
                       edgecolor="white", linewidth=0.6, zorder=3)
            n_kept = int(kept.sum())
            ax.set_title(f"{name} dist — {label}: {n_kept} tokens kept", fontsize=11)
            ax.set_ylim(0, max(base.max() * 1.15, 0.15))
            _style_axis(ax)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(VOCAB, rotation=45, ha="right", fontsize=8)
    axes[0][0].set_ylabel("probability")
    axes[1][0].set_ylabel("probability")
    fig.suptitle(
        "top-k keeps a FIXED count (3 either way); top-p ADAPTS: 1 token when peaked, 9 when flat",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "dec_topk_vs_topp.png")


def fig_nucleus_adapts() -> None:
    """Nucleus size vs distribution peakiness: small when peaked, large when flat (top-k flat)."""
    # Sweep a family of distributions from peaked -> flat by interpolating the peaked logits
    # toward a uniform vector; plot the resulting nucleus size and the fixed top-k line.
    steps = np.linspace(0.0, 1.0, 9)  # 0 = fully peaked, 1 = fully flat (uniform)
    uniform = torch.full_like(PEAKED, float(PEAKED.mean()))
    nucleus_sizes = []
    entropies = []
    for s in steps:
        logits = (1 - s) * PEAKED + s * uniform  # blend peaked -> uniform
        nucleus_sizes.append(nucleus_size(logits, TOP_P))
        entropies.append(entropy_bits(F.softmax(logits, dim=-1)))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(entropies, nucleus_sizes, "-o", color=GREEN, linewidth=2.6, markersize=7,
            label=f"top-p (p={TOP_P}) nucleus size — ADAPTS", zorder=4)
    ax.axhline(TOP_K, color=BLUE, linestyle="--", linewidth=2.4,
               label=f"top-k (k={TOP_K}) — FIXED, ignores shape", zorder=3)
    ax.set_xlabel("distribution entropy (bits) — left = peaked, right = flat")
    ax.set_ylabel("tokens kept")
    ax.set_title("top-p widens as the distribution flattens; top-k is stuck at k")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "dec_nucleus_adapts.png")


def fig_entropy_vs_temperature() -> None:
    """Entropy as a function of temperature for the peaked and flat distributions."""
    temps = np.linspace(0.2, 3.0, 60)
    h_peaked = [entropy_bits(softmax_with_temperature(PEAKED, float(t))) for t in temps]
    h_flat = [entropy_bits(softmax_with_temperature(FLAT, float(t))) for t in temps]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(temps, h_peaked, color=PURPLE, linewidth=2.6, label="peaked distribution", zorder=4)
    ax.plot(temps, h_flat, color=AMBER, linewidth=2.6, label="flat distribution", zorder=4)
    ax.axhline(np.log2(len(VOCAB)), color=SLATE, linestyle=":", linewidth=1.8,
               label=f"max entropy log2({len(VOCAB)}) = {np.log2(len(VOCAB)):.2f} bits", zorder=2)
    # Mark the three temperatures the page/demo quote, with the peaked-dist entropies.
    for t in TEMPERATURES:
        h = entropy_bits(softmax_with_temperature(PEAKED, t))
        ax.plot(t, h, "o", color=PURPLE, markersize=8, zorder=5)
        ax.annotate(f"T={t}\n{h:.2f} bits", (t, h), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color=INK)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("entropy (bits)")
    ax.set_title("Higher temperature = higher entropy = flatter sampling distribution")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "dec_entropy_vs_temperature.png")


def main() -> None:
    fig_temperature_softmax()
    fig_greedy_vs_sampling()
    fig_topk_vs_topp()
    fig_nucleus_adapts()
    fig_entropy_vs_temperature()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

"""Normalization concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/11-Normalization.md:
  1. norm_axes.png   -- the iconic figure: over WHICH axes each norm computes its
     mean/var. A grid of (batch N x channels C), with the normalized region shaded
     for BatchNorm / LayerNorm / InstanceNorm / GroupNorm.
  2. norm_beforeafter.png -- activations before (drifted, wide) vs after (mean 0,
     var 1) normalization, then the learnable gamma/beta re-scale.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def norm_axes():
    N, C = 4, 6                      # batch size, channels
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8))
    panels = [
        ("BatchNorm", "per channel, over the batch", lambda n, c: c == 2, BLUE),
        ("LayerNorm", "per sample, over channels", lambda n, c: n == 1, PURPLE),
        ("InstanceNorm", "per sample & channel", lambda n, c: n == 1 and c == 2, GREEN),
        ("GroupNorm", "per sample, over a channel group", lambda n, c: n == 1 and 1 <= c <= 3, AMBER),
    ]
    for ax, (name, sub, sel, col) in zip(axes, panels):
        for n in range(N):
            for c in range(C):
                on = sel(n, c)
                ax.add_patch(Rectangle((n, C - 1 - c), 0.92, 0.92,
                                       facecolor=col if on else "#e9ecf0",
                                       edgecolor="white", lw=1.2, alpha=0.95 if on else 1))
        ax.set_xlim(-0.2, N + 0.2); ax.set_ylim(-0.2, C + 0.2); ax.set_aspect("equal")
        ax.set_title(name, fontsize=12.5, fontweight="bold")
        ax.text(N / 2, -0.9, sub, ha="center", fontsize=9, color=SLATE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if name == "BatchNorm":
            ax.text(-0.7, C / 2, "channels ↑", rotation=90, va="center", fontsize=8.5, color=SLATE)
            ax.text(N / 2, -1.55, "(columns = batch samples)", ha="center", fontsize=8, color=SLATE, style="italic")
    fig.suptitle("Which axes each norm reduces over (shaded = normalized together)",
                 fontsize=13.5, fontweight="bold", y=1.06)
    fig.tight_layout(); fig.savefig(f"{OUT}/norm_axes.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote norm_axes.png")


def norm_beforeafter():
    rng = np.random.default_rng(0)
    x = rng.normal(4.0, 2.5, 4000)                 # drifted, wide activations
    xn = (x - x.mean()) / x.std()                  # normalized: mean 0, var 1
    xg = 1.4 * xn + 0.5                            # learnable scale/shift gamma=1.4, beta=0.5
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bins = np.linspace(-6, 12, 70)
    ax.hist(x, bins=bins, color=RED, alpha=0.55, label="before: drifted (mean 4, std 2.5)")
    ax.hist(xn, bins=bins, color=GREEN, alpha=0.6, label="after normalize: mean 0, std 1")
    ax.hist(xg, bins=bins, color=PURPLE, alpha=0.55, label="after γ·x̂+β (learnable, γ=1.4, β=0.5)")
    ax.axvline(0, color=SLATE, ls=":", lw=1.2)
    ax.set_xlabel("activation value"); ax.set_ylabel("count")
    ax.set_title("Normalize to mean 0 / var 1 — then let γ, β put back what's useful",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/norm_beforeafter.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote norm_beforeafter.png")


if __name__ == "__main__":
    norm_axes()
    norm_beforeafter()
    print("OUT:", OUT)

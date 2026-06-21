"""Dropout concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/10-Dropout.md:
  1. dropout_subnet.png -- full network (test) vs a sampled thinned sub-network
     (training): ~half the hidden units zeroed, their edges removed. Each step
     trains a different sub-network -> an implicit ensemble.
  2. dropout_scaling.png -- inverted dropout: drop a fraction p, scale survivors
     by 1/(1-p), so the EXPECTED activation is unchanged and test time stays clean.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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


def _draw_net(ax, dropped, title, tcol):
    layers = [3, 5, 5, 2]
    xs = [0, 1.4, 2.8, 4.2]
    pos = {}
    for li, (n, x) in enumerate(zip(layers, xs)):
        ys = np.linspace(0.5, 4.5, n)
        for ni, y in enumerate(ys):
            pos[(li, ni)] = (x, y)
    # edges
    for li in range(len(layers) - 1):
        for a in range(layers[li]):
            for b in range(layers[li + 1]):
                dead = (li, a) in dropped or (li + 1, b) in dropped
                ax.plot([pos[(li, a)][0], pos[(li + 1, b)][0]],
                        [pos[(li, a)][1], pos[(li + 1, b)][1]],
                        color="#dfe3e8" if dead else SLATE, lw=0.6, alpha=0.35 if dead else 0.5, zorder=1)
    # nodes
    for (li, ni), (x, y) in pos.items():
        is_drop = (li, ni) in dropped
        col = "#d7dbe0" if is_drop else (BLUE if li == 0 else GREEN if li == len(layers)-1 else PURPLE)
        ax.add_patch(Circle((x, y), 0.18, facecolor=col, edgecolor="white", lw=1.4, zorder=3))
        if is_drop:
            ax.plot([x-0.13, x+0.13], [y-0.13, y+0.13], color=RED, lw=2, zorder=4)
            ax.plot([x-0.13, x+0.13], [y+0.13, y-0.13], color=RED, lw=2, zorder=4)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=tcol)
    ax.set_xlim(-0.4, 4.6); ax.set_ylim(-0.1, 5.3); ax.axis("off"); ax.set_aspect("equal")


def dropout_subnet():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    _draw_net(ax1, set(), "full network — test time (all units kept)", GREEN)
    dropped = {(1, 0), (1, 3), (2, 1), (2, 4)}        # ~half the hidden units dropped
    _draw_net(ax2, dropped, "training pass — a thinned sub-network (p ≈ 0.5)", RED)
    fig.suptitle("Dropout samples a different sub-network every forward pass (an implicit ensemble)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_subnet.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dropout_subnet.png")


def dropout_scaling():
    rng = np.random.default_rng(3)
    a = np.abs(rng.normal(1.0, 0.3, 8)) + 0.4         # activations
    p = 0.4
    one = a * np.array([1, 0, 1, 1, 0, 1, 0, 1], float) / (1 - p)   # one inverted-dropout pass
    masks = (rng.random((20000, 8)) > p).astype(float)             # average over many masks
    avg = (masks * a / (1 - p)).mean(0)                            # E[inverted dropout] per unit
    idx = np.arange(8)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for ax, (vals, ttl, col) in zip(axes, [
            (a, "original activations", SLATE),
            (one, f"one training pass (drop p={p},\nsurvivors ×1/(1−p))", RED),
            (avg, "average over many passes\n→ matches the original", GREEN)]):
        ax.bar(idx, vals, color=col, alpha=0.85)
        ax.set_title(ttl, fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("unit"); _despine(ax); ax.set_xticks(idx)
    axes[0].set_ylabel("activation")
    fig.suptitle("Inverted dropout: each unit's EXPECTED activation is unchanged, so test time needs no rescaling",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dropout_scaling.png")


if __name__ == "__main__":
    dropout_subnet()
    dropout_scaling()
    print("OUT:", OUT)

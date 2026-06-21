"""Weight-initialization concept-page diagrams (muted palette, parallel scale).

Two figures for 05. Deep_Learning/concepts/05-Weight-Initialization.md:
  1. init_std_depth.png -- activation std through a 20-layer ReLU net for four
     inits: too-small (collapses to 0), too-large (explodes), Xavier (decays a
     bit on ReLU), He (stays ~constant — the right scale for ReLU).
  2. init_hist.png -- activation histogram at a deep layer: too-small piles at 0
     (dead net), He keeps a healthy spread.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _forward(scale_fn, L=20, W=256, mode="relu"):
    """Forward a batch through L ReLU layers; return per-layer activation std (and last acts)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((W, 256))                 # batch of 256
    stds = [a.std()]
    last = a
    for _ in range(L):
        Wl = rng.standard_normal((W, W)) * scale_fn(W)
        z = Wl @ a
        a = np.maximum(0, z) if mode == "relu" else np.tanh(z)
        stds.append(a.std()); last = a
    return np.array(stds), last


def init_std_depth():
    inits = [
        ("too small (×0.01)", lambda n: 0.01, RED),
        ("too large (×1.0)", lambda n: 1.0, AMBER),
        ("Xavier (1/√n)", lambda n: 1/np.sqrt(n), BLUE),
        ("He (√(2/n))", lambda n: np.sqrt(2/n), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    layers = np.arange(0, 21)
    for name, fn, col in inits:
        stds, _ = _forward(fn)
        ax.semilogy(layers, np.clip(stds, 1e-30, None), color=col, lw=2.3, marker="o", ms=3.5, label=name)
    ax.axhspan(0.3, 3, color=GREEN, alpha=0.06)
    ax.set_xlabel("layer depth"); ax.set_ylabel("activation std  (log scale)")
    ax.set_title("Initialization scale decides whether the signal survives depth",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center left"); _despine(ax)
    ax.set_ylim(1e-12, 1e6)
    fig.tight_layout(); fig.savefig(f"{OUT}/init_std_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote init_std_depth.png")


def init_hist():
    _, small = _forward(lambda n: 0.05, L=15)
    _, he = _forward(lambda n: np.sqrt(2/n), L=15)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
    ax1.hist(small.flatten(), bins=60, color=RED, alpha=0.8)
    ax1.set_title("too-small init: activations collapse to ~0\n(dead network, no signal)",
                  fontsize=11, fontweight="bold", color=RED)
    ax1.set_xlabel("activation at layer 15"); ax1.set_ylabel("count"); _despine(ax1)
    ax2.hist(he.flatten(), bins=60, color=GREEN, alpha=0.8)
    ax2.set_title("He init: healthy spread of activations\n(signal preserved)",
                  fontsize=11, fontweight="bold", color=GREEN)
    ax2.set_xlabel("activation at layer 15"); _despine(ax2)
    fig.suptitle("Same 15-layer ReLU net, two initializations", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/init_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote init_hist.png")


if __name__ == "__main__":
    init_std_depth()
    init_hist()
    print("OUT:", OUT)

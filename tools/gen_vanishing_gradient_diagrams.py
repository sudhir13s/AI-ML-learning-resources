"""Vanishing/exploding-gradients concept-page diagrams (muted palette, parallel scale).

Two figures for 05. Deep_Learning/concepts/06-Vanishing-Exploding-Gradients.md:
  1. veg_mechanism.png -- backprop multiplies a factor r per layer: r<1 vanishes,
     r=1 stable, r>1 explodes (clean exponentials on a log axis).
  2. veg_measured.png  -- a REAL 25-layer net: gradient L2-norm per layer for
     sigmoid (vanishes), ReLU+He init (stable), and ReLU+He+residual (flat highway).
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


def veg_mechanism():
    n = np.arange(0, 26)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.semilogy(n, 0.7 ** n, color=BLUE, lw=2.4, marker="o", ms=4, label="r = 0.7  → vanishes (×0.7/layer)")
    ax.semilogy(n, 1.0 ** n, color=GREEN, lw=2.4, marker="s", ms=4, label="r = 1.0  → stable")
    ax.semilogy(n, 1.3 ** n, color=RED, lw=2.4, marker="^", ms=4, label="r = 1.3  → explodes (×1.3/layer)")
    ax.set_xlabel("layers back-propagated through (depth)")
    ax.set_ylabel("gradient magnitude  (log scale)")
    ax.set_title("Backprop multiplies one factor per layer — depth compounds it",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="center left"); _despine(ax)
    ax.axhspan(1e-3, 1e3, color="none")
    fig.tight_layout(); fig.savefig(f"{OUT}/veg_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote veg_mechanism.png")


def _sim(L=25, W=64, mode="sigmoid", seed=0):
    """Forward+backward through L layers; return gradient L2-norm at each layer."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((W, 1))
    Ws, zs, acts = [], [], [x]
    for l in range(L):
        if mode == "sigmoid":
            Wl = rng.standard_normal((W, W)) * (1.0 / np.sqrt(W))   # Xavier-ish
        else:
            # He init for ReLU; residual branches use a smaller scale so the
            # identity path dominates (as in real ResNets) -> a genuinely flat highway
            scale = np.sqrt(2.0 / W) * (0.3 if mode == "residual" else 1.0)
            Wl = rng.standard_normal((W, W)) * scale
        z = Wl @ acts[-1]
        if mode == "sigmoid":
            a = 1 / (1 + np.exp(-z))
        else:
            a = np.maximum(0, z)
            if mode == "residual" and acts[-1].shape == a.shape:
                a = a + acts[-1]                                    # skip connection
        Ws.append(Wl); zs.append(z); acts.append(a)
    # backward from a unit upstream gradient
    g = np.ones((W, 1)); norms = []
    for l in reversed(range(L)):
        if mode == "sigmoid":
            s = 1 / (1 + np.exp(-zs[l])); g = Ws[l].T @ (g * (s * (1 - s)))
        else:
            d = (zs[l] > 0).astype(float)
            main = Ws[l].T @ (g * d)                                # gradient through the weight
            g = main + (g if mode == "residual" else 0)            # skip path BYPASSES the weight
        norms.append(np.linalg.norm(g))
    return np.array(norms[::-1])


def veg_measured():
    L = 25
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    layers = np.arange(1, L + 1)
    ax.semilogy(layers, _sim(L, mode="sigmoid"), color=BLUE, lw=2.3, marker="o", ms=3.5,
                label="sigmoid + Xavier → vanishes")
    ax.semilogy(layers, _sim(L, mode="relu"), color=AMBER, lw=2.3, marker="s", ms=3.5,
                label="ReLU + He init → stable")
    ax.semilogy(layers, _sim(L, mode="residual"), color=GREEN, lw=2.3, marker="^", ms=3.5,
                label="ReLU + He + residual → flat highway")
    ax.set_xlabel("layer (1 = input side, deepest to back-propagate to)")
    ax.set_ylabel("gradient L2-norm reaching this layer  (log scale)")
    ax.set_title("Measured: the fixes that keep the gradient alive through 25 layers",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/veg_measured.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote veg_measured.png")


if __name__ == "__main__":
    veg_mechanism()
    veg_measured()
    print("OUT:", OUT)

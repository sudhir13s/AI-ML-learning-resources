"""Backpropagation concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/02-Backpropagation-and-Computational-Graphs.md:
  1. bp_vanishing.png  -- gradient norm per layer in a deep net: sigmoid gradients
     vanish in early layers; ReLU keeps them alive. The reason activation choice
     and normalization matter for deep backprop.
  2. bp_gradcheck.png  -- analytic backprop gradients vs numerical finite-difference
     gradients fall on the y = x line: how you know your backward pass is correct.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def vanishing():
    torch.manual_seed(0)
    depth, width = 20, 64
    def grad_norms(act):
        layers = []
        for _ in range(depth):
            lin = nn.Linear(width, width)
            nn.init.normal_(lin.weight, std=1.0 / width ** 0.5)  # same neutral init for both
            nn.init.zeros_(lin.bias)
            layers += [lin, act()]
        net = nn.Sequential(*layers)
        x = torch.randn(64, width)
        net(x).pow(2).mean().backward()
        return [lin.weight.grad.norm().item()
                for lin in net if isinstance(lin, nn.Linear)]
    sig = grad_norms(nn.Sigmoid)
    relu = grad_norms(nn.ReLU)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    L = np.arange(1, depth + 1)
    ax.semilogy(L, sig, color=RED, lw=2.5, marker="o", ms=3.5, label="Sigmoid → gradients vanish")
    ax.semilogy(L, relu, color=GREEN, lw=2.5, marker="o", ms=3.5, label="ReLU → gradients survive")
    ax.set_xlabel("layer (1 = closest to input)"); ax.set_ylabel("gradient norm (log scale)")
    ax.set_title("Why depth is hard: gradients shrink as they flow backward", fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right"); _despine(ax)
    ax.annotate("early layers get almost\nno gradient → barely learn", (3, sig[2]),
                textcoords="offset points", xytext=(34, 16), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_vanishing.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_vanishing.png")


def gradcheck():
    rng = np.random.default_rng(0)
    # tiny MLP: x(8) -> W1(8x16) -> relu -> W2(16x1) -> MSE vs target
    n_in, n_h = 8, 16
    W1 = rng.standard_normal((n_in, n_h)) * 0.5
    W2 = rng.standard_normal((n_h, 1)) * 0.5
    x = rng.standard_normal(n_in); y = np.array([1.0])

    def forward(W1, W2):
        h = np.maximum(0, x @ W1)          # relu
        out = h @ W2
        return 0.5 * np.sum((out - y) ** 2), h, out

    loss, h, out = forward(W1, W2)
    # analytic backprop
    dout = (out - y)                       # (1,)
    dW2 = np.outer(h, dout)                # (n_h,1)
    dh = (W2 @ dout) * (h > 0)             # relu grad
    dW1 = np.outer(x, dh)                  # (n_in,n_h)
    analytic = np.concatenate([dW1.ravel(), dW2.ravel()])

    # numerical finite-difference gradient
    eps = 1e-5
    numeric = []
    for W, shape in [(W1, W1.shape), (W2, W2.shape)]:
        flat = W.ravel().copy()
        for i in range(flat.size):
            up = flat.copy(); up[i] += eps
            dn = flat.copy(); dn[i] -= eps
            if shape == W1.shape:
                lu = forward(up.reshape(shape), W2)[0]; ld = forward(dn.reshape(shape), W2)[0]
            else:
                lu = forward(W1, up.reshape(shape))[0]; ld = forward(W1, dn.reshape(shape))[0]
            numeric.append((lu - ld) / (2 * eps))
    numeric = np.array(numeric)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    lo, hi = analytic.min() - 0.2, analytic.max() + 0.2
    ax.plot([lo, hi], [lo, hi], color=SLATE, ls="--", lw=1.6, label="y = x (perfect match)")
    ax.scatter(analytic, numeric, color=PURPLE, s=42, alpha=0.85, edgecolor="white", zorder=5)
    ax.set_xlabel("analytic gradient (backprop)"); ax.set_ylabel("numerical gradient (finite diff)")
    ax.set_title("Gradient check: backprop matches finite differences", fontsize=13, fontweight="bold")
    md = np.max(np.abs(analytic - numeric))
    ax.text(0.04, 0.92, f"max abs diff: {md:.2e}", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=GREEN)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right"); _despine(ax); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_gradcheck.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_gradcheck.png")


def gates():
    """The four local-gradient 'gates': how add / multiply / max / copy route gradient."""
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    up = 2.0  # incoming upstream gradient, shown on every gate

    def draw(ax, title, fwd, back, color):
        ax.add_patch(plt.Circle((0.5, 0.55), 0.16, color=color, zorder=3))
        ax.text(0.5, 0.55, title[0], ha="center", va="center", color="white",
                fontsize=18, fontweight="bold", zorder=4)
        ax.annotate("", (0.34, 0.62), (0.05, 0.78), arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.4))
        ax.annotate("", (0.34, 0.48), (0.05, 0.30), arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.4))
        ax.annotate("", (0.95, 0.55), (0.66, 0.55), arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.2))
        ax.text(0.80, 0.62, f"↑{up:g}", color=AMBER, fontsize=11, fontweight="bold")  # upstream
        ax.text(0.02, 0.83, fwd[0], color=BLUE, fontsize=9.5)
        ax.text(0.02, 0.20, fwd[1], color=BLUE, fontsize=9.5)
        ax.text(0.13, 0.70, back[0], color=GREEN, fontsize=10, fontweight="bold")   # back to in1
        ax.text(0.13, 0.36, back[1], color=GREEN, fontsize=10, fontweight="bold")   # back to in2
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    draw(axes[0], "add: distributor", ["x=3", "y=4"], ["←2", "←2"], PURPLE)
    draw(axes[1], "multiply: swapper", ["x=3", "y=4"], ["←8", "←6"], GREEN)
    draw(axes[2], "max/ReLU: router", ["x=3", "y=4 (max ✓)"], ["←0", "←2"], NAVY)
    draw(axes[3], "copy: adder", ["used twice", " "], ["←2+2", "=4"], AMBER)
    fig.suptitle("Local-gradient gates: how each op routes the upstream gradient (↑2)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_gates.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_gates.png")


if __name__ == "__main__":
    vanishing()
    gradcheck()
    gates()
    print("OUT:", OUT)

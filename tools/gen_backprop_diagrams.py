"""Backpropagation concept-page diagrams (muted palette, parallel matplotlib scale).

Figures for 05. Deep_Learning/concepts/02-Backpropagation-and-Computational-Graphs.md:
  1. bp_vanishing.png   -- gradient norm per layer in a deep net: sigmoid gradients
     vanish in early layers; ReLU keeps them alive.
  2. bp_gradcheck.png   -- analytic backprop gradients vs numerical finite-difference
     gradients fall on the y = x line: how you know your backward pass is correct.
  3. bp_gates.png       -- the four local-gradient "gates" (add/multiply/max/copy).
  4. bp_compgraph.png   -- the tiny 2->2->2 net's computational graph with forward
     VALUES and backward GRADIENTS annotated on every edge (the by-hand worked example).
  5. bp_fwd_vs_rev.png  -- forward-mode vs reverse-mode cost: how many passes each needs
     as #inputs grows with one scalar output -> reverse wins.
  6. bp_delta_recurrence.png -- the delta error pulled back through layers (the
     recurrence delta^l = (W^{l+1}^T delta^{l+1}) (.) sigma'(z^l)), measured magnitudes.
  7. bp_gradcheck_eps.png -- gradient-check relative error vs epsilon: the U-curve
     (too-large eps = truncation error, too-small = floating-point round-off).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    n_in, n_h = 8, 16
    W1 = rng.standard_normal((n_in, n_h)) * 0.5
    W2 = rng.standard_normal((n_h, 1)) * 0.5
    x = rng.standard_normal(n_in); y = np.array([1.0])

    def forward(W1, W2):
        h = np.maximum(0, x @ W1)
        out = h @ W2
        return 0.5 * np.sum((out - y) ** 2), h, out

    loss, h, out = forward(W1, W2)
    dout = (out - y)
    dW2 = np.outer(h, dout)
    dh = (W2 @ dout) * (h > 0)
    dW1 = np.outer(x, dh)
    analytic = np.concatenate([dW1.ravel(), dW2.ravel()])

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
    up = 2.0

    def draw(ax, title, fwd, back, color):
        ax.add_patch(plt.Circle((0.5, 0.55), 0.16, color=color, zorder=3))
        ax.text(0.5, 0.55, title[0], ha="center", va="center", color="white",
                fontsize=18, fontweight="bold", zorder=4)
        ax.annotate("", (0.34, 0.62), (0.05, 0.78), arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.4))
        ax.annotate("", (0.34, 0.48), (0.05, 0.30), arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.4))
        ax.annotate("", (0.95, 0.55), (0.66, 0.55), arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.2))
        ax.text(0.80, 0.62, f"↑{up:g}", color=AMBER, fontsize=11, fontweight="bold")
        ax.text(0.02, 0.83, fwd[0], color=BLUE, fontsize=9.5)
        ax.text(0.02, 0.20, fwd[1], color=BLUE, fontsize=9.5)
        ax.text(0.13, 0.70, back[0], color=GREEN, fontsize=10, fontweight="bold")
        ax.text(0.13, 0.36, back[1], color=GREEN, fontsize=10, fontweight="bold")
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


def compgraph():
    """The tiny 2->2->2 net's computational graph with forward values (blue/navy) and
    backward gradients (green) annotated -- the by-hand worked example."""
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 6); ax.axis("off")

    def box(cx, cy, w, h, color, top, mid, bot=""):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=color, ec="white", lw=1.5, zorder=3))
        ax.text(cx, cy + h * 0.27, top, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.02, mid, ha="center", va="center", color="white",
                fontsize=8.4, zorder=4)
        if bot:
            ax.text(cx, cy - h * 0.30, bot, ha="center", va="center", color="white",
                    fontsize=8.4, zorder=4)

    y = 3.25
    box(1.15, y, 1.5, 1.6, BLUE,   "x", "[1, 2]")
    box(3.5, y, 2.0, 1.8, PURPLE,  "z¹ = xW¹+b¹", "[0.60, 1.30]", "→ tanh")
    box(5.95, y, 1.7, 1.8, NAVY,   "a¹", "[0.537,", "0.862]")
    box(8.35, y, 2.0, 1.8, PURPLE, "z² = a¹W²+b²", "[0.541, 0.512]", "→ softmax")
    box(10.9, y, 1.7, 1.8, GREEN,  "p", "[0.507, 0.493]", "L = 0.679")

    for x0, x1 in [(1.9, 2.5), (4.5, 5.1), (6.8, 7.35), (9.35, 10.05)]:
        ax.add_patch(FancyArrowPatch((x0, y + 0.3), (x1, y + 0.3),
                     arrowstyle="-|>", mutation_scale=14, color=SLATE, lw=2, zorder=2))

    deltas = [("δ²=p−y\n[−0.493, 0.493]", 10.05, 9.35),
              ("∂L/∂a¹\n[−0.197, 0.049]", 7.35, 6.8),
              ("δ¹=∂L/∂z¹\n[−0.140, 0.013]", 5.1, 4.5),
              ("∂L/∂x", 2.5, 1.9)]
    for lbl, x0, x1 in deltas:
        ax.add_patch(FancyArrowPatch((x0, y - 0.6), (x1, y - 0.6),
                     arrowstyle="-|>", mutation_scale=13, color=GREEN, lw=2,
                     linestyle=(0, (4, 2)), zorder=2))
        ax.text((x0 + x1) / 2, y - 1.22, lbl, ha="center", va="center",
                color=GREEN, fontsize=8.0, fontweight="bold")

    ax.text(6.0, 5.6, "FORWARD  →   compute values, cache z¹, a¹",
            ha="center", color=SLATE, fontsize=10.5, fontweight="bold")
    ax.text(6.0, 0.95, "←  BACKWARD   pull δ back: matmul ×Wᵀ, then ⊙ σ′",
            ha="center", color=GREEN, fontsize=10.5, fontweight="bold")
    ax.text(6.0, 0.3, "weight grads:  ∂L/∂W² = a¹ᵀδ²,   ∂L/∂W¹ = xᵀδ¹   (outer products)",
            ha="center", color=AMBER, fontsize=9.5, fontweight="bold")

    fig.suptitle("Computational graph of the 2→2→2 net: forward values, backward gradients",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_compgraph.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_compgraph.png")


def fwd_vs_rev():
    """Number of graph passes forward-mode vs reverse-mode needs to get ALL gradients of a
    scalar loss, as #inputs (parameters) grows. Reverse = 1 always; forward = #inputs."""
    n_inputs = np.array([1, 2, 5, 10, 50, 100, 500, 1000])
    forward_passes = n_inputs.astype(float)
    reverse_passes = np.ones_like(forward_passes)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.loglog(n_inputs, forward_passes, color=RED, lw=2.5, marker="o", ms=5,
              label="forward mode: 1 pass per input → ∝ #inputs")
    ax.loglog(n_inputs, reverse_passes, color=GREEN, lw=2.5, marker="s", ms=5,
              label="reverse mode (backprop): 1 pass total")
    ax.set_xlabel("number of inputs / parameters (one scalar loss output)")
    ax.set_ylabel("passes needed for ALL gradients")
    ax.set_title("Why reverse mode wins for deep learning (many inputs, one loss)",
                 fontsize=13, fontweight="bold")
    ax.annotate("a real net has millions of\nparameters → forward mode is hopeless",
                (500, 500), textcoords="offset points", xytext=(-220, 22),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax)
    ax.set_ylim(0.6, 3000)
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_fwd_vs_rev.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_fwd_vs_rev.png")


def delta_recurrence():
    """Measured |delta^l| per layer for tanh vs ReLU, illustrating the recurrence
    delta^l = (W^{l+1}^T delta^{l+1}) (.) sigma'(z^l) growing or shrinking with depth."""
    depth, width = 12, 48

    def delta_mags(act):
        torch.manual_seed(1)
        h = torch.randn(1, width)
        zs = []
        for _ in range(depth):
            W = torch.randn(width, width) / width ** 0.5
            W.requires_grad_(True)
            z = h @ W; z.retain_grad(); zs.append(z)
            h = act(z)
        h.pow(2).sum().backward()
        return [z.grad.norm().item() for z in zs]

    tanh = delta_mags(torch.tanh)
    relu = delta_mags(torch.relu)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    L = np.arange(1, depth + 1)
    ax.semilogy(L, tanh, color=PURPLE, lw=2.5, marker="o", ms=4, label="tanh: |δˡ| decays backward")
    ax.semilogy(L, relu, color=GREEN, lw=2.5, marker="s", ms=4, label="ReLU: |δˡ| stays stable")
    ax.set_xlabel("layer l  (12 = output side, 1 = input side)")
    ax.set_ylabel("|δˡ| = ‖∂L/∂zˡ‖  (log scale)")
    ax.set_title("The δ recurrence: error pulled back layer by layer (×Wᵀ, ⊙σ′)",
                 fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    ax.set_ylim(0.06, max(tanh) * 3)
    ax.annotate("each step: ×Wᵀ then ⊙σ′(z)\n→ magnitude can shrink or grow",
                (4, tanh[8]), textcoords="offset points", xytext=(-30, -54),
                fontsize=9, color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.legend(frameon=False, fontsize=9.5, loc="lower left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_delta_recurrence.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_delta_recurrence.png")


def gradcheck_eps():
    """The grad-check U-curve: relative error of the centered finite difference vs eps.
    Too-large eps -> truncation O(eps^2); too-small -> floating-point round-off."""
    def f(w): return 0.5 * (np.tanh(3 * w) - 0.5) ** 2
    w = 0.7
    analytic = (np.tanh(3 * w) - 0.5) * (1 - np.tanh(3 * w) ** 2) * 3
    eps = np.logspace(-1, -13, 40)
    relerr = []
    for e in eps:
        num = (f(w + e) - f(w - e)) / (2 * e)
        relerr.append(abs(analytic - num) / (abs(analytic) + abs(num) + 1e-30))
    relerr = np.array(relerr)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.loglog(eps, relerr, color=BLUE, lw=2.5, marker="o", ms=3.5)
    imin = int(np.argmin(relerr))
    ax.scatter([eps[imin]], [relerr[imin]], color=GREEN, s=90, zorder=6,
               edgecolor="white", label=f"sweet spot ≈ {eps[imin]:.0e}  (rel err {relerr[imin]:.0e})")
    ax.set_xlabel("ε (finite-difference step)"); ax.set_ylabel("relative error vs analytic gradient")
    ax.set_title("Gradient check: the ε U-curve (truncation vs round-off)",
                 fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    ax.annotate("large ε:\ntruncation error O(ε²)", (eps[2], relerr[2]),
                textcoords="offset points", xytext=(30, -6), fontsize=9, color=RED, fontweight="bold")
    ax.annotate("tiny ε:\nround-off error", (eps[-3], relerr[-3]),
                textcoords="offset points", xytext=(-20, 12), fontsize=9, color=AMBER, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bp_gradcheck_eps.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bp_gradcheck_eps.png")


if __name__ == "__main__":
    vanishing()
    gradcheck()
    gates()
    compgraph()
    fwd_vs_rev()
    delta_recurrence()
    gradcheck_eps()
    print("OUT:", OUT)

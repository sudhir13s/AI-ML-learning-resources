"""Weight-initialization concept-page diagrams (muted palette, parallel scale).

Four MEASURED figures for 05. Deep_Learning/concepts/05-Weight-Initialization.md.
All numbers come from real forward/backward passes through deep nets — nothing
is hand-drawn.

  1. init_std_depth.png   -- activation std through a 30-layer ReLU net for four
     inits: naive-small (collapses to ~0), naive-large (explodes), Xavier
     (decays on ReLU — missing the ×2), He (stays ~1, the right scale for ReLU).
  2. init_grad_std_depth.png -- gradient std across the same depths: the SAME
     scale problem hits the backward pass — small init vanishes the gradient,
     large explodes it, He keeps it flat.
  3. init_hist.png        -- activation histogram at a deep layer for THREE
     regimes: saturated (tanh, too-large → piled at ±1), dead (ReLU, too-small →
     piled at 0), healthy (He ReLU → ~half-normal, signal preserved).
  4. init_dead_relu.png   -- measured dead-ReLU fraction per layer (bad vs He)
     AND a measured loss-vs-epoch curve on a small task (naive vs Xavier vs He),
     showing bad init simply fails to train.

Run via:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_weight_init_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Shared measured forward/backward pass through an L-layer net.
# --------------------------------------------------------------------------- #
def _forward_relu(scale_fn, L=30, W=512, batch=512, seed=0):
    """Return per-layer activation std and the final-layer activations (measured)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((W, batch)).astype(np.float64)  # input std = 1
    stds = [a.std()]
    last = a
    for _ in range(L):
        Wl = rng.standard_normal((W, W)) * scale_fn(W)
        a = np.maximum(0.0, Wl @ a)
        stds.append(a.std())
        last = a
    return np.array(stds), last


def _grad_std_relu(scale_fn, L=30, W=512, batch=256, seed=1):
    """Backprop a unit loss through L ReLU layers; return per-layer gradient std."""
    rng = np.random.default_rng(seed)
    Ws, masks, acts = [], [], []
    a = rng.standard_normal((W, batch))
    acts.append(a)
    for _ in range(L):
        Wl = rng.standard_normal((W, W)) * scale_fn(W)
        z = Wl @ a
        m = (z > 0).astype(np.float64)
        a = z * m
        Ws.append(Wl); masks.append(m); acts.append(a)
    # backward: seed with a realistic unit-variance upstream gradient
    # (a constant ones-vector has std 0 and would create a spurious endpoint dip)
    g = rng.standard_normal(acts[-1].shape)
    grad_stds = [g.std()]
    for l in range(L - 1, -1, -1):
        g = g * masks[l]                # through ReLU
        g = Ws[l].T @ g                 # through the linear layer
        grad_stds.append(g.std())
    return np.array(grad_stds[::-1])    # index 0 = input side


# --------------------------------------------------------------------------- #
# Figure 1 — activation std across depth.
# --------------------------------------------------------------------------- #
def init_std_depth():
    inits = [
        ("naive small (×0.02)", lambda n: 0.02, RED),
        ("naive large (×0.20)", lambda n: 0.20, AMBER),
        ("Xavier (1/√n)", lambda n: 1 / np.sqrt(n), BLUE),
        ("He (√(2/n))", lambda n: np.sqrt(2 / n), GREEN),
    ]
    L = 30
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    layers = np.arange(0, L + 1)
    for name, fn, col in inits:
        stds, _ = _forward_relu(fn, L=L)
        ax.semilogy(layers, np.clip(stds, 1e-30, None), color=col, lw=2.3,
                    marker="o", ms=3.2, label=name)
    ax.axhspan(0.3, 3, color=GREEN, alpha=0.06)
    ax.text(0.5, 1.0, "healthy band", color=GREEN, fontsize=9, va="center")
    ax.set_xlabel("layer depth")
    ax.set_ylabel("activation std  (log scale)")
    ax.set_title("Initialization scale decides whether the signal survives depth",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center left")
    _despine(ax)
    ax.set_ylim(1e-18, 1e10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/init_std_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote init_std_depth.png")


# --------------------------------------------------------------------------- #
# Figure 2 — gradient std across depth (the backward pass).
# --------------------------------------------------------------------------- #
def init_grad_std_depth():
    inits = [
        ("naive small (×0.02)", lambda n: 0.02, RED),
        ("naive large (×0.20)", lambda n: 0.20, AMBER),
        ("Xavier (1/√n)", lambda n: 1 / np.sqrt(n), BLUE),
        ("He (√(2/n))", lambda n: np.sqrt(2 / n), GREEN),
    ]
    L = 30
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    layers = np.arange(0, L + 1)
    for name, fn, col in inits:
        gstds = _grad_std_relu(fn, L=L)
        ax.semilogy(layers, np.clip(gstds, 1e-30, None), color=col, lw=2.3,
                    marker="s", ms=3.2, label=name)
    ax.axhspan(0.3, 3, color=GREEN, alpha=0.06)
    ax.set_xlabel("layer depth  (0 = input side, 30 = loss side)")
    ax.set_ylabel("gradient std  (log scale)")
    ax.set_title("The same scale problem hits the BACKWARD pass",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center left")
    _despine(ax)
    ax.set_ylim(1e-18, 1e10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/init_grad_std_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote init_grad_std_depth.png")


# --------------------------------------------------------------------------- #
# Figure 3 — activation histograms: saturated vs dead vs healthy.
# --------------------------------------------------------------------------- #
def _forward_tanh(scale_fn, L=15, W=512, batch=512, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((W, batch))
    for _ in range(L):
        Wl = rng.standard_normal((W, W)) * scale_fn(W)
        a = np.tanh(Wl @ a)
    return a


def init_hist():
    # saturated: tanh with too-large init -> pile at +-1
    sat = _forward_tanh(lambda n: 0.30, L=15)
    # dead: ReLU with too-small init -> collapse to ~0
    _, dead = _forward_relu(lambda n: 0.04, L=15)
    # healthy: ReLU with He -> half-normal-ish spread, std ~1
    _, healthy = _forward_relu(lambda n: np.sqrt(2 / 512), L=15)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax1.hist(sat.flatten(), bins=60, color=AMBER, alpha=0.85)
    ax1.set_title("saturated (tanh, too large)\nmass piled at ±1 — gradient dies",
                  fontsize=10.5, fontweight="bold", color=AMBER)
    ax1.set_xlabel("activation at layer 15"); ax1.set_ylabel("count"); _despine(ax1)

    ax2.hist(dead.flatten(), bins=60, color=RED, alpha=0.85)
    ax2.set_title("dead (ReLU, too small)\nmass piled at 0 — no signal",
                  fontsize=10.5, fontweight="bold", color=RED)
    ax2.set_xlabel("activation at layer 15"); _despine(ax2)

    ax3.hist(healthy.flatten(), bins=60, color=GREEN, alpha=0.85)
    ax3.set_title("healthy (He ReLU)\nstd ≈ 1, half-normal spread",
                  fontsize=10.5, fontweight="bold", color=GREEN)
    ax3.set_xlabel("activation at layer 15"); _despine(ax3)

    fig.suptitle("Same depth, three initialization regimes — the histogram tells the story",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUT}/init_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote init_hist.png")


# --------------------------------------------------------------------------- #
# Figure 4 — dead-ReLU fraction per layer + measured loss-vs-epoch (torch).
# --------------------------------------------------------------------------- #
def _dead_fraction(scale_fn, bias=0.0, L=20, W=512, batch=512, seed=3):
    """Fraction of units that are zero for the WHOLE batch (permanently dead) per layer.
    A negative bias offset is the classic real-world cause of dead ReLUs; we contrast
    it against zero-bias He init."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((W, batch))
    fracs = []
    for _ in range(L):
        Wl = rng.standard_normal((W, W)) * scale_fn(W)
        z = Wl @ a + bias
        a = np.maximum(0.0, z)
        # a unit is "dead at this layer" if it is 0 for the whole batch
        fracs.append(float((a.sum(axis=1) == 0).mean()))
    return np.array(fracs)


class _Net(nn.Module):
    def __init__(self, depth=8, width=128, init="he"):
        super().__init__()
        layers = []
        d = 20
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            d = width
        self.hidden = nn.ModuleList(layers)
        self.head = nn.Linear(width, 1)
        self._init(init)

    def _init(self, init):
        for lin in list(self.hidden) + [self.head]:
            if init == "naive_small":
                nn.init.normal_(lin.weight, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(lin.weight)
            elif init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        for lin in self.hidden:
            x = F.relu(lin(x))
        return self.head(x)


def _train_curve(init, epochs=60, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(7)
    X = torch.randn(1024, 20, generator=g)
    true_w = torch.randn(20, 1, generator=g)
    y = torch.tanh(X @ true_w) + 0.1 * torch.randn(1024, 1, generator=g)
    net = _Net(depth=8, width=128, init=init)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.array(losses)


def init_dead_relu():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # left: dead-ReLU fraction per layer — bad (negative-bias) vs He(zero bias)
    L = 20
    he = lambda n: np.sqrt(2 / n)
    layers = np.arange(1, L + 1)
    for name, bias, col in [
        ("bad init (He weights, bias = −2)", -2.0, RED),
        ("bad init (He weights, bias = −1)", -1.0, AMBER),
        ("He init (bias = 0)", 0.0, GREEN),
    ]:
        fr = _dead_fraction(he, bias=bias, L=L)
        ax1.plot(layers, fr * 100, color=col, lw=2.3, marker="o", ms=3.2, label=name)
    ax1.set_xlabel("layer depth")
    ax1.set_ylabel("dead units  (% zero for whole batch)")
    ax1.set_title("A bad (negative-bias) init kills ReLU units; He keeps them alive",
                  fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_ylim(-3, 103)
    _despine(ax1)

    # right: measured loss-vs-epoch (torch)
    for name, init, col in [
        ("naive small (std 0.01)", "naive_small", RED),
        ("Xavier", "xavier", BLUE),
        ("He / Kaiming", "he", GREEN),
    ]:
        losses = _train_curve(init)
        ax2.plot(np.arange(len(losses)), losses, color=col, lw=2.3, label=name)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training MSE loss")
    ax2.set_title("Same 8-layer net trains or stalls by init alone",
                  fontsize=11.5, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9.5, loc="upper right")
    _despine(ax2)

    fig.suptitle("Init is not cosmetic: it decides whether the network learns at all",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/init_dead_relu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote init_dead_relu.png")


if __name__ == "__main__":
    init_std_depth()
    init_grad_std_depth()
    init_hist()
    init_dead_relu()
    print("OUT:", OUT)

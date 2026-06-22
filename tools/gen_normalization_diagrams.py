"""Normalization concept-page diagrams (muted palette, parallel matplotlib scale).

Figures for 05. Deep_Learning/concepts/11-Normalization.md:
  1. norm_axes.png   -- the iconic figure: over WHICH axes each norm computes its
     mean/var. A grid of (batch N x channels C), with the normalized region shaded
     for BatchNorm / LayerNorm / InstanceNorm / GroupNorm.
  2. norm_beforeafter.png -- activations before (drifted, wide) vs after (mean 0,
     var 1) normalization, then the learnable gamma/beta re-scale.
  3. norm_training_curve.png -- MEASURED: deep MLP trained with LayerNorm vs without,
     at a learning rate the un-normalized net can't tolerate. Normalized = faster,
     lower, stable; plain net stalls/diverges.
  4. norm_prepost_gradient.png -- MEASURED: gradient L2-norm by layer depth at init,
     pre-norm vs post-norm transformer-style blocks. Post-norm grows toward output
     (needs warmup); pre-norm stays flat.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn

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


def norm_training_curve():
    """MEASURED: train a deep MLP with vs without LayerNorm at an aggressive LR."""
    torch.manual_seed(0)
    np.random.seed(0)
    n, d_in, d_h, depth = 512, 20, 64, 12
    X = torch.randn(n, d_in)
    true_w = torch.randn(d_in, 1)
    y = (X @ true_w + 0.1 * torch.randn(n, 1))           # regression target

    def make_mlp(use_norm):
        layers = [nn.Linear(d_in, d_h), nn.ReLU()]
        for _ in range(depth):
            layers.append(nn.Linear(d_h, d_h))
            if use_norm:
                layers.append(nn.LayerNorm(d_h))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_h, 1))
        return nn.Sequential(*layers)

    def train(use_norm, lr, steps=300):
        torch.manual_seed(1)
        net = make_mlp(use_norm)
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        lossfn = nn.MSELoss()
        hist = []
        for _ in range(steps):
            opt.zero_grad()
            loss = lossfn(net(X), y)
            loss.backward()
            opt.step()
            hist.append(min(loss.item(), 1e3) if torch.isfinite(loss) else 1e3)
        return hist

    lr = 0.03                                            # plain net stalls; norm net converges fast
    h_norm = train(True, lr)
    h_plain = train(False, lr)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(h_plain, color=RED, lw=2.2, label=f"no normalization (LR={lr})")
    ax.plot(h_norm, color=GREEN, lw=2.2, label=f"LayerNorm in every block (LR={lr})")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Normalization stabilizes a deep net at a high learning rate (measured)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10); _despine(ax)
    ax.text(0.98, 0.5, f"final loss\nplain  {h_plain[-1]:.2f}\nnorm   {h_norm[-1]:.3f}",
            transform=ax.transAxes, ha="right", va="center", fontsize=9.5, color=SLATE,
            bbox=dict(boxstyle="round", fc="#f2f4f7", ec=SLATE, alpha=0.9))
    fig.tight_layout(); fig.savefig(f"{OUT}/norm_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote norm_training_curve.png  (plain final {h_plain[-1]:.2f}, norm final {h_norm[-1]:.3f})")


def norm_prepost_gradient():
    """MEASURED: gradient L2-norm per block at init, pre-norm vs post-norm."""
    torch.manual_seed(0)
    n, d, depth = 64, 128, 30

    class Block(nn.Module):
        def __init__(self, d, pre):
            super().__init__()
            self.pre = pre
            self.norm = nn.LayerNorm(d)
            self.lin1 = nn.Linear(d, 4 * d); self.lin2 = nn.Linear(4 * d, d)
            self.act = nn.GELU()

        def sub(self, x):
            return self.lin2(self.act(self.lin1(x)))

        def forward(self, x):
            if self.pre:                                 # pre-norm: norm inside the branch
                return x + self.sub(self.norm(x))
            return self.norm(x + self.sub(x))            # post-norm: norm on the stream

    def grad_by_depth(pre):
        torch.manual_seed(2)
        blocks = nn.ModuleList([Block(d, pre) for _ in range(depth)])
        # a small output head + target so the loss is a realistic supervised scalar
        head = nn.Linear(d, d)
        x = torch.randn(n, d)
        target = torch.randn(n, d)
        h = x
        for b in blocks:
            h = b(h)
        loss = (head(h) - target).pow(2).mean()
        loss.backward()
        # gradient magnitude flowing into each block's first linear layer
        return [b.lin1.weight.grad.norm().item() for b in blocks]

    g_pre = grad_by_depth(True)
    g_post = grad_by_depth(False)

    pre_ratio = g_pre[-1] / g_pre[0]
    post_ratio = g_post[-1] / g_post[0]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    xs = np.arange(1, depth + 1)
    ax.plot(xs, g_post, "o-", color=RED, lw=2.0, ms=4,
            label=f"post-norm (grows toward output; out/in = {post_ratio:.2f}×)")
    ax.plot(xs, g_pre, "s-", color=GREEN, lw=2.0, ms=4,
            label=f"pre-norm (flat / slightly decreasing; out/in = {pre_ratio:.2f}×)")
    ax.set_xlabel(f"block depth (1 = input side, {depth} = output side)")
    ax.set_ylabel("gradient L2-norm into block")
    ax.set_title("Post-norm gradients rise toward the output; pre-norm stays flat (measured, at init)",
                 fontsize=12.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/norm_prepost_gradient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote norm_prepost_gradient.png  (pre range {min(g_pre):.2e}-{max(g_pre):.2e}, "
          f"post range {min(g_post):.2e}-{max(g_post):.2e})")


if __name__ == "__main__":
    norm_axes()
    norm_beforeafter()
    norm_training_curve()
    norm_prepost_gradient()
    print("OUT:", OUT)

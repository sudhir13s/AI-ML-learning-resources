"""Residual/skip-connection concept-page diagrams (muted palette, parallel scale).

Two figures for 05. Deep_Learning/concepts/18-Residual-Skip-Connections.md, both
from REAL torch runs (no fabricated curves):
  1. res_gradflow.png  -- gradient L2-norm per layer through a 30-layer net:
     plain (vanishes toward early layers) vs residual (flat highway).
  2. res_degradation.png -- training loss of a deep PLAIN net (stalls high — the
     degradation problem) vs the same depth WITH residuals (trains down).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
torch.manual_seed(0)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


class DeepNet(nn.Module):
    def __init__(self, depth=30, width=64, residual=False):
        super().__init__()
        self.residual = residual
        self.inp = nn.Linear(1, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, 1)
    def forward(self, x):
        h = torch.tanh(self.inp(x)); self.acts = []
        for blk in self.blocks:
            f = torch.tanh(blk(h))
            h = h + f if self.residual else f          # the skip connection
            self.acts.append(h)
        return self.out(h)


def res_gradflow():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = torch.randn(64, 1); y = torch.sin(x)
    for residual, col, lab in [(False, BLUE, "plain net → gradient vanishes"),
                               (True, GREEN, "residual net → flat highway")]:
        net = DeepNet(depth=30, residual=residual)
        loss = ((net(x) - y) ** 2).mean(); net.zero_grad(); loss.backward()
        norms = [blk.weight.grad.norm().item() for blk in net.blocks]
        ax.semilogy(range(1, 31), norms, color=col, lw=2.3, marker="o", ms=3.5, label=lab)
    ax.set_xlabel("block (1 = input side)"); ax.set_ylabel("gradient L2-norm  (log scale)")
    ax.set_title("Residual connections keep the gradient alive through 30 layers",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/res_gradflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote res_gradflow.png")


def _train(residual, steps=400):
    torch.manual_seed(0)
    net = DeepNet(depth=30, residual=residual)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    x = torch.linspace(-3, 3, 128).unsqueeze(1); y = torch.sin(x)
    losses = []
    for _ in range(steps):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def res_degradation():
    plain = _train(False); res = _train(True)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.semilogy(plain, color=RED, lw=2.3, label="30-layer PLAIN net → stalls high (degradation)")
    ax.semilogy(res, color=GREEN, lw=2.3, label="30-layer net WITH residuals → trains down")
    ax.set_xlabel("training step"); ax.set_ylabel("training loss  (log scale)")
    ax.set_title("The degradation problem: a deep plain net won't even fit the training data",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/res_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote res_degradation.png")


if __name__ == "__main__":
    res_gradflow()
    res_degradation()
    print("OUT:", OUT)

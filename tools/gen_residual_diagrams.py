"""Residual/skip-connection concept-page diagrams (muted palette, parallel scale).

Four figures for 05. Deep_Learning/concepts/18-Residual-Skip-Connections.md.

MEASURED (real torch runs, no fabricated curves):
  1. res_degradation.png -- training loss of a deep PLAIN net (stalls high — the
     degradation problem) vs the same depth WITH residuals (trains down).
  2. res_gradflow.png  -- gradient L2-norm per layer through a 30-layer net:
     plain (vanishes toward early layers) vs residual (flat highway).

SCHEMATIC (hand-drawn with matplotlib, palette-only):
  3. res_block.png     -- the residual block x -> F(x) -> (+x) -> y, identity
     shortcut highlighted as the gradient highway.
  4. res_stream.png    -- the residual stream in a transformer: every sublayer
     READS from and ADDS back to a running stream (the model's communication bus).

Palette (muted, color:#fff text where labels sit on fills):
  BLUE input/data · PURPLE process · GREEN output/success · RED danger ·
  SLATE frozen/static · AMBER highlight · NAVY alt.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


# --------------------------------------------------------------------------- #
# Shared model
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 1. Gradient flow (measured)
# --------------------------------------------------------------------------- #
def res_gradflow():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = torch.randn(64, 1); y = torch.sin(x)
    summary = {}
    for residual, col, lab in [(False, RED, "plain net → gradient vanishes"),
                               (True, GREEN, "residual net → flat highway")]:
        net = DeepNet(depth=30, residual=residual)
        loss = ((net(x) - y) ** 2).mean(); net.zero_grad(); loss.backward()
        norms = [blk.weight.grad.norm().item() for blk in net.blocks]
        summary[residual] = (norms[0], norms[-1])
        ax.semilogy(range(1, 31), norms, color=col, lw=2.3, marker="o", ms=3.5, label=lab)
    ax.set_xlabel("block index  (1 = input side / earliest layer)")
    ax.set_ylabel("gradient L2-norm  (log scale)")
    ax.set_title("Residual connections keep the gradient alive through 30 layers",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right"); _despine(ax)
    ax.annotate("plain: ~1e-7 at block 1\n(vanished — can't learn)",
                xy=(1, summary[False][0]), xytext=(6, summary[False][0] * 60),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    fig.tight_layout(); fig.savefig(f"{OUT}/res_gradflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote res_gradflow.png | plain block1={summary[False][0]:.2e} "
          f"res block1={summary[True][0]:.2e}")


# --------------------------------------------------------------------------- #
# 2. Degradation (measured)
# --------------------------------------------------------------------------- #
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
    plt.close(fig)
    print(f"wrote res_degradation.png | plain final={plain[-1]:.3e} res final={res[-1]:.3e}")


# --------------------------------------------------------------------------- #
# helpers for schematic figures
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fill, fc="#fff", fs=11, weight="normal", rad=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={rad}",
                                linewidth=0, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=fc, fontsize=fs, fontweight=weight, zorder=3)


def _arrow(ax, p0, p1, color="#3a3a3a", lw=2.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# --------------------------------------------------------------------------- #
# 3. The residual block (schematic)
# --------------------------------------------------------------------------- #
def res_block():
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 10); ax.axis("off")

    cx = 4.6  # centre of the main path
    # main (residual) path up the centre
    _box(ax, cx - 1.4, 0.55, 2.8, 0.95, "input  x", BLUE, fs=12, weight="bold")
    _box(ax, cx - 1.6, 2.45, 3.2, 0.9, "weight layer  (conv / linear)", PURPLE, fs=10)
    _box(ax, cx - 1.6, 3.85, 3.2, 0.9, "BN → ReLU", PURPLE, fs=10)
    _box(ax, cx - 1.6, 5.25, 3.2, 0.9, "weight layer", PURPLE, fs=10)
    # add node
    ax.add_patch(plt.Circle((cx, 7.4), 0.42, color=AMBER, zorder=3))
    ax.text(cx, 7.4, "+", ha="center", va="center", color="#fff", fontsize=20, fontweight="bold", zorder=4)
    _box(ax, cx - 1.9, 8.55, 3.8, 0.9, "ReLU  →  y = F(x) + x", GREEN, fs=11, weight="bold")

    # arrows on the main path
    _arrow(ax, (cx, 1.5), (cx, 2.45))
    _arrow(ax, (cx, 3.35), (cx, 3.85))
    _arrow(ax, (cx, 4.75), (cx, 5.25))
    _arrow(ax, (cx, 6.15), (cx, 6.98), color=PURPLE, lw=2.4)
    ax.text(cx + 0.25, 6.55, "F(x)", color=PURPLE, fontsize=11, fontweight="bold", va="center")
    _arrow(ax, (cx, 7.82), (cx, 8.55))

    # identity shortcut: x up the right side, highlighted
    rx = 8.7
    _arrow(ax, (cx + 1.4, 1.02), (rx, 1.02), color=NAVY, lw=2.8)      # x -> right
    _arrow(ax, (rx, 1.02), (rx, 7.4), color=NAVY, lw=2.8)            # up
    _arrow(ax, (rx, 7.4), (cx + 0.42, 7.4), color=NAVY, lw=2.8)      # into +
    ax.text(rx + 0.35, 4.3, "identity shortcut\ncarries x unchanged\n(the gradient highway)",
            color=NAVY, fontsize=9.5, ha="center", va="center", rotation=90, fontweight="bold")

    ax.set_title("The residual block:  learn the correction F(x), then add the input back",
                 fontsize=12.5, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/res_block.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote res_block.png")


# --------------------------------------------------------------------------- #
# 4. The residual stream in a transformer (schematic)
# --------------------------------------------------------------------------- #
def res_stream():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 10); ax.axis("off")

    stream_x = 2.4
    # the running stream: a thick vertical bus
    ax.add_patch(FancyBboxPatch((stream_x - 0.45, 0.5), 0.9, 8.6,
                                boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=SLATE, linewidth=0, zorder=1))
    ax.text(stream_x, 0.15, "embeddings", ha="center", va="top", fontsize=9.5, color=SLATE, fontweight="bold")
    ax.text(stream_x - 1.15, 5.0, "residual stream\n(running sum / working memory)",
            ha="center", va="center", fontsize=10, color=SLATE, fontweight="bold", rotation=90)
    ax.text(stream_x, 9.35, "logits", ha="center", va="bottom", fontsize=9.5, color=SLATE, fontweight="bold")

    sublayers = [
        (1.7, "Attention(Norm(x))", PURPLE),
        (3.5, "FFN(Norm(x))", BLUE),
        (5.3, "Attention(Norm(x))", PURPLE),
        (7.1, "FFN(Norm(x))", BLUE),
    ]
    for y, label, col in sublayers:
        # READ branch off the stream
        _arrow(ax, (stream_x + 0.45, y), (6.4, y), color=col, lw=1.8, rad=0.0)
        _box(ax, 6.4, y - 0.42, 3.7, 0.84, label, col, fs=10)
        # write (add) back into the stream
        _arrow(ax, (6.4, y + 0.30), (stream_x + 0.45, y + 0.7), color=col, lw=1.8, rad=-0.25, style="-|>")
        # the + node on the stream
        ax.add_patch(plt.Circle((stream_x, y + 0.7), 0.26, color=AMBER, zorder=4))
        ax.text(stream_x, y + 0.7, "+", ha="center", va="center", color="#fff",
                fontsize=13, fontweight="bold", zorder=5)

    ax.text(10.3, 4.4, "each sublayer READS\nthe stream, computes a\ndelta, and ADDS it back:\nx ← x + Sublayer(Norm(x))",
            ha="left", va="center", fontsize=9.5, color="#3a3a3a")

    ax.set_title("The residual stream: a transformer's communication bus",
                 fontsize=12.5, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/res_stream.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote res_stream.png")


if __name__ == "__main__":
    res_degradation()
    res_gradflow()
    res_block()
    res_stream()
    print("OUT:", OUT)

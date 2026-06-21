"""RNN/LSTM concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/14-RNN-LSTM-GRU.md:
  1. rnn_gradient_time.png -- REAL torch measurement: gradient of the last step's
     loss w.r.t. the input at lag k. Plain RNN vanishes with the time-lag; LSTM
     keeps it alive (the long-range-dependency fix).
  2. lstm_cell.png -- schematic of one LSTM cell: the cell-state line across the
     top, the forget/input/output gates, and the additive cell update.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
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


def rnn_gradient_time():
    """For each cell type, measure |d loss_T / d x_t| as a function of lag T-t."""
    T, H = 40, 32
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    lstm = nn.LSTMCell(1, H)
    with torch.no_grad():                              # forget-gate bias high (the standard trick)
        lstm.bias_ih[H:2*H].fill_(3.0); lstm.bias_hh[H:2*H].fill_(3.0)  # f ≈ sigmoid(6) ≈ 1 → cell holds
    for name, cell, col in [("plain RNN", nn.RNNCell(1, H), BLUE),
                            ("LSTM (forget-bias init)", lstm, GREEN)]:
        xs = [torch.zeros(1, 1, requires_grad=True) for _ in range(T)]
        h = torch.zeros(1, H); c = torch.zeros(1, H)
        for t in range(T):
            if name.startswith("LSTM"):
                h, c = cell(xs[t], (h, c))
            else:
                h = cell(xs[t], h)
        h.sum().backward()
        grads = [x.grad.abs().sum().item() + 1e-30 for x in xs]   # sensitivity to each input
        lags = list(range(T - 1, -1, -1))                          # lag = T - t
        ax.semilogy(lags, grads, color=col, lw=2.3, marker="o", ms=3, label=name)
    ax.set_xlabel("time-lag between input and output (steps back)")
    ax.set_ylabel("|∂ output / ∂ input|  (log scale)")
    ax.set_title("Why plain RNNs forget: gradient vanishes with the time-lag; LSTM holds it",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/rnn_gradient_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rnn_gradient_time.png")


def lstm_cell():
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    # cell-state highway across the top
    ax.annotate("", xy=(9.6, 4.4), xytext=(0.4, 4.4),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3))
    ax.text(0.4, 4.7, "C₍t−1₎  cell state (the memory highway)", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(9.0, 4.7, "Cₜ", fontsize=11, color=GREEN, fontweight="bold")
    # forget multiply, input add on the highway
    for x, sym, col in [(3.0, "×", RED), (5.6, "+", AMBER)]:
        ax.add_patch(Circle((x, 4.4), 0.26, facecolor=col, edgecolor="white", lw=1.5, zorder=4))
        ax.text(x, 4.4, sym, ha="center", va="center", color="#fff", fontsize=13, fontweight="bold", zorder=5)
    # gates row
    gates = [("forget gate\nfₜ = σ(...)", 3.0, RED, "how much old\nmemory to keep"),
             ("input gate\niₜ = σ(...)", 5.0, BLUE, "how much new\ninfo to write"),
             ("candidate\nC̃ₜ = tanh(...)", 6.2, PURPLE, "the new\ncontent"),
             ("output gate\noₜ = σ(...)", 8.2, NAVY, "what to expose\nas hidden hₜ")]
    for label, x, col, sub in gates:
        ax.add_patch(FancyBboxPatch((x-0.7, 1.7), 1.4, 0.9, boxstyle="round,pad=0.06",
                                    facecolor=col, edgecolor="white", lw=1.5))
        ax.text(x, 2.15, label, ha="center", va="center", color="#fff", fontsize=8.5, fontweight="bold")
    # inputs
    ax.text(0.5, 1.0, "inputs:  hₜ₋₁  (previous hidden)   +   xₜ  (current token)",
            fontsize=10, color=SLATE, fontweight="bold")
    ax.annotate("", xy=(3.0, 4.14), xytext=(3.0, 2.6), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.annotate("", xy=(5.6, 4.14), xytext=(5.6, 2.6), arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
    # hidden output
    ax.add_patch(Circle((8.2, 4.4), 0.0))
    ax.annotate("", xy=(8.2, 0.7), xytext=(8.2, 1.7), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8))
    ax.text(8.2, 0.45, "hₜ = oₜ ⊙ tanh(Cₜ)", ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.text(5.0, 3.05, "Cₜ = fₜ ⊙ C₍t−1₎  +  iₜ ⊙ C̃ₜ   ← the ADDITIVE update is the gradient highway",
            ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.set_title("Inside an LSTM cell: gates decide what to forget, write, and output",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/lstm_cell.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lstm_cell.png")


if __name__ == "__main__":
    rnn_gradient_time()
    lstm_cell()
    print("OUT:", OUT)

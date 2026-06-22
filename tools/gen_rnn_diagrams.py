"""RNN / LSTM / GRU concept-page diagrams (muted palette, parallel type scale).

Figures for 05. Deep_Learning/concepts/14-RNN-LSTM-GRU.md:
  1. rnn_unrolled.png       -- the recurrence unrolled through time: shared weights,
                               hidden state threaded across steps (schematic).
  2. lstm_cell.png          -- annotated LSTM cell: the cell-state highway with a
                               forget-multiply + input-add, and the three gates below.
  3. rnn_gradient_time.png  -- MEASURED gradient magnitude vs timesteps-back, plain
                               RNN (vanishes) vs LSTM with forget-bias init (flat).
  4. copy_task.png          -- MEASURED accuracy on a long-range copy/memorization
                               task vs sequence length: LSTM remembers, RNN forgets.
  5. gru_vs_lstm.png        -- side-by-side gate schematic, LSTM (3 gates + cell)
                               vs GRU (2 gates, merged state).

Run with the torch-enabled env:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_rnn_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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


def _box(ax, xy, w, h, color, text, tcolor="#fff", fs=11, fw="bold", rounded=0.12):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounded}",
                       linewidth=1.5, edgecolor="#222", facecolor=color, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tcolor, fontsize=fs, fontweight=fw, zorder=3)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p0, p1, color="#333", lw=1.8, style="-|>", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, zorder=1,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# 1. RNN unrolled through time
# ---------------------------------------------------------------------------
def rnn_unrolled():
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.set_xlim(0, 9.8); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.set_title("A vanilla RNN unrolled through time — one shared cell applied at every step",
                 fontsize=14, fontweight="bold", pad=8)

    xs = [1.2, 3.9, 6.6]
    sub = ["₁", "₂", "₃"]
    for i, x in enumerate(xs):
        _box(ax, (x, 0.45), 1.3, 0.7, BLUE, f"x{sub[i]}", fs=12)          # input
        _box(ax, (x, 1.8), 1.3, 0.9, PURPLE, f"h{sub[i]}\ntanh", fs=11)   # hidden
        _box(ax, (x, 3.45), 1.3, 0.7, GREEN, f"y{sub[i]}", fs=12)         # output
        _arrow(ax, (x + 0.65, 1.15), (x + 0.65, 1.8), color=BLUE)
        ax.text(x + 0.98, 1.42, "Wₓₕ", fontsize=9, color=BLUE, ha="left")
        _arrow(ax, (x + 0.65, 2.7), (x + 0.65, 3.45), color=GREEN)
        ax.text(x + 0.98, 3.0, "W_hy", fontsize=9, color=GREEN, ha="left")
        ax.text(x + 0.65, 0.2, f"t = {i+1}", fontsize=10, color="#444", ha="center")

    for i in range(len(xs) - 1):                                          # recurrent W_hh
        _arrow(ax, (xs[i] + 1.3, 2.25), (xs[i + 1], 2.25), color=RED, lw=2.2)
        ax.text((xs[i] + xs[i + 1] + 1.3) / 2, 2.46, "W_hh", fontsize=9.5,
                color=RED, ha="center", fontweight="bold")

    _arrow(ax, (0.25, 2.25), (xs[0], 2.25), color=SLATE, lw=2.0)
    ax.text(0.2, 2.5, "h₀", fontsize=10, color=SLATE, ha="center")
    _arrow(ax, (xs[-1] + 1.3, 2.25), (xs[-1] + 2.0, 2.25), color=RED, lw=2.0)
    ax.text(xs[-1] + 2.3, 2.25, "...", fontsize=15, color=RED, va="center")

    ax.text(4.9, 0.05, "the SAME Wₓₕ, W_hh, W_hy are reused at every step (weight sharing)",
            fontsize=10, color=RED, ha="center", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/rnn_unrolled.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rnn_unrolled.png")


# ---------------------------------------------------------------------------
# 2. LSTM cell internals
# ---------------------------------------------------------------------------
def lstm_cell():
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("Inside an LSTM cell: the cell-state highway (C) with a forget-multiply and an input-add",
                 fontsize=13.5, fontweight="bold", pad=8)

    hy = 4.6
    _arrow(ax, (0.3, hy), (2.7, hy), color=GREEN, lw=3.2)
    ax.text(0.3, hy + 0.3, "C₍t−1₎", fontsize=11, color=GREEN, fontweight="bold")
    mul = Circle((3.0, hy), 0.23, facecolor=RED, edgecolor="#222", lw=1.5, zorder=3)
    ax.add_patch(mul); ax.text(3.0, hy, "×", color="#fff", ha="center", va="center",
                               fontsize=15, fontweight="bold", zorder=4)
    _arrow(ax, (3.23, hy), (5.5, hy), color=GREEN, lw=3.2)
    add = Circle((5.8, hy), 0.23, facecolor=NAVY, edgecolor="#222", lw=1.5, zorder=3)
    ax.add_patch(add); ax.text(5.8, hy, "+", color="#fff", ha="center", va="center",
                               fontsize=15, fontweight="bold", zorder=4)
    _arrow(ax, (6.03, hy), (9.8, hy), color=GREEN, lw=3.2)
    ax.text(8.9, hy + 0.3, "Cₜ", fontsize=11, color=GREEN, fontweight="bold")
    ax.text(5.0, 5.2, "additive memory highway:  Cₜ = fₜ ⊙ C₍t−1₎ + iₜ ⊙ gₜ",
            fontsize=10.5, color=GREEN, ha="center", fontweight="bold", style="italic")

    gy = 2.3
    _box(ax, (2.4, gy), 1.2, 0.8, AMBER, "fₜ\nσ", fs=12)      # forget
    _box(ax, (4.1, gy), 1.2, 0.8, BLUE, "iₜ\nσ", fs=12)       # input
    _box(ax, (5.5, gy), 1.2, 0.8, PURPLE, "gₜ\ntanh", fs=11)  # candidate
    _box(ax, (7.5, gy), 1.2, 0.8, SLATE, "oₜ\nσ", fs=12)      # output

    _arrow(ax, (3.0, gy + 0.8), (3.0, hy - 0.23), color=AMBER, lw=2.0)
    _arrow(ax, (4.7, gy + 0.8), (5.62, hy - 0.23), color=BLUE, lw=2.0, rad=-0.18)
    _arrow(ax, (6.1, gy + 0.8), (5.95, hy - 0.23), color=PURPLE, lw=2.0, rad=0.12)

    _box(ax, (0.3, gy - 0.05), 1.2, 0.45, NAVY, "h₍t−1₎", fs=10)
    _box(ax, (0.3, gy + 0.55), 1.2, 0.45, BLUE, "xₜ", fs=11)
    for gx in (2.4, 4.1, 5.5, 7.5):
        _arrow(ax, (1.5, gy + 0.4), (gx, gy + 0.4), color="#888", lw=1.1)

    th = Circle((8.1, hy - 0.95), 0.27, facecolor=GREEN, edgecolor="#222", lw=1.4, zorder=3)
    ax.add_patch(th); ax.text(8.1, hy - 0.95, "tanh", color="#fff", ha="center",
                              va="center", fontsize=8.5, fontweight="bold", zorder=4)
    _arrow(ax, (8.1, hy - 0.05), (8.1, hy - 0.68), color=GREEN, lw=2.0)
    omul = Circle((8.1, 1.55), 0.23, facecolor=RED, edgecolor="#222", lw=1.5, zorder=3)
    ax.add_patch(omul); ax.text(8.1, 1.55, "×", color="#fff", ha="center", va="center",
                                fontsize=14, fontweight="bold", zorder=4)
    _arrow(ax, (8.1, hy - 1.22), (8.1, 1.78), color=GREEN, lw=2.0)
    _arrow(ax, (8.1, gy), (8.1, 1.78), color=SLATE, lw=2.0)
    _box(ax, (7.45, 0.4), 1.3, 0.65, PURPLE, "hₜ", fs=12)
    _arrow(ax, (8.1, 1.32), (8.1, 1.05), color=PURPLE, lw=2.0)
    ax.text(6.95, 0.72, "hₜ = oₜ ⊙ tanh(Cₜ)", fontsize=9.5, color=PURPLE, ha="right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/lstm_cell.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lstm_cell.png")


# ---------------------------------------------------------------------------
# 3 & 4. MEASURED diagrams (need torch)
# ---------------------------------------------------------------------------
def _measure_gradient_time(T=40, H=32, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    def run(kind):
        if kind == "RNN":
            cell = nn.RNNCell(1, H)
        else:
            cell = nn.LSTMCell(1, H)
            with torch.no_grad():
                cell.bias_ih[H:2 * H].fill_(3.0)
                cell.bias_hh[H:2 * H].fill_(3.0)
        xs = [torch.zeros(1, 1, requires_grad=True) for _ in range(T)]
        h = torch.zeros(1, H); c = torch.zeros(1, H)
        for t in range(T):
            if kind == "RNN":
                h = cell(xs[t], h)
            else:
                h, c = cell(xs[t], (h, c))
        h.sum().backward()
        return np.array([xs[T - 1 - k].grad.abs().sum().item() for k in range(T)])

    return run("RNN"), run("LSTM")


def rnn_gradient_time():
    rnn, lstm = _measure_gradient_time()
    lags = np.arange(len(rnn))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    eps = 1e-30
    ax.semilogy(lags, np.maximum(rnn, eps), color=RED, lw=2.4, marker="o", ms=4,
                label="plain RNN — vanishes exponentially")
    ax.semilogy(lags, np.maximum(lstm, eps), color=GREEN, lw=2.4, marker="s", ms=4,
                label="LSTM (forget-bias ≈ 1) — flat highway")
    ax.set_xlabel("timesteps back from the output  (lag k)")
    ax.set_ylabel("|∂ output / ∂ input(t−k)|   (log scale)")
    ax.set_title("Gradient through time: the RNN forgets, the LSTM remembers (measured)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10.5, loc="lower left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/rnn_gradient_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rnn_gradient_time.png")
    return rnn, lstm


def _train_copy_task(T, H=48, n_train=3000, n_test=600, epochs=120, kind="LSTM", seed=0):
    """Carry a single signal bit (planted at position 0) across T steps to the last
    step's readout. Pure long-range memorization; chance = 50%."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    def make(n):
        x = torch.zeros(n, T, 1)                          # noise channel ~ 0
        bit = (torch.rand(n, 1) > 0.5).float()
        x[:, 0, 0] = bit[:, 0] * 2 - 1                     # +-1 marker at t=0
        return x, bit

    Xtr, ytr = make(n_train); Xte, yte = make(n_test)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = (nn.LSTM(1, H, batch_first=True) if kind == "LSTM"
                        else nn.RNN(1, H, batch_first=True, nonlinearity="tanh"))
            if kind == "LSTM":
                for name, p in self.rnn.named_parameters():
                    if "bias_ih" in name or "bias_hh" in name:
                        n = p.shape[0]
                        p.data[n // 4:n // 2].fill_(1.0)      # forget-bias init
            self.head = nn.Linear(H, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        opt.zero_grad()
        lossf(net(Xtr), ytr).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # clip explosions
        opt.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(net(Xte)) > 0.5).float() == yte).float().mean().item()
    return acc


def copy_task():
    lengths = [5, 10, 20, 40, 70, 100]
    rnn_acc, lstm_acc = [], []
    for T in lengths:
        rnn_acc.append(_train_copy_task(T, kind="RNN"))
        lstm_acc.append(_train_copy_task(T, kind="LSTM"))
        print(f"  T={T:3d}  RNN={rnn_acc[-1]:.2f}  LSTM={lstm_acc[-1]:.2f}")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(lengths, np.array(rnn_acc) * 100, color=RED, lw=2.4, marker="o", ms=6,
            label="plain RNN")
    ax.plot(lengths, np.array(lstm_acc) * 100, color=GREEN, lw=2.4, marker="s", ms=6,
            label="LSTM")
    ax.axhline(50, color=SLATE, ls="--", lw=1.2)
    ax.text(60, 46.5, "chance (50%)", color=SLATE, fontsize=9, ha="center")
    ax.set_xlabel("sequence length  (distance from the cue to the answer)")
    ax.set_ylabel("test accuracy  (%)")
    ax.set_ylim(40, 103)
    ax.set_title("Long-range copy task: the LSTM holds the memory, the RNN collapses to chance",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, loc="lower left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/copy_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote copy_task.png")
    return lengths, rnn_acc, lstm_acc


# ---------------------------------------------------------------------------
# 5. GRU vs LSTM gate schematic
# ---------------------------------------------------------------------------
def gru_vs_lstm():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax in axes:
        ax.set_xlim(0, 5); ax.set_ylim(-0.3, 4.6); ax.axis("off")

    ax = axes[0]
    ax.set_title("LSTM — 3 gates + separate cell state", fontsize=13, fontweight="bold")
    _arrow(ax, (0.4, 3.6), (4.6, 3.6), color=GREEN, lw=3.0)
    ax.text(0.4, 3.85, "cell state C (memory highway)", fontsize=9.5, color=GREEN, fontweight="bold")
    _box(ax, (0.4, 1.8), 1.0, 0.7, AMBER, "forget\nfₜ", fs=9.5)
    _box(ax, (1.6, 1.8), 1.0, 0.7, BLUE, "input\niₜ", fs=9.5)
    _box(ax, (2.8, 1.8), 1.0, 0.7, SLATE, "output\noₜ", fs=9.5)
    _box(ax, (1.0, 0.5), 1.4, 0.65, PURPLE, "candidate gₜ", fs=9.5)
    _box(ax, (3.4, 0.5), 1.1, 0.65, PURPLE, "hₜ", fs=11)
    ax.text(2.5, 2.9, "Cₜ = f⊙C₍t−1₎ + i⊙gₜ", fontsize=9, color="#333", ha="center")
    ax.text(2.5, -0.15, "4 weight blocks", fontsize=9.5, color=RED, ha="center",
            style="italic", fontweight="bold")

    ax = axes[1]
    ax.set_title("GRU — 2 gates, one merged state", fontsize=13, fontweight="bold")
    _arrow(ax, (0.4, 3.6), (4.6, 3.6), color=PURPLE, lw=3.0)
    ax.text(0.4, 3.85, "hidden state h (no separate cell)", fontsize=9.5,
            color=PURPLE, fontweight="bold")
    _box(ax, (0.7, 1.8), 1.05, 0.7, AMBER, "reset\nrₜ", fs=9.5)
    _box(ax, (2.3, 1.8), 1.05, 0.7, BLUE, "update\nzₜ", fs=9.5)
    _box(ax, (1.4, 0.5), 1.5, 0.65, PURPLE, "candidate h̃ₜ", fs=9.5)
    ax.text(2.5, 2.9, "hₜ = (1−z)⊙h₍t−1₎ + z⊙h̃ₜ", fontsize=9, color="#333", ha="center")
    ax.text(2.5, -0.15, "3 weight blocks — ~25% fewer params",
            fontsize=9.5, color=GREEN, ha="center", style="italic", fontweight="bold")

    fig.suptitle("LSTM vs GRU: the GRU drops the output gate and merges cell + hidden state",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gru_vs_lstm.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gru_vs_lstm.png")


if __name__ == "__main__":
    rnn_unrolled()
    lstm_cell()
    gru_vs_lstm()
    g_rnn, g_lstm = rnn_gradient_time()
    print(f"  [grad@lag39]  RNN={g_rnn[-1]:.2e}  LSTM={g_lstm[-1]:.2e}")
    copy_task()
    print("all RNN diagrams written to", OUT)

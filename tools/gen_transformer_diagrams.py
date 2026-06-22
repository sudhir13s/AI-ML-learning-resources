"""Transformer-architecture concept-page diagrams (muted palette, parallel matplotlib scale).

Visuals for 05. Deep_Learning/concepts/16-Transformer-Architecture.md:
  1. tf_positional_encoding.png -- sinusoidal positional encoding: heatmap over
     (position x dimension) + a few dimensions as waves of different frequency.
  2. tf_params.png             -- where a transformer block's parameters live:
     attention vs FFN per block, and how the total scales with depth.
  3. tf_flop_crossover.png      -- attention (O(n^2 d)) vs FFN (O(n d^2)) FLOPs per
     block as sequence length grows, with the n = 2*d_ff crossover marked; plus the
     FLOP split as a function of model width at a fixed sequence length.
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


def positional_encoding():
    pos_n, d = 60, 64
    pos = np.arange(pos_n)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    PE = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    im = ax1.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax1.set_xlabel("embedding dimension"); ax1.set_ylabel("token position")
    ax1.set_title("Sinusoidal positional encoding $PE[pos, i]$", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    for dim, c, lab in [(0, BLUE, "dim 0 (fast)"), (4, PURPLE, "dim 4"), (20, GREEN, "dim 20 (slow)")]:
        ax2.plot(np.arange(pos_n), PE[:, dim], color=c, lw=2.2, label=lab)
    ax2.set_xlabel("token position"); ax2.set_ylabel("value")
    ax2.set_title("Each dimension is a wave of a different frequency", fontsize=13, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04)); _despine(ax2)
    ax2.set_ylim(-1.15, 1.35)
    fig.tight_layout(); fig.savefig(f"{OUT}/tf_positional_encoding.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tf_positional_encoding.png")


def params():
    d_model, d_ff, n_heads = 768, 3072, 12          # BERT-base-ish block
    attn = 4 * d_model * d_model                      # Wq,Wk,Wv,Wo
    ffn = 2 * d_model * d_ff                           # two linear layers (4x expansion)
    ln = 2 * 2 * d_model                               # two LayerNorms (gamma,beta)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    comps = ["Attention\n(Wq,Wk,Wv,Wo)", "Feed-forward\n(2 layers, 4× wide)", "LayerNorm"]
    vals = [attn / 1e6, ffn / 1e6, ln / 1e6]
    bars = ax1.bar(comps, vals, 0.6, color=[PURPLE, GREEN, SLATE], edgecolor="white")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.1, f"{v:.2f}M", ha="center", fontweight="bold", color="#222", fontsize=10)
    ax1.set_ylabel("parameters (millions)")
    ax1.set_title("One block's parameters: FFN dominates", fontsize=13, fontweight="bold")
    ax1.text(1, ffn/1e6 * 0.5, "≈ 2× attention", ha="center", color="white", fontweight="bold", fontsize=10)
    _despine(ax1)

    layers = np.arange(1, 25)
    per_block = (attn + ffn + ln) / 1e6
    ax2.plot(layers, layers * per_block, color=NAVY, lw=2.6, marker="o", ms=3)
    ax2.set_xlabel("number of layers (depth)"); ax2.set_ylabel("transformer-block params (M)")
    ax2.set_title("Params scale linearly with depth", fontsize=13, fontweight="bold")
    ax2.annotate(f"12 layers ≈ {12*per_block:.0f}M\n(BERT-base block stack)", (12, 12*per_block),
                 textcoords="offset points", xytext=(-95, 18), fontsize=9, fontweight="bold", color="#222",
                 arrowprops=dict(arrowstyle="->", color=SLATE))
    _despine(ax2)
    fig.tight_layout(); fig.savefig(f"{OUT}/tf_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tf_params.png")


def flop_crossover():
    """Per-block forward FLOPs: attention vs FFN. Counting matmul FLOPs as 2*M*N*K.

    Attention (the two n x n matmuls, batch=1, summed over heads):
        scores  QK^T : 2 * n * n * d
        weighted sum : 2 * n * n * d        ->  total 4 * n^2 * d
    The four projections (Wq,Wk,Wv,Wo) add 8 * n * d^2 but are O(n d^2) like the FFN;
    we attribute them to the per-token "projection+FFN" side and isolate the genuinely
    quadratic core attention here so the crossover with the FFN is the n^2-vs-d^2 story.

    FFN (two linear layers, d -> 4d -> d):
        2 * n * d * 4d  +  2 * n * 4d * d   =  16 * n * d^2
    Crossover (4 n^2 d == 16 n d^2)  ->  n = 4d.  i.e. attention's quadratic core only
    overtakes the FFN once the sequence is several times the model width.
    """
    d = 768
    n = np.linspace(1, 16384, 400)
    attn_flops = 4 * n**2 * d                 # core (n x n) attention
    ffn_flops = 16 * n * d**2                 # two FFN matmuls (4x expansion)
    crossover_n = 4 * d                        # 4 n^2 d = 16 n d^2  ->  n = 4d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))

    ax1.plot(n, ffn_flops / 1e9, color=GREEN, lw=2.6, label="FFN  (16·n·d²)")
    ax1.plot(n, attn_flops / 1e9, color=PURPLE, lw=2.6, label="core attention  (4·n²·d)")
    ax1.axvline(crossover_n, color=RED, ls="--", lw=1.8)
    ax1.annotate(f"crossover\nn = 4d = {crossover_n}", (crossover_n, ffn_flops.max()/1e9*0.62),
                 color=RED, fontsize=9.5, fontweight="bold", ha="left",
                 xytext=(crossover_n + 700, ffn_flops.max()/1e9*0.62), va="center")
    ax1.fill_between(n, 0, ffn_flops/1e9, where=(n < crossover_n), color=GREEN, alpha=0.07)
    ax1.set_xlabel("sequence length  n  (d = 768 fixed)")
    ax1.set_ylabel("forward GFLOPs per block")
    ax1.set_title("FFN dominates until n ≈ 4d, then attention wins", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax1)

    # right: at a fixed, realistic n, how the per-block FLOP budget splits vs model width
    n_fixed = 2048
    widths = np.array([256, 512, 768, 1024, 2048, 4096])
    attn_w = 4 * n_fixed**2 * widths
    ffn_w = 16 * n_fixed * widths**2
    frac_attn = attn_w / (attn_w + ffn_w) * 100
    ax2.plot(widths, frac_attn, color=PURPLE, lw=2.6, marker="o", ms=5, label="attention share")
    ax2.plot(widths, 100 - frac_attn, color=GREEN, lw=2.6, marker="s", ms=5, label="FFN share")
    ax2.set_xlabel(f"model width  d   (n = {n_fixed} fixed)")
    ax2.set_ylabel("share of per-block FLOPs (%)")
    ax2.set_title("Wider models push the budget toward the FFN", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 100); ax2.legend(frameon=False, fontsize=9.5, loc="center right"); _despine(ax2)
    fig.tight_layout(); fig.savefig(f"{OUT}/tf_flop_crossover.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote tf_flop_crossover.png")


if __name__ == "__main__":
    positional_encoding()
    params()
    flop_crossover()
    print("OUT:", OUT)

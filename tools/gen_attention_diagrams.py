"""Attention-mechanism concept-page diagrams (muted palette, parallel matplotlib scale).

Visuals for 05. Deep_Learning/concepts/15-Attention-Mechanism.md:
  1. attn_softmax_scaling.png -- the MATH: why divide by sqrt(d_k). Left: variance of
     Q·K grows linearly with d_k. Right: softmax of the same scores saturates to near
     one-hot when unscaled, stays smooth when scaled.
  2. attn_heatmap.png         -- INTUITION: an (illustrative) attention weight matrix,
     each query row a distribution over key columns; one long-range link highlighted.
  3. attn_path_parallel.png   -- WHY: RNN vs self-attention on max path length between
     two tokens and sequential ops (O(n) vs O(1)).
  4. attn_dataflow.png        -- the scaled-dot-product DATAFLOW with every shape labelled
     (Q,K,V -> QKᵀ -> scale -> mask -> softmax -> ·V), schematic boxes + arrows.
  5. attn_softmax_entropy.png -- MEASURED: as d_k grows, the UNSCALED softmax collapses
     (entropy -> 0, max-weight -> 1) while the SCALED one stays high-entropy/trainable.
  6. attn_cost_growth.png     -- MEASURED: O(n^2) score-matrix memory & FLOPs vs sequence
     length, the quadratic wall that motivates FlashAttention / sparse / linear / KV-cache.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


# ---- 1. Why scale by sqrt(d_k) ----------------------------------------------
def softmax_scaling():
    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # (a) variance of dot product grows ~ d_k (unit-variance q,k entries)
    dks = np.arange(1, 257)
    emp = []
    for dk in dks[::8]:
        q = rng.standard_normal((2000, dk)); k = rng.standard_normal((2000, dk))
        emp.append(np.var(np.sum(q * k, axis=1)))
    ax1.plot(dks, dks, color=SLATE, lw=2.2, ls="--", label="theory: Var = d_k")
    ax1.scatter(dks[::8], emp, color=BLUE, s=26, zorder=5, label="empirical")
    ax1.set_xlabel("head dimension $d_k$"); ax1.set_ylabel("Var($q\\cdot k$)")
    ax1.set_title("Dot-product variance grows with $d_k$", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax1)

    # (b) softmax of one score vector: unscaled vs /sqrt(d_k)
    dk = 64
    scores = rng.standard_normal(12) * np.sqrt(dk)   # raw QK scores, std ~ sqrt(dk)
    def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
    w_raw = softmax(scores)
    w_scaled = softmax(scores / np.sqrt(dk))
    x = np.arange(12); w = 0.4
    ax2.bar(x - w/2, w_raw, w, color=RED, label="unscaled (saturates)", edgecolor="white")
    ax2.bar(x + w/2, w_scaled, w, color=GREEN, label="scaled by $1/\\sqrt{d_k}$", edgecolor="white")
    ax2.set_xlabel("key position"); ax2.set_ylabel("attention weight")
    ax2.set_title("Scaling keeps softmax from going one-hot", fontsize=13, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9.5); _despine(ax2)
    ax2.annotate("unscaled: one key dominates →\nvanishing gradients", (np.argmax(w_raw), w_raw.max()),
                 textcoords="offset points", xytext=(-6, -34), fontsize=8.5, color=RED, fontweight="bold",
                 ha="center", arrowprops=dict(arrowstyle="->", color=RED))
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_softmax_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_softmax_scaling.png")


# ---- 2. Attention heatmap (illustrative) ------------------------------------
def heatmap():
    toks = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
    n = len(toks)
    W = np.zeros((n, n))
    for i in range(n):                       # causal: token i attends to <= i
        base = np.zeros(n)
        for j in range(i + 1):
            base[j] = np.exp(-0.6 * (i - j))  # local recency bias
        base[i] += 0.3                        # self
        W[i] = base
    # inject the long-range link: "it" (idx 7) -> "animal" (idx 1)
    W[7, 1] += 1.6
    W = W / W.sum(axis=1, keepdims=True)
    W = np.where(np.tril(np.ones((n, n))) > 0, W, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(W, cmap="BuPu", aspect="equal", vmin=0, vmax=0.7)
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=9)
    ax.set_xlabel("key (attended to)"); ax.set_ylabel("query (attending)")
    ax.set_title("Illustrative self-attention weights: 'it' → 'animal'", fontsize=13, fontweight="bold")
    ax.add_patch(plt.Rectangle((1 - 0.5, 7 - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("weight", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_heatmap.png")


# ---- 3. RNN vs self-attention: path length & parallelism --------------------
def path_parallel():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    n = np.arange(1, 65)
    ax1.plot(n, n, color=RED, lw=2.6, label="RNN: O(n)")
    ax1.plot(n, np.ones_like(n), color=GREEN, lw=2.6, label="Self-attention: O(1)")
    ax1.set_xlabel("distance between two tokens"); ax1.set_ylabel("steps for signal to travel")
    ax1.set_title("Max path length between tokens", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5); _despine(ax1)
    ax1.annotate("every token reaches\nevery other in ONE hop", (40, 1), textcoords="offset points",
                 xytext=(-30, 38), fontsize=9, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN))

    cats = ["RNN", "Self-attention"]
    seq_ops = [r"$O(n)$", r"$O(1)$"]; path = [r"$O(n)$", r"$O(1)$"]
    par = ["no", "yes"]
    ax2.axis("off")
    cell = [[seq_ops[0], path[0], par[0]], [seq_ops[1], path[1], par[1]]]
    tbl = ax2.table(cellText=cell, rowLabels=cats,
                    colLabels=["Sequential ops", "Path length", "Parallel?"],
                    cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.4)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("white")
        if r == 0: cellobj.set_facecolor(SLATE); cellobj.set_text_props(color="white", fontweight="bold")
        elif r == 1: cellobj.set_facecolor("#e9e4f0")
        elif r == 2: cellobj.set_facecolor("#dff0e6")
    ax2.set_title("Why attention beat RNNs", fontsize=13, fontweight="bold", y=0.86)
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_path_parallel.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_path_parallel.png")


# ---- 4. Scaled-dot-product dataflow with shapes -----------------------------
def dataflow():
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, label, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=fc, ec="white", lw=1.5))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                color="white", fontsize=9.0, style="italic")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=15, color=SLATE, lw=1.8))

    # inputs Q,K,V (left column)
    box(0.2, 3.5, 1.7, 1.0, "Q", "[n, d_k]", BLUE)
    box(0.2, 2.0, 1.7, 1.0, "K", "[m, d_k]", BLUE)
    box(0.2, 0.5, 1.7, 1.0, "V", "[m, d_v]", BLUE)

    # QKᵀ
    box(2.6, 2.75, 1.9, 1.0, "scores = QKᵀ", "[n, m]", PURPLE)
    arrow(1.9, 4.0, 2.6, 3.4)
    arrow(1.9, 2.5, 2.6, 3.1)

    # scale
    box(5.1, 2.75, 1.7, 1.0, "÷ √d_k", "[n, m]", AMBER)
    arrow(4.5, 3.25, 5.1, 3.25)

    # mask
    box(7.3, 2.75, 1.7, 1.0, "+ mask", "−∞ → 0", RED)
    arrow(6.8, 3.25, 7.3, 3.25)

    # softmax
    box(9.5, 2.75, 2.0, 1.0, "softmax", "rows sum→1", GREEN)
    arrow(9.0, 3.25, 9.5, 3.25)

    # weighted sum with V -> output
    box(7.3, 0.5, 2.0, 1.0, "weights · V", "[n, d_v]", PURPLE)
    box(9.9, 0.5, 1.7, 1.0, "output", "[n, d_v]", GREEN)
    arrow(10.5, 2.75, 9.0, 1.5)      # softmax weights down into ·V
    arrow(1.9, 1.0, 7.3, 1.0)        # V into ·V
    arrow(9.3, 1.0, 9.9, 1.0)        # ·V into output

    ax.text(6.0, 4.7, "Scaled dot-product attention — dataflow with shapes",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(6.0, 0.05, "Attention(Q,K,V) = softmax( QKᵀ / √d_k + mask ) · V",
            ha="center", fontsize=11, style="italic", color=SLATE)
    fig.savefig(f"{OUT}/attn_dataflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_dataflow.png")


# ---- 5. Softmax entropy/sharpness vs d_k (measured) -------------------------
def softmax_entropy():
    rng = np.random.default_rng(7)
    dks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    m = 16          # keys to attend over
    trials = 4000
    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)
    ent_raw, ent_scaled, max_raw, max_scaled = [], [], [], []
    max_ent = np.log(m)
    for dk in dks:
        # raw QK scores for unit-variance q,k: variance d_k  => std sqrt(dk)
        s = rng.standard_normal((trials, m)) * np.sqrt(dk)
        wr = softmax(s); ws = softmax(s / np.sqrt(dk))
        ent_raw.append(np.mean(-(wr * np.log(wr + 1e-12)).sum(-1)) / max_ent)
        ent_scaled.append(np.mean(-(ws * np.log(ws + 1e-12)).sum(-1)) / max_ent)
        max_raw.append(np.mean(wr.max(-1))); max_scaled.append(np.mean(ws.max(-1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax1.semilogx(dks, ent_raw, "o-", color=RED, lw=2.4, base=2, label="unscaled")
    ax1.semilogx(dks, ent_scaled, "s-", color=GREEN, lw=2.4, base=2, label="scaled $1/\\sqrt{d_k}$")
    ax1.set_xlabel("head dimension $d_k$"); ax1.set_ylabel("normalized entropy (1 = uniform)")
    ax1.set_title("Unscaled softmax collapses as $d_k$ grows", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.05); ax1.legend(frameon=False, fontsize=9.5); _despine(ax1)

    ax2.semilogx(dks, max_raw, "o-", color=RED, lw=2.4, base=2, label="unscaled")
    ax2.semilogx(dks, max_scaled, "s-", color=GREEN, lw=2.4, base=2, label="scaled $1/\\sqrt{d_k}$")
    ax2.axhline(1 / m, color=SLATE, ls=":", lw=1.5, label=f"uniform = 1/{m}")
    ax2.set_xlabel("head dimension $d_k$"); ax2.set_ylabel("mean max attention weight")
    ax2.set_title("...toward one-hot (max weight → 1)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05); ax2.legend(frameon=False, fontsize=9.5); _despine(ax2)
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_softmax_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_softmax_entropy.png")


# ---- 6. O(n^2) cost growth (measured arithmetic) ----------------------------
def cost_growth():
    d = 64
    n = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    # attention score-matrix memory (one head, fp16 = 2 bytes): n*n*2 bytes
    mem_mib = (n.astype(float) ** 2 * 2) / (1024 ** 2)
    # self-attention FLOPs ~ 2 * (QKᵀ: n*n*d) + 2 * (·V: n*n*d) = 4 n^2 d
    flops_g = (4.0 * n.astype(float) ** 2 * d) / 1e9
    # linear reference (O(n)) for contrast, scaled to meet at n=128
    lin = mem_mib[0] * (n / n[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax1.loglog(n, mem_mib, "o-", color=RED, lw=2.5, base=2, label="full attention  $O(n^2)$")
    ax1.loglog(n, lin, "--", color=GREEN, lw=2.2, base=2, label="linear / sparse  $O(n)$")
    ax1.set_xlabel("sequence length $n$"); ax1.set_ylabel("score-matrix memory (MiB, fp16, 1 head)")
    ax1.set_title("The quadratic memory wall", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5); ax1.grid(True, which="both", alpha=0.2)
    ax1.annotate("32k tokens →\n~2 GiB per head per layer", (n[-1], mem_mib[-1]),
                 textcoords="offset points", xytext=(-160, -6), fontsize=8.5,
                 color=RED, fontweight="bold")

    ax2.loglog(n, flops_g, "o-", color=PURPLE, lw=2.5, base=2)
    ax2.set_xlabel("sequence length $n$"); ax2.set_ylabel("attention FLOPs (GFLOP, 1 head, $d{=}64$)")
    ax2.set_title("Compute also grows as $O(n^2 d)$", fontsize=13, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.annotate("doubling $n$ →\n4× the work", (n[-3], flops_g[-3]),
                 textcoords="offset points", xytext=(-120, 34), fontsize=9,
                 color=PURPLE, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PURPLE))
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_cost_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote attn_cost_growth.png")


if __name__ == "__main__":
    softmax_scaling()
    heatmap()
    path_parallel()
    dataflow()
    softmax_entropy()
    cost_growth()
    print("OUT:", OUT)

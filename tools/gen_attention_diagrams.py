"""Attention-mechanism concept-page diagrams (muted palette, parallel matplotlib scale).

Three visuals for 05. Deep_Learning/concepts/15-Attention-Mechanism.md:
  1. attn_softmax_scaling.png -- the MATH: why divide by sqrt(d_k). Left: variance of
     Q·K grows linearly with d_k. Right: softmax of the same scores saturates to near
     one-hot when unscaled, stays smooth when scaled.
  2. attn_heatmap.png         -- INTUITION: an (illustrative) attention weight matrix,
     each query row a distribution over key columns; one long-range link highlighted.
  3. attn_path_parallel.png   -- WHY: RNN vs self-attention on max path length between
     two tokens and sequential ops (O(n) vs O(1)).
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
    rng = np.random.default_rng(3)
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

    labels = ["Sequential\nops", "Path\nlength", "Parallel\nover tokens?"]
    rnn = [1, 1, 0]; attn = [0, 0, 1]   # qualitative: green good
    # show as a small comparison table-ish bar
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


if __name__ == "__main__":
    softmax_scaling()
    heatmap()
    path_parallel()
    print("OUT:", OUT)

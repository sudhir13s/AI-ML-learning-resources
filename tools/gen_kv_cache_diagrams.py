"""KV-Cache concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 09. LLMs/concepts/05-KV-Cache.md:
  1. kv_recompute_waste.png  -- the PROBLEM: per-step K/V projection work,
     without cache (grows with position) vs with cache (constant).
  2. kv_memory_growth.png    -- the MATH: cache size (GB) vs context length
     for 7B / 13B / 70B(MHA), linear growth.
  3. kv_mha_mqa_gqa.png      -- the APPLICATION: cache size shrinks with fewer
     KV heads (MHA -> GQA-8 -> MQA).
  4. kv_attention_cache.png  -- WHY ONLY K,V: decode-step attention matrix --
     last query row active, all past K/V columns reused from cache, past query
     rows greyed (never needed again).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "09. LLMs", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 1. Recompute waste: per-step K/V projection work -----------------------
def recompute_waste():
    n = np.arange(1, 33)
    without = n.astype(float)          # recompute K,V for all positions each step
    with_c = np.ones_like(n, dtype=float)   # only the 1 new token
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(n, with_c, without, color=RED, alpha=0.16, label="redundant work")
    ax.plot(n, without, color=RED, lw=2.6, marker="o", ms=3, label="Without cache (recompute all past)")
    ax.plot(n, with_c, color=GREEN, lw=2.6, marker="o", ms=3, label="With cache (only the new token)")
    ax.annotate("each new token re-projects\nevery previous token's K,V",
                (24, 24), textcoords="offset points", xytext=(-150, -6), fontsize=9.5,
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("constant: 1 token / step", (28, 1), textcoords="offset points",
                xytext=(-30, 26), fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("Decode step (sequence position)")
    ax.set_ylabel("K/V projections computed this step")
    ax.set_title("The waste KV cache removes: O(n) recompute per step → O(1)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_xlim(1, 32); ax.set_ylim(0, 33); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/kv_recompute_waste.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kv_recompute_waste.png")


# ---- 2. Memory growth vs context length -------------------------------------
def memory_growth():
    # bytes/token = 2 (K,V) * layers * kv_heads * head_dim * dtype_bytes ; fp16=2
    def per_tok_mb(L, kvh, d, b=2):
        return 2 * L * kvh * d * b / 1e6
    ctx = np.linspace(0, 32768, 200)
    models = [
        ("Llama-2-7B (MHA, 32×32×128)", per_tok_mb(32, 32, 128), BLUE),
        ("Llama-2-13B (MHA, 40×40×128)", per_tok_mb(40, 40, 128), PURPLE),
        ("Llama-2-70B (MHA-equiv, 80×64×128)", per_tok_mb(80, 64, 128), RED),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for name, mb, c in models:
        ax.plot(ctx, ctx * mb / 1024, color=c, lw=2.6, label=f"{name} — {mb*1024:.0f} KB/token")
    ax.axvline(4096, color=SLATE, ls="--", lw=1.4)
    ax.text(4300, 0.5, "4K context", color=SLATE, fontsize=9.5, fontweight="bold")
    ax.scatter([4096], [4096 * per_tok_mb(32, 32, 128) / 1024], color=AMBER, s=60, zorder=5, edgecolor="white")
    ax.annotate("7B @ 4K ≈ 2 GB", (4096, 4096 * per_tok_mb(32, 32, 128) / 1024),
                textcoords="offset points", xytext=(20, 6), fontsize=9.5, fontweight="bold", color="#222")
    ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("KV cache size (GB, batch = 1)")
    ax.set_title("KV cache grows linearly with context — and rivals the weights", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9); _despine(ax)
    ax.set_xlim(0, 32768); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(f"{OUT}/kv_memory_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kv_memory_growth.png")


# ---- 3. MHA vs GQA vs MQA cache size ----------------------------------------
def mha_mqa_gqa():
    # 70B-ish: L=80, head_dim=128, fp16, seq=4096, batch=1; vary kv heads
    L, d, seq, b = 80, 128, 4096, 2
    def gb(kvh):
        return 2 * L * kvh * d * seq * b / 1e9
    labels = ["MHA\n(64 KV heads)", "GQA-8\n(8 KV heads)", "MQA\n(1 KV head)"]
    vals = [gb(64), gb(8), gb(1)]
    colors = [RED, GREEN, NAVY]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.55, color=colors, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f} GB", ha="center", fontweight="bold", color="#222", fontsize=10)
    ax.text(1, vals[1] + max(vals) * 0.13, "8× smaller\n(LLaMA-2/3 choice)", ha="center", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(2, vals[2] + max(vals) * 0.13, "64× smaller", ha="center", color=NAVY, fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("KV cache (GB) @ 4K ctx, batch 1")
    ax.set_title("Fewer KV heads = smaller cache: why GQA/MQA exist", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/kv_mha_mqa_gqa.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kv_mha_mqa_gqa.png")


# ---- 4. Attention-cache schematic: why only K,V (not Q) ----------------------
def attention_cache():
    n = 6  # tokens so far; we are generating token index n-1 (the last row)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    cur = n - 1
    for i in range(n):           # query rows
        for j in range(n):       # key columns
            if j > i:
                continue          # causal mask (upper triangle empty)
            if i == cur:
                fc, alpha = GREEN, 0.85   # active query row (this step)
            else:
                fc, alpha = SLATE, 0.18   # past query rows: never recomputed
            ax.add_patch(Rectangle((j, n - 1 - i), 0.92, 0.92, facecolor=fc, alpha=alpha, edgecolor="white"))
    # highlight the cached K/V columns (all past positions) along the active row
    ax.add_patch(Rectangle((-0.05, -0.05, ), n, 0.0, fill=False))
    ax.annotate("current query q₅ (computed now)", (cur + 0.5, n - 1 - cur + 0.5),
                xytext=(cur - 4.6, n - 1 - cur - 0.9), textcoords="data", fontsize=9.5,
                color=GREEN, fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("K,V for all past tokens:\nread straight from cache", (2.0, n + 0.15),
                xytext=(0.2, n + 0.7), textcoords="data", fontsize=9.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("past query rows:\nnever needed again\n(Q is not cached)", (0.5, n - 1 - 0 + 0.5),
                xytext=(n + 0.2, n - 1.5), textcoords="data", fontsize=9, color=SLATE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlim(-0.3, n + 3.2); ax.set_ylim(-0.3, n + 1.2)
    ax.set_xticks(np.arange(n) + 0.46); ax.set_xticklabels([f"k/v$_{i}$" for i in range(n)], fontsize=9)
    ax.set_yticks(np.arange(n) + 0.46); ax.set_yticklabels([f"q$_{n-1-i}$" for i in range(n)], fontsize=9)
    ax.set_title("Why cache K and V but not Q (one decode step)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); _despine(ax)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/kv_attention_cache.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kv_attention_cache.png")


if __name__ == "__main__":
    recompute_waste()
    memory_growth()
    mha_mqa_gqa()
    attention_cache()
    print("OUT:", OUT)

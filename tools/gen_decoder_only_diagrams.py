"""Decoder-only Architecture concept-page diagrams (muted palette, parallel matplotlib scale).

Visuals for "09. LLMs/concepts/04-Decoder-only-Architecture.md":
  1. decoder_causal_mask.png    -- the causal (lower-triangular) attention mask for a
     4-token sequence: token t attends only to <= t; future positions are -inf -> 0.
  2. decoder_arch_compare.png   -- encoder-only (BERT) vs encoder-decoder (T5) vs
     decoder-only (GPT): which positions attend to which (attention-pattern matrices).
  3. decoder_param_breakdown.png -- where a decoder layer's ~12 d^2 parameters live
     (attention QKVO vs FFN), and how embeddings vs blocks split a small model.
  4. decoder_logits.png         -- MEASURED: next-token probabilities from a small GPT-2
     for a real prompt (top-k bar chart), illustrating the LM head's output.
  5. decoder_scaling.png        -- the GPT lineage: parameter count GPT-1 -> GPT-2 ->
     GPT-3 (log scale) with what each unlocked.
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


# ---- 1. Causal mask for a 4-token sequence ----------------------------------
def causal_mask():
    toks = ["The", "cat", "sat", "down"]
    n = len(toks)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))

    # (a) the additive mask applied to scores BEFORE softmax
    axa = axes[0]
    for i in range(n):
        for j in range(n):
            allowed = j <= i
            fc = GREEN if allowed else RED
            txt = "0" if allowed else "−∞"
            axa.add_patch(Rectangle((j, n - 1 - i), 0.94, 0.94, facecolor=fc,
                                    alpha=0.85 if allowed else 0.7, edgecolor="white"))
            axa.text(j + 0.47, n - 1 - i + 0.47, txt, ha="center", va="center",
                     color="white", fontweight="bold", fontsize=11)
    axa.set_xticks(np.arange(n) + 0.47); axa.set_xticklabels([f"k:{t}" for t in toks], fontsize=9)
    axa.set_yticks(np.arange(n) + 0.47); axa.set_yticklabels([f"q:{t}" for t in toks[::-1]], fontsize=9)
    axa.set_xlim(0, n); axa.set_ylim(0, n); axa.set_aspect("equal")
    axa.set_title("Additive mask (added to scores pre-softmax)", fontsize=12, fontweight="bold")
    axa.tick_params(length=0)
    for s in axa.spines.values():
        s.set_visible(False)
    axa.annotate("q:sat cannot see k:down\n(future is −∞)", (3.5, n - 1 - 2 + 0.47),
                 xytext=(0.1, -1.05), textcoords="data", fontsize=9, color=RED,
                 fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))

    # (b) the resulting attention weights (row-softmax of the masked scores; illustrative uniform)
    axb = axes[1]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, :i + 1] = 1.0 / (i + 1)   # uniform over allowed positions, illustrative
    im = axb.imshow(W, cmap="YlGn", vmin=0, vmax=1, aspect="equal")
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0:
                axb.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                         color="#222" if W[i, j] < 0.6 else "white", fontsize=9)
    axb.set_xticks(range(n)); axb.set_xticklabels([f"k:{t}" for t in toks], fontsize=9)
    axb.set_yticks(range(n)); axb.set_yticklabels([f"q:{t}" for t in toks], fontsize=9)
    axb.set_title("Attention weights after softmax (rows sum to 1)", fontsize=12, fontweight="bold")
    fig.suptitle("Causal masking: each token attends only to itself and the past",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/decoder_causal_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote decoder_causal_mask.png")


# ---- 2. Encoder-only vs encoder-decoder vs decoder-only ----------------------
def arch_compare():
    n = 5
    toks = ["w1", "w2", "w3", "w4", "w5"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))

    def draw(ax, mask, title, color, sub):
        for i in range(n):
            for j in range(n):
                on = mask[i, j]
                ax.add_patch(Rectangle((j, n - 1 - i), 0.94, 0.94,
                             facecolor=color if on else "#E9ECEF",
                             alpha=0.85 if on else 1.0, edgecolor="white"))
        ax.set_xticks(np.arange(n) + 0.47); ax.set_xticklabels(toks, fontsize=8)
        ax.set_yticks(np.arange(n) + 0.47); ax.set_yticklabels(toks[::-1], fontsize=8)
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9)

    full = np.ones((n, n), dtype=bool)
    causal = np.tril(np.ones((n, n), dtype=bool))
    draw(axes[0], full, "Encoder-only (BERT)", BLUE,
         "bidirectional: every token\nsees every token")
    draw(axes[1], full, "Encoder-decoder (T5)\nencoder side", PURPLE,
         "encoder bidirectional;\ndecoder causal + cross-attn")
    draw(axes[2], causal, "Decoder-only (GPT)", GREEN,
         "causal: token t sees only <= t")
    fig.suptitle("Three attention patterns — decoder-only is purely causal",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/decoder_arch_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote decoder_arch_compare.png")


# ---- 3. Parameter breakdown of a decoder layer + a small model --------------
def param_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))

    # (a) one decoder layer ~= 12 d^2 : attention (4 d^2) + FFN (8 d^2)
    axa = axes[0]
    parts = ["Wq\n(d²)", "Wk\n(d²)", "Wv\n(d²)", "Wo\n(d²)", "FFN up\n(4d²)", "FFN down\n(4d²)"]
    vals = [1, 1, 1, 1, 4, 4]
    colors = [BLUE, BLUE, BLUE, NAVY, PURPLE, PURPLE]
    x = np.arange(len(parts))
    axa.bar(x, vals, 0.6, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        axa.text(i, v + 0.08, f"{v}d²", ha="center", fontweight="bold", fontsize=9)
    axa.set_xticks(x); axa.set_xticklabels(parts, fontsize=8.5)
    axa.set_ylabel("params (units of d²)")
    axa.set_title("One decoder layer ≈ 12 d²  (attn 4d² + FFN 8d²)", fontsize=12, fontweight="bold")
    axa.set_ylim(0, 5); _despine(axa)
    axa.text(0.5, 4.4, "attention\n4d²", ha="center", color=BLUE, fontsize=9, fontweight="bold")
    axa.text(4.5, 4.6, "FFN\n8d²", ha="center", color=PURPLE, fontsize=9, fontweight="bold")

    # (b) GPT-2 small (124M): embeddings vs blocks
    axb = axes[1]
    d, L, V, ctx = 768, 12, 50257, 1024
    emb = V * d
    pos = ctx * d
    per_layer = 12 * d * d + 13 * d  # ~12 d^2 + biases/norms (approx)
    blocks = L * per_layer
    total = emb + pos + blocks  # tied head -> no extra
    sizes = [emb, pos, blocks]
    labels = [f"token emb\n(tied w/ head)\n{emb/1e6:.1f}M",
              f"pos emb\n{pos/1e6:.2f}M",
              f"{L} blocks\n{blocks/1e6:.1f}M"]
    axb.pie(sizes, labels=labels, colors=[GREEN, AMBER, PURPLE],
            autopct=lambda p: f"{p:.0f}%", textprops={"fontsize": 9}, startangle=90,
            wedgeprops=dict(edgecolor="white"))
    axb.set_title(f"GPT-2 small param split — {total/1e6:.0f}M total", fontsize=12, fontweight="bold")
    fig.suptitle("Where the parameters live in a decoder-only model",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/decoder_param_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote decoder_param_breakdown.png")


# ---- 4. MEASURED next-token logits from a small GPT-2 -----------------------
def measured_logits():
    prompt = "The capital of France is"
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1]          # last-position logits over vocab
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, 8)
        words = [tok.decode([i]).strip() or "·" for i in topk.indices.tolist()]
        vals = topk.values.tolist()
        measured = True
    except Exception as e:                              # offline / no weights: constructed
        print("  (measured GPT-2 unavailable, using illustrative values):", e)
        words = ["Paris", "the", "a", "located", "now", "France", "one", "in"]
        vals = [0.41, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.018]
        measured = False

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    y = np.arange(len(words))[::-1]
    bars = ax.barh(y, vals, color=[GREEN if i == 0 else BLUE for i in range(len(words))],
                   edgecolor="white")
    for yi, v, w in zip(y, vals, words):
        ax.text(v + max(vals) * 0.01, yi, f"{v*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([f"'{w}'" for w in words], fontsize=10)
    ax.set_xlabel("next-token probability (softmax of LM-head logits)")
    tag = "GPT-2 (124M), measured" if measured else "illustrative (GPT-2 offline)"
    ax.set_title(f"LM head output for  “{prompt} __”   —  {tag}", fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.18); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/decoder_logits.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote decoder_logits.png")


# ---- 5. GPT lineage scaling --------------------------------------------------
def scaling_lineage():
    models = [
        ("GPT-1\n2018", 117e6, "generative pretrain\n+ fine-tune", BLUE),
        ("GPT-2\n2019", 1.5e9, "zero-shot,\ncoherent text", PURPLE),
        ("GPT-3\n2020", 175e9, "in-context\nlearning emerges", GREEN),
        ("GPT-4\n2023", 1.0e12, "reasoning,\nmultimodal*", AMBER),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(models))
    vals = [m[1] for m in models]
    colors = [m[3] for m in models]
    ax.bar(x, vals, 0.55, color=colors, edgecolor="white")
    ax.set_yscale("log")
    for i, (name, v, note, c) in enumerate(models):
        label = f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.0f}M"
        if v >= 1e12:
            label = "~1T*"
        ax.text(i, v * 1.4, label, ha="center", fontweight="bold", fontsize=10)
        ax.text(i, v * 0.045, note, ha="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in models], fontsize=9.5)
    ax.set_ylabel("parameters (log scale)")
    ax.set_title("The GPT lineage: same decoder-only shape, ~1000× scale → emergent abilities",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(5e7, 5e12); _despine(ax)
    ax.text(0.99, 0.02, "*GPT-4 size unofficial / estimated", transform=ax.transAxes,
            ha="right", fontsize=8, color=SLATE, style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/decoder_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote decoder_scaling.png")


if __name__ == "__main__":
    causal_mask()
    arch_compare()
    param_breakdown()
    measured_logits()
    scaling_lineage()
    print("OUT:", OUT)

"""Contextual-embeddings concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 06. NLP/concepts/06-Contextual-Embeddings-ELMo-BERT.md:
  1. ctx_static_vs_contextual.png -- MEASURED. "bank" projected to 2D: layer-0 input
     embedding is ONE fixed point (static); the last-layer contextual vectors split into
     a river cluster and a money/finance cluster. PCA of real BERT vectors.
  2. ctx_mlm_objective.png        -- schematic of the masked-LM objective: mask 15% of
     tokens, feed BIDIRECTIONAL context (left + right) into the encoder, predict the
     masked word from the softmax over the vocabulary.
  3. ctx_bert_vs_gpt.png          -- schematic: BERT (bidirectional encoder, sees both
     sides, [MASK] objective, understanding) vs GPT (causal decoder, sees left only,
     next-token objective, generation).
  4. ctx_layer_probe.png          -- MEASURED. cos(river-bank, money-bank) per BERT layer:
     1.0 at the input embedding (static), diverging through depth as context is mixed in.

Run with the project's Python 3.12 env:
  ~/.uv/envs/ml-py312/bin/python3 tools/gen_contextual_embeddings_diagrams.py
Requires: torch, transformers, scikit-learn, matplotlib. If the BERT download fails the
measured figures fall back to clearly-labelled illustrative data (printed to stderr).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images"))
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Shared: load BERT once (used by the two measured figures). Returns None on failure.
# ---------------------------------------------------------------------------
def _load_bert():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, logging
        logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True).eval()
        return torch, tok, model
    except Exception as e:  # offline / no weights
        print(f"[warn] BERT unavailable ({type(e).__name__}: {e}); using illustrative data", file=sys.stderr)
        return None


def _word_vec(torch, tok, model, sentence, word, layer):
    enc = tok(sentence, return_tensors="pt")
    ids = enc["input_ids"][0]
    with torch.no_grad():
        out = model(**enc)
    pos = (ids == tok.convert_tokens_to_ids(word)).nonzero(as_tuple=True)[0][0]
    return out.hidden_states[layer][0][pos].numpy()


# ---------------------------------------------------------------------------
# 1. Static (one fixed point) vs contextual (context-dependent cluster) -- MEASURED
# ---------------------------------------------------------------------------
def static_vs_contextual():
    bundle = _load_bert()
    river_sents = [
        "I sat on the river bank watching the water.",
        "We fished from the muddy bank of the stream.",
        "Willows leaned over the grassy bank of the creek.",
        "The boat drifted toward the far bank of the river.",
    ]
    money_sents = [
        "I deposited cash at the bank downtown.",
        "The bank approved my mortgage loan.",
        "She works as a teller at the bank.",
        "I withdrew money from the bank yesterday.",
    ]
    if bundle is not None:
        torch, tok, model = bundle
        ctx = np.array(
            [_word_vec(torch, tok, model, s, "bank", -1) for s in river_sents]
            + [_word_vec(torch, tok, model, s, "bank", -1) for s in money_sents]
        )
        static = _word_vec(torch, tok, model, river_sents[0], "bank", 0)  # layer-0 = context-free
        from sklearn.decomposition import PCA
        pts = PCA(n_components=2).fit_transform(np.vstack([ctx, static]))
        ctx2d, static2d = pts[:-1], pts[-1]
        measured = True
    else:  # illustrative fallback
        rng = np.random.default_rng(0)
        ctx2d = np.vstack([rng.normal([-1.6, 0.4], 0.25, (4, 2)), rng.normal([1.6, -0.3], 0.25, (4, 2))])
        static2d = np.array([0.0, 0.0]); measured = True if False else False

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    riv, mon = ctx2d[:4], ctx2d[4:]
    ax.scatter(riv[:, 0], riv[:, 1], s=120, color=BLUE, edgecolor="white", zorder=4, label='"bank" — river sense (contextual)')
    ax.scatter(mon[:, 0], mon[:, 1], s=120, color=GREEN, edgecolor="white", zorder=4, label='"bank" — money sense (contextual)')
    ax.scatter([static2d[0]], [static2d[1]], s=360, marker="*", color=RED, edgecolor="white",
               zorder=5, label='static word2vec/GloVe (one fixed vector)')
    for cluster, col, lab in [(riv, BLUE, "river"), (mon, GREEN, "money")]:
        c = cluster.mean(0)
        ax.annotate(lab, c, textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=11, fontweight="bold", color=col)
    ax.annotate("static: ONE vector\n(both senses collapse here)", static2d,
                textcoords="offset points", xytext=(0, -38), ha="center", fontsize=9.5,
                color=RED, fontweight="bold")
    ax.set_title('Static vs contextual: the SAME word "bank" in 8 sentences (BERT, PCA to 2D)',
                 fontsize=13.0, fontweight="bold")
    ax.set_xlabel("PC 1   (measured: BERT-base last-layer vectors)" if measured else "(illustrative)")
    ax.set_yticks([]); _despine(ax, keep_left=False); ax.tick_params(length=0)
    # pad limits so the star + annotation never collide with points/legend
    xs_all = np.append(ctx2d[:, 0], static2d[0]); ys_all = np.append(ctx2d[:, 1], static2d[1])
    ax.set_xlim(xs_all.min() - 2.2, xs_all.max() + 2.2)
    ax.set_ylim(ys_all.min() - 1.6, ys_all.max() + 2.4)
    ax.legend(loc="lower center", fontsize=9.0, framealpha=0.9, ncol=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/ctx_static_vs_contextual.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ctx_static_vs_contextual.png")


# ---------------------------------------------------------------------------
# 2. The masked-LM objective -- schematic
# ---------------------------------------------------------------------------
def mlm_objective():
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.4); ax.axis("off")
    tokens = ["I", "deposited", "cash", "at", "the", "[MASK]"]
    cols = [SLATE, SLATE, SLATE, SLATE, SLATE, RED]
    xs = np.linspace(1.2, 9.0, len(tokens))
    # input token row
    for x, t, c in zip(xs, tokens, cols):
        ax.add_patch(FancyBboxPatch((x - 0.7, 0.6), 1.4, 0.9, boxstyle="round,pad=0.04",
                                    fc=c, ec="white", lw=1.2))
        ax.text(x, 1.05, t, ha="center", va="center", color="#fff", fontsize=10.0, fontweight="bold")
    ax.text(xs.mean(), 0.05, "input tokens (15% chosen for masking → here: 'bank')",
            ha="center", fontsize=9, color=SLATE)
    # encoder block
    ax.add_patch(FancyBboxPatch((0.6, 2.55), 9.4, 1.55, boxstyle="round,pad=0.05", fc=PURPLE, ec="white", lw=1.4))
    ax.text(5.3, 3.3, "BERT encoder — every token attends to EVERY other\n(bidirectional self-attention)",
            ha="center", va="center", color="#fff", fontsize=10.5, fontweight="bold")
    # input -> encoder arrows
    for x in xs:
        ax.add_patch(FancyArrowPatch((x, 1.55), (x, 2.5), arrowstyle="->", color=SLATE, lw=1.3, mutation_scale=11))
    # bidirectional context band ABOVE the encoder (no overlap with the label)
    ax.add_patch(FancyArrowPatch((1.0, 4.45), (9.6, 4.45), arrowstyle="<->", color=AMBER, lw=2.2, mutation_scale=14))
    ax.text(5.3, 4.75, "left context  ⟷  right context  both feed the masked position",
            ha="center", fontsize=9.5, color=AMBER, fontweight="bold")
    # prediction head over the masked column
    mx = xs[-1]
    ax.add_patch(FancyArrowPatch((mx, 4.15), (mx, 5.55), arrowstyle="->", color=RED, lw=2.0, mutation_scale=14))
    ax.text(mx + 0.25, 4.85, "predict the\nmasked word", ha="left", va="center", fontsize=8.8,
            color=RED, fontweight="bold")
    bx = min(mx, 8.0)  # keep the box inside the canvas
    ax.add_patch(FancyBboxPatch((bx - 1.85, 5.6), 3.7, 2.4, boxstyle="round,pad=0.05", fc=GREEN, ec="white", lw=1.4))
    ax.text(bx, 7.6, "softmax over the\n~30k-word vocabulary", ha="center", color="#fff",
            fontsize=9.3, fontweight="bold")
    preds = [("bank", 0.43), ("atm", 0.19), ("hotel", 0.06)]
    for i, (w, p) in enumerate(preds):
        ax.text(bx, 6.85 - i * 0.4, f"{w:>6}   {p:.2f}", ha="center", color="#fff",
                fontsize=9.5, fontweight="bold" if i == 0 else "normal")
    ax.set_title("The masked language model (MLM) objective: predict hidden words from BOTH sides",
                 fontsize=13.0, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/ctx_mlm_objective.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ctx_mlm_objective.png")


# ---------------------------------------------------------------------------
# 3. BERT (bidirectional encoder) vs GPT (causal decoder) -- schematic
# ---------------------------------------------------------------------------
def bert_vs_gpt():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    toks = ["the", "river", "bank", "was", "muddy"]
    for ax, title, mode, col in [(axes[0], "BERT — bidirectional encoder", "bi", BLUE),
                                 (axes[1], "GPT — causal (left-to-right) decoder", "causal", GREEN)]:
        ax.set_xlim(-0.5, len(toks) - 0.5); ax.set_ylim(-0.5, 4.2); ax.axis("off")
        focus = 2  # "bank"
        for i, t in enumerate(toks):
            c = AMBER if i == focus else SLATE
            ax.add_patch(FancyBboxPatch((i - 0.42, 0.0), 0.84, 0.7, boxstyle="round,pad=0.03",
                                        fc=c, ec="white", lw=1.1))
            ax.text(i, 0.35, t, ha="center", va="center", color="#fff", fontsize=10, fontweight="bold")
        # which tokens "bank" can see
        seen = range(len(toks)) if mode == "bi" else range(focus + 1)
        for j in seen:
            if j == focus:
                continue
            ax.add_patch(FancyArrowPatch((focus, 0.9), (j, 0.9), arrowstyle="-|>",
                                         connectionstyle="arc3,rad=0.45", color=col, lw=1.6, mutation_scale=11))
        objective = "[MASK] → predict hidden token\n(bidirectional · understanding)" if mode == "bi" else "→ predict NEXT token\n(causal · generation)"
        ax.add_patch(FancyBboxPatch((len(toks) / 2 - 2.1, 2.95), 3.2, 0.95, boxstyle="round,pad=0.04",
                                    fc=col, ec="white", lw=1.2))
        ax.text(len(toks) / 2 - 0.5, 3.42, objective, ha="center", va="center", fontsize=9.0,
                color="#fff", fontweight="bold")
        ax.text(focus, 1.9, '"bank" attends to:', ha="center", fontsize=9, color="#333")
        ax.text(len(toks) / 2 - 0.5, -0.35,
                "both left AND right context" if mode == "bi" else "ONLY left context (future masked out)",
                ha="center", fontsize=9, color=col, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
    fig.suptitle("Same word, opposite designs: encoder reads both sides; decoder reads only the past",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/ctx_bert_vs_gpt.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ctx_bert_vs_gpt.png")


# ---------------------------------------------------------------------------
# 4. Per-layer probe: cos(river-bank, money-bank) by layer -- MEASURED
# ---------------------------------------------------------------------------
def layer_probe():
    bundle = _load_bert()
    s1 = "I sat on the river bank watching the water."
    s2 = "I deposited cash at the bank downtown."
    if bundle is not None:
        torch, tok, model = bundle
        import torch.nn.functional as F
        sims = []
        for L in range(13):
            a = torch.tensor(_word_vec(torch, tok, model, s1, "bank", L))
            b = torch.tensor(_word_vec(torch, tok, model, s2, "bank", L))
            sims.append(F.cosine_similarity(a, b, dim=0).item())
        measured = True
    else:
        sims = [1.0, 0.73, 0.64, 0.56, 0.56, 0.54, 0.53, 0.47, 0.45, 0.41, 0.42, 0.47, 0.48]; measured = False

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    layers = list(range(13))
    ax.plot(layers, sims, "-o", color=PURPLE, lw=2.4, markersize=7, markerfacecolor=AMBER, markeredgecolor="white")
    ax.axhline(1.0, color=SLATE, ls="--", lw=1.2)
    ax.annotate("layer 0 = input embedding:\nidentical (static, context-free)", (0, 1.0),
                textcoords="offset points", xytext=(26, -30), fontsize=9.5, color=SLATE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SLATE))
    lo = int(np.argmin(sims))
    ax.annotate(f"most disambiguated\n(cos ≈ {sims[lo]:.2f})", (lo, sims[lo]),
                textcoords="offset points", xytext=(-4, 22), ha="center", fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("BERT layer (0 = input embeddings, 12 = final)")
    ax.set_ylabel('cos( "bank" river , "bank" money )')
    ax.set_title("Context is mixed in with DEPTH: the two senses of 'bank' pull apart layer by layer"
                 if measured else "(illustrative) two senses pull apart with depth",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(0.3, 1.08); ax.set_xticks(layers); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/ctx_layer_probe.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ctx_layer_probe.png")


if __name__ == "__main__":
    static_vs_contextual()
    mlm_objective()
    bert_vs_gpt()
    layer_probe()
    print("all contextual-embeddings diagrams written to", OUT)

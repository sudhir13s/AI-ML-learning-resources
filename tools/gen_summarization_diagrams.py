"""Text-Summarization concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 06. NLP/concepts/13-Text-Summarization.md:
  1. summ_extractive_vs_abstractive.png -- schematic: extractive SELECTS sentences
     verbatim from the source; abstractive GENERATES new text that may paraphrase.
  2. summ_textrank_graph.png -- the MEASURED TextRank/LexRank sentence-similarity graph
     on a tiny 5-sentence document, node sizes = PageRank centrality, top-2 highlighted.
  3. summ_pointer_generator.png -- schematic of the pointer-generator copy/generate
     soft switch p_gen mixing the vocabulary distribution and the copy (attention)
     distribution into the final distribution.
  4. summ_rouge_compare.png -- MEASURED ROUGE-1/2/L of an extractive (TextRank top-2)
     summary vs an abstractive (distilbart-cnn) summary against a reference, on a tiny doc.

Run with:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_summarization_diagrams.py
If the HF model download fails, figure 4 falls back to extractive-only measured ROUGE
and prints a note (and the page text says the same).
"""
import os
import re
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# The tiny shared 5-sentence document used by the measured figures (and page).
# ---------------------------------------------------------------------------
DOC = [
    "Solar power capacity grew sharply across Europe last year.",          # S1
    "Solar energy installations expanded rapidly throughout Europe in 2024.",  # S2
    "Engineers warn the aging power grid struggles to absorb the new supply.",  # S3
    "Grid operators say the network cannot easily handle the added solar load.",  # S4
    "A local bakery announced a new sourdough recipe on Tuesday.",          # S5
]
REFERENCE = "Solar power grew fast in Europe but the grid struggles to absorb it."


def _tfidf_cosine_graph(sentences):
    """Build the sentence-similarity graph exactly as TextRank/LexRank do:
    TF-IDF vectors, cosine similarity edges. Returns (G, scores, sim)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    G = nx.from_numpy_array(sim)
    # PageRank with the cosine weights = TextRank centrality
    scores = nx.pagerank(G, weight="weight")
    return G, scores, sim


# ---------------------------------------------------------------------------
# Figure 1: extractive vs abstractive (schematic)
# ---------------------------------------------------------------------------
def fig_extractive_vs_abstractive():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    src_lines = ["S1  Solar capacity grew sharply in Europe.",
                 "S2  Installations expanded rapidly in 2024.",
                 "S3  The aging grid struggles to absorb supply.",
                 "S4  Operators say the network can't handle it.",
                 "S5  A bakery announced a new sourdough recipe."]

    # ----- Extractive panel -----
    ax = axes[0]
    ax.set_title("Extractive: SELECT salient sentences verbatim",
                 fontsize=13.5, fontweight="bold", color=BLUE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    keep = {0, 2}  # selected sentences
    for i, ln in enumerate(src_lines):
        y = 8.6 - i * 1.05
        chosen = i in keep
        fc = GREEN if chosen else "#E8EAED"
        tc = "#fff" if chosen else "#555"
        box = FancyBboxPatch((0.3, y - 0.36), 9.4, 0.78,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.4, edgecolor=GREEN if chosen else "#BBB",
                             facecolor=fc)
        ax.add_patch(box)
        ax.text(0.55, y, ln, va="center", ha="left", fontsize=10.0,
                color=tc, fontweight="bold" if chosen else "normal")
        if chosen:
            ax.text(9.95, y, "✓", va="center", ha="right", fontsize=14,
                    color=GREEN, fontweight="bold")
    ax.text(5, 2.7, "Summary = S1 + S3, copied word-for-word",
            ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
    ax.text(5, 1.9, "100% faithful by construction · cannot fuse or compress",
            ha="center", fontsize=9.5, color="#555", style="italic")
    box = FancyBboxPatch((0.6, 0.5), 8.8, 1.0,
                         boxstyle="round,pad=0.05,rounding_size=0.1",
                         linewidth=1.6, edgecolor=GREEN, facecolor="#EAF4EF")
    ax.add_patch(box)
    ax.text(5, 1.0, "Solar capacity grew sharply in Europe. The aging grid\nstruggles to absorb supply.",
            ha="center", va="center", fontsize=9.5, color="#1E6A4A")

    # ----- Abstractive panel -----
    ax = axes[1]
    ax.set_title("Abstractive: GENERATE new text (may paraphrase / fuse)",
                 fontsize=13.5, fontweight="bold", color=PURPLE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    for i, ln in enumerate(src_lines):
        y = 8.6 - i * 1.05
        box = FancyBboxPatch((0.3, y - 0.36), 9.4, 0.78,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.2, edgecolor="#BBB", facecolor="#E8EAED")
        ax.add_patch(box)
        ax.text(0.55, y, ln, va="center", ha="left", fontsize=10.0, color="#555")
    # encoder-decoder blob
    enc = FancyBboxPatch((3.1, 2.85), 3.8, 0.95,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         linewidth=1.6, edgecolor=PURPLE, facecolor=PURPLE)
    ax.add_patch(enc)
    ax.text(5.0, 3.32, "seq2seq encoder–decoder", ha="center", va="center",
            fontsize=10.5, color="#fff", fontweight="bold")
    ax.annotate("", xy=(5.0, 3.85), xytext=(5.0, 4.05),
                arrowprops=dict(arrowstyle="-", color="#999"))
    arr = FancyArrowPatch((5.0, 2.8), (5.0, 1.6), arrowstyle="-|>",
                          mutation_scale=18, color=PURPLE, lw=2.0)
    ax.add_patch(arr)
    box = FancyBboxPatch((0.6, 0.5), 8.8, 1.0,
                         boxstyle="round,pad=0.05,rounding_size=0.1",
                         linewidth=1.6, edgecolor=PURPLE, facecolor="#EEEAF4")
    ax.add_patch(box)
    ax.text(5, 1.0, "Solar power surged across Europe, but the strained grid\ncan barely keep up.",
            ha="center", va="center", fontsize=9.5, color="#4D3A7A", fontweight="bold")
    ax.text(5, 2.0, "fuses S1+S3, coins “surged”, “strained” (not in source)",
            ha="center", fontsize=9.0, color=RED, style="italic")

    fig.suptitle("Two paradigms of summarization", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/summ_extractive_vs_abstractive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote summ_extractive_vs_abstractive.png")


# ---------------------------------------------------------------------------
# Figure 2: MEASURED TextRank sentence-similarity graph
# ---------------------------------------------------------------------------
def fig_textrank_graph():
    G, scores, sim = _tfidf_cosine_graph(DOC)
    labels = {i: f"S{i+1}" for i in range(len(DOC))}
    ranked = sorted(scores, key=scores.get, reverse=True)
    top2 = set(ranked[:2])

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    pos = nx.spring_layout(G, seed=7, weight="weight", k=1.4)

    # edges: width & alpha by cosine similarity
    for u, v in G.edges():
        w = sim[u, v]
        if w < 1e-6:
            continue
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=SLATE, lw=0.6 + 6.5 * w, alpha=0.25 + 0.6 * w, zorder=1)
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        if w > 0.05:
            ax.text(mx, my, f"{w:.2f}", fontsize=8.5, color="#444", zorder=3,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # nodes: size & color by PageRank centrality
    smax = max(scores.values())
    for i in G.nodes():
        s = scores[i]
        size = 1400 + 9000 * (s)
        color = GREEN if i in top2 else BLUE
        ax.scatter(*pos[i], s=size, color=color, edgecolor="#fff", linewidth=2.0, zorder=2)
        ax.text(*pos[i], f"{labels[i]}\n{s:.3f}", ha="center", va="center",
                fontsize=10.5, color="#fff", fontweight="bold", zorder=4)

    ax.set_title("TextRank/LexRank: PageRank centrality on a TF-IDF cosine sentence graph",
                 fontsize=13.0, fontweight="bold")
    # side legend with the actual sentences
    leg = "\n".join([f"S{i+1}: {DOC[i][:46]}{'…' if len(DOC[i])>46 else ''}"
                     for i in range(len(DOC))])
    ax.text(0.5, -0.02, leg, transform=ax.transAxes, fontsize=8.4, color="#333",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4F5F7", ec="#DDD"))
    ax.text(0.0, 1.0, f"top-2 selected: S{ranked[0]+1}, S{ranked[1]+1}  (highest centrality)",
            transform=ax.transAxes, fontsize=10, color=GREEN, fontweight="bold", va="top")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT}/summ_textrank_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote summ_textrank_graph.png  scores=",
          {f"S{i+1}": round(scores[i], 3) for i in range(len(DOC))},
          "top2=", [f"S{i+1}" for i in ranked[:2]])
    return scores, sim, ranked


# ---------------------------------------------------------------------------
# Figure 3: pointer-generator copy/generate switch (schematic)
# ---------------------------------------------------------------------------
def fig_pointer_generator():
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Pointer-generator: a soft switch $p_{gen}$ between GENERATING and COPYING",
                 fontsize=13.5, fontweight="bold")

    def box(x, y, w, h, text, fc, ec, tc="#fff", fs=10.0, bold=True):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

    # decoder state
    box(0.4, 4.4, 2.6, 1.2, "decoder state\n$s_t$ + context $h_t^*$", PURPLE, "#4D3A7A")
    # vocab distribution
    box(4.0, 7.4, 3.4, 1.3, "$P_{vocab}$\ngenerate from vocabulary", BLUE, "#2A5B86")
    # copy distribution
    box(4.0, 1.2, 3.4, 1.3, "attention $a_t$ over source\n= copy distribution", AMBER, "#6A5518")
    # p_gen gate
    box(4.0, 4.45, 3.4, 1.1, "$p_{gen}=\\sigma(w^\\top[s_t,h_t^*,x_t])$", GREEN, "#1E6A4A")

    # arrows from decoder
    for ty in (8.05, 5.0, 1.85):
        arr = FancyArrowPatch((3.0, 5.0), (4.0, ty), arrowstyle="-|>",
                              mutation_scale=14, color="#888", lw=1.6,
                              connectionstyle="arc3,rad=0.0")
        ax.add_patch(arr)

    # final distribution
    box(8.6, 4.2, 3.0, 1.6,
        "$P(w)=p_{gen}P_{vocab}(w)$\n$+(1{-}p_{gen})\\sum_{i:w_i=w} a_t^i$",
        NAVY, "#1A4B70", fs=9.6)

    a1 = FancyArrowPatch((7.4, 8.05), (9.9, 5.8), arrowstyle="-|>",
                         mutation_scale=14, color=BLUE, lw=2.0,
                         connectionstyle="arc3,rad=-0.2")
    ax.add_patch(a1)
    ax.text(8.7, 7.4, "$\\times\\,p_{gen}$", fontsize=10, color=BLUE, fontweight="bold")
    a2 = FancyArrowPatch((7.4, 1.85), (9.9, 4.2), arrowstyle="-|>",
                         mutation_scale=14, color=AMBER, lw=2.0,
                         connectionstyle="arc3,rad=0.2")
    ax.add_patch(a2)
    ax.text(8.5, 2.5, "$\\times\\,(1{-}p_{gen})$", fontsize=10, color="#6A5518", fontweight="bold")
    a3 = FancyArrowPatch((7.4, 5.0), (8.6, 5.0), arrowstyle="-|>",
                         mutation_scale=14, color=GREEN, lw=2.0)
    ax.add_patch(a3)

    ax.text(6.0, 0.45,
            "$p_{gen}\\to1$: write a vocab word (paraphrase).   "
            "$p_{gen}\\to0$: copy a source word → solves OOV / rare names.",
            ha="center", fontsize=9.8, color="#333", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/summ_pointer_generator.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote summ_pointer_generator.png")


# ---------------------------------------------------------------------------
# Figure 4: MEASURED ROUGE extractive vs abstractive
# ---------------------------------------------------------------------------
def _rouge(pred, ref):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = sc.score(ref, pred)
    return {k: r[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


def fig_rouge_compare(scores, ranked):
    # Extractive summary = top-2 TextRank sentences, in document order
    top2_idx = sorted(ranked[:2])
    extractive = " ".join(DOC[i] for i in top2_idx)

    # Abstractive summary via distilbart-cnn (small). Fall back if download fails.
    abstractive = None
    note = ""
    long_doc = " ".join(DOC[:4])  # the on-topic part (drop the bakery distractor)
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        mname = "sshleifer/distilbart-cnn-12-6"
        tok = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForSeq2SeqLM.from_pretrained(mname)
        ids = tok(long_doc, return_tensors="pt", truncation=True)
        out = model.generate(**ids, max_length=40, min_length=12, num_beams=4)
        abstractive = tok.decode(out[0], skip_special_tokens=True).strip()
        note = "abstractive = distilbart-cnn-12-6 (measured)"
    except Exception as e:  # pragma: no cover - network dependent
        note = f"abstractive model unavailable ({type(e).__name__}); extractive-only shown"
        print("WARN:", note)

    ext_r = _rouge(extractive, REFERENCE)
    abs_r = _rouge(abstractive, REFERENCE) if abstractive else None

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    keys = ["rouge1", "rouge2", "rougeL"]
    x = np.arange(len(metrics))
    if abs_r:
        w = 0.36
        ax.bar(x - w / 2, [ext_r[k] for k in keys], w, color=GREEN,
               edgecolor="#1E6A4A", label="extractive (TextRank top-2)")
        ax.bar(x + w / 2, [abs_r[k] for k in keys], w, color=PURPLE,
               edgecolor="#4D3A7A", label="abstractive (distilbart-cnn)")
        for i, k in enumerate(keys):
            ax.text(i - w / 2, ext_r[k] + 0.01, f"{ext_r[k]:.2f}", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
            ax.text(i + w / 2, abs_r[k] + 0.01, f"{abs_r[k]:.2f}", ha="center", fontsize=9.5, color=PURPLE, fontweight="bold")
    else:
        ax.bar(x, [ext_r[k] for k in keys], 0.5, color=GREEN,
               edgecolor="#1E6A4A", label="extractive (TextRank top-2)")
        for i, k in enumerate(keys):
            ax.text(i, ext_r[k] + 0.01, f"{ext_r[k]:.2f}", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("ROUGE F1 (vs reference)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Measured ROUGE vs a reference summary", fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.text(0.0, -0.20, f"reference: “{REFERENCE}”", transform=ax.transAxes,
            fontsize=8.6, color="#333", style="italic")
    ax.text(0.0, -0.27, f"extractive: “{extractive[:78]}…”", transform=ax.transAxes,
            fontsize=8.6, color=GREEN)
    if abstractive:
        ax.text(0.0, -0.34, f"abstractive: “{abstractive[:78]}{'…' if len(abstractive)>78 else ''}”",
                transform=ax.transAxes, fontsize=8.6, color=PURPLE)
    _despine(ax)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(f"{OUT}/summ_rouge_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote summ_rouge_compare.png")
    print("  extractive:", extractive)
    print("  extractive ROUGE:", {k: round(v, 3) for k, v in ext_r.items()})
    if abstractive:
        print("  abstractive:", abstractive)
        print("  abstractive ROUGE:", {k: round(v, 3) for k, v in abs_r.items()})


if __name__ == "__main__":
    fig_extractive_vs_abstractive()
    scores, sim, ranked = fig_textrank_graph()
    fig_pointer_generator()
    fig_rouge_compare(scores, ranked)
    print("\nAll diagrams written to", OUT)

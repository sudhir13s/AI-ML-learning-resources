"""Reproducible figure generator for 07-Sentence-and-Document-Embeddings.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the sentences, the SBERT (or synthetic-fallback) vectors, the cosines, the SIF weights,
the pooling Spearman scores, and the STS correlation are all IMPORTED from `sentence_embeddings.py`,
so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_07.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `se_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced (measured = from the live backend; illustrative = a labelled schematic):
  se_mean_pool_orderblind.png  -- measured: mean-pool gives "dog bit man"="man bit dog"; encoder splits
  se_sif_weights.png           -- measured: SIF weight a/(a+p(w)) curve + the "the cat sat" worked example
  se_anisotropy.png            -- illustrative: the narrow-cone vs isotropic geometry, schematic
  se_paraphrase_separation.png -- measured: paraphrase vs unrelated cosine, encoder vs mean-pool
  se_pooling_compare.png       -- measured: STS Spearman for mean vs max vs CLS pooling
  se_projection.png            -- measured: PCA of 3 topics x 3 paraphrases -> clusters separate
  se_search.png                -- measured: semantic-search cosine ranking over the support corpus
  se_sts_spearman.png          -- measured: cosine vs gold scatter + Spearman on the STS-like set
  se_triplet_loss.png          -- illustrative: triplet loss = hinge on (d_an - d_ap), the two cases
  se_bi_vs_cross.png           -- illustrative: bi-encoder (cache+cosine) vs cross-encoder (joint)
  se_doc2vec.png               -- illustrative: PV-DM (order-aware) vs PV-DBOW (bag) paragraph vectors
  se_colbert_maxsim.png        -- illustrative: ColBERT MaxSim heatmap (query x doc tokens, row-max summed)

When the real model is present these are MEASURED from all-MiniLM-L6-v2; offline they fall back to the
deterministic synthetic encoder and the figures still render (titles note the backend). Verified on
Python 3.12 / numpy 2.x / matplotlib 3.x, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from sentence_embeddings import (
    ORDER_A,
    ORDER_B,
    PARAPHRASE_SET,
    SEARCH_QUERY,
    SIF_A,
    colbert_maxsim,
    load_encoder,
    order_blindness_demo,
    paraphrase_separation,
    pooling_comparison,
    semantic_search,
    sif_weight,
    sif_worked_example,
    sts_evaluation,
    topic_projection,
    triplet_loss,
)

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150

# One shared backend for every figure, so all numbers come from the same model load.
ENCODER = load_encoder()
BACKEND_TAG = "all-MiniLM-L6-v2, measured" if ENCODER.is_real else "synthetic fallback"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}")


def fig_mean_pool_orderblind() -> None:
    """Measured: mean-pool gives the two reversed sentences the SAME vector; the encoder separates them."""
    ob = order_blindness_demo(ENCODER)
    labels = ["mean-pool of\nstatic word vectors", f"trained encoder\n({BACKEND_TAG.split(',')[0]})"]
    values = [ob["mean_pool_cosine"], ob["encoder_cosine"]]
    colors = [RED, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", color=INK,
                fontsize=12, fontweight="bold")
    ax.axhline(1.0, color=SLATE, lw=0.9, ls=":")
    ax.annotate("perfectly identical:\nthe two vectors are equal",
                xy=(0, 1.0), xytext=(0.0, 1.16), ha="center", color=RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_ylim(0, 1.30)
    ax.set_ylabel(f'cosine("{ORDER_A}",\n"{ORDER_B}")', fontsize=9.5)
    ax.set_title(
        "Same words, opposite meaning: mean-pooling is order-blind\n"
        "averaging is permutation-invariant, so it cannot tell the two apart; a real encoder can",
        fontsize=11,
    )
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, "se_mean_pool_orderblind.png")


def fig_sif_weights() -> None:
    """Measured: the SIF weight curve a/(a+p(w)) across frequencies, with the 'the cat sat' words marked.

    The inset bars show the worked example: plain mean's x-coordinate (dominated by the stopword 'the')
    vs the SIF-weighted average's x-coordinate (pulled back toward content words).
    """
    ex = sif_worked_example()
    probs = np.logspace(-4.2, -0.6, 200)  # p(w) from very rare to very common
    weights = np.array([sif_weight(p) for p in probs])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    ax.plot(probs, weights, color=PURPLE, lw=2.6, zorder=3)
    word_colors = {"the": RED, "cat": GREEN, "sat": BLUE}
    for w, p, raw in zip(ex["words"], ex["probs"], ex["weights"]):
        ax.scatter([p], [raw], s=120, color=word_colors[w], edgecolor="white", linewidth=1.4, zorder=4)
        ax.annotate(f"'{w}'\np={p:g}\nw={raw:.3f}", xy=(p, raw),
                    xytext=(p * (0.32 if w == "the" else 1.6), raw + (0.10 if w == "the" else 0.04)),
                    fontsize=8.5, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xscale("log")
    ax.set_xlabel("unigram probability  p(w)   (log scale)")
    ax.set_ylabel(r"SIF weight  $\dfrac{a}{a+p(w)}$" + f"   (a={SIF_A:g})")
    ax.set_ylim(0, 1.05)
    ax.set_title("SIF down-weights frequent words smoothly\ncommon 'the' ~0.02; rare 'cat' ~0.77", fontsize=10.5)
    _style_axis(ax)

    # Right: the worked-example x-coordinate (the stopword 'the' direction) plain vs SIF.
    xs = [ex["plain"][0], ex["sif"][0]]
    bars = ax2.bar(["plain\nmean", "SIF\nweighted"], xs, color=[RED, GREEN], edgecolor="white", width=0.55)
    for bar, v in zip(bars, xs):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", color=INK,
                 fontsize=11, fontweight="bold")
    ax2.set_ylabel("x-coordinate  =  pull toward stopword 'the'")
    ax2.set_ylim(0, 0.46)
    ax2.set_title("'the cat sat': SIF cuts the\nstopword's pull ~4x (0.40 -> 0.10)", fontsize=10)
    _style_axis(ax2)
    ax2.grid(axis="x", visible=False)
    fig.suptitle(f"Smooth inverse frequency (SIF) weighting  ({BACKEND_TAG.split(',')[0]} not needed — pure math)",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "se_sif_weights.png")


def fig_anisotropy() -> None:
    """Illustrative: raw mean-pooled BERT vectors crowd into a narrow cone (all cosines high); a
    fine-tuned encoder spreads them isotropically so cosine recovers meaning.

    Schematic 2-D directions, not measured -- it visualizes WHY anisotropy breaks cosine. The numbers
    (avg random-pair cosine) are the page's quoted ranges, labelled illustrative.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    # Left: anisotropic cone -- vectors clustered around a single direction.
    cone_dir = np.array([1.0, 0.35])
    cone_dir /= np.linalg.norm(cone_dir)
    angles = rng.normal(0, 0.16, 22)  # tight angular spread -> narrow cone
    base = np.arctan2(cone_dir[1], cone_dir[0])
    # Right: isotropic -- vectors spread over all directions.
    iso_angles = rng.uniform(0, 2 * np.pi, 22)
    for ax, angset, title, color, avg in (
        (axes[0], base + angles, "raw mean-pooled BERT: anisotropic cone\nall cosines high — unrelated looks similar", RED, "0.3–0.6"),
        (axes[1], iso_angles, "after SBERT / SimCSE: isotropic\ncosine separates related from unrelated", GREEN, "~0.0"),
    ):
        for a in angset:
            ax.plot([0, np.cos(a)], [0, np.sin(a)], color=color, lw=1.6, alpha=0.7, zorder=2)
            ax.scatter([np.cos(a)], [np.sin(a)], s=26, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([0], [0], s=40, color=INK, zorder=4)
        circ = plt.Circle((0, 0), 1.0, fill=False, color=GRID, lw=1.0)
        ax.add_patch(circ)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.text(0, -1.18, f"avg cosine of RANDOM pairs ≈ {avg}", ha="center", color=color, fontsize=9.5, fontweight="bold")
    fig.suptitle("Anisotropy: why raw BERT sentence vectors are bad for cosine (illustrative)",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "se_anisotropy.png")


def fig_paraphrase_separation() -> None:
    """Measured: paraphrase vs unrelated cosine -- the trained encoder separates them, mean-pool does not."""
    ps = paraphrase_separation(ENCODER)
    groups = ["paraphrase\n(same meaning)", "unrelated\n(different meaning)"]
    encoder_vals = [ps["encoder_paraphrase"], ps["encoder_unrelated"]]
    meanpool_vals = [ps["meanpool_paraphrase"], ps["meanpool_unrelated"]]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - width / 2, meanpool_vals, width, color=SLATE, label="mean-pool of static word vectors")
    ax.bar(x + width / 2, encoder_vals, width, color=GREEN, label=f"trained encoder ({BACKEND_TAG.split(',')[0]})")
    for xi, (m, e) in enumerate(zip(meanpool_vals, encoder_vals)):
        ax.text(xi - width / 2, m + 0.02 * np.sign(m or 1), f"{m:.2f}", ha="center", color=INK, fontsize=9.5)
        ax.text(xi + width / 2, e + 0.02 * np.sign(e or 1), f"{e:.2f}", ha="center", color=INK, fontsize=9.5)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(min(min(meanpool_vals), min(encoder_vals)) - 0.15, 1.05)
    ax.set_title(
        "A trained encoder separates meaning; averaging does not\n"
        "the encoder pushes paraphrases high and unrelated low; mean-pool barely distinguishes them",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, "se_paraphrase_separation.png")


def fig_pooling_compare() -> None:
    """Measured: STS Spearman under mean vs max vs CLS pooling -- mean usually wins."""
    pooling = pooling_comparison(ENCODER)
    strategies = ["mean", "max", "CLS\n(first token)"]
    keys = ["mean", "max", "cls"]
    values = [pooling[k] for k in keys]
    best = int(np.argmax(values))
    colors = [GREEN if i == best else BLUE for i in range(3)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(strategies, values, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012 * np.sign(v or 1), f"{v:.3f}",
                ha="center", color=INK, fontsize=11, fontweight="bold")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("STS Spearman rank correlation\n(cosine vs gold similarity)")
    ax.set_ylim(min(0, min(values)) - 0.05, max(values) + 0.12)
    note = "mean pooling wins" if keys[best] == "mean" else f"{keys[best]} wins on this tiny set"
    ax.set_title(
        f"Pooling strategy changes the vector — and the score ({BACKEND_TAG})\n"
        f"on this small STS-like set, {note} (mean is the robust default at scale)",
        fontsize=10.5,
    )
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, "se_pooling_compare.png")


def fig_projection() -> None:
    """Measured: PCA of 3 topics x 3 paraphrases -- paraphrases cluster, topics separate."""
    coords, labels, sents = topic_projection(ENCODER)
    topic_colors = {"weather": BLUE, "finance": GREEN, "cooking": AMBER}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for topic in ("weather", "finance", "cooking"):
        idx = [i for i, t in enumerate(labels) if t == topic]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=150, color=topic_colors[topic],
                   edgecolor="white", linewidth=1.4, zorder=3, label=topic)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(
        f"Meaning becomes geometry: 3 topics x 3 paraphrases ({BACKEND_TAG})\n"
        "paraphrases land together; the topics separate — search, clustering & dedup are just geometry here",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="best", fontsize=10, title="topic")
    _style_axis(ax)
    _save(fig, "se_projection.png")


def fig_search() -> None:
    """Measured: cosine ranking of the support corpus to the query (the two password docs rank high)."""
    ranking = semantic_search(ENCODER)
    sims = [s for s, _ in ranking]
    docs = [d for _, d in ranking]
    short = [d[:46] + ("…" if len(d) > 46 else "") for d in docs]
    colors = [GREEN if s > 0.3 else SLATE for s in sims]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ypos = np.arange(len(sims))[::-1]
    bars = ax.barh(ypos, sims, color=colors, edgecolor="white")
    for bar, s in zip(bars, sims):
        off = 0.012 if s >= 0 else -0.012
        ax.text(s + off, bar.get_y() + bar.get_height() / 2, f"{s:+.3f}",
                va="center", ha="left" if s >= 0 else "right", color=INK, fontsize=9.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(short, fontsize=8.5)
    ax.axvline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("cosine similarity to query")
    ax.set_xlim(min(sims) - 0.08, max(sims) + 0.16)
    ax.set_title(
        f'Semantic search: "{SEARCH_QUERY}" ({BACKEND_TAG})\n'
        'the "forgot password" doc ranks 2nd WITHOUT the word "reset" — a meaning match, not a keyword match',
        fontsize=10,
    )
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "se_search.png")


def fig_sts_spearman() -> None:
    """Measured: cosine vs gold scatter over the STS-like set, with the Spearman correlation."""
    sts = sts_evaluation(ENCODER)
    cos = sts["cosines"]
    gold = sts["gold"]
    r = sts["spearman"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(gold, cos, s=110, color=PURPLE, edgecolor="white", linewidth=1.3, zorder=3)
    # A light trend guide (least-squares line) to show the monotone relationship.
    coef = np.polyfit(gold, cos, 1)
    xs = np.linspace(gold.min(), gold.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color=SLATE, lw=1.4, ls="--", zorder=2)
    ax.set_xlabel("gold similarity (human-style label, 0–1)")
    ax.set_ylabel("model cosine similarity")
    ax.set_title(
        f"STS evaluation: cosine tracks human similarity ({BACKEND_TAG})\n"
        f"Spearman rank correlation r = {r:.3f} — ordering similarity correctly is what STS measures",
        fontsize=10.5,
    )
    _style_axis(ax)
    _save(fig, "se_sts_spearman.png")


def fig_triplet_loss() -> None:
    """Illustrative: triplet loss = hinge max(0, d_ap - d_an + margin); the solved vs violated cases."""
    margin = 0.3
    gap = np.linspace(-0.8, 0.8, 200)  # gap = d_an - d_ap (positive = positive is closer = good)
    loss = np.array([max(0.0, -g + margin) for g in gap])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(gap, loss, color=PURPLE, lw=2.8, zorder=3)
    ax.axvline(margin, color=SLATE, lw=1.0, ls=":")
    ax.text(margin + 0.02, 0.9, f"loss = 0 once gap ≥ margin ({margin})", color=INK, fontsize=9.5)
    # Mark the two page worked-example cases (using distance d = 1 - cos).
    solved_gap = 0.884 - 0.013   # d_an - d_ap
    viol_gap = 0.200 - 0.400
    for g, lab, color in (
        (solved_gap, f"solved\nL={triplet_loss(0.013, 0.884):.2f}", GREEN),
        (viol_gap, f"violated\nL={triplet_loss(0.400, 0.200):.2f}", RED),
    ):
        ax.scatter([g], [max(0.0, -g + margin)], s=140, color=color, edgecolor="white", linewidth=1.4, zorder=4)
        ax.annotate(lab, xy=(g, max(0.0, -g + margin)),
                    xytext=(g, max(0.0, -g + margin) + 0.18), ha="center", color=color, fontsize=9.5,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=color))
    ax.set_xlabel("gap  =  d(anchor, negative) − d(anchor, positive)   (larger = positive is closer = better)")
    ax.set_ylabel("triplet loss")
    ax.set_ylim(-0.05, 1.2)
    ax.set_title(
        "Triplet loss is a hinge: zero once the positive clears the margin (illustrative)\n"
        "left of the margin the gradient pulls the positive in and pushes the negative out",
        fontsize=10.5,
    )
    _style_axis(ax)
    _save(fig, "se_triplet_loss.png")


def fig_bi_vs_cross() -> None:
    """Illustrative: bi-encoder (encode each text once -> cache + cosine) vs cross-encoder (joint pass)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))

    # Left: bi-encoder.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    def box(ax, x, y, w, h, text, color, fontsize=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", lw=1.5, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fontsize, zorder=3)
    box(ax, 0.4, 7.6, 3.2, 1.2, "text A", BLUE)
    box(ax, 6.4, 7.6, 3.2, 1.2, "text B", BLUE)
    box(ax, 0.4, 5.4, 3.2, 1.2, "encoder", PURPLE)
    box(ax, 6.4, 5.4, 3.2, 1.2, "encoder\n(same weights)", PURPLE)
    box(ax, 0.4, 3.4, 3.2, 1.1, "vector u\n(cacheable)", SLATE)
    box(ax, 6.4, 3.4, 3.2, 1.1, "vector v\n(cacheable)", SLATE)
    box(ax, 2.9, 1.2, 4.2, 1.2, "cosine(u, v)\none cheap dot product", GREEN)
    for x in (2.0, 8.0):
        ax.annotate("", xy=(x, 6.65), xytext=(x, 7.55), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
        ax.annotate("", xy=(x, 4.55), xytext=(x, 5.35), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(3.7, 2.45), xytext=(2.0, 3.35), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(6.3, 2.45), xytext=(8.0, 3.35), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.set_title("Bi-encoder (SBERT): encode once, compare forever\nindexable → first-stage RETRIEVAL", fontsize=10.5, color=INK)

    # Right: cross-encoder.
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box(ax, 1.6, 7.6, 6.8, 1.2, "[CLS] text A [SEP] text B", BLUE)
    box(ax, 1.6, 5.0, 6.8, 1.6, "one transformer\n(full cross-attention: A sees B)", PURPLE)
    box(ax, 2.9, 2.2, 4.2, 1.2, "one relevance score\n(not a reusable vector)", RED)
    ax.annotate("", xy=(5.0, 6.65), xytext=(5.0, 7.55), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(5.0, 3.45), xytext=(5.0, 4.95), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.set_title("Cross-encoder: feed the PAIR jointly\nmost accurate, not indexable → RERANK", fontsize=10.5, color=INK)

    fig.suptitle("Two ways to compare two texts (illustrative): bi-encoder retrieves, cross-encoder reranks",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "se_bi_vs_cross.png")


def fig_doc2vec() -> None:
    """Illustrative: PV-DM (paragraph vector + context -> next word, order-aware) vs PV-DBOW
    (paragraph vector alone -> sampled words, bag). Two panels of the Doc2Vec training objectives."""

    def box(ax, x, y, w, h, text, color, fontsize=9.0):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", lw=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white",
                fontsize=fontsize, zorder=3)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))

    # --- Left: PV-DM (Distributed Memory) -- paragraph vector + ordered context predict the next word.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box(ax, 0.3, 7.4, 2.7, 1.1, "paragraph\nvector D", PURPLE)
    box(ax, 3.3, 7.4, 1.9, 1.1, "w(t-2)\n'the'", BLUE)
    box(ax, 5.4, 7.4, 1.9, 1.1, "w(t-1)\n'cat'", BLUE)
    box(ax, 7.5, 7.4, 2.0, 1.1, "(position kept:\nordered)", SLATE, fontsize=8.0)
    box(ax, 2.8, 4.6, 4.4, 1.2, "concatenate / average\n→ classifier", AMBER)
    box(ax, 3.7, 2.0, 2.6, 1.2, "predict next\nword: 'sat'", GREEN)
    for x in (1.65, 4.25, 6.35):
        ax.annotate("", xy=(x, 5.8), xytext=(x, 7.35), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.annotate("", xy=(5.0, 3.25), xytext=(5.0, 4.55), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.set_title("PV-DM (Distributed Memory)\nparagraph vector + ORDERED context → next word",
                 fontsize=10.5, color=INK)
    ax.text(5.0, 0.7, "keeps some word ORDER (like CBOW)", ha="center", color=GREEN, fontsize=9.5,
            fontweight="bold")

    # --- Right: PV-DBOW -- paragraph vector ALONE predicts randomly sampled words (bag, no order).
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box(ax, 3.4, 7.4, 3.2, 1.2, "paragraph\nvector D", PURPLE)
    box(ax, 0.5, 3.6, 2.6, 1.1, "'cat'", BLUE)
    box(ax, 3.7, 3.6, 2.6, 1.1, "'sat'", BLUE)
    box(ax, 6.9, 3.6, 2.6, 1.1, "'mat'", BLUE)
    for x in (1.8, 5.0, 8.2):
        ax.annotate("", xy=(x, 4.75), xytext=(5.0, 7.35), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.text(5.0, 2.4, "predict words SAMPLED from the doc\n(no context, no order)", ha="center",
            color=INK, fontsize=9.0)
    ax.set_title("PV-DBOW (Distributed Bag-of-Words)\nparagraph vector ALONE → sampled words",
                 fontsize=10.5, color=INK)
    ax.text(5.0, 0.7, "discards order (like skip-gram), lighter & faster", ha="center", color=RED,
            fontsize=9.5, fontweight="bold")

    fig.suptitle("Doc2Vec / Paragraph Vectors (Le & Mikolov 2014): learn a document vector, two ways (illustrative)",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "se_doc2vec.png")


def fig_colbert_maxsim() -> None:
    """Illustrative: ColBERT late interaction. A query-token x doc-token cosine heatmap; for each query
    token (row) the MAX over doc tokens is boxed (MaxSim), and those maxima sum to the score."""
    cb = colbert_maxsim()
    matrix = cb["matrix"]
    q_tokens = cb["query_tokens"]
    d_tokens = cb["doc_tokens"]
    row_argmax = cb["row_argmax"]
    row_max = cb["row_max"]
    n_q, n_d = matrix.shape

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    im = ax.imshow(matrix, cmap="PuBuGn", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_d))
    ax.set_xticklabels(d_tokens, fontsize=10)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(q_tokens, fontsize=10)
    ax.set_xlabel("document tokens", color=INK)
    ax.set_ylabel("query tokens", color=INK)
    for i in range(n_q):
        for j in range(n_d):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.4 else INK, fontsize=10)
        # Box the per-row MaxSim cell (the chosen doc token for this query token).
        j = int(row_argmax[i])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=RED, lw=3.0, zorder=4))
    ax.set_title(
        "ColBERT late interaction: MaxSim per query token, then sum (illustrative)\n"
        "red box = max similarity over doc tokens for that query token",
        fontsize=10.5, color=INK,
    )
    ax.tick_params(colors=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine(q token, d token)")
    # Annotate the MaxSim sum to the right of the heatmap.
    maxsim_text = "  +  ".join(f"{m:.2f}" for m in row_max)
    ax.text(0.5, -0.30, f"score  =  $\\sum_i \\max_j \\cos(q_i, d_j)$  =  {maxsim_text}  =  {cb['score']:.2f}",
            transform=ax.transAxes, ha="center", color=INK, fontsize=10.5)
    fig.tight_layout()
    _save(fig, "se_colbert_maxsim.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    tag = "real" if ENCODER.is_real else "synthetic fallback"
    print(f"backend: {ENCODER.name} ({tag})")
    fig_mean_pool_orderblind()
    fig_sif_weights()
    fig_anisotropy()
    fig_paraphrase_separation()
    fig_pooling_compare()
    fig_projection()
    fig_search()
    fig_sts_spearman()
    fig_triplet_loss()
    fig_bi_vs_cross()
    fig_doc2vec()
    fig_colbert_maxsim()
    print("done.")


if __name__ == "__main__":
    main()

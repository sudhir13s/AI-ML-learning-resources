"""Reproducible figure generator for 06-Contextual-Embeddings-ELMo-BERT.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the sentences, the BERT (or synthetic-fallback) vectors, the cosines, the layer-probe
curve, and the masked-LM fills are all IMPORTED from `contextual_embeddings.py`, so the figures
cannot silently drift from the prose or the demo. Run:

    python make_figures_06.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `ctx_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced (measured = from the live backend; illustrative = a labelled schematic):
  ctx_static_vs_contextual.png    -- measured: 8 'bank' vectors in 2-D PCA -> river vs money clusters
  ctx_static_vs_contextual_bars.png -- measured: same-sense vs cross-sense cosine, contextual vs static
  ctx_sense_cosine_matrix.png     -- measured: pairwise cosine across the 4 'bank' sentences
  ctx_layer_probe.png             -- measured: cos(river bank, money bank) at every layer (depth curve)
  ctx_mlm_objective.png           -- measured: BERT's top masked-LM predictions for one [MASK] prompt
  ctx_mlm_80_10_10.png            -- illustrative: the 15% selection and 80/10/10 corruption split
  ctx_elmo_layer_weighting.png    -- illustrative: ELMo's learned softmax layer weights, by task
  ctx_bert_vs_gpt.png             -- illustrative: bidirectional (BERT) vs causal (GPT) attention masks

When the real model is present these are MEASURED from BERT-base; offline they fall back to the
deterministic synthetic model and the figures still render (titles note the backend). Verified on
Python 3.12 / numpy 2.x / matplotlib 3.x, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from contextual_embeddings import (
    LOAN,
    MLM_PROMPTS,
    MONEY,
    MONEY2,
    PROBE_WORD,
    RIVER,
    bert_config,
    collect_sense_vectors,
    cosine,
    layer_probe,
    load_contextual_model,
    pca_2d,
    sense_cosines,
    static_baseline_point,
    top_fill,
    word_vector,
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
BACKEND = load_contextual_model()
BACKEND_TAG = "BERT-base, measured" if BACKEND.is_real else "synthetic fallback"


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


def fig_static_vs_contextual() -> None:
    """Measured: 8 'bank' vectors in 2-D PCA split into a river cluster and a money cluster.

    A static embedding gives ONE vector for 'bank' regardless of sentence -- drawn as the red star
    (the centroid of the contextual vectors); it cannot be in two clusters at once.
    """
    matrix, senses, _ = collect_sense_vectors(BACKEND)
    coords = pca_2d(matrix)
    static_pt = static_baseline_point(matrix)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for sense, color, label in (("river", BLUE, "'bank' (river sense)"), ("money", GREEN, "'bank' (money sense)")):
        idx = [i for i, s in enumerate(senses) if s == sense]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=130, color=color, edgecolor="white",
            linewidth=1.4, zorder=3, label=label,
        )
    ax.scatter(
        [static_pt[0]], [static_pt[1]], marker="*", s=520, color=RED, edgecolor="white",
        linewidth=1.4, zorder=4, label="one static 'bank' vector\n(word2vec/GloVe: cannot split)",
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(
        f"Same word, two senses, two regions of space ({BACKEND_TAG})\n"
        "contextual vectors cluster by sense; a single static vector cannot",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="best", fontsize=9)
    _style_axis(ax)
    _save(fig, "ctx_static_vs_contextual.png")


def fig_static_vs_contextual_bars() -> None:
    """Measured: same-sense vs cross-sense cosine, contextual (last layer) vs static (layer 0).

    The static bars are both ~1.0 (the same string -> the same vector); the contextual bars separate
    same-sense (high) from cross-sense (low). This is the polysemy fix as two pairs of bars.
    """
    cos = sense_cosines(BACKEND)
    # Static (layer-0) cosines for the same pairs: the input embedding is the same vector for 'bank'.
    static_same = cosine(
        word_vector(BACKEND, MONEY, PROBE_WORD, layer=0),
        word_vector(BACKEND, MONEY2, PROBE_WORD, layer=0),
    )
    static_cross = cos["river_money_layer0"]
    labels = ["same sense\n(money / money)", "different sense\n(river / money)"]
    contextual = [cos["money_money2"], cos["river_money"]]
    static = [static_same, static_cross]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2, static, width, color=SLATE, label="static (layer 0 / word2vec-like)")
    ax.bar(x + width / 2, contextual, width, color=GREEN, label="contextual (BERT last layer)")
    for xi, (s, c) in enumerate(zip(static, contextual)):
        ax.text(xi - width / 2, s + 0.02, f"{s:.2f}", ha="center", color=INK, fontsize=9.5)
        ax.text(xi + width / 2, c + 0.02, f"{c:.2f}", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("cosine similarity")
    ax.set_title(
        f"Static can't tell the senses apart; contextual can ({BACKEND_TAG})\n"
        "static cosine stays high for both pairs; contextual splits same (high) from different (low)",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper center", fontsize=9, ncol=2)
    _style_axis(ax)
    _save(fig, "ctx_static_vs_contextual_bars.png")


def fig_sense_cosine_matrix() -> None:
    """Measured: pairwise cosine between 'bank' across the 4 named sentences (river/money/loan/money2)."""
    sentences = [
        ("river", RIVER),
        ("money", MONEY),
        ("loan", LOAN),
        ("teller", MONEY2),
    ]
    vecs = [word_vector(BACKEND, s, PROBE_WORD) for _, s in sentences]
    n = len(vecs)
    mat = np.array([[cosine(vecs[i], vecs[j]) for j in range(n)] for i in range(n)])
    labels = [name for name, _ in sentences]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(mat, cmap="PuBuGn", vmin=mat.min(), vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                color="white" if mat[i, j] > (mat.min() + 1.0) / 2 else INK,
                fontsize=11, fontweight="bold",
            )
    ax.set_title(
        f"Cosine between 'bank' across four sentences ({BACKEND_TAG})\n"
        "money/loan/teller (financial) cluster high; river sits apart",
        fontsize=10.5,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    _save(fig, "ctx_sense_cosine_matrix.png")


def fig_layer_probe() -> None:
    """Measured: cos(river bank, money bank) at every layer -- contextualization grows with depth."""
    probe = layer_probe(BACKEND, RIVER, MONEY)
    cosines = probe.per_layer_cosine
    layers = np.arange(len(cosines))
    min_layer = int(np.argmin(cosines))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(layers, cosines, color=PURPLE, lw=2.6, marker="o", markersize=6, markeredgecolor="white", zorder=3)
    ax.axhline(1.0, color=SLATE, lw=0.9, ls=":")
    ax.annotate(
        "layer 0 = static input\n(identical, cos = 1.00)",
        xy=(0, cosines[0]), xytext=(1.3, max(0.66, min(cosines) + 0.12)),
        color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="->", color=SLATE),
    )
    ax.annotate(
        f"most disambiguated\n(layer {min_layer}, cos = {cosines[min_layer]:.2f})",
        xy=(min_layer, cosines[min_layer]),
        xytext=(min_layer - 1.6, cosines[min_layer] + 0.17),
        color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="->", color=SLATE), ha="center",
    )
    ax.set_xlabel("BERT layer (0 = input embedding, 12 = final)")
    ax.set_ylabel("cosine(river-'bank', money-'bank')")
    ax.set_ylim(min(cosines) - 0.06, 1.08)
    ax.set_title(
        f"Contextualization is built by depth ({BACKEND_TAG})\n"
        "static at layer 0; self-attention pulls the two senses apart through the stack",
        fontsize=11,
    )
    _style_axis(ax)
    _save(fig, "ctx_layer_probe.png")


def fig_mlm_objective() -> None:
    """Measured: BERT's top masked-LM predictions for 'I deposited cash at the [MASK] downtown.'"""
    prompt = MLM_PROMPTS[0]
    preds = top_fill(BACKEND, prompt, k=5)
    words = [w for w, _ in preds]
    probs = [p for _, p in preds]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [GREEN if i == 0 else BLUE for i in range(len(words))]
    bars = ax.barh(range(len(words))[::-1], probs, color=colors, edgecolor="white")
    for bar, p in zip(bars, probs):
        ax.text(p + 0.005, bar.get_y() + bar.get_height() / 2, f"{p:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(words))[::-1])
    ax.set_yticklabels(words)
    ax.set_xlabel("softmax probability over the WordPiece vocabulary")
    ax.set_xlim(0, max(probs) * 1.25)
    ax.set_title(
        f"Masked-LM fills the blank from both-sided context ({BACKEND_TAG})\n"
        "'I deposited cash at the [MASK] downtown.' -> top prediction 'bank'",
        fontsize=11,
    )
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "ctx_mlm_objective.png")


def fig_mlm_80_10_10() -> None:
    """Illustrative: the 15% selection, then the 80/10/10 corruption split applied to chosen tokens."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # Left: 15% of positions chosen for prediction (vs 85% untouched in the input).
    chosen = 15
    untouched = 85
    ax.barh([1.6], [untouched], color=SLATE, edgecolor="white", height=0.5, label="85% untouched (input)")
    ax.barh([1.6], [chosen], left=[untouched], color=AMBER, edgecolor="white", height=0.5, label="15% selected for prediction")
    # Right: of the chosen 15%, the 80/10/10 split.
    split = [(80, GREEN, "80% -> [MASK]"), (10, BLUE, "10% -> random token"), (10, PURPLE, "10% -> kept unchanged")]
    left = 0.0
    for frac, color, lab in split:
        ax.barh([0.4], [frac], left=[left], color=color, edgecolor="white", height=0.5, label=lab)
        ax.text(left + frac / 2, 0.4, f"{frac}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += frac
    ax.text(50, 2.15, "Step 1 — select 15% of token positions", ha="center", color=INK, fontsize=10)
    ax.text(50, 0.95, "Step 2 — corrupt the selected 15% by 80/10/10", ha="center", color=INK, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("percent of tokens")
    ax.set_title(
        "BERT's masking recipe: 15% selected, then 80/10/10 corrupted (illustrative)\n"
        "the mix forces good representations at EVERY position, not just at [MASK]",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)
    _save(fig, "ctx_mlm_80_10_10.png")


def fig_elmo_layer_weighting() -> None:
    """Illustrative: ELMo's learned softmax weights over biLM layers differ by downstream task.

    Lower layers lean syntactic, higher layers semantic; a parsing task loads low, a sense task loads
    high. The bars are an illustrative softmax over 3 layers (input + 2 biLSTM) consistent with the
    page's claim, not measured ELMo weights.
    """
    layers = ["layer 0\n(char-CNN)", "layer 1\n(biLSTM, syntactic)", "layer 2\n(biLSTM, semantic)"]
    # Illustrative softmax weights: a syntactic task (POS/parse) loads the lower biLSTM; a semantic
    # task (word sense / coreference) loads the upper biLSTM. Each set sums to 1 (softmax).
    syntactic = np.array([0.20, 0.55, 0.25])
    semantic = np.array([0.15, 0.30, 0.55])
    x = np.arange(len(layers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - width / 2, syntactic, width, color=BLUE, label="syntactic task (e.g. POS / parsing)")
    ax.bar(x + width / 2, semantic, width, color=PURPLE, label="semantic task (e.g. word sense / coref)")
    for xi, (a, b) in enumerate(zip(syntactic, semantic)):
        ax.text(xi - width / 2, a + 0.01, f"{a:.2f}", ha="center", color=INK, fontsize=9)
        ax.text(xi + width / 2, b + 0.01, f"{b:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel(r"learned softmax weight  $s_j^{\mathrm{task}}$")
    ax.set_title(
        "ELMo learns a per-task softmax over biLM layers (illustrative)\n"
        "syntactic tasks load the lower layer; semantic tasks load the upper layer",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style_axis(ax)
    _save(fig, "ctx_elmo_layer_weighting.png")


def fig_bert_vs_gpt() -> None:
    """Illustrative: BERT's full (bidirectional) attention mask vs GPT's causal (lower-triangular) mask."""
    tokens = ["I", "ran", "to", "the", "bank"]
    n = len(tokens)
    full = np.ones((n, n))
    causal = np.tril(np.ones((n, n)))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, mask, title, color in (
        (axes[0], full, "BERT (encoder): bidirectional\nevery token attends both ways", BLUE),
        (axes[1], causal, "GPT (decoder): causal\neach token sees only its left context", GREEN),
    ):
        cmap = plt.matplotlib.colors.ListedColormap(["#EAEDF1", color])
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(tokens, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("attends to (key)", color=INK, fontsize=9)
        ax.set_ylabel("query token", color=INK, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "•" if mask[i, j] else "", ha="center", va="center", color="white", fontsize=12)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.tick_params(colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Bidirectional (masked-LM) vs causal (next-token) attention — the two horns of the dilemma",
        fontsize=11.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "ctx_bert_vs_gpt.png")


def main() -> None:
    cfg = bert_config()  # touch the config so the import is exercised; reported once for context
    print(f"writing figures to {OUT_DIR}")
    print(f"backend: {BACKEND.name} ({'real' if BACKEND.is_real else 'synthetic fallback'})  "
          f"BERT-base ref: {cfg['layers']} layers, hidden {cfg['hidden']}")
    fig_static_vs_contextual()
    fig_static_vs_contextual_bars()
    fig_sense_cosine_matrix()
    fig_layer_probe()
    fig_mlm_objective()
    fig_mlm_80_10_10()
    fig_elmo_layer_weighting()
    fig_bert_vs_gpt()
    print("done.")


if __name__ == "__main__":
    main()

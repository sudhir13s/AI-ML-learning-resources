"""Reproducible figure generator for 06-NLP / 11-Question-Answering.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the span decode, the start/end logits, the SQuAD EM/F1 scorer, the retriever cosines,
and the recall/hop curves are all IMPORTED from `question_answering.py`, so the figures cannot
silently drift from the prose or the demo. Run:

    python make_figures_11.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `qa_`. The palette matches the chapter's Mermaid diagrams (muted, white text
on fills).

Figures produced (measured = from the live backend / scorer; illustrative = a labelled schematic):
  qa_taxonomy.png            -- illustrative: the QA design space (answer type x knowledge source)
  qa_span_logits_heatmap.png -- measured: start/end logits over the hand-worked passage (MANDATORY)
  qa_joint_span_score.png    -- measured: the joint span-score matrix start_i+end_j, valid i<=j region
  qa_span_head.png           -- measured: start/end probability bars over a real passage
  qa_em_f1.png               -- measured: Exact Match vs token-F1 for the five worked pairs
  qa_em_f1_leniency.png      -- measured: how F1 stays high (and EM collapses) as extra tokens creep in
  qa_retrieve_read.png       -- measured: retriever cosine per passage -> the read span
  qa_retrieval_ceiling.png   -- illustrative: end-to-end accuracy capped by retriever recall@k
  qa_multihop_compounding.png-- illustrative: multiplicative success decay across reasoning hops

When the real fine-tuned QA model is cached these figures are MEASURED from it; offline they fall
back to the deterministic synthetic span model and still render (titles note the backend). Verified
on Python 3.12 / numpy 2.x / matplotlib 3.x, CPU, deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from question_answering import (
    EM_F1_PAIRS,
    HAND_END,
    HAND_START,
    HAND_TOKENS,
    RETRIEVER_CORPUS,
    RETRIEVER_QUESTION,
    best_valid_span,
    exact_match,
    load_qa_model,
    retrieve_top_passage,
    softmax,
    token_f1,
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

# One shared backend, so every measured figure comes from the same model load.
BACKEND = load_qa_model()
BACKEND_TAG = "real model, measured" if BACKEND.is_real else "synthetic fallback"


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
    print(f"wrote images/{name}")


# =================================================================================================
# 1. The QA design-space grid (illustrative)
# =================================================================================================


def fig_taxonomy() -> None:
    """The two-axis QA design space: answer type (rows) x knowledge source (columns)."""
    rows = ["Extractive\n(span)", "Abstractive\n(generate)", "Multiple-choice\n/ yes-no"]
    cols = ["Reading comp.\n(given passage)", "Closed-book\n(parameters)", "Open-domain\n(corpus)"]
    cell = [
        ["SQuAD\nBERT span head", "—", "DPR + span\nreader"],
        ["LLM reads\nthe document", "T5 / GPT\nclosed-book", "RAG\nretrieve+generate"],
        ["RACE / BoolQ\n[CLS] classifier", "MMLU\nfrom weights", "ARC\nretrieve+classify"],
    ]
    colors = [
        [GREEN, SLATE, GREEN],
        [AMBER, PURPLE, AMBER],
        [BLUE, PURPLE, BLUE],
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    n_r, n_c = len(rows), len(cols)
    for r in range(n_r):
        for c in range(n_c):
            ax.add_patch(Rectangle((c, n_r - 1 - r), 1, 1, facecolor=colors[r][c], edgecolor="white", linewidth=2))
            ax.text(c + 0.5, n_r - 1 - r + 0.5, cell[r][c], ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.set_xticks([c + 0.5 for c in range(n_c)])
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks([n_r - 1 - r + 0.5 for r in range(n_r)])
    ax.set_yticklabels(rows, fontsize=10)
    ax.set_xlim(0, n_c)
    ax.set_ylim(0, n_r)
    ax.set_xlabel("knowledge source  ->", fontsize=11, weight="bold")
    ax.set_ylabel("answer shape  ->", fontsize=11, weight="bold")
    ax.set_title(
        "The QA design space: pick the architecture from the cell\n"
        "two orthogonal axes -- what shape is the answer, and where does the knowledge live",
        fontsize=12, color=INK,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "qa_taxonomy.png")


# =================================================================================================
# 2. Span-logit heatmap (MANDATORY) -- start/end logits over the hand-worked passage
# =================================================================================================


def fig_span_logits_heatmap() -> None:
    """Measured: the start and end logits over the 7 hand-worked tokens, as a heatmap.

    This is the raw material of the decode: two rows (S.h_i and E.h_i), one column per token. The
    bright cell in each row is where that head wants to point -- both land on 'Paris', so the
    argmax-pair span is the single token 'Paris'.
    """
    matrix = np.vstack([HAND_START, HAND_END])  # row 0 = start logits, row 1 = end logits
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    im = ax.imshow(matrix, aspect="auto", cmap="BuPu", vmin=-3.5, vmax=4.5)
    ax.set_xticks(range(len(HAND_TOKENS)))
    ax.set_xticklabels(HAND_TOKENS, fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["start  S·h_i", "end  E·h_j"], fontsize=11)
    for r in range(2):
        for c in range(len(HAND_TOKENS)):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    color="white" if val > 0.5 else INK, fontsize=10, weight="bold")
    # box the argmax cell in each row
    i, j, _ = best_valid_span(HAND_START, HAND_END, l_max=30)
    ax.add_patch(Rectangle((i - 0.5, -0.5), 1, 1, fill=False, edgecolor=GREEN, linewidth=3))
    ax.add_patch(Rectangle((j - 0.5, 0.5), 1, 1, fill=False, edgecolor=RED, linewidth=3))
    ax.set_title(
        "Start/end logits over the passage -- the raw material of the span decode (measured)\n"
        f"both heads peak on 'Paris' (boxed) -> argmax-pair span = ({i},{j}) = 'Paris'",
        fontsize=11, color=INK,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("logit", color=INK)
    ax.tick_params(length=0)
    _save(fig, "qa_span_logits_heatmap.png")


# =================================================================================================
# 3. Joint span-score matrix start_i + end_j over the valid i<=j region
# =================================================================================================


def fig_joint_span_score() -> None:
    """Measured: the joint span score start_i+end_j for every (i,j); the lower triangle is invalid."""
    n = len(HAND_TOKENS)
    scores = HAND_START[:, None] + HAND_END[None, :]  # scores[i, j] = start_i + end_j
    masked = np.ma.masked_where(np.arange(n)[:, None] > np.arange(n)[None, :], scores)  # j>=i only
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cmap = plt.cm.BuPu.copy()
    cmap.set_bad(color="#E8E8EC")  # grey out the invalid j<i half
    im = ax.imshow(masked, cmap=cmap, vmin=-6, vmax=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(HAND_TOKENS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(HAND_TOKENS, fontsize=9)
    ax.set_xlabel("end position j  (E·h_j)")
    ax.set_ylabel("start position i  (S·h_i)")
    i_star, j_star, best = best_valid_span(HAND_START, HAND_END, l_max=30)
    ax.add_patch(Rectangle((j_star - 0.5, i_star - 0.5), 1, 1, fill=False, edgecolor=GREEN, linewidth=3))
    ax.text(j_star, i_star, "argmax", ha="center", va="center", color=GREEN, fontsize=8, weight="bold")
    # annotate the invalid region once
    ax.text(1.2, 4.6, "j < i:\ninvalid\n(backwards span)", color=SLATE, fontsize=9, ha="center", style="italic")
    ax.set_title(
        "Joint span score  start_i + end_j  over valid spans (j >= i)\n"
        f"grey = invalid backwards spans; argmax = ('Paris','Paris'), score {best:.1f} (measured)",
        fontsize=10.5, color=INK,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("span score", color=INK)
    ax.tick_params(length=0)
    _save(fig, "qa_joint_span_score.png")


# =================================================================================================
# 4. Span-head probability bars over a real passage
# =================================================================================================


def _eiffel_distributions() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (passage tokens, P_start, P_end) for the Eiffel passage from the live or synthetic model."""
    question = "Where is the Eiffel Tower located?"
    passage = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    if BACKEND.is_real:
        import torch

        inp = BACKEND.tokenizer(question, passage, return_tensors="pt")
        with torch.no_grad():
            out = BACKEND.model(**inp)
        seq = inp.sequence_ids(0)
        ids = inp["input_ids"][0]
        start = out.start_logits[0].numpy()
        end = out.end_logits[0].numpy()
        idx = [k for k, s in enumerate(seq) if s == 1]
        toks = [BACKEND.tokenizer.decode([ids[k]]).strip() for k in idx]
        return toks, softmax(start)[idx], softmax(end)[idx]
    from question_answering import _synthetic_logits

    tokens, start, end, _ = _synthetic_logits(question, passage)
    return tokens, softmax(start), softmax(end)


def fig_span_head() -> None:
    """Measured: start (green) and end (red) probability over the passage tokens of a real model."""
    toks, p_start, p_end = _eiffel_distributions()
    x = np.arange(len(toks))
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.bar(x - 0.2, p_start, width=0.4, color=GREEN, label="P_start (answer begins here)")
    ax.bar(x + 0.2, p_end, width=0.4, color=RED, label="P_end (answer ends here)")
    i_s, i_e = int(np.argmax(p_start)), int(np.argmax(p_end))
    ax.axvspan(i_s - 0.5, i_e + 0.5, color=AMBER, alpha=0.12, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(
        f"A trained span head points: one sharp start peak, one sharp end peak ({BACKEND_TAG})\n"
        "the shaded region is the predicted span argmax_{j>=i}(S·h_i + E·h_j)",
        fontsize=11, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_span_head.png")


# =================================================================================================
# 5. EM vs F1 bars for the five worked pairs
# =================================================================================================


def fig_em_f1() -> None:
    """Measured: Exact Match (slate) vs token-F1 (green) for the five worked prediction/gold pairs."""
    labels = [f"{p!r}\nvs {g!r}" for p, g in EM_F1_PAIRS]
    ems = [exact_match(p, g) for p, g in EM_F1_PAIRS]
    f1s = [token_f1(p, g) for p, g in EM_F1_PAIRS]
    y = np.arange(len(EM_F1_PAIRS))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.barh(y + 0.2, ems, height=0.38, color=SLATE, label="Exact Match (all-or-nothing)")
    ax.barh(y - 0.2, f1s, height=0.38, color=GREEN, label="token F1 (partial credit)")
    for k, (em, f1) in enumerate(zip(ems, f1s)):
        ax.text(em + 0.02, k + 0.2, f"{em:.0f}", va="center", fontsize=9, color=SLATE, weight="bold")
        ax.text(f1 + 0.02, k - 0.2, f"{f1:.2f}", va="center", fontsize=9, color=GREEN, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("score")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title(
        "EM is brutal; F1 gives partial credit (measured by the SQuAD scorer)\n"
        "'Paris, France' vs 'Paris': EM=0 but F1=0.67 -- and 'the Champ de Mars' scores a perfect 1 via normalization",
        fontsize=10.5, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_em_f1.png")


# =================================================================================================
# 6. F1 leniency curve -- F1 stays high as extra tokens creep in, EM collapses immediately
# =================================================================================================


def fig_em_f1_leniency() -> None:
    """Measured: append k extra (wrong) tokens to a correct answer and watch EM die while F1 decays."""
    gold = "Leonardo da Vinci"  # 3 gold tokens
    # every appended token is a real content word, so EM dies at the very first one (k=1)
    extras = ["painter", "Florentine", "Renaissance", "master", "around", "1503", "Italy", "Tuscany", "Europe"]
    ks = list(range(len(extras) + 1))
    ems, f1s = [], []
    for k in ks:
        pred = gold + (" " + " ".join(extras[:k]) if k else "")
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(ks, f1s, marker="o", color=GREEN, linewidth=2.4, label="token F1")
    ax.plot(ks, ems, marker="s", color=SLATE, linewidth=2.4, label="Exact Match")
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(len(extras) - 0.2, 0.52, "F1=0.5 (precision halved)", color=RED, fontsize=8.5, ha="right")
    ax.set_xlabel("extra tokens appended to the correct answer 'Leonardo da Vinci'")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(
        "Why F1 is lenient and EM is not (measured)\n"
        "one extra token kills EM (1 -> 0); F1 only sags, because recall stays 1.0 and precision drops slowly",
        fontsize=10.5, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_em_f1_leniency.png")


# =================================================================================================
# 7. Retriever cosine per passage -> the read span
# =================================================================================================


def fig_retrieve_read() -> None:
    """Measured: cosine of the question against each corpus passage; the top one feeds the reader."""
    top_idx, scores = retrieve_top_passage(RETRIEVER_QUESTION, RETRIEVER_CORPUS)
    labels = [f"p{k}: {p[:30]}..." for k, p in enumerate(RETRIEVER_CORPUS)]
    colors = [GREEN if k == top_idx else SLATE for k in range(len(RETRIEVER_CORPUS))]
    y = np.arange(len(RETRIEVER_CORPUS))
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.barh(y, scores, color=colors, height=0.6)
    for k, s in enumerate(scores):
        ax.text(s + 0.005, k, f"{s:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cosine similarity to the question")
    ax.set_xlim(0, max(scores) * 1.25)
    ax.set_title(
        f"Retrieve, then read: '{RETRIEVER_QUESTION}' (measured)\n"
        "the top-cosine passage (green) is handed to the span reader -> 'Leonardo da Vinci'",
        fontsize=10.5, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_retrieve_read.png")


# =================================================================================================
# 8. Retrieval recall ceiling -- the reader can never beat what the retriever surfaced
# =================================================================================================


def fig_retrieval_ceiling() -> None:
    """Illustrative: end-to-end accuracy is capped by retriever recall@k -- a hard ceiling."""
    ks = np.array([1, 3, 5, 10, 20])
    recall = np.array([0.55, 0.74, 0.82, 0.89, 0.93])  # representative DPR-style recall@k curve
    reader_given_hit = 0.80  # a strong reader, conditional on the evidence being present
    end_to_end = recall * reader_given_hit
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ks, recall, marker="o", color=BLUE, linewidth=2.4, label="retriever recall@k (the ceiling)")
    ax.plot(ks, end_to_end, marker="s", color=GREEN, linewidth=2.4, label="end-to-end QA accuracy")
    ax.fill_between(ks, end_to_end, recall, color=RED, alpha=0.10)
    ax.text(11, 0.78, "reader can only\nlose from here", color=RED, fontsize=9, ha="center")
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylabel("accuracy")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title(
        "Open-domain QA is bottlenecked by the retriever (illustrative)\n"
        "if the answer-bearing passage isn't in the top-k, a perfect reader still scores zero",
        fontsize=10.5, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_retrieval_ceiling.png")


# =================================================================================================
# 9. Multi-hop compounding -- success decays multiplicatively with hops
# =================================================================================================


def fig_multihop_compounding() -> None:
    """Illustrative: per-hop retrieval success p compounds as p^hops -- why multi-hop is hard."""
    hops = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for p, color, label in ((0.95, GREEN, "p=0.95 per hop"), (0.90, AMBER, "p=0.90 per hop"), (0.80, RED, "p=0.80 per hop")):
        chain = p**hops
        ax.plot(hops, chain, marker="o", color=color, linewidth=2.4, label=label)
        ax.text(hops[-1] + 0.05, chain[-1], f"{chain[-1]:.2f}", va="center", color=color, fontsize=9, weight="bold")
    ax.set_xlabel("number of reasoning hops (retrieve -> read -> reformulate -> retrieve ...)")
    ax.set_ylabel("probability the whole chain succeeds")
    ax.set_xticks(hops)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=10, title="per-hop retrieval recall")
    ax.set_title(
        "Multi-hop error compounds: chain success = p^hops (illustrative)\n"
        "even 90%-per-hop retrieval drops below 60% by hop 5 -- before any reader error",
        fontsize=10.5, color=INK,
    )
    _style_axis(ax)
    _save(fig, "qa_multihop_compounding.png")


def main() -> None:
    print(f"backend: {BACKEND_TAG}")
    print(f"writing figures to {OUT_DIR}")
    fig_taxonomy()
    fig_span_logits_heatmap()
    fig_joint_span_score()
    fig_span_head()
    fig_em_f1()
    fig_em_f1_leniency()
    fig_retrieve_read()
    fig_retrieval_ceiling()
    fig_multihop_compounding()
    print("done.")


if __name__ == "__main__":
    main()

"""Static figure generator for 01-RAG-Fundamentals -- every number comes from the REAL pipeline.

Imports the same `RagPipeline` the page and notebook use (rag_fundamentals.py), builds the real
FAISS index over the real Wikipedia corpus, and plots the actual retrieval scores, recall curves,
and reranker output. No hand-typed values, no synthetic corpus. Writes muted-palette PNGs to the
shared chapter image dir (../../images/) with the per-chapter prefix `rag01_`.

    python make_figures_01.py

Figures produced (all from real data unless marked illustrative):
  rag01_similarity_bars.png    -- real cosine of the top retrieved passages to the headline query.
  rag01_embedding_space.png    -- a real corpus sample + the query in 2D (PCA); top-k highlighted.
  rag01_recall_vs_k.png        -- real dense vs reranked recall@k over 200 real QA pairs.
  rag01_rerank_win.png         -- the real Lincoln case: dense mis-ranks, the cross-encoder fixes it.
  rag01_prompt_anatomy.png     -- the real augmented prompt dissected into instruction/context/question.
  rag01_lost_in_the_middle.png -- the U-shaped 'lost in the middle' curve (illustrative, Liu et al. 2024).

Recall measurement is the slow part (~25 s: it cross-encodes 200x20 pairs); the result is cached
to `_recall_cache.json` next to this script so re-runs are fast. Delete that file to recompute.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x. Headless (Agg); no display needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from rag_fundamentals import RERANK_CANDIDATES, TOP_K, RagPipeline, load_corpus

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / corpus
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # retrieved / output
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # frozen / distractor
AMBER = "#7A6528"  # highlight / query
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
CACHE = Path(__file__).resolve().parent / "_recall_cache.json"
DPI = 110

HEADLINE_QUERY = "What was reversed about the temperature scale in 1745?"
RERANK_QUERY = "When was Abraham Lincoln inaugurated as president?"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of an (n, d) matrix to 2D via the top-2 principal components (plotting only).

    Pure-numpy PCA: centre, take the right singular vectors, keep 2. This is a *visualization* of
    the 384-d embedding geometry -- the retriever itself never reduces dimensions.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T  # (n, 2): coordinates in the top-2 PC plane


def fig_similarity_bars(pipeline: RagPipeline) -> None:
    """Real cosine of the top retrieved passages to the headline query -- why doc[144] wins."""
    hits = pipeline.retrieve(HEADLINE_QUERY, k=8)
    scores = [h.score for h in hits]
    labels = [f"doc[{h.doc_id}]" for h in hits]
    colors = [GREEN if rank < TOP_K else SLATE for rank in range(len(hits))]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    bars = ax.bar(labels, scores, color=colors, edgecolor=INK, linewidth=0.8, width=0.66)
    for bar, score in zip(bars, scores):
        ax.annotate(f"{score:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=8, color=INK, ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points")
    ax.axvline(TOP_K - 0.5, color=RED, linewidth=1.6, linestyle="--", alpha=0.85)
    ax.annotate(f"top-{TOP_K} cutoff", (TOP_K - 0.5, max(scores) * 0.95), color=RED,
                fontsize=9, ha="left", va="center", xytext=(6, 0), textcoords="offset points")
    ax.set_title("Real cosine similarity to the query, sorted -- with the top-k cut", fontsize=12, pad=12)
    ax.set_ylabel("cosine similarity (all-MiniLM-L6-v2)")
    ax.set_ylim(0, max(scores) * 1.2)
    ax.tick_params(axis="x", labelrotation=20)
    _save(fig, "rag01_similarity_bars.png")


def fig_embedding_space(pipeline: RagPipeline) -> None:
    """A real corpus sample + the query in 2D (PCA); the query's top-k highlighted.

    Plotting all 3,200 points is unreadable, so we sample: the real top-k retrieved passages, a
    handful of the next-nearest (near-miss) passages, and a random background of unrelated ones.
    The geometry is the real 384-d embedding space projected to 2D -- retrieval is 'find the
    nearest points', shown literally.
    """
    query_vec = pipeline.embed_query(HEADLINE_QUERY)[0]
    hits = pipeline.retrieve(HEADLINE_QUERY, k=12)  # top-k + near-misses
    top_ids = [h.doc_id for h in hits[:TOP_K]]
    near_ids = [h.doc_id for h in hits[TOP_K:12]]
    rng = np.random.default_rng(0)
    background_ids = [
        i for i in rng.choice(len(pipeline.passages), size=40, replace=False)
        if i not in top_ids and i not in near_ids
    ][:30]

    ids = top_ids + near_ids + background_ids
    vecs = np.vstack([pipeline.embeddings[ids], query_vec[None, :]])
    coords = _pca_2d(vecs)
    doc_xy, q_xy = coords[:-1], coords[-1]
    n_top, n_near = len(top_ids), len(near_ids)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    _style_axis(ax)
    # background (unrelated) passages
    ax.scatter(doc_xy[n_top + n_near:, 0], doc_xy[n_top + n_near:, 1], s=70, color=SLATE,
               alpha=0.35, edgecolors="none", zorder=2, label="unrelated passages")
    # near-miss passages
    ax.scatter(doc_xy[n_top:n_top + n_near, 0], doc_xy[n_top:n_top + n_near, 1], s=110, color=BLUE,
               alpha=0.6, edgecolors=INK, linewidths=0.8, zorder=3, label="near-miss (ranks 4-12)")
    # top-k retrieved, with beams to the query
    # stagger label offsets so adjacent top-k labels don't overlap
    label_offsets = [(10, 6), (10, -12), (-14, 10)]
    for j in range(n_top):
        x, y = doc_xy[j]
        ax.plot([q_xy[0], x], [q_xy[1], y], color=GREEN, linewidth=1.4, alpha=0.5, zorder=2)
        ax.scatter(x, y, s=240, color=GREEN, alpha=0.95, edgecolors=INK, linewidths=1.0, zorder=4)
        dx, dy = label_offsets[j % len(label_offsets)]
        ax.annotate(f"doc[{top_ids[j]}]", (x, y), fontsize=8, color=INK,
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
                    xytext=(dx, dy), textcoords="offset points", zorder=5)
    ax.scatter(*q_xy, s=430, marker="*", color=AMBER, edgecolors=INK, linewidths=1.2, zorder=6)
    ax.annotate("query", q_xy, fontsize=10, color=INK, ha="center", va="bottom",
                xytext=(0, 14), textcoords="offset points", fontweight="bold", zorder=6)
    ax.set_title("Retrieval is nearest-neighbour search in the real 384-d embedding space",
                 fontsize=11.5, pad=12)
    ax.set_xlabel("principal component 1 (2D projection of 384-d vectors)")
    ax.set_ylabel("principal component 2")
    green_proxy = ax.scatter([], [], s=150, color=GREEN, edgecolors=INK, label=f"top-{TOP_K} retrieved")
    star_proxy = ax.scatter([], [], s=220, marker="*", color=AMBER, edgecolors=INK, label="query")
    handles = [star_proxy, green_proxy,
               ax.scatter([], [], s=110, color=BLUE, edgecolors=INK, label="near-miss (ranks 4-12)"),
               ax.scatter([], [], s=70, color=SLATE, label="unrelated passages")]
    ax.legend(handles=handles, loc="best", framealpha=0.95, fontsize=8.5)
    _save(fig, "rag01_embedding_space.png")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _compute_recall(pipeline: RagPipeline, qa: list[dict[str, str]], *, n: int = 200) -> dict:
    """Measure real recall@k, dense vs reranked, over `n` real factoid QA pairs.

    Gold signal: a passage is 'correct' if its text contains the (normalized) answer string. We
    keep only short, non-yes/no answers so containment is a meaningful proxy for 'this passage
    supports the answer'. Dense: recall@k for k in a grid. Reranked: take the top-20 dense hits,
    cross-encoder rerank, measure recall@{1,3,5}.
    """
    sample: list[tuple[str, set[int]]] = []
    for row in qa:
        answer = row["answer"].strip()
        if answer.lower() in ("yes", "no") or len(answer.split()) > 6:
            continue
        norm_answer = _norm(answer)
        if len(norm_answer) < 2:
            continue
        gold = {i for i, p in enumerate(pipeline.passages) if norm_answer in _norm(p)}
        if gold:
            sample.append((row["question"], gold))
        if len(sample) >= n:
            break

    questions = [q for q, _ in sample]
    q_emb = pipeline.embedder.encode(
        questions, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    ).astype("float32")
    ks = [1, 2, 3, 5, 10, 20]
    _, ids = pipeline.index.search(q_emb, max(ks))

    dense_recall = {}
    for k in ks:
        hits = sum(
            len({int(x) for x in ids[row, :k]} & gold) > 0 for row, (_, gold) in enumerate(sample)
        )
        dense_recall[k] = hits / len(sample)

    rerank_recall = {1: 0, 3: 0, 5: 0}
    for row, (question, gold) in enumerate(sample):
        candidate_ids = [int(x) for x in ids[row, :RERANK_CANDIDATES]]
        ce_scores = pipeline._reranker.predict(  # reranker is initialised by the caller
            [(question, pipeline.passages[i]) for i in candidate_ids]
        )
        ordered = [candidate_ids[j] for j in np.argsort(ce_scores)[::-1]]
        for k in rerank_recall:
            rerank_recall[k] += int(len(set(ordered[:k]) & gold) > 0)
    for k in rerank_recall:
        rerank_recall[k] /= len(sample)

    return {"n": len(sample), "ks": ks, "dense": dense_recall, "rerank": rerank_recall}


def fig_recall_vs_k(pipeline: RagPipeline, qa: list[dict[str, str]]) -> None:
    """Real recall@k over 200 real QA pairs: dense curve + the reranked points that lift the left end."""
    if CACHE.exists():
        data = json.loads(CACHE.read_text())
    else:
        from sentence_transformers import CrossEncoder

        from rag_fundamentals import RERANK_MODEL_ID

        pipeline._reranker = CrossEncoder(RERANK_MODEL_ID)
        data = _compute_recall(pipeline, qa)
        CACHE.write_text(json.dumps(data))
    ks = data["ks"]
    dense = [data["dense"][str(k)] if isinstance(next(iter(data["dense"])), str) else data["dense"][k] for k in ks]
    rerank = data["rerank"]

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    _style_axis(ax)
    ax.plot(ks, dense, marker="o", color=PURPLE, linewidth=2.2, markersize=7,
            markeredgecolor=INK, markerfacecolor=BLUE, label="dense (bi-encoder)")
    for k, r in zip(ks, dense):
        ax.annotate(f"{r:.2f}", (k, r), fontsize=8, color=INK, ha="center", va="top",
                    xytext=(0, -8), textcoords="offset points")
    # reranked points at k = 1,3,5 (rerank of top-20 dense)
    rr_ks = [1, 3, 5]
    rr_vals = [rerank[str(k)] if str(k) in rerank else rerank[k] for k in rr_ks]
    ax.plot(rr_ks, rr_vals, marker="D", color=GREEN, linewidth=2.2, markersize=8,
            markeredgecolor=INK, markerfacecolor=GREEN, label="+ cross-encoder rerank (top-20)")
    for k, r in zip(rr_ks, rr_vals):
        ax.annotate(f"{r:.2f}", (k, r), fontsize=8.5, color=GREEN, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", fontweight="bold")
    ax.set_title(f"Real recall@k on {data['n']} Wikipedia QA pairs -- rerank lifts the left end",
                 fontsize=11.5, pad=12)
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylabel("recall@k  (fraction of answers retrieved)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ks)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    _save(fig, "rag01_recall_vs_k.png")


def fig_rerank_win(pipeline: RagPipeline) -> None:
    """The real Lincoln case: dense mis-ranks a title chunk #1; the cross-encoder fixes the order."""
    dense = pipeline.retrieve(RERANK_QUERY, k=5)
    reranked = pipeline.rerank(RERANK_QUERY, dense, k=5)
    dense_ids = [h.doc_id for h in dense]
    rerank_ids = [h.doc_id for h in reranked]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # left: dense order (by cosine)
    _style_axis(ax_l)
    labels_l = [f"doc[{h.doc_id}]" for h in dense]
    colors_l = [RED if h.doc_id == dense_ids[0] else BLUE for h in dense]
    ax_l.barh(range(len(dense))[::-1], [h.score for h in dense], color=colors_l, edgecolor=INK,
              linewidth=0.8, height=0.6)
    ax_l.set_yticks(range(len(dense))[::-1])
    ax_l.set_yticklabels(labels_l)
    for rank, h in enumerate(dense):
        ax_l.annotate(f"{h.score:.3f}", (h.score, len(dense) - 1 - rank), fontsize=8, color=INK,
                      va="center", ha="left", xytext=(3, 0), textcoords="offset points")
    ax_l.set_title("Dense (bi-encoder cosine)\n#1 = doc[288] 'Young Abraham Lincoln' -- useless",
                   fontsize=10.5, pad=8, color=INK)
    ax_l.set_xlabel("cosine similarity")

    # right: reranked order (by cross-encoder logit)
    _style_axis(ax_r)
    labels_r = [f"doc[{h.doc_id}]" for h in reranked]
    colors_r = [GREEN if h.doc_id == rerank_ids[0] else SLATE for h in reranked]
    ax_r.barh(range(len(reranked))[::-1], [h.score for h in reranked], color=colors_r,
              edgecolor=INK, linewidth=0.8, height=0.6)
    ax_r.set_yticks(range(len(reranked))[::-1])
    ax_r.set_yticklabels(labels_r)
    for rank, h in enumerate(reranked):
        ax_r.annotate(f"{h.score:.2f}", (h.score, len(reranked) - 1 - rank), fontsize=8, color=INK,
                      va="center", ha="left" if h.score > 0 else "right",
                      xytext=(3 if h.score > 0 else -3, 0), textcoords="offset points")
    ax_r.set_title("+ cross-encoder rerank\n#1 = doc[322] the actual inauguration passage",
                   fontsize=10.5, pad=8, color=INK)
    ax_r.set_xlabel("cross-encoder relevance logit")
    ax_r.axvline(0, color=GRID, linewidth=1.0)
    fig.suptitle("Reranking fixes a real retrieval miss (query: Lincoln's inauguration date)",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "rag01_rerank_win.png")


def fig_prompt_anatomy(pipeline: RagPipeline) -> None:
    """The real augmented prompt for the headline query, dissected into instruction/context/question."""
    hits = pipeline.retrieve(HEADLINE_QUERY, k=TOP_K)

    def _wrap(text: str, width: int = 78) -> str:
        words, lines, line = text.split(), [], ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        return "\n".join(lines)

    passages = [_wrap(h.text[:220] + ("..." if len(h.text) > 220 else "")) for h in hits]

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    fig.subplots_adjust(bottom=0.08, top=0.93)
    ax.axis("off")
    blocks = [
        ("INSTRUCTION", "Answer using ONLY the context. If absent, say you don't know. Cite [n].", AMBER),
        ("CONTEXT  (real retrieved passages -- the grounding)",
         "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)), GREEN),
        ("QUESTION", HEADLINE_QUERY, BLUE),
    ]
    y = 0.97
    for title, body, color in blocks:
        n_lines = body.count("\n") + 1
        height = 0.075 + 0.050 * n_lines
        ax.add_patch(plt.Rectangle((0.02, y - height), 0.96, height, transform=ax.transAxes,
                                   facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.6))
        ax.text(0.04, y - 0.025, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                color=color, va="top")
        ax.text(0.04, y - 0.062, body, transform=ax.transAxes, fontsize=8.5, color=INK, va="top",
                family="monospace")
        y -= height + 0.026
    ax.text(0.04, max(y - 0.005, 0.03),
            "this whole block -- not just the question -- is what the LLM generates from",
            transform=ax.transAxes, fontsize=9.5, style="italic", color=INK, va="top")
    ax.set_title("Anatomy of the real augmented prompt", fontsize=13, color=INK, pad=8)
    _save(fig, "rag01_prompt_anatomy.png")


def fig_lost_in_the_middle() -> None:
    """The U-shaped 'lost in the middle' accuracy curve (illustrative shape, Liu et al. 2024)."""
    positions = np.linspace(0, 1, 11)
    accuracy = 0.55 + 0.40 * (2 * positions - 1) ** 2

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _style_axis(ax)
    ax.plot(positions, accuracy, marker="o", color=RED, linewidth=2.4, markersize=7,
            markeredgecolor=INK, markerfacecolor=AMBER)
    ax.fill_between(positions, accuracy, 0.5, where=(accuracy < 0.75), color=RED, alpha=0.08)
    ax.annotate("buried in the middle\n-> often ignored", (0.5, accuracy[5]), color=RED, fontsize=9.5,
                ha="center", va="bottom", xytext=(0, 18), textcoords="offset points")
    ax.annotate("start", (0.0, accuracy[0]), color=INK, fontsize=9, ha="left", va="bottom",
                xytext=(4, 6), textcoords="offset points")
    ax.annotate("end", (1.0, accuracy[-1]), color=INK, fontsize=9, ha="right", va="bottom",
                xytext=(-4, 6), textcoords="offset points")
    ax.set_title("Lost in the middle -- position of evidence matters (illustrative)", fontsize=12, pad=12)
    ax.set_xlabel("position of the relevant passage within the context  (0 = start, 1 = end)")
    ax.set_ylabel("answer accuracy (normalized)")
    ax.set_ylim(0.45, 1.0)
    _save(fig, "rag01_lost_in_the_middle.png")


ANIM_CACHE = Path(__file__).resolve().parent / "_anim_cache.json"


def write_anim_cache(pipeline: RagPipeline) -> None:
    """Persist the real geometry + answer the animation needs, so the GIF build stays fast.

    We store the headline query, its top-3 retrieved passages (id + text + cosine), and a small
    PCA layout of the query with those passages + a background sample -- all from the REAL index.
    The animation reads this instead of re-embedding 3,200 passages and re-calling the LLM.
    """
    hits = pipeline.retrieve(HEADLINE_QUERY, k=TOP_K)
    query_vec = pipeline.embed_query(HEADLINE_QUERY)[0]
    rng = np.random.default_rng(1)
    bg_ids = [int(i) for i in rng.choice(len(pipeline.passages), size=24, replace=False)
              if i not in [h.doc_id for h in hits]][:18]
    layout_ids = [h.doc_id for h in hits] + bg_ids
    vecs = np.vstack([pipeline.embeddings[layout_ids], query_vec[None, :]])
    coords = _pca_2d(vecs)
    # A concise, real grounded answer for the GIF (kept short so it fits the panel).
    result = pipeline.answer(HEADLINE_QUERY, rerank=False, max_new_tokens=60)
    ANIM_CACHE.write_text(json.dumps({
        "query": HEADLINE_QUERY,
        "top": [{"id": h.doc_id, "text": h.text, "cos": round(h.score, 3)} for h in hits],
        "n_top": TOP_K,
        "doc_xy": coords[:-1].tolist(),
        "q_xy": coords[-1].tolist(),
        "answer": result.answer,
        "model": result.model,
    }))
    print(f"wrote {ANIM_CACHE}")


def main() -> None:
    passages, qa = load_corpus()
    pipeline = RagPipeline(passages)
    fig_similarity_bars(pipeline)
    fig_embedding_space(pipeline)
    fig_rerank_win(pipeline)
    fig_recall_vs_k(pipeline, qa)
    fig_prompt_anatomy(pipeline)
    fig_lost_in_the_middle()
    write_anim_cache(pipeline)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

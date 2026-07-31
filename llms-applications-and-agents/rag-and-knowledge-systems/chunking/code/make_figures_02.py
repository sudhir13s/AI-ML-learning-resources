"""Static figure generator for 02-Document-Chunking-Strategies.

Imports the SAME canonical functions the page and notebook use (document_chunking.py) so every
plotted number is the chapter's own — no hand-typed values. Writes muted-palette PNGs to the shared
chapter image dir (../../images/) with the per-chapter prefix `rag02_`.

    python make_figures_02.py

Figures produced:
  rag02_fixed_split_shatter.png   -- the naive fixed split drawn over the text: a boundary lands
                                     INSIDE "March 3rd, 2024", so neither chunk holds the fact whole.
  rag02_size_recall_precision.png -- chunk size vs recall (does a chunk hold the whole answer?) and
                                     precision (how much of the chunk is the answer) -- the core knob.
  rag02_overlap_diagram.png       -- fixed chunks WITHOUT vs WITH overlap; overlap re-includes the
                                     boundary region so a straddling fact survives in one chunk.
  rag02_semantic_trace.png        -- adjacent-sentence cosine similarity across the document, with the
                                     percentile cut line; dips at section boundaries become chunk cuts.
  rag02_strategy_compare.png      -- per-strategy: #chunks and whether the answer survived + was
                                     retrieved top-1 -- the measured quality difference.
  rag02_overlap_token_cost.png    -- overlap fraction vs the extra tokens (re-embedded duplicate text)
                                     it costs -- the price of robustness.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x. Headless (Agg); no display needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from document_chunking import (
    ANSWER_PHRASE,
    DOCUMENT,
    FIXED_CHUNK_CHARS,
    SEMANTIC_PERCENTILE,
    adjacent_similarities,
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    compute_idf,
    evaluate_strategy,
    split_sentences,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / document
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # good / answer survives
RED = "#8B3B4A"  # bad / fact shattered
SLATE = "#4A5B6E"  # neutral / chunk fill
AMBER = "#7A6528"  # highlight / boundary
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110


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


def fig_fixed_split_shatter() -> None:
    """Draw the naive fixed split over the document; show the boundary cutting through the date.

    Uses the renderer to measure the exact pixel extent of the text BEFORE the cut, so the red cut
    bar lands precisely between 'Marc' and 'h 3rd, 2024' — no hand-tuned character-width guessing.
    """
    start = DOCUMENT.index("It was launched")
    excerpt = DOCUMENT[start : start + 78].replace("\n", " ")
    cut_offset = DOCUMENT.index("March") + 4 - start  # boundary falls after 'Marc' (inside "March")
    before, after = excerpt[:cut_offset], excerpt[cut_offset:]

    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    ax.axis("off")
    fig.canvas.draw()  # realise the renderer so we can measure text extents
    renderer = fig.canvas.get_renderer()

    ax.text(0.015, 0.80, "Naive fixed-size cut (size=140) lands inside the fact:", fontsize=11,
            fontweight="bold", color=INK, transform=ax.transAxes)
    x0, y = 0.015, 0.46
    t_before = ax.text(x0, y, before, fontsize=12, family="monospace", color=SLATE,
                       transform=ax.transAxes, va="center")
    # measure 'before' width in axes-fraction to position the cut bar and the 'after' span exactly
    bbox = t_before.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    cut_x = inv.transform((bbox.x1, bbox.y0))[0]
    ax.text(cut_x, y, after, fontsize=12, family="monospace", color=RED, transform=ax.transAxes,
            va="center")
    ax.axvline(cut_x, ymin=0.30, ymax=0.66, color=RED, linewidth=2.6)
    ax.text(cut_x, 0.74, "chunk boundary", fontsize=9, color=RED, ha="center", transform=ax.transAxes)
    ax.text(0.015, 0.12,
            f"→ '{ANSWER_PHRASE}' is split 'Marc' | 'h 3rd, 2024'. Neither chunk holds the fact "
            "whole, so retrieval can never return it.",
            fontsize=9.5, style="italic", color=INK, transform=ax.transAxes, va="center")
    ax.set_title("Cut mid-fact → the fact becomes unretrievable", fontsize=12.5, pad=10)
    _save(fig, "rag02_fixed_split_shatter.png")


def fig_size_recall_precision() -> None:
    """Chunk size vs recall (answer survives whole) and precision (answer's share of the chunk)."""
    sizes = np.array([40, 60, 80, 100, 120, 140, 160, 200, 260, 340, 440, 560])
    answer_len = len(ANSWER_PHRASE)
    recall, precision = [], []
    for size in sizes:
        chunks = chunk_fixed(DOCUMENT, size=int(size), overlap=0)
        # recall: does any chunk contain the whole answer phrase? (1.0 if survivable, else 0.0)
        survives = any(ANSWER_PHRASE in c for c in chunks)
        recall.append(1.0 if survives else 0.0)
        # precision proxy: answer length / chunk length for the chunk holding it (signal density);
        # if it doesn't survive, the best a chunk can offer is a fragment -> 0 usable precision
        if survives:
            holder = next(c for c in chunks if ANSWER_PHRASE in c)
            precision.append(answer_len / len(holder))
        else:
            precision.append(0.0)

    # the smooth EXPECTED recall 1 - f/S, averaged over random fact offsets (only valid for S > f)
    size_curve = np.linspace(answer_len + 1, sizes.max(), 200)
    expected_recall = 1.0 - answer_len / size_curve

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _style_axis(ax)
    ax.plot(size_curve, expected_recall, color=GREEN, linewidth=1.4, linestyle="--", alpha=0.45,
            label="expected recall  1 − f/S  (avg over offsets)")
    ax.plot(sizes, recall, marker="o", color=GREEN, linewidth=2.2, markersize=6,
            markeredgecolor=INK, label="realized recall (this fact): survives whole")
    ax.plot(sizes, precision, marker="s", color=BLUE, linewidth=2.2, markersize=6,
            markeredgecolor=INK, label="precision: answer's share of the chunk")
    ax.axvline(FIXED_CHUNK_CHARS, color=RED, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.annotate(f"naive size={FIXED_CHUNK_CHARS}\n(splits the fact)", (FIXED_CHUNK_CHARS, 0.5),
                color=RED, fontsize=8.5, ha="left", va="center",
                xytext=(8, 0), textcoords="offset points")
    ax.set_title("Chunk size: the recall vs precision tradeoff", fontsize=12, pad=12)
    ax.set_xlabel("chunk size (characters)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="center right", framealpha=0.95, fontsize=9)
    _save(fig, "rag02_size_recall_precision.png")


def fig_overlap_diagram() -> None:
    """Fixed chunks without vs with overlap — overlap re-includes the boundary region."""
    fig, (ax_no, ax_yes) = plt.subplots(2, 1, figsize=(8.8, 5.0))
    doc_len = 10  # ten abstract "tokens" for a clean schematic
    chunk_w = 4

    def _draw(ax, overlap, title):
        ax.set_xlim(-0.5, doc_len + 0.5)
        ax.set_ylim(-2.3, 1.2)
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        # the document as a strip of token cells
        for t in range(doc_len):
            ax.add_patch(plt.Rectangle((t, 0.3), 0.9, 0.5, facecolor="#EEF1F4", edgecolor=GRID))
        # the fact spans tokens 3-4 (the boundary region)
        ax.add_patch(plt.Rectangle((3, 0.3), 1.9, 0.5, facecolor=AMBER, alpha=0.35, edgecolor=AMBER))
        ax.text(3.95, 0.55, "fact", fontsize=8.5, ha="center", va="center", color=INK)
        step = chunk_w - overlap
        colors = [BLUE, PURPLE, SLATE, AMBER, BLUE]
        fact_lo, fact_hi = 3, 4  # the fact spans tokens 3-4
        start, ci = 0, 0
        while start < doc_len:
            width = min(chunk_w, doc_len - start)
            # does this chunk window [start, start+chunk_w) fully contain BOTH fact tokens?
            holds_fact = start <= fact_lo and fact_hi <= start + chunk_w - 1
            color = colors[ci % len(colors)]
            row_y = -1.2 - (ci % 2) * 0.62  # stagger rows so overlapping chunks are both visible
            ax.add_patch(plt.Rectangle((start, row_y), width, 0.5,
                                       facecolor=GREEN if holds_fact else color,
                                       alpha=0.6 if holds_fact else 0.45,
                                       edgecolor=GREEN if holds_fact else color,
                                       linewidth=2.6 if holds_fact else 1.0))
            tag = f"chunk {ci}" + ("  ✓ holds fact" if holds_fact else "")
            ax.text(start + width / 2, row_y + 0.25, tag, fontsize=8, ha="center", va="center",
                    color=INK, fontweight="bold" if holds_fact else "normal")
            start += step
            ci += 1

    _draw(ax_no, 0, "WITHOUT overlap — the fact (tokens 3–4) is split across chunk 0 | chunk 1")
    _draw(ax_yes, 2, "WITH overlap=2 — chunk 1 re-includes tokens 2–3, so it lands the fact whole (green)")
    fig.suptitle("Overlap rescues facts that straddle a chunk boundary", fontsize=12.5, color=INK, y=1.0)
    _save(fig, "rag02_overlap_diagram.png")


def fig_semantic_trace() -> None:
    """Adjacent-sentence cosine similarity across the doc, with the percentile cut line."""
    sentences = split_sentences(DOCUMENT)
    idf = compute_idf(sentences)
    sims = adjacent_similarities(sentences, idf)
    threshold = float(np.percentile(sims, SEMANTIC_PERCENTILE))
    gap_idx = np.arange(len(sims))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    _style_axis(ax)
    ax.plot(gap_idx, sims, marker="o", color=PURPLE, linewidth=2.2, markersize=8,
            markeredgecolor=INK, markerfacecolor=BLUE, zorder=3)
    ax.axhline(threshold, color=RED, linewidth=1.6, linestyle="--", alpha=0.85)
    ax.annotate(f"{SEMANTIC_PERCENTILE}th-percentile cut", (len(sims) - 1, threshold), color=RED,
                fontsize=9, ha="right", va="bottom", xytext=(0, 6), textcoords="offset points")
    for i, sim in enumerate(sims):
        if sim <= threshold:  # a dip that becomes a chunk boundary
            ax.scatter(i, sim, s=240, facecolors="none", edgecolors=RED, linewidths=2.2, zorder=4)
            ax.annotate("cut", (i, sim), color=RED, fontsize=9, ha="center", va="top",
                        xytext=(0, -10), textcoords="offset points")
    ax.set_title("Semantic chunking cuts where adjacent sentences diverge", fontsize=12, pad=12)
    ax.set_xlabel("gap between sentence i and i+1")
    ax.set_ylabel("cosine similarity of adjacent sentences")
    ax.set_xticks(gap_idx)
    ax.set_xticklabels([f"{i}|{i + 1}" for i in gap_idx])
    _save(fig, "rag02_semantic_trace.png")


def fig_strategy_compare() -> None:
    """Per-strategy #chunks and whether the answer survived + was retrieved top-1."""
    idf_sent = compute_idf(split_sentences(DOCUMENT))
    strategies = {
        "fixed\n(no overlap)": chunk_fixed(DOCUMENT, overlap=0),
        "fixed\n(+overlap)": chunk_fixed(DOCUMENT, overlap=40),
        "recursive": chunk_recursive(DOCUMENT),
        "semantic": chunk_semantic(DOCUMENT, idf_sent),
    }
    names, n_chunks, retrieved_ok = [], [], []
    for name, chunks in strategies.items():
        res = evaluate_strategy(name, chunks, compute_idf(chunks))
        names.append(name)
        n_chunks.append(res.n_chunks)
        retrieved_ok.append(res.answer_intact)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style_axis(ax)
    colors = [GREEN if ok else RED for ok in retrieved_ok]
    bars = ax.bar(names, n_chunks, color=colors, edgecolor=INK, linewidth=0.8, width=0.6)
    for bar, ok in zip(bars, retrieved_ok):
        label = "answer\nretrieved ✓" if ok else "answer\nlost ✗"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=8.5, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points")
    ax.set_title("Same document, four strategies: chunk count and whether the fact survived",
                 fontsize=11.5, pad=12)
    ax.set_ylabel("number of chunks")
    ax.set_ylim(0, max(n_chunks) * 1.3)
    green_proxy = plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor=INK, label="answer retrieved top-1")
    red_proxy = plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor=INK, label="answer lost / not top-1")
    ax.legend(handles=[green_proxy, red_proxy], loc="upper right", framealpha=0.95, fontsize=9)
    _save(fig, "rag02_strategy_compare.png")


def fig_overlap_token_cost() -> None:
    """Overlap fraction vs the extra tokens it costs (re-embedded duplicated boundary text)."""
    # With chunk size S and overlap O, step = S-O, so #chunks ~ L/(S-O); duplicated chars ~ O per gap.
    # Extra fraction over the no-overlap baseline = O / (S - O): the share of work that is re-embedding.
    overlaps = np.linspace(0, 0.6, 13)  # overlap as a FRACTION of chunk size
    extra_fraction = overlaps / (1 - overlaps)  # O/(S-O) with O,S in units of S -> overlap/(1-overlap)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _style_axis(ax)
    ax.plot(overlaps * 100, extra_fraction * 100, marker="o", color=AMBER, linewidth=2.2,
            markersize=6, markeredgecolor=INK, markerfacecolor=RED)
    ax.set_title("Overlap is not free: extra tokens embedded vs overlap fraction", fontsize=12, pad=12)
    ax.set_xlabel("overlap (% of chunk size)")
    ax.set_ylabel("extra tokens embedded (% over no-overlap)")
    # mark a common practical choice (~10-15% overlap)
    ax.axvline(12.5, color=SLATE, linewidth=1.4, linestyle=":", alpha=0.8)
    ax.annotate("typical ~10–15%\n(≈14% extra cost)", (12.5, extra_fraction[3] * 100 + 8),
                color=INK, fontsize=8.5, ha="left", va="bottom",
                xytext=(6, 0), textcoords="offset points")
    _save(fig, "rag02_overlap_token_cost.png")


def main() -> None:
    fig_fixed_split_shatter()
    fig_size_recall_precision()
    fig_overlap_diagram()
    fig_semantic_trace()
    fig_strategy_compare()
    fig_overlap_token_cost()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

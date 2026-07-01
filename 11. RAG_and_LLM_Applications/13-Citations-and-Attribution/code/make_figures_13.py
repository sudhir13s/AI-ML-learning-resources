"""Static figure generator for 13-Citations-and-Attribution.

Imports the SAME canonical functions the page and notebook use (citations_attribution.py, which
reuses ch5) so every plotted number is the chapter's own -- no hand-typed values. Writes
muted-palette PNGs to the shared chapter image dir (../../images/) with the per-chapter prefix
`rag13_`.

    python make_figures_13.py

Figures produced:
  rag13_attribution_pipeline.png  -- the mechanism map: retrieve -> generate -> decompose into
                                     claims -> attribute each claim to its best passage -> verify
                                     (claim entailed by cited passage?). Schematic diagram.
  rag13_cited_answer.png          -- the reader-facing artifact: the answer with each grounded claim
                                     drawn to the passage it cites, and the hallucinated claim flagged
                                     UNCITABLE in red -- the whole point, on the demo's real verdicts.
  rag13_attribution_heatmap.png   -- the claim x passage cosine matrix: which passage each claim
                                     attributes to (argmax highlighted), the hallucination's row all
                                     below the bar.
  rag13_precision_recall.png      -- citation precision & recall on the clean (thresholded) attributor
                                     vs an over-citing one (threshold 0): precision drops, recall holds.
  rag13_coarse_vs_fine.png        -- coarse (which document) vs fine (which exact span) attribution --
                                     the granularity axis, schematic.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from citations_attribution import (
    ANSWER,
    QUESTION,
    SUPPORT_THRESHOLD,
    DenseRetriever,
    attribute_claims,
    build_golds,
    citation_precision,
    citation_recall,
    full_corpus,
    retrieve_passages,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # retrieval / data / passages
PURPLE = "#5D4A8A"  # process / generation
GREEN = "#2E7A5A"  # cited / supported / good
RED = "#8B3B4A"  # uncitable / hallucination / bad
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / threshold
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


def _box(ax, x, y, w, h, text, color, tcol="white", fs=8.6):
    """A filled rounded box with centred text -- the flow-diagram primitive."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcol, fontsize=fs, fontweight="bold")


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ================================================================================================
# Figure 1 -- the attribution mechanism map
# ================================================================================================


def fig_attribution_pipeline() -> None:
    """The post-hoc attribution pipeline: retrieve -> generate -> decompose -> attribute -> verify.

    Schematic mechanism diagram (labelled). The point: attribution is a stage AFTER generation that
    maps each claim back to the passage that supports it, and the VERIFY step (is the claim entailed
    by the cited passage?) is what a cosine matcher only approximates -- an NLI judge does it right.
    """
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Post-hoc attribution: attach every claim to the passage it came from",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # top pipeline row
    _box(ax, 0.01, 0.60, 0.15, 0.16, "question +\nretrieved\npassages", BLUE, fs=8.2)
    _box(ax, 0.20, 0.60, 0.15, 0.16, "GENERATE\nanswer", PURPLE, fs=8.6)
    _box(ax, 0.39, 0.60, 0.17, 0.16, "DECOMPOSE\ninto claims", SLATE, fs=8.4)
    _box(ax, 0.60, 0.60, 0.18, 0.16, "ATTRIBUTE\neach claim →\nbest passage", BLUE, fs=8.0)
    _box(ax, 0.82, 0.60, 0.16, 0.16, "VERIFY\nclaim ⊆\ncited passage?", GREEN, fs=8.0)
    for x0, x1 in ((0.16, 0.20), (0.35, 0.39), (0.56, 0.60), (0.78, 0.82)):
        ax.annotate("", xy=(x1, 0.68), xytext=(x0, 0.68), arrowprops=dict(arrowstyle="->", color=INK, lw=1.7))

    # verify branch: pass -> cite; fail -> uncitable
    ax.annotate("", xy=(0.83, 0.40), xytext=(0.88, 0.58), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax.annotate("", xy=(0.97, 0.40), xytext=(0.92, 0.58), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    _box(ax, 0.74, 0.28, 0.11, 0.12, "assign\n[n]", GREEN, fs=8.4)
    _box(ax, 0.88, 0.28, 0.11, 0.12, "flag\nUNCITABLE", RED, fs=7.8)

    # entailment caveat callout under ATTRIBUTE / VERIFY
    ax.add_patch(plt.Rectangle((0.58, 0.02), 0.40, 0.20, facecolor=AMBER, alpha=0.10, edgecolor=AMBER, linewidth=1.3))
    ax.text(0.78, 0.16, "the honest bar: does the passage ENTAIL the claim?", ha="center",
            fontsize=8.4, color=AMBER, fontweight="bold")
    ax.text(0.78, 0.085, "cosine ≈ topical match (our proxy) — an NLI model\n"
            "checks true entailment (ALCE / AIS). cosine ≠ entailment.",
            ha="center", va="center", fontsize=7.8, color=INK)

    # left callout: post-hoc vs generation-time
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.53, 0.20, facecolor=PURPLE, alpha=0.08, edgecolor=PURPLE, linewidth=1.3))
    ax.text(0.275, 0.16, "two regimes", ha="center", fontsize=8.6, color=PURPLE, fontweight="bold")
    ax.text(0.275, 0.085, "POST-HOC: map claims back AFTER generation (this page — works on any output)\n"
            "GENERATION-TIME: model emits [1][2] AS it writes (Anthropic / Vertex / LlamaIndex)",
            ha="center", va="center", fontsize=7.6, color=INK)
    _save(fig, "rag13_attribution_pipeline.png")


# ================================================================================================
# Figure 2 -- the cited answer: claims drawn to their passages, hallucination flagged red
# ================================================================================================


def fig_cited_answer(dense: DenseRetriever, passages: tuple[str, ...]) -> None:
    """The reader-facing artifact: each claim drawn to the passage it cites; the hallucination in red.

    Runs the REAL attributor. Left column = the answer's claims (green if cited, red if uncitable);
    right column = the retrieved passages [1..k]; a line joins each cited claim to its passage with
    the cosine on it. The uncitable claim has NO line and is boxed red -- the unverifiable claim, made
    obvious. Every claim, citation and cosine is the demo's real output.
    """
    attributions = attribute_claims(dense, ANSWER, passages)
    n_claims = len(attributions)
    n_pass = len(passages)

    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Every claim attached to its source — the hallucination has nowhere to hide",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.18, 0.9, "ANSWER claims", ha="center", fontsize=10.0, color=INK, fontweight="bold")
    ax.text(0.78, 0.9, "RETRIEVED passages", ha="center", fontsize=10.0, color=BLUE, fontweight="bold")

    claim_x, claim_w = 0.02, 0.34
    pass_x, pass_w = 0.60, 0.39
    claim_ys = np.linspace(0.72, 0.14, n_claims)
    pass_ys = np.linspace(0.74, 0.12, n_pass)

    # passages on the right
    pass_centers = {}
    for i, (y, passage) in enumerate(zip(pass_ys, passages), start=1):
        ax.add_patch(plt.Rectangle((pass_x, y - 0.06), pass_w, 0.10, facecolor=BLUE, alpha=0.12,
                     edgecolor=BLUE, linewidth=1.2))
        ax.text(pass_x + 0.015, y - 0.01, f"[{i}] " + _short(passage, 54), ha="left", va="center",
                fontsize=8.0, color=INK)
        pass_centers[i] = (pass_x, y - 0.01)

    # claims on the left, each drawn to its cited passage
    for attr, y in zip(attributions, claim_ys):
        cited = attr.citation is not None
        face = GREEN if cited else RED
        ax.add_patch(plt.Rectangle((claim_x, y - 0.06), claim_w, 0.10, facecolor=face, alpha=0.14,
                     edgecolor=face, linewidth=1.4))
        marker = f"[{attr.citation}]" if cited else "[?]"
        ax.text(claim_x + 0.015, y - 0.01, _short(attr.claim, 44), ha="left", va="center",
                fontsize=8.0, color=INK)
        ax.text(claim_x + claim_w - 0.012, y - 0.01, marker, ha="right", va="center",
                fontsize=9.0, color=face, fontweight="bold")
        if cited:
            px, py = pass_centers[attr.citation]
            ax.annotate("", xy=(px, py), xytext=(claim_x + claim_w, y - 0.01),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.7, alpha=0.85))
            # place the cosine label just off the claim end (not the crossing midpoint) so the two
            # arrows' labels never collide where the lines cross
            ax.text(claim_x + claim_w + 0.035, y - 0.01, f"cos {attr.best_score:.2f}",
                    ha="left", va="center", fontsize=7.8, color=GREEN, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=GREEN, alpha=0.92, linewidth=0.8))
        else:
            ax.text(claim_x + claim_w + 0.035, y - 0.01, f"UNCITABLE — best cos {attr.best_score:.2f} < {SUPPORT_THRESHOLD}",
                    ha="left", va="center", fontsize=7.8, color=RED, fontweight="bold", style="italic")

    ax.text(0.5, 0.03, "grounded claims cite a real passage (green);  the fabricated claim matches none "
            "→ flagged [?] (red).  cosines are real all-MiniLM output.",
            ha="center", fontsize=8.6, color=INK, style="italic")
    _save(fig, "rag13_cited_answer.png")


# ================================================================================================
# Figure 3 -- the claim x passage attribution heatmap
# ================================================================================================


def fig_attribution_heatmap(dense: DenseRetriever, passages: tuple[str, ...]) -> None:
    """The claim x passage cosine matrix: each claim's attribution row, argmax highlighted.

    Runs the REAL attributor and plots the full claim-vs-passage cosine matrix. The argmax cell in
    each row (the cited passage) is outlined; the hallucination's row is ALL below the threshold, so
    it has no cell above the bar -> uncitable. This is the attribution decision, as a heatmap.
    """
    attributions = attribute_claims(dense, ANSWER, passages)
    matrix = np.array([attr.scores for attr in attributions])  # (n_claims, n_passages), real cosines
    n_claims, n_pass = matrix.shape

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    im = ax.imshow(matrix, cmap="BuGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_pass))
    ax.set_xticklabels([f"[{i+1}]\n{_short(p, 22)}" for i, p in enumerate(passages)], fontsize=7.6)
    ax.set_yticks(range(n_claims))
    ax.set_yticklabels([f"claim {i+1}\n{_short(a.claim, 26)}" for i, a in enumerate(attributions)], fontsize=7.8)
    ax.set_xlabel("retrieved passage", fontsize=9.5, color=INK)
    ax.set_title("Attribution matrix: each claim's cosine to every passage (argmax = the citation)",
                 fontsize=11.0, color=INK, fontweight="bold", pad=12)

    for i in range(n_claims):
        best_j = int(np.argmax(matrix[i]))
        for j in range(n_pass):
            val = matrix[i, j]
            is_cited = (j == best_j) and attributions[i].citation is not None
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.6,
                    color="white" if val > 0.55 else INK,
                    fontweight="bold" if is_cited else "normal")
            if is_cited:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=GREEN, linewidth=3.0))
            elif j == best_j and attributions[i].citation is None:  # uncitable row: argmax still below bar
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=RED,
                             linewidth=2.6, linestyle="--"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"cosine (support threshold τ = {SUPPORT_THRESHOLD})", fontsize=8.6, color=INK)
    cbar.ax.axhline(SUPPORT_THRESHOLD, color=AMBER, linewidth=2.0)
    ax.text(0.5, -0.30, "green box = citation assigned;  red dashed = argmax still below τ → UNCITABLE "
            "(the hallucination)", transform=ax.transAxes, ha="center", fontsize=8.4, color=INK, style="italic")
    _save(fig, "rag13_attribution_heatmap.png")


# ================================================================================================
# Figure 4 -- citation precision & recall: thresholded vs over-citing
# ================================================================================================


def fig_precision_recall(dense: DenseRetriever, passages: tuple[str, ...]) -> None:
    """Citation precision & recall: the clean thresholded attributor vs an over-citing one.

    Runs the REAL metrics both ways. Thresholded (τ=0.5): precision 1.0, recall 1.0 -- cites only the
    grounded claims. Over-citing (τ=0): the hallucination is force-cited, adding a false citation, so
    precision drops while recall (supportable claims only) holds. The precision/recall dial, measured.
    """
    golds = build_golds(passages)
    thresholded = attribute_claims(dense, ANSWER, passages, threshold=SUPPORT_THRESHOLD)
    overcited = attribute_claims(dense, ANSWER, passages, threshold=0.0)
    p_thr, r_thr = citation_precision(thresholded, golds), citation_recall(thresholded, golds)
    p_over, r_over = citation_precision(overcited, golds), citation_recall(overcited, golds)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _style_axis(ax)
    groups = ["citation precision", "citation recall"]
    x = np.arange(len(groups))
    width = 0.36
    thr_vals = [p_thr, r_thr]
    over_vals = [p_over, r_over]
    b1 = ax.bar(x - width / 2, thr_vals, width, label=f"thresholded (τ={SUPPORT_THRESHOLD})",
                color=GREEN, edgecolor=INK, linewidth=0.9)
    b2 = ax.bar(x + width / 2, over_vals, width, label="over-citing (τ=0, cite everything)",
                color=RED, edgecolor=INK, linewidth=0.9)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                        fontsize=9.0, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.0)
    ax.set_ylim(0, 1.28)  # extra headroom so the legend clears the tallest bars
    ax.set_ylabel("score", fontsize=10.0)
    ax.set_title("Over-citation tanks precision, leaves recall unchanged (measured)",
                 fontsize=11.5, color=INK, fontweight="bold", pad=10)
    ax.legend(fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.95)
    # annotation sits in the empty valley between the precision drop and the recall bars
    ax.annotate(f"forcing a citation on the\nhallucination = a FALSE cite\n→ precision {p_thr:.2f} → {p_over:.2f}",
                xy=(0 + width / 2, p_over), xytext=(0.32, 0.28), textcoords="data", fontsize=8.4, color=RED,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=RED, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    _save(fig, "rag13_precision_recall.png")


# ================================================================================================
# Figure 5 -- coarse (which document) vs fine (which span) attribution
# ================================================================================================


def fig_coarse_vs_fine(passages: tuple[str, ...]) -> None:
    """Coarse (which document) vs fine (which exact span) attribution -- the granularity axis.

    Schematic. Left: coarse attribution points a claim at a whole document ("[Doc A]") -- cheap, but
    the reader must still hunt the sentence. Right: fine attribution points at the exact span (the
    highlighted sentence) -- what the Anthropic Citations API returns via start/end char indices, the
    verifiable ideal. The claim + passage shown are the demo's real resolution claim + its source.
    """
    claim = "Its hyperspectral imager has a ground resolution of 4 meters."
    passage = passages[0] if "resolution" in passages[0] else next(p for p in passages if "resolution" in p)
    span_start = passage.find("ground resolution")
    span_end = passage.find("meters.") + len("meters.")
    before, span = passage[:span_start], passage[span_start:span_end]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for ax in (ax_l, ax_r):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # claim shown on both
    for ax, title, tcol in ((ax_l, "COARSE — cite the document", SLATE), (ax_r, "FINE — cite the exact span", GREEN)):
        ax.text(0.5, 0.93, title, ha="center", fontsize=11.5, color=tcol, fontweight="bold")
        ax.add_patch(plt.Rectangle((0.05, 0.66), 0.9, 0.14, facecolor=PURPLE, alpha=0.12,
                     edgecolor=PURPLE, linewidth=1.2))
        ax.text(0.5, 0.73, "claim: " + claim, ha="center", va="center", fontsize=8.4, color=INK)

    # coarse: arrow to the whole doc box
    ax_l.annotate("", xy=(0.5, 0.42), xytext=(0.5, 0.65), arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))
    ax_l.text(0.5, 0.53, "cite [Doc]", ha="center", fontsize=8.6, color=SLATE, fontweight="bold")
    ax_l.add_patch(plt.Rectangle((0.08, 0.14), 0.84, 0.26, facecolor=SLATE, alpha=0.12,
                   edgecolor=SLATE, linewidth=1.4))
    ax_l.text(0.5, 0.27, _short(passage, 66), ha="center", va="center", fontsize=8.0, color=INK)
    ax_l.text(0.5, 0.055, "reader still has to FIND the sentence — coarse but cheap",
              ha="center", fontsize=8.0, color=SLATE, style="italic")

    # fine: arrow to the highlighted span within the doc
    ax_r.annotate("", xy=(0.5, 0.42), xytext=(0.5, 0.65), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax_r.text(0.5, 0.53, "cite [Doc: chars n–m]", ha="center", fontsize=8.6, color=GREEN, fontweight="bold")
    ax_r.add_patch(plt.Rectangle((0.08, 0.14), 0.84, 0.26, facecolor=BLUE, alpha=0.08,
                   edgecolor=BLUE, linewidth=1.4))
    # render the passage with the cited span highlighted (three text pieces)
    ax_r.text(0.10, 0.27, _short(before, 26), ha="left", va="center", fontsize=7.8, color=SLATE)
    ax_r.add_patch(plt.Rectangle((0.10, 0.185), 0.80, 0.055, facecolor=GREEN, alpha=0.28, edgecolor=GREEN, linewidth=1.2))
    ax_r.text(0.5, 0.212, span, ha="center", va="center", fontsize=8.0, color=INK, fontweight="bold")
    ax_r.text(0.5, 0.055, "reader jumps straight to the exact evidence — the verifiable ideal",
              ha="center", fontsize=8.0, color=GREEN, style="italic")

    fig.suptitle("Attribution granularity: coarse (which document) vs fine (which exact span)",
                 fontsize=12.5, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag13_coarse_vs_fine.png")


def main() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    passages, _ = retrieve_passages(dense, corpus, k=3)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend} | question: {QUESTION}")
    fig_attribution_pipeline()
    fig_cited_answer(dense, passages)
    fig_attribution_heatmap(dense, passages)
    fig_precision_recall(dense, passages)
    fig_coarse_vs_fine(passages)


if __name__ == "__main__":
    main()

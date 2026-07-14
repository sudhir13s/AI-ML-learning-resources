"""Static figure generator for 08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.

Imports the SAME canonical functions the page and notebook use (advanced_rag.py, which reuses ch5 &
ch7) so every plotted number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs
to the shared chapter image dir (../../images/) with the per-chapter prefix `rag08_`.

    python make_figures_08.py

Figures produced:
  rag08_parent_child.png       -- the child->parent hierarchy + retrieve-small-feed-large: retrieve on
                                  small children, return the deduped parent sections for generation.
  rag08_precision_context.png  -- precision (sharp child retrieval, stays on target) x context (the
                                  parent delivers Nx more chars): the measured multiplier.
  rag08_ragfusion_recovery.png -- the vague query MISSES the gold child at top-2; RAG-Fusion recovers
                                  it (recall 0 -> 1) -- the ch7 machinery, applied to children.
  rag08_selfrag_loop.png       -- the Self-RAG reflect loop: Retrieve? -> generate -> ISREL/ISSUP/ISUSE
                                  critique -> accept or regenerate/retrieve-more (reflection tokens).
  rag08_support_check.png      -- measured support scores (grounded / off-topic / same-structure swap)
                                  vs the threshold, incl. the honest cosine-vs-entailment limitation.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from advanced_rag import (
    RRF_K,
    SUPPORT_THRESHOLD,
    ParentDocumentRetriever,
    build_hierarchy,
    recall_at_k,
    retrieve_multiquery,
    support_score,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / child
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # parent / grounded / accept
RED = "#8B3B4A"  # miss / reject / danger
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / critique
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


def fig_parent_child(retriever: ParentDocumentRetriever) -> None:
    """The child->parent hierarchy and the retrieve-small / read-large flow, with the measured hit.

    Left: every child sentence (blue) mapped to its parent section (green) -- the index. Right: the
    flow for the demo query: retrieve the top child (highlighted), then RETURN its deduped parent for
    generation. The whole point: sharp retrieval on a small unit, full context from the larger parent.
    """
    parents, children = retriever.parents, retriever.children
    query = "What keeps the local solar time of each pass roughly constant?"
    result = retriever.retrieve(query, k=3)
    top_child = result.child_indices[0]

    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1.06)  # a little headroom so the right-hand flow column never clips
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.985, "Parent-document retrieval: retrieve small children, feed the larger parents",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # --- left column: children; middle: parents ---
    n_children = len(children)
    child_y = np.linspace(0.86, 0.10, n_children)
    parent_of = [c.parent_id for c in children]
    # parent block y-centres (average of their children's y)
    parent_ys = {}
    for pid in range(len(parents)):
        ys = [child_y[i] for i in range(n_children) if parent_of[i] == pid]
        parent_ys[pid] = float(np.mean(ys))

    def short(text, n=46):
        return text if len(text) <= n else text[: n - 1] + "…"

    # draw parent boxes (green) in the middle
    for pid, p in enumerate(parents):
        y = parent_ys[pid]
        ax.add_patch(plt.Rectangle((0.46, y - 0.11), 0.30, 0.22, facecolor=GREEN, alpha=0.16,
                     edgecolor=GREEN, linewidth=1.4))
        ax.text(0.61, y + 0.075, f"PARENT: # {p.heading}", ha="center", fontsize=8.6,
                fontweight="bold", color=GREEN)
    # draw children (blue) on the left, link each to its parent
    for i, c in enumerate(children):
        y = child_y[i]
        is_top = i == top_child
        face = AMBER if is_top else BLUE
        ax.text(0.02, y, f"child[{i}]: {short(c.text)}", ha="left", va="center", fontsize=7.6,
                color=INK, bbox=dict(boxstyle="round,pad=0.28", facecolor=face,
                alpha=0.22 if not is_top else 0.35, edgecolor=face))
        ax.annotate("", xy=(0.46, parent_ys[c.parent_id]), xytext=(0.44, y),
                    arrowprops=dict(arrowstyle="->", color=GRID if not is_top else AMBER,
                    lw=0.8 if not is_top else 1.8))

    # --- right: the flow for the demo query ---
    ax.annotate("", xy=(0.80, 0.5), xytext=(0.77, 0.5),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    steps = [
        (f"query:\n{short(query, 30)}", SLATE, 0.86),
        (f"retrieve #1 child[{top_child}]\n(cos {result.child_scores[0]:.3f}) — SHARP", AMBER, 0.63),
        (f"map → dedupe parents\n{list(result.parent_ids)}", PURPLE, 0.40),
        (f"feed PARENT # {parents[result.parent_ids[0]].heading}\nto LLM — FULL CONTEXT", GREEN, 0.15),
    ]
    step_x, step_w = 0.80, 0.24
    step_cx = step_x + step_w / 2
    for text, color, y in steps:
        ax.add_patch(plt.Rectangle((step_x, y - 0.075), step_w, 0.13, facecolor=color, alpha=0.9,
                     edgecolor=INK, linewidth=1.0))
        ax.text(step_cx, y, text, ha="center", va="center", color="white", fontsize=7.6, fontweight="bold")
    for y0, y1 in ((0.785, 0.705), (0.555, 0.475), (0.325, 0.225)):
        ax.annotate("", xy=(step_cx, y1), xytext=(step_cx, y0),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    _save(fig, "rag08_parent_child.png")


def fig_precision_context(retriever: ParentDocumentRetriever) -> None:
    """Precision (sharp child retrieval) x Context (the parent delivers Nx more chars) -- measured.

    Left bar-pair: chars fed to the LLM by child-only vs parent-doc (the measured context multiplier).
    Right: the retrieval stays sharp -- the top child is in the correct target section either way, so
    parent-doc buys context WITHOUT sacrificing retrieval precision. Numbers are the chapter's own.
    """
    parents, children = retriever.parents, retriever.children
    query = "What keeps the local solar time of each pass roughly constant?"
    result = retriever.retrieve(query, k=3)
    top_child = children[result.child_indices[0]]
    parent = parents[top_child.parent_id]
    child_chars = len(top_child.text)
    parent_chars = len(parent.text)
    mult = parent_chars / child_chars

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    _style_axis(ax)
    bars = ax.bar(["child-only\n(top-1 sentence)", "parent-document\n(full section)"],
                  [child_chars, parent_chars], color=[BLUE, GREEN], edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, val in zip(bars, [child_chars, parent_chars]):
        ax.annotate(f"{val} chars", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=10, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontweight="bold")
    ax.annotate(f"{mult:.1f}× more context,\nSAME sharp retrieval\n(both hit the '{parent.heading}' section)",
                xy=(1, parent_chars), xytext=(0.28, parent_chars * 1.08), fontsize=9.5, color=GREEN,
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4, alpha=0.7))
    ax.set_ylabel("characters of context fed to the LLM")
    ax.set_title("Retrieve small, read large: parent-doc adds context, keeps retrieval sharp (measured)",
                 fontsize=11, pad=12)
    ax.set_ylim(0, parent_chars * 1.35)
    _save(fig, "rag08_precision_context.png")


def fig_ragfusion_recovery(retriever: ParentDocumentRetriever) -> None:
    """The vague query MISSES the gold child at top-2; RAG-Fusion recovers it (recall 0 -> 1).

    Reuses ch7's retrieve_multiquery over the child index. Left: raw single-query top-2 (gold absent,
    red). Right: fused reformulations' top-2 (gold present, green). The recall bars below make the
    0 -> 1 lift explicit -- the chapter's measured numbers.
    """
    children = retriever.children
    fusion_k = 2
    vague = "What are the imaging capabilities of Helios-7?"
    reformulations = (
        "How many spectral bands does Helios-7 capture?",
        "What wavelength range does the imager cover?",
        "What kind of imager is on Helios-7?",
    )
    gold = next(i for i, c in enumerate(children) if "spectral bands" in c.text)
    raw = retriever._dense.search(vague, k=fusion_k).indices  # noqa: SLF001
    fused = retrieve_multiquery(retriever._dense, (vague, *reformulations), k=fusion_k, k_rrf=RRF_K)  # noqa: SLF001
    raw_recall = recall_at_k(raw, gold)
    fused_recall = recall_at_k(fused, gold)

    fig, (ax_lists, ax_rec) = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.5, 1.0]})

    # --- left: the two top-2 lists ---
    ax_lists.axis("off")
    ax_lists.set_xlim(0, 1)
    ax_lists.set_ylim(0, 1)
    ax_lists.text(0.5, 0.96, f"vague query: “{vague}”", ha="center", fontsize=9.5, style="italic", color=INK)
    ax_lists.text(0.25, 0.82, "RAW single query", ha="center", fontsize=10, fontweight="bold", color=RED)
    ax_lists.text(0.75, 0.82, "RAG-Fusion (multi-query + RRF)", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    for rank, (rc, fc) in enumerate(zip(raw, fused), start=1):
        y = 0.66 - (rank - 1) * 0.16
        for x, cid in ((0.25, rc), (0.75, fc)):
            is_gold = cid == gold
            face = GREEN if is_gold else SLATE
            label = f"#{rank} child[{cid}]" + ("  ← GOLD" if is_gold else "")
            ax_lists.text(x, y, label, ha="center", fontsize=9.5, color=INK,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=face,
                          alpha=0.2 if not is_gold else 0.32, edgecolor=face))
    ax_lists.text(0.25, 0.20, "gold MISSING\n(the specific fact\nnever surfaces)", ha="center",
                  fontsize=8.8, color=RED, fontweight="bold")
    ax_lists.text(0.75, 0.20, "gold RECOVERED\n(a reformulation's\nlist carries it up)", ha="center",
                  fontsize=8.8, color=GREEN, fontweight="bold")

    # --- right: recall bars ---
    _style_axis(ax_rec)
    bars = ax_rec.bar(["raw\nquery", "RAG-\nFusion"], [raw_recall, fused_recall],
                      color=[RED, GREEN], edgecolor=INK, linewidth=0.9, width=0.5)
    for bar, val in zip(bars, [raw_recall, fused_recall]):
        ax_rec.annotate(f"{val:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        fontsize=13, color=INK, ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontweight="bold")
    ax_rec.set_ylabel(f"recall@{fusion_k}  (gold child found?)")
    ax_rec.set_ylim(0, 1.25)
    ax_rec.set_title("recall 0 → 1", fontsize=11, pad=8)
    fig.suptitle("RAG-Fusion recovers a fact the single vague query misses (measured, top-2)",
                 fontsize=11.5, y=1.0, color=INK, fontweight="bold")
    _save(fig, "rag08_ragfusion_recovery.png")


def fig_selfrag_loop() -> None:
    """The Self-RAG reflect loop: Retrieve? -> generate -> ISREL/ISSUP/ISUSE critique -> accept/redo.

    A schematic of the reflection-token control flow (Asai et al. 2023). Not a measurement -- it is the
    mechanism diagram; the reflection-token DEFINITIONS are labelled from the paper. The measured
    grounding signal (ISSUP proxy) is the separate support-check figure.
    """
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "Self-RAG: retrieve on demand, then critique your own grounding",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    def box(x, y, w, h, text, color, tcol="white", fs=8.6):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcol, fontsize=fs, fontweight="bold")

    box(0.36, 0.83, 0.28, 0.10, "query x", SLATE)
    box(0.30, 0.65, 0.40, 0.11, "Retrieve?  (reflection token)\nyes / no / continue", AMBER)
    ax.annotate("", xy=(0.5, 0.76), xytext=(0.5, 0.83), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    # no-retrieve branch
    box(0.03, 0.46, 0.24, 0.10, "no → answer from\nparametric memory", PURPLE, fs=8.0)
    ax.annotate("no (self-contained)", xy=(0.15, 0.56), xytext=(0.30, 0.66),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.3), fontsize=7.8, color=SLATE, ha="center")
    # yes-retrieve branch
    box(0.55, 0.46, 0.30, 0.10, "yes → retrieve passages d", BLUE, fs=8.2)
    ax.annotate("yes (needs grounding)", xy=(0.70, 0.56), xytext=(0.66, 0.645),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3), fontsize=7.8, color=BLUE, ha="center")
    box(0.55, 0.28, 0.30, 0.12, "generate answer y\n+ critique tokens:\nISREL · ISSUP · ISUSE", GREEN, fs=8.0)
    ax.annotate("", xy=(0.70, 0.40), xytext=(0.70, 0.46), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    box(0.55, 0.09, 0.30, 0.12, "grounded & useful?\naccept  —  else regenerate\nor retrieve more", RED, fs=8.0)
    ax.annotate("", xy=(0.70, 0.21), xytext=(0.70, 0.28), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    # feedback loop
    ax.annotate("", xy=(0.55, 0.34), xytext=(0.50, 0.15),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, connectionstyle="arc3,rad=-0.4"))
    ax.text(0.44, 0.24, "not supported →\nregenerate", fontsize=7.6, color=RED, ha="center", style="italic")

    # reflection-token legend (definitions from Asai et al. 2023)
    ax.text(0.02, 0.045, "Reflection tokens (Asai et al. 2023):  "
            "Retrieve = when to retrieve (yes/no/continue) · "
            "ISREL = passage relevant? · ISSUP = output supported by evidence? · "
            "ISUSE = response useful? (1–5)",
            fontsize=7.4, color=INK, ha="left", va="bottom", wrap=True)
    _save(fig, "rag08_selfrag_loop.png")


def fig_support_check(retriever: ParentDocumentRetriever) -> None:
    """Measured support scores (grounded / off-topic / same-structure swap) vs the threshold.

    The chapter's own numbers: the grounded claim clears the bar, the off-topic hallucination is
    rejected, and the same-structure date-swap SLIPS PAST it -- the honest cosine-vs-entailment limit
    (why Self-RAG trains an ISSUP token instead of thresholding cosine). Colours encode accept/reject.
    """
    parents, children = retriever.parents, retriever.children  # noqa: F841 -- parents used via retriever
    launch_res = retriever.retrieve("When was Helios-7 launched?", k=3)
    ctx = retriever.parent_context(launch_res)
    dense = retriever._dense  # noqa: SLF001
    claims = [
        ("grounded\n(true)", "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport.", True),
        ("off-topic\nhallucination", "Photosynthesis converts carbon dioxide and water into glucose.", False),
        ("same-structure\nfalse (date swap)", "Helios-7 launched in July 2021 from Cape Canaveral.", None),
    ]
    labels = [c[0] for c in claims]
    scores = [support_score(dense, c[1], ctx) for c in claims]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _style_axis(ax)
    colors = []
    for s in scores:
        colors.append(GREEN if s >= SUPPORT_THRESHOLD else RED)
    bars = ax.bar(labels, scores, color=colors, edgecolor=INK, linewidth=0.9, width=0.6)
    for bar, s in zip(bars, scores):
        verdict = "ACCEPT" if s >= SUPPORT_THRESHOLD else "REJECT"
        ax.annotate(f"{s:.3f}\n{verdict}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=9, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontweight="bold")
    ax.axhline(SUPPORT_THRESHOLD, color=INK, linewidth=1.3, linestyle="--", alpha=0.7)
    ax.annotate(f"support threshold = {SUPPORT_THRESHOLD}", (2.4, SUPPORT_THRESHOLD), color=INK,
                fontsize=8.6, ha="right", va="bottom", xytext=(0, 3), textcoords="offset points")
    # flag the limitation on the third bar
    ax.annotate("cosine ≈ topical, not factual:\nthis false claim SLIPS PAST\n(why Self-RAG trains ISSUP)",
                xy=(2, scores[2]), xytext=(1.3, 0.90), fontsize=8.3, color=RED, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, alpha=0.8))
    ax.set_ylabel("support score  (max cosine of claim to any context sentence)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Self-RAG support check (ISSUP proxy): grounded ✓, off-topic ✗ — and its honest limit",
                 fontsize=10.5, pad=12)
    _save(fig, "rag08_support_check.png")


def main() -> None:
    parents, children = build_hierarchy()
    retriever = ParentDocumentRetriever(parents, children)
    print(f"dense lens: {retriever.backend}")
    fig_parent_child(retriever)
    fig_precision_context(retriever)
    fig_ragfusion_recovery(retriever)
    fig_selfrag_loop()
    fig_support_check(retriever)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

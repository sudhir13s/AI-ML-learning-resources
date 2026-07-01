"""Static figure generator for 12-Long-Context-vs-RAG.

Imports the SAME canonical functions the page and notebook use (long_context_vs_rag.py, which reuses
ch5) so every OUR-MEASUREMENT number is the chapter's own -- no hand-typed values. CITED-EXTERNAL
numbers (Liu et al. U-curve, RULER, provider sizes) come from the module's clearly-labelled constants
and every figure that uses them says "external" on its face. Writes muted-palette PNGs to the shared
chapter image dir (../../images/) with the per-chapter prefix `rag12_`.

    python make_figures_12.py

Figures produced:
  rag12_cost_crossover.png    -- OUR MEASUREMENT: per-query token cost, stuff-everything (rises with
                                 corpus) vs RAG (flat); the crossover where RAG wins, and the widening gap.
  rag12_lost_in_middle.png    -- CITED EXTERNAL (Liu et al. 2023): the accuracy-vs-position U-curve
                                 (worst in the middle). Labelled as their reported data, not ours.
  rag12_dilution_proxy.png    -- OUR MEASUREMENT: the gold's retrieval margin shrinking as distractor
                                 context grows (an encoder proxy for context dilution).
  rag12_effective_context.png -- CITED EXTERNAL (RULER, Hsieh et al. 2024): effective context < advertised
                                 window. Illustrative of their finding, labelled external.
  rag12_decision_map.png      -- the stuff / RAG / hybrid decision map by corpus size vs window & cost.
  rag12_cost_multiplier.png   -- OUR MEASUREMENT: stuffing's cost multiplier vs RAG at growing corpus sizes.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from long_context_vs_rag import (
    LIU_U_CURVE,
    PROVIDERS,
    RETRIEVE_K,
    TOKENS_PER_CHUNK,
    DenseRetriever,
    cost_crossover_chunks,
    full_corpus,
    measure_dilution,
    query_cost_usd,
    rag_tokens,
    stuff_tokens,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # RAG / retrieve
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # good / RAG-wins
RED = "#8B3B4A"  # stuff / expensive / worst
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / crossover
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


def _ours_tag(ax, y: float = 0.02, va: str = "bottom") -> None:
    """Stamp 'OUR MEASUREMENT' on a figure built from the chapter's own computed numbers."""
    ax.text(0.99, y, "OUR MEASUREMENT (real, reproducible)", transform=ax.transAxes, ha="right",
            va=va, fontsize=7.4, color=GREEN, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))


def _external_tag(ax, source: str, y: float = 0.02, va: str = "bottom") -> None:
    """Stamp a 'CITED EXTERNAL' banner on a figure built from someone else's reported numbers."""
    ax.text(0.99, y, f"CITED EXTERNAL — {source} (not our measurement)", transform=ax.transAxes,
            ha="right", va=va, fontsize=7.2, color=RED, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))


def fig_cost_crossover() -> None:
    """OUR MEASUREMENT: per-query token cost of stuff-everything (rises) vs RAG (flat), + the crossover."""
    price = 3.00
    chunk_sizes = np.array([1, 2, 5, 8, 10, 20, 50, 100, 300, 1000])
    stuff_costs = [query_cost_usd(stuff_tokens(int(c)), price) for c in chunk_sizes]
    rag_cost = query_cost_usd(rag_tokens(), price)
    crossover = cost_crossover_chunks()

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    _style_axis(ax)
    ax.plot(chunk_sizes, stuff_costs, "-o", color=RED, linewidth=2.0, markersize=4.5,
            label="stuff whole corpus (grows per chunk, every query)")
    ax.axhline(rag_cost, color=BLUE, linewidth=2.0, linestyle="-",
               label=f"RAG: retrieve k={RETRIEVE_K} (flat = {rag_tokens()} tok/query)")
    ax.axvline(crossover, color=AMBER, linewidth=1.4, linestyle="--")
    ax.annotate(f"crossover ≈ {crossover} chunks\n(beyond here RAG is cheaper,\nand the gap only widens)",
                xy=(crossover, rag_cost), xytext=(crossover * 3.5, rag_cost * 4.5), fontsize=8.6,
                color=AMBER, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (chunks, log scale)")
    ax.set_ylabel("input cost per query, USD (log scale)")
    ax.legend(loc="upper left", fontsize=8.6, framealpha=0.9)
    ax.set_title("Cost per query: stuffing scales with the corpus; RAG stays flat (measured)",
                 fontsize=11, pad=10)
    _ours_tag(ax)
    _save(fig, "rag12_cost_crossover.png")


def fig_cost_multiplier() -> None:
    """OUR MEASUREMENT: stuffing's per-query token cost as a MULTIPLE of RAG's, at growing corpus sizes."""
    sizes = [10, 100, 1_000, 10_000, 100_000]
    rag_toks = rag_tokens()
    multipliers = [stuff_tokens(s) / rag_toks for s in sizes]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    _style_axis(ax)
    bars = ax.bar([f"{s:,}" for s in sizes], multipliers, color=RED, edgecolor=INK, linewidth=0.9, width=0.6)
    for bar, m in zip(bars, multipliers):
        ax.annotate(f"{m:,.0f}×", (bar.get_x() + bar.get_width() / 2, bar.get_height()), fontsize=9.5,
                    color=INK, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                    fontweight="bold")
    ax.axhline(1.0, color=BLUE, linewidth=1.6, linestyle="--")
    ax.text(0.02, 1.0, " RAG baseline (1×)", transform=ax.get_yaxis_transform(), color=BLUE,
            fontsize=8.4, va="bottom", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (chunks)")
    ax.set_ylabel("stuffing cost ÷ RAG cost per query (log)")
    ax.set_title("The cost of stuffing everything, relative to RAG — it explodes with corpus size (measured)",
                 fontsize=10.5, pad=10)
    ax.text(0.02, 0.97, "OUR MEASUREMENT (real, reproducible)", transform=ax.transAxes, ha="left",
            va="top", fontsize=7.4, color=GREEN, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
    _save(fig, "rag12_cost_multiplier.png")


def fig_lost_in_middle() -> None:
    """CITED EXTERNAL (Liu et al. 2023): the accuracy-vs-position U-curve — labelled as their data."""
    positions = [p for p, _ in LIU_U_CURVE]
    accs = [a for _, a in LIU_U_CURVE]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    _style_axis(ax)
    ax.plot(positions, accs, "-o", color=PURPLE, linewidth=2.2, markersize=7)
    worst_i = int(np.argmin(accs))
    ax.annotate("lost in the middle\n(worst accuracy)", xy=(positions[worst_i], accs[worst_i]),
                xytext=(positions[worst_i], accs[worst_i] + 12), fontsize=9.0, color=RED, ha="center",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate("best at the start", xy=(positions[0], accs[0]), xytext=(positions[0] + 1, accs[0] - 8),
                fontsize=8.4, color=GREEN, fontweight="bold")
    ax.annotate("nearly as good at the end", xy=(positions[-1], accs[-1]),
                xytext=(positions[-1] - 8, accs[-1] - 9), fontsize=8.4, color=GREEN, fontweight="bold")
    ax.set_xlabel("position of the relevant document within the context (of 20)")
    ax.set_ylabel("answer accuracy (%)")
    ax.set_ylim(min(accs) - 18, max(accs) + 18)
    ax.set_title("Lost in the middle: accuracy dips when the answer sits mid-context",
                 fontsize=11, pad=10)
    _external_tag(ax, "Liu et al. 2023, arXiv:2307.03172")
    _save(fig, "rag12_lost_in_middle.png")


def fig_dilution_proxy(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """OUR MEASUREMENT: the gold's retrieval margin shrinking as distractor context grows."""
    query = "When was the Helios-7 satellite launched?"
    gold = corpus[0]
    counts = (1, 5, 20, 100, 500)  # start at 1 so the margin curve is a clean shrink (0 = focused baseline)
    points = measure_dilution(dense, query, gold, corpus, counts)
    focused = measure_dilution(dense, query, gold, corpus, (0,))[0].margin  # RAG focused baseline

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    _style_axis(ax)
    xs = [p.n_distractors for p in points]
    margins = [p.margin for p in points]
    ax.plot(xs, margins, "-o", color=RED, linewidth=2.0, markersize=6,
            label="stuffed context: gold's margin over best distractor")
    ax.axhline(focused, color=BLUE, linewidth=2.0, linestyle="--",
               label=f"RAG focused (gold alone): margin {focused:.2f}")
    for p in points:
        ax.annotate(f"{p.margin:.2f}", (p.n_distractors, p.margin), fontsize=8.0, color=INK,
                    ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("# distractor passages surrounding the gold (log scale)")
    ax.set_ylabel("gold's cosine margin over best distractor")
    ax.legend(loc="upper right", fontsize=8.4, framealpha=0.9)
    ax.set_title("Context dilution (encoder proxy): more irrelevant context erodes the gold's lead",
                 fontsize=10.5, pad=10)
    _ours_tag(ax, y=0.55, va="center")
    _save(fig, "rag12_dilution_proxy.png")


def fig_effective_context() -> None:
    """CITED EXTERNAL (RULER, Hsieh et al. 2024): effective context < advertised window — labelled external.

    Illustrative of RULER's finding that a model's USABLE context is well below its advertised window.
    The bars pair each advertised window with an illustrative smaller effective length (labelled as
    external / illustrative of RULER's result, NOT our measurement or an exact per-model claim).
    """
    # advertised windows (verified provider docs) paired with an illustrative "effective" fraction to
    # depict RULER's qualitative finding (usable < advertised). Labelled external + illustrative.
    labels = ["128k\n(GPT-4o)", "200k\n(Claude)", "1M\n(Gemini 1.5)"]
    advertised = [128, 200, 1000]  # in thousands of tokens
    effective = [64, 120, 500]  # illustrative of RULER's "usable < advertised" — NOT exact per-model

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    _style_axis(ax)
    b1 = ax.bar(x - width / 2, advertised, width, label="advertised window", color=SLATE,
                edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + width / 2, effective, width, label="effective context (illustrative of RULER)",
                color=AMBER, edgecolor=INK, linewidth=0.8)
    for bars, vals in ((b1, advertised), (b2, effective)):
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v}k", (bar.get_x() + bar.get_width() / 2, bar.get_height()), fontsize=8.6,
                        color=INK, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                        fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("context length (thousands of tokens)")
    ax.legend(loc="upper left", fontsize=8.4, framealpha=0.9)
    ax.set_title("Advertised ≠ effective: usable context is smaller than the window (RULER)",
                 fontsize=10.5, pad=10)
    ax.text(0.4, 0.72, "CITED EXTERNAL — RULER, Hsieh et al. 2024 (illustrative,\nnot our measurement)",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.4, color=RED, style="italic",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.9))
    _save(fig, "rag12_effective_context.png")


def fig_decision_map() -> None:
    """The stuff / RAG / hybrid decision map by corpus size vs window & cost — schematic mechanism."""
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Stuff, retrieve, or hybrid? A cost-and-accuracy decision",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    def box(x, y, w, h, text, color, fs=8.8):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fs,
                fontweight="bold")

    box(0.36, 0.83, 0.28, 0.09, "a query over a corpus", SLATE)
    box(0.30, 0.63, 0.40, 0.10, "corpus ≪ window\nAND cost irrelevant?", PURPLE, fs=8.4)
    ax.annotate("", xy=(0.5, 0.73), xytext=(0.5, 0.83), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))

    # yes -> stuff
    box(0.03, 0.44, 0.26, 0.10, "STUFF it\nwhole corpus in the prompt\n(simplest, no retrieval)", GREEN, fs=8.0)
    ax.annotate("yes", xy=(0.16, 0.54), xytext=(0.34, 0.65),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4), fontsize=8.4, color=GREEN, ha="center")
    # no -> retrieve
    box(0.55, 0.44, 0.42, 0.10, "corpus ≫ window OR cost/latency matter\n→ RETRIEVE (RAG): top-k only", BLUE, fs=8.0)
    ax.annotate("no", xy=(0.74, 0.54), xytext=(0.66, 0.63),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4), fontsize=8.4, color=BLUE, ha="center")
    # hybrid
    box(0.55, 0.24, 0.42, 0.10, "need MANY relevant chunks?\n→ HYBRID: retrieve many → long window", AMBER, fs=8.0)
    ax.annotate("", xy=(0.76, 0.34), xytext=(0.76, 0.44), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))

    ax.text(0.5, 0.10, "why not just stuff everything into a 2M window? — cost scales per token per query, "
            "the model is lost-in-the-middle,\nand effective context < advertised (RULER). Retrieval puts the "
            "few right chunks where the model reads best.",
            ha="center", fontsize=8.6, color=INK, style="italic")
    _save(fig, "rag12_decision_map.png")


def main() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"dense lens: {dense.backend}")
    print(f"providers: {[p.name for p in PROVIDERS]} | chunk={TOKENS_PER_CHUNK} tok")
    fig_cost_crossover()
    fig_cost_multiplier()
    fig_lost_in_middle()
    fig_dilution_proxy(dense, corpus)
    fig_effective_context()
    fig_decision_map()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

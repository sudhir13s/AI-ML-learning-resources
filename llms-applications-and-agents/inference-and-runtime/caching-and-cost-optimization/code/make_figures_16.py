"""Static figure generator for 16-Caching-and-Cost-Optimization.

Imports the SAME canonical functions the page and notebook use (caching_cost.py, which reuses ch5's
encoder + ch12's cost model) so every plotted number is the chapter's own -- no hand-typed values.
Writes muted-palette PNGs to the shared chapter image dir (../../images/) with the per-chapter prefix
`rag16_`.

    python make_figures_16.py

Figures produced:
  rag16_cache_flow.png      -- the semantic-cache mechanism: query -> embed -> nearest cached (cosine
                               >= tau ? HIT : MISS -> compute + store). Schematic mechanism diagram.
  rag16_cost_vs_hitrate.png -- expected per-query cost (and % saved) as the hit rate rises 0 -> 1; the
                               stream's measured operating point marked.
  rag16_tradeoff.png        -- the false-hit / miss tradeoff as the cache threshold tau sweeps (the
                               real rates from the pipeline).
  rag16_latency.png         -- per-query latency: a MISS (full call) vs a HIT (lookup only), and the
                               stream's no-cache vs cached totals.
  rag16_cache_layers.png    -- the cache-layers stack: semantic / prompt (prefix) / embedding /
                               retrieval caching, each cutting a different cost. Schematic.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from caching_cost import (
    CACHE_THRESHOLD,
    FULL_CALL_MS,
    HIT_MS,
    QUERY_STREAM,
    STREAM_ANSWERS,
    DenseRetriever,
    build_probes,
    expected_cost_per_query,
    false_hit_and_miss_rates,
    full_call_cost_usd,
    full_corpus,
    hit_cost_usd,
    run_stream,
    savings_fraction,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / cache / good
PURPLE = "#5D4A8A"  # process / compute
GREEN = "#2E7A5A"  # hit / saved / good
RED = "#8B3B4A"  # miss / false-hit / cost
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # threshold / highlight
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
# Figure 1 -- the semantic-cache mechanism
# ================================================================================================


def fig_cache_flow() -> None:
    """The semantic-cache mechanism: query -> embed -> nearest cached -> HIT (cheap) or MISS (compute).

    Schematic mechanism diagram (labelled). The point: a new query embeds, is compared to every cached
    query by cosine, and either HITS (return the stored answer for ~free) or MISSES (compute the full
    answer, then store it). Matches the page's Mermaid diagram.
    """
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Semantic cache: serve a stored answer when a new query is close enough",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    _box(ax, 0.01, 0.60, 0.13, 0.16, "query", SLATE, fs=9.0)
    _box(ax, 0.18, 0.60, 0.14, 0.16, "EMBED\n(ch5 encoder)", BLUE, fs=8.2)
    _box(ax, 0.36, 0.60, 0.19, 0.16, "nearest cached\nquery (cosine)", BLUE, fs=8.2)
    ax.add_patch(plt.Rectangle((0.59, 0.58), 0.16, 0.20, facecolor=AMBER, alpha=0.14, edgecolor=AMBER, linewidth=1.6))
    ax.text(0.67, 0.68, "cosine ≥ τ ?", ha="center", va="center", fontsize=9.0, color=AMBER, fontweight="bold")
    for x0, x1 in ((0.14, 0.18), (0.32, 0.36), (0.55, 0.59)):
        ax.annotate("", xy=(x1, 0.68), xytext=(x0, 0.68), arrowprops=dict(arrowstyle="->", color=INK, lw=1.7))

    # HIT branch (up) and MISS branch (down)
    ax.annotate("", xy=(0.80, 0.72), xytext=(0.75, 0.70), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    _box(ax, 0.80, 0.66, 0.18, 0.12, "HIT → return\ncached answer (~free)", GREEN, fs=7.8)
    ax.annotate("", xy=(0.80, 0.46), xytext=(0.75, 0.64), arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    _box(ax, 0.80, 0.40, 0.18, 0.12, "MISS → compute\n(full call) + store", RED, fs=7.8)

    # the cache store, below
    ax.add_patch(plt.Rectangle((0.30, 0.16), 0.40, 0.16, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, linewidth=1.4))
    ax.text(0.50, 0.28, "cache store: (query embedding → answer) pairs", ha="center", fontsize=8.6, color=BLUE, fontweight="bold")
    ax.text(0.50, 0.21, "TTL / invalidate when the underlying docs change", ha="center", fontsize=7.8, color=INK, style="italic")
    ax.annotate("", xy=(0.45, 0.58), xytext=(0.45, 0.33), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2, alpha=0.7))
    ax.annotate("", xy=(0.88, 0.39), xytext=(0.70, 0.24), arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, alpha=0.6))

    ax.text(0.5, 0.04, "the win = hit_rate × (full-call cost − hit cost);  a HIT pays only the lookup embedding",
            ha="center", fontsize=8.8, color=INK, style="italic")
    _save(fig, "rag16_cache_flow.png")


# ================================================================================================
# Figure 2 -- expected cost vs hit rate
# ================================================================================================


def fig_cost_vs_hitrate(stream_hit_rate: float) -> None:
    """Expected per-query cost (and % saved) as the hit rate rises 0 -> 1; the stream's point marked.

    Runs the REAL cost model. As the hit rate rises, expected per-query cost falls linearly from the
    full-call cost toward the (tiny) hit cost; savings ~ hit rate. The stream's measured hit rate is
    marked with its actual saving.
    """
    hit_rates = np.linspace(0.0, 1.0, 51)
    costs = np.array([expected_cost_per_query(h) for h in hit_rates])
    full = full_call_cost_usd()

    fig, ax1 = plt.subplots(figsize=(10.4, 5.8))
    _style_axis(ax1)
    ax1.plot(hit_rates, costs * 1e3, "-", color=BLUE, linewidth=2.4, label="expected cost per query")
    ax1.axhline(full * 1e3, color=RED, linewidth=1.4, linestyle=":", label=f"no-cache (full call) = ${full * 1e3:.3f}/1k")
    ax1.set_xlabel("cache hit rate", fontsize=10.5)
    ax1.set_ylabel("expected cost per query (milli-USD)", fontsize=10.5, color=BLUE)
    ax1.set_ylim(0, full * 1e3 * 1.1)
    ax1.set_title("Expected per-query cost falls linearly with the hit rate (measured cost model)",
                  fontsize=11.5, color=INK, fontweight="bold", pad=10)

    # mark the stream's operating point
    op_cost = expected_cost_per_query(stream_hit_rate)
    ax1.scatter([stream_hit_rate], [op_cost * 1e3], s=170, color=GREEN, edgecolor=INK, linewidth=1.3, zorder=6)
    ax1.annotate(f"this stream:\nhit rate {stream_hit_rate:.3f}\n→ {savings_fraction(stream_hit_rate):.1%} saved",
                 xy=(stream_hit_rate, op_cost * 1e3), xytext=(stream_hit_rate - 0.02, full * 1e3 * 0.62),
                 fontsize=8.6, color=GREEN, ha="right", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN, alpha=0.95))
    ax1.legend(fontsize=8.8, loc="upper right", framealpha=0.95)
    _save(fig, "rag16_cost_vs_hitrate.png")


# ================================================================================================
# Figure 3 -- the false-hit / miss tradeoff
# ================================================================================================


def fig_tradeoff(dense: DenseRetriever) -> None:
    """The false-hit / miss tradeoff as the cache threshold tau sweeps (the real rates from the pipeline).

    Runs the REAL false_hit_and_miss_rates over a fine tau sweep. false-hit (wrong answer served) falls
    with tau; miss (real paraphrase not caught) rises with tau. The default tau=0.8 is marked. A false
    hit is the dangerous error -- it serves a confidently WRONG cached answer.
    """
    probes = build_probes()
    taus = np.round(np.arange(0.50, 0.96, 0.05), 2)
    fhs, misses = [], []
    for tau in taus:
        fh, miss = false_hit_and_miss_rates(dense, probes, float(tau))
        fhs.append(fh)
        misses.append(miss)

    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    _style_axis(ax)
    ax.plot(taus, fhs, "-o", color=RED, linewidth=2.2, markersize=6, label="false-hit (wrong cached answer served)")
    ax.plot(taus, misses, "-s", color=BLUE, linewidth=2.2, markersize=6, label="miss (real paraphrase not cached)")
    ax.axvline(CACHE_THRESHOLD, color=AMBER, linewidth=1.8, linestyle="--")
    ax.text(CACHE_THRESHOLD + 0.004, 0.9, f"default τ = {CACHE_THRESHOLD}", color=AMBER,
            fontsize=8.6, fontweight="bold", rotation=90, va="top")
    ax.set_xlabel("cache threshold τ", fontsize=10.5)
    ax.set_ylabel("rate over the labelled probes", fontsize=10.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Raising τ cuts false hits but misses more paraphrases (measured)",
                 fontsize=11.5, color=INK, fontweight="bold", pad=10)
    ax.legend(fontsize=8.6, loc="center right", framealpha=0.95)
    ax.text(0.5, -0.16, "a false hit serves a confidently WRONG cached answer — cosine matches TOPIC, not exact intent",
            transform=ax.transAxes, ha="center", fontsize=8.2, color=INK, style="italic")
    _save(fig, "rag16_tradeoff.png")


# ================================================================================================
# Figure 4 -- latency: miss vs hit, and the stream totals
# ================================================================================================


def fig_latency(dense: DenseRetriever) -> None:
    """Per-query latency (a MISS's full call vs a HIT's lookup) and the stream's no-cache vs cached totals.

    Runs the REAL stream. Left: per-query latency, a full call vs a cache hit (the modelled constants).
    Right: the whole stream's total latency, no-cache vs cached -- the wall-clock the cache saves.
    """
    result = run_stream(dense, QUERY_STREAM, STREAM_ANSWERS)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1, 1]})

    # LEFT: per-query latency
    _style_axis(ax_l)
    bars = ax_l.bar(["MISS\n(full call)", "HIT\n(lookup only)"], [FULL_CALL_MS, HIT_MS],
                    color=[RED, GREEN], edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, v in zip(bars, [FULL_CALL_MS, HIT_MS]):
        ax_l.annotate(f"{v:.0f} ms", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                      fontsize=9.4, color=INK, fontweight="bold")
    ax_l.set_ylabel("latency per query (ms, modelled)", fontsize=10.0)
    ax_l.set_ylim(0, FULL_CALL_MS * 1.15)
    ax_l.set_title(f"A cache hit is ~{FULL_CALL_MS / HIT_MS:.0f}× faster than a full call", fontsize=10.5, color=INK, fontweight="bold")

    # RIGHT: whole-stream totals
    _style_axis(ax_r)
    bars = ax_r.bar(["no cache", "with cache"], [result.latency_no_cache_ms, result.latency_cached_ms],
                    color=[RED, GREEN], edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, v in zip(bars, [result.latency_no_cache_ms, result.latency_cached_ms]):
        ax_r.annotate(f"{v:.0f} ms", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                      fontsize=9.4, color=INK, fontweight="bold")
    ax_r.set_ylabel("total latency, 8-query stream (ms)", fontsize=10.0)
    ax_r.set_ylim(0, result.latency_no_cache_ms * 1.15)
    ax_r.set_title(f"stream: {result.latency_saved_ms / result.latency_no_cache_ms:.0%} of wall-clock saved",
                   fontsize=10.5, color=INK, fontweight="bold")

    fig.suptitle("Latency: a cache hit skips the full retrieve+generate round trip (measured stream)",
                 fontsize=12.0, y=1.01, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "rag16_latency.png")


# ================================================================================================
# Figure 5 -- the cache-layers stack
# ================================================================================================


def fig_cache_layers() -> None:
    """The cache-layers stack: semantic / prompt (prefix) / embedding / retrieval, each cutting a cost.

    Schematic (labelled). The four caching layers of a RAG app, each removing a different repeated
    cost: semantic (whole-answer reuse by similarity), prompt/prefix (provider KV reuse of a fixed
    prefix), embedding (don't re-embed the same text), retrieval (reuse a query's retrieved chunks).
    """
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Cache at every layer: each removes a different repeated cost",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    layers = [
        ("SEMANTIC cache", "serve a cached ANSWER when a new query is a near-paraphrase (GPTCache / Redis)",
         "skips the whole retrieve+generate", GREEN, 0.72),
        ("PROMPT / prefix cache", "provider reuses the KV of a fixed prefix (system prompt, docs) — Anthropic / OpenAI",
         "cuts input token cost 50–90%", BLUE, 0.55),
        ("EMBEDDING cache", "don't re-embed the same text (queries, chunks) — memoize by content hash",
         "skips repeated encoder calls", PURPLE, 0.38),
        ("RETRIEVAL cache", "reuse a query's retrieved chunk IDs when the same query recurs",
         "skips the vector search", SLATE, 0.21),
    ]
    for title, desc, saves, col, y in layers:
        ax.add_patch(plt.Rectangle((0.04, y - 0.065), 0.62, 0.12, facecolor=col, alpha=0.92, edgecolor=INK, linewidth=1.0))
        ax.text(0.06, y + 0.008, title, ha="left", va="center", fontsize=9.4, color="white", fontweight="bold")
        ax.text(0.06, y - 0.038, _short(desc, 78), ha="left", va="center", fontsize=7.2, color="white")
        ax.text(0.70, y, "→ " + saves, ha="left", va="center", fontsize=8.4, color=col, fontweight="bold")

    ax.text(0.5, 0.05, "the layers compose: a semantic HIT skips everything; a MISS still benefits from "
            "prompt + embedding + retrieval caches", ha="center", fontsize=8.6, color=INK, style="italic")
    _save(fig, "rag16_cache_layers.png")


def main() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    result = run_stream(dense, QUERY_STREAM, STREAM_ANSWERS)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend} | stream hit rate: {result.hit_rate:.3f} "
          f"| full ${full_call_cost_usd():.6f} | hit ${hit_cost_usd():.6f}")
    fig_cache_flow()
    fig_cost_vs_hitrate(result.hit_rate)
    fig_tradeoff(dense)
    fig_latency(dense)
    fig_cache_layers()


if __name__ == "__main__":
    main()

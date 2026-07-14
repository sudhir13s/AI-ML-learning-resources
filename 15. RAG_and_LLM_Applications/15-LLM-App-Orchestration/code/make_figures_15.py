"""Static figure generator for 15-LLM-App-Orchestration.

Imports the SAME canonical functions the page and notebook use (orchestration.py, which wires ch5/6/14)
so every plotted number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to the
shared chapter image dir (../../images/) with the per-chapter prefix `rag15_`.

    python make_figures_15.py

Figures produced:
  rag15_pipeline.png       -- the wired mini-RAG app as a chain: route -> retrieve -> rerank -> guardrail
                              -> generate, with the state threaded through. Schematic mechanism diagram.
  rag15_topologies.png     -- the three orchestration shapes: sequential CHAIN, ROUTED branch, cyclic GRAPH.
  rag15_app_trace.png      -- the REAL end-to-end trace of the fact query: each step's actual output line.
  rag15_router.png         -- the router's cosine scores for a fact query vs a chit-chat query (real argmax).
  rag15_retry.png          -- the retry/fallback flow: naive glue CRASHES; the wrapped step recovers on attempt 2.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from orchestration import (
    CHITCHAT_QUERY,
    FACT_QUERY,
    AppState,
    FlakyStep,
    StepError,
    build_app,
    full_corpus,
    with_retry,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / retrieval / state
PURPLE = "#5D4A8A"  # process / generate
GREEN = "#2E7A5A"  # good / emit / success
RED = "#8B3B4A"  # abstain / crash / bad
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # router / highlight
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
# Figure 1 -- the wired mini-RAG app as a chain, with state threaded through
# ================================================================================================


def fig_pipeline() -> None:
    """The wired app as a chain: route -> retrieve -> rerank -> guardrail -> generate, state threaded.

    Schematic mechanism diagram (labelled). The point: a real app is a chain of typed steps each
    transforming a shared STATE; the orchestrator threads the state and records a trace. Matches the
    page's Mermaid diagram.
    """
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "An LLM app is a CHAIN of typed steps, each transforming a shared state",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    steps = [
        (0.01, "ROUTE", "pick the path", AMBER),
        (0.21, "RETRIEVE", "ch5 dense top-k", BLUE),
        (0.41, "RERANK", "ch6 cross-encoder", BLUE),
        (0.61, "GUARDRAIL", "ch14 context ≥ τ?", GREEN),
        (0.81, "GENERATE", "answer / abstain", PURPLE),
    ]
    w = 0.17
    for x, title, sub, col in steps:
        _box(ax, x, 0.58, w, 0.18, f"{title}\n{sub}", col, fs=8.2)
    for x, _, _, _ in steps[1:]:
        ax.annotate("", xy=(x, 0.67), xytext=(x - 0.03, 0.67), arrowprops=dict(arrowstyle="->", color=INK, lw=1.7))

    # the state ribbon threaded underneath
    ax.add_patch(plt.Rectangle((0.01, 0.30), 0.97, 0.14, facecolor=BLUE, alpha=0.08, edgecolor=BLUE, linewidth=1.3))
    ax.text(0.02, 0.37, "state:", ha="left", va="center", fontsize=8.6, color=BLUE, fontweight="bold")
    ax.text(0.5, 0.37, "query → +route → +retrieved → +reranked/context → +grounding/abstained → +answer",
            ha="center", va="center", fontsize=8.4, color=INK)
    for x, _, _, _ in steps:
        ax.annotate("", xy=(x + w / 2, 0.45), xytext=(x + w / 2, 0.57), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1, alpha=0.6))

    ax.text(0.5, 0.16, "each step returns a NEW (immutable) state — so no step can silently clobber "
            "another's field, and the trace records every version", ha="center", fontsize=8.6, color=INK, style="italic")
    ax.text(0.5, 0.05, "separate the WHAT (the steps) from the HOW (the orchestrator's run loop: threading, "
            "retries, tracing)", ha="center", fontsize=8.4, color=SLATE, style="italic")
    _save(fig, "rag15_pipeline.png")


# ================================================================================================
# Figure 2 -- the three orchestration topologies
# ================================================================================================


def fig_topologies() -> None:
    """The three shapes: sequential CHAIN, ROUTED branch, cyclic GRAPH -- the vocabulary of orchestration.

    Schematic. Left: a linear chain A->B->C. Middle: a router branching to B1 or B2. Right: a stateful
    graph with a cycle (a retry/agent loop). Increasing power, left to right.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 1) CHAIN
    ax = axes[0]
    ax.text(0.5, 0.93, "CHAIN (sequential)", ha="center", fontsize=11.0, color=BLUE, fontweight="bold")
    for i, (y, lbl) in enumerate([(0.68, "A"), (0.45, "B"), (0.22, "C")]):
        _box(ax, 0.38, y, 0.24, 0.13, lbl, BLUE, fs=10.0)
        if i < 2:
            ax.annotate("", xy=(0.5, y - 0.02), xytext=(0.5, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(0.5, 0.06, "f∘g∘h over a typed state", ha="center", fontsize=8.4, color=SLATE, style="italic")

    # 2) ROUTED
    ax = axes[1]
    ax.text(0.5, 0.93, "ROUTED (branch)", ha="center", fontsize=11.0, color=AMBER, fontweight="bold")
    _box(ax, 0.36, 0.70, 0.28, 0.13, "ROUTE", AMBER, fs=9.5)
    _box(ax, 0.08, 0.40, 0.34, 0.13, "path B1", GREEN, fs=9.0)
    _box(ax, 0.58, 0.40, 0.34, 0.13, "path B2", PURPLE, fs=9.0)
    ax.annotate("", xy=(0.25, 0.53), xytext=(0.44, 0.69), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.annotate("", xy=(0.75, 0.53), xytext=(0.56, 0.69), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.6))
    ax.text(0.5, 0.06, "pick by scoring the input", ha="center", fontsize=8.4, color=SLATE, style="italic")

    # 3) GRAPH (cyclic)
    ax = axes[2]
    ax.text(0.5, 0.93, "GRAPH (stateful, cyclic)", ha="center", fontsize=11.0, color=RED, fontweight="bold")
    _box(ax, 0.36, 0.70, 0.28, 0.13, "step", BLUE, fs=9.5)
    _box(ax, 0.36, 0.40, 0.28, 0.13, "check", GREEN, fs=9.5)
    ax.annotate("", xy=(0.5, 0.53), xytext=(0.5, 0.69), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    # the loop-back edge
    ax.annotate("", xy=(0.64, 0.76), xytext=(0.64, 0.46),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, connectionstyle="arc3,rad=-0.6"))
    ax.text(0.83, 0.61, "retry /\nloop", ha="center", fontsize=8.0, color=RED, fontweight="bold")
    _box(ax, 0.30, 0.14, 0.40, 0.11, "done (step budget)", SLATE, fs=8.2)
    ax.annotate("", xy=(0.5, 0.25), xytext=(0.5, 0.39), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(0.5, 0.03, "cycles + a step budget", ha="center", fontsize=8.4, color=SLATE, style="italic")

    fig.suptitle("Three orchestration shapes: chain → routed → cyclic graph (increasing power)",
                 fontsize=12.5, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag15_topologies.png")


# ================================================================================================
# Figure 3 -- the REAL end-to-end app trace
# ================================================================================================


def fig_app_trace(app, query: str) -> None:
    """The REAL trace of the fact query through the wired app: each step's actual output line.

    Runs the real app and renders its trace verbatim -- route, retrieve, rerank, guardrail, generate --
    each as a row, so the reader sees exactly what ran and in what order. The grounding number and the
    doc indices are the app's real output.
    """
    result = app.run(query)
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "The wired app's REAL trace: one query, five steps, one observable log",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.02, 0.87, f"query: {query}", ha="left", fontsize=9.0, color=BLUE, fontweight="bold")

    step_colors = {"route": AMBER, "retrieve": BLUE, "rerank": BLUE, "guardrail": GREEN, "generate": PURPLE}
    ys = np.linspace(0.72, 0.18, len(result.trace))
    for y, line in zip(ys, result.trace):
        step = line.split("]")[0].strip("[").split()[0]  # the [name] at the line start
        col = step_colors.get(step, SLATE)
        ax.add_patch(plt.Rectangle((0.03, y - 0.045), 0.94, 0.08, facecolor=col, alpha=0.11, edgecolor=col, linewidth=1.2))
        ax.text(0.05, y, line, ha="left", va="center", fontsize=8.4, color=INK, family="monospace")

    ax.text(0.02, 0.09, f"ANSWER: {_short(result.answer, 78)}", ha="left", fontsize=8.8, color=GREEN, fontweight="bold")
    ax.text(0.5, 0.02, "naming every step is what makes the pipeline OBSERVABLE — you can see which "
            "step ran, and which would fail", ha="center", fontsize=8.4, color=SLATE, style="italic")
    _save(fig, "rag15_app_trace.png")


# ================================================================================================
# Figure 4 -- the router's cosine scores
# ================================================================================================


def fig_router(app) -> None:
    """The router's cosine scores for a fact query vs a chit-chat query -- the routing decision, measured.

    Runs the REAL router. Grouped bars: each query's cosine to each route description; the argmax (chosen)
    bar is outlined. The fact query peaks on 'rag', the chit-chat on 'direct' -- a real embedding decision.
    """
    routes = app.router.routes
    queries = [("fact query", FACT_QUERY), ("chit-chat query", CHITCHAT_QUERY)]
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    _style_axis(ax)
    x = np.arange(len(routes))
    width = 0.36
    palette = [BLUE, GREEN]
    for qi, (qlabel, q) in enumerate(queries):
        scores = app.router.route_scores(q)
        chosen = int(np.argmax(scores))
        bars = ax.bar(x + (qi - 0.5) * width, scores, width, label=qlabel, color=palette[qi],
                      edgecolor=INK, linewidth=0.9)
        for j, bar in enumerate(bars):
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                        fontsize=8.4, color=INK, fontweight="bold")
            if j == chosen:  # outline the chosen (argmax) route for this query
                bar.set_edgecolor(AMBER)
                bar.set_linewidth(2.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"route: {r.name}\n({_short(r.description, 28)})" for r in routes], fontsize=8.0)
    ax.set_ylabel("cosine(query, route description)", fontsize=10.0)
    ax.set_ylim(0, max(0.65, ax.get_ylim()[1]))
    ax.set_title("The router argmaxes cosine(query, route description) — a real embedding decision",
                 fontsize=11.0, color=INK, fontweight="bold", pad=10)
    ax.legend(fontsize=8.8, loc="upper right", framealpha=0.95)
    ax.text(0.5, -0.16, "amber outline = the chosen route (argmax): the fact query → rag, the chit-chat → direct",
            transform=ax.transAxes, ha="center", fontsize=8.2, color=INK, style="italic")
    _save(fig, "rag15_router.png")


# ================================================================================================
# Figure 5 -- the retry / fallback flow
# ================================================================================================


def fig_retry() -> None:
    """The retry flow: naive glue CRASHES on a transient fault; the wrapped step recovers on attempt 2.

    Runs the REAL FlakyStep both ways. Left: the naive (unwrapped) step raises and the pipeline dies.
    Right: with_retry catches the StepError, retries, and succeeds -- the real trace lines shown.
    """
    # naive: crashes
    naive = FlakyStep(fail_times=1)
    crashed_msg = ""
    try:
        naive(AppState(query="probe"))
    except StepError as err:
        crashed_msg = str(err)
    # wrapped: recovers
    flaky = FlakyStep(fail_times=1)
    recovered = with_retry(flaky, max_attempts=3)(AppState(query="probe"))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax in (ax_l, ax_r):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # LEFT: naive crash
    ax_l.text(0.5, 0.92, "NAIVE glue: one transient fault kills the pipeline", ha="center",
              fontsize=10.5, color=RED, fontweight="bold")
    _box(ax_l, 0.30, 0.62, 0.40, 0.13, "step (attempt 1)", BLUE, fs=8.6)
    ax_l.annotate("", xy=(0.5, 0.50), xytext=(0.5, 0.61), arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax_l.add_patch(plt.Rectangle((0.20, 0.30), 0.60, 0.18, facecolor=RED, alpha=0.14, edgecolor=RED, linewidth=1.5))
    ax_l.text(0.5, 0.39, f"✗ CRASH\n{crashed_msg}", ha="center", va="center", fontsize=8.6, color=RED, fontweight="bold")
    ax_l.text(0.5, 0.14, "no retry, no fallback → the whole request fails", ha="center", fontsize=8.2, color=SLATE, style="italic")

    # RIGHT: retry recovers
    ax_r.text(0.5, 0.92, "with_retry: catch, back off, retry → recover", ha="center",
              fontsize=10.5, color=GREEN, fontweight="bold")
    rows = [
        ("attempt 1: transient fault", RED),
        ("↓ retry (backoff)", AMBER),
        ("attempt 2: SUCCESS", GREEN),
    ]
    ys = [0.68, 0.50, 0.32]
    for (label, col), y in zip(rows, ys):
        ax_r.add_patch(plt.Rectangle((0.14, y - 0.05), 0.72, 0.10, facecolor=col, alpha=0.14, edgecolor=col, linewidth=1.3))
        ax_r.text(0.5, y, label, ha="center", va="center", fontsize=8.8, color=col, fontweight="bold")
    ax_r.text(0.5, 0.14, "the same fault is RECOVERED — the request succeeds", ha="center", fontsize=8.2, color=SLATE, style="italic")

    # assert-backed footnote: the recovered trace really contains the two logged attempts
    ok = any("failed attempt 1" in ln for ln in recovered.trace) and any("succeeded on attempt 2" in ln for ln in recovered.trace)
    fig.suptitle(f"Retry turns a transient failure from a crash into a recovery (trace-verified: {ok})",
                 fontsize=12.0, y=1.02, color=INK, fontweight="bold")
    _save(fig, "rag15_retry.png")


def main() -> None:
    corpus = full_corpus()
    app = build_app(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {app.dense.backend}")
    fig_pipeline()
    fig_topologies()
    fig_app_trace(app, FACT_QUERY)
    fig_router(app)
    fig_retry()


if __name__ == "__main__":
    main()

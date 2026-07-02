"""Figure generator for 02-ReAct-Reason-and-Act — every number from the REAL agent run.

All measured figures come from the same real module the chapter and notebook use
(``react_agent.py``): a real ``Qwen/Qwen2.5-1.5B-Instruct`` driven greedily through a real
Thought -> Action -> Observation loop against the real ``calculator`` and ``wiki`` tools. Nothing
is hand-typed — the trace panels are a real solved trace, and the comparison bars are the real
exact-match accuracy of ReAct vs a reason-only baseline on the real evaluation set.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``agentic02_``:

  agentic02_react_trace.png   -- a REAL solved multi-hop trace rendered as stacked
                                 Thought / Action / Observation cards (the loop, seen).
  agentic02_react_vs_direct.png -- REAL exact-match accuracy: ReAct (tools) vs reason-only
                                 (no tools) on the real multi-step eval set.
  agentic02_per_question.png  -- REAL per-question outcome grid (correct/wrong) for both methods,
                                 with the ReAct step count — where reason-only fails and ReAct fixes it.

    python make_figures_02.py

Verified on Python 3.12 / torch 2.12 / transformers 5.10 / matplotlib 3.10.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# This generator lives in the domain-level ``12. Agentic_AI/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so
# the ``react_agent`` import below resolves regardless of the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "02-ReAct-Reason-and-Act" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from react_agent import (  # noqa: E402  (resolved via the sys.path insert above)
    EVAL_SET,
    LanguageModel,
    compare_react_vs_direct,
    run_react,
)

# ---- Palette (muted; matches the chapter's Mermaid classDefs) -----------------------------------
BLUE = "#3A6B96"  # data / question
PURPLE = "#5D4A8A"  # thought (reasoning)
AMBER = "#7A6528"  # action (tool call)
GREEN = "#2E7A5A"  # observation / correct
RED = "#8B3B4A"  # wrong / the foil
SLATE = "#4A5B6E"  # neutral
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines
PAPER = "#F7F8FA"  # card fill

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 130
IMG_PREFIX = "agentic02_"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink labels."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{IMG_PREFIX}{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


# ==================================================================================================
# Figure 1 — a REAL solved trace, rendered as Thought / Action / Observation cards
# ==================================================================================================
def fig_react_trace(llm: LanguageModel) -> None:
    """Render one REAL solved multi-hop trace as stacked, colour-coded cards — the loop made visible."""
    result = run_react(
        llm, "In what year was the Eiffel Tower completed, and what is that year plus 100?"
    )

    # flatten the real transcript into typed rows the figure draws as cards
    rows: list[tuple[str, str]] = [("Question", result.question)]
    for step in result.steps:
        for line in step.text.splitlines():
            line = line.strip()
            if line.startswith("Thought:"):
                rows.append(("Thought", line[len("Thought:") :].strip()))
            elif line.startswith("Action:"):
                rows.append(("Action", line[len("Action:") :].strip()))
        if step.observation is not None:
            rows.append(("Observation", step.observation))
    if result.answer is not None:
        rows.append(("Answer", result.answer))

    kind_color = {
        "Question": BLUE,
        "Thought": PURPLE,
        "Action": AMBER,
        "Observation": GREEN,
        "Answer": SLATE,
    }

    fig, ax = plt.subplots(figsize=(9.2, 0.92 * len(rows) + 0.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows))
    ax.axis("off")
    ax.set_title(
        "A real ReAct trace: reason -> act -> observe, until the answer is grounded",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=12,
    )

    for i, (kind, body) in enumerate(rows):
        y = len(rows) - 1 - i
        color = kind_color[kind]
        box = FancyBboxPatch(
            (0.15, y + 0.12),
            9.7,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=color,
            facecolor=PAPER,
        )
        ax.add_patch(box)
        ax.text(0.42, y + 0.5, kind, fontsize=9.5, fontweight="bold", color=color, va="center")
        wrapped = textwrap.fill(body, width=74)
        ax.text(2.35, y + 0.5, wrapped, fontsize=8.6, color=INK, va="center")
        # a little "flow" arrow between cards to show the loop advancing downward
        if i < len(rows) - 1:
            ax.annotate(
                "",
                xy=(0.28, y + 0.06),
                xytext=(0.28, y + 0.14),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.1),
            )
    _save(fig, "react_trace")


# ==================================================================================================
# Figure 2 & 3 — the REAL ReAct-vs-direct comparison
# ==================================================================================================
def fig_comparison(llm: LanguageModel) -> None:
    """Two figures from ONE real comparison run: overall accuracy bars + per-question outcome grid."""
    rows = compare_react_vs_direct(llm, EVAL_SET)
    n = len(rows)
    react_correct = sum(r.react_correct for r in rows)
    direct_correct = sum(r.direct_correct for r in rows)

    # ---- Figure 2: overall accuracy bars ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    methods = ["ReAct\n(reason + act + tools)", "Reason-only\n(no tools)"]
    accs = [react_correct / n, direct_correct / n]
    bars = ax.bar(methods, accs, color=[GREEN, RED], width=0.56, edgecolor="white", linewidth=1.5)
    for bar, correct in zip(bars, (react_correct, direct_correct)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.0%}\n({correct}/{n})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(
        f"Grounding in real observations fixes the errors\nreason-only makes ({n} real multi-step questions)",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    _style_axis(ax)
    _save(fig, "react_vs_direct")

    # ---- Figure 3: per-question outcome grid ----
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 3)
    # extra headroom on top so the title clears the header row, and a ~6% bottom margin
    # (floor below y=0.5, the last row) so no renderer clips the final "17^3" row.
    ax.set_ylim(-0.4, n + 1.1)
    ax.axis("off")
    ax.text(
        0.05,
        n + 0.85,
        "Per-question outcome: where reason-only fails and ReAct recovers",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    # header
    ax.text(0.05, n + 0.2, "question (real, multi-step)", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(2.05, n + 0.2, "reason-only", fontsize=9.5, fontweight="bold", color=INK, ha="center")
    ax.text(2.6, n + 0.2, "ReAct", fontsize=9.5, fontweight="bold", color=INK, ha="center")
    for i, r in enumerate(rows):
        y = n - 1 - i + 0.5
        q = textwrap.shorten(r.question, width=58)
        ax.text(0.05, y, q, fontsize=8.4, color=INK, va="center")
        for x, ok in ((2.05, r.direct_correct), (2.6, r.react_correct)):
            ax.scatter(
                x,
                y,
                s=340,
                marker="o",
                color=GREEN if ok else RED,
                edgecolor="white",
                linewidth=1.4,
                zorder=3,
            )
            ax.text(x, y, "✓" if ok else "✗", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(2.92, y, f"{r.react_steps} steps", fontsize=7.6, color=SLATE, va="center")
    _save(fig, "per_question")


def main() -> None:
    print("Loading the real model to regenerate figures from the real agent run ...")
    llm = LanguageModel()
    print(f"  model {llm.model_id} on {llm.device}")
    fig_react_trace(llm)
    fig_comparison(llm)
    print("Done — all figures regenerated from the real ReAct run.")


if __name__ == "__main__":
    main()

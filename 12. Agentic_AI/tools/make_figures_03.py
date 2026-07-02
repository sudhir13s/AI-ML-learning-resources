"""Figure generator for 03-Tool-Use-and-Function-Calling — every number from the REAL agent run.

All figures come from the same real module the chapter and notebook use (``function_calling_agent.py``):
a real ``Qwen/Qwen2.5-1.5B-Instruct`` driven greedily through the real structured function-calling
protocol (JSON-schema tools -> ``<tool_call>`` JSON -> schema-validated dispatch -> tool-role result ->
answer) against the real ``calculator`` / ``convert_units`` / ``get_exchange_rate`` tools. Nothing is
hand-typed — the trace panels show a real solved trace (with the real JSON the model emitted), and the
reliability bars are the real "did we get a dispatchable call?" rates for structured vs text-parsing.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``agentic03_``:

  agentic03_call_trace.png     -- a REAL single tool call rendered as protocol cards:
                                  User -> assistant tool_call (the real JSON) -> tool result -> answer.
  agentic03_parallel_trace.png -- a REAL parallel multi-tool trace: two independent tool_calls emitted
                                  in ONE turn, each with its own tool-result message, then the answer.
  agentic03_reliability.png    -- REAL "dispatchable call" rate: structured function-calling (JSON,
                                  schema-validated) vs a ReAct-style text protocol (regex-parsed prose).

    python make_figures_03.py

Verified on Python 3.12 / torch 2.12 / transformers 5.10 / matplotlib 3.10.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# This generator lives in the domain-level ``12. Agentic_AI/tools/`` folder, while the chapter module it
# demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``function_calling_agent`` import below resolves regardless of the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "03-Tool-Use-and-Function-Calling" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from function_calling_agent import (  # noqa: E402  (resolved via the sys.path insert above)
    ToolCallingModel,
    compare_structured_vs_text,
    run_agent,
)

# ---- Palette (muted; matches the chapter's Mermaid classDefs) -----------------------------------
BLUE = "#3A6B96"  # data / user query
PURPLE = "#5D4A8A"  # the model's tool_call (the structured request)
AMBER = "#7A6528"  # schema / dispatch
GREEN = "#2E7A5A"  # tool result / success
RED = "#8B3B4A"  # failure / the text foil
SLATE = "#4A5B6E"  # neutral / answer
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines
PAPER = "#F7F8FA"  # card fill

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 130
IMG_PREFIX = "agentic03_"


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


def _render_cards(rows: list[tuple[str, str]], title: str, name: str) -> None:
    """Render a list of (kind, body) protocol cards, stacked and colour-coded — the loop made visible."""
    kind_color = {
        "User": BLUE,
        "tool_call": PURPLE,
        "Tool result": GREEN,
        "Assistant": SLATE,
    }
    fig, ax = plt.subplots(figsize=(9.6, 0.94 * len(rows) + 0.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=12)
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
        ax.text(0.42, y + 0.5, kind, fontsize=9.3, fontweight="bold", color=color, va="center")
        wrapped = textwrap.fill(body, width=68)
        # monospace for the structured JSON so it reads as a data payload, prose otherwise
        mono = kind in ("tool_call", "Tool result")
        ax.text(
            2.55,
            y + 0.5,
            wrapped,
            fontsize=8.3 if mono else 8.7,
            color=INK,
            va="center",
            family="monospace" if mono else "sans-serif",
        )
        if i < len(rows) - 1:
            ax.annotate(
                "",
                xy=(0.28, y + 0.06),
                xytext=(0.28, y + 0.14),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.1),
            )
    _save(fig, name)


# ==================================================================================================
# Figure 1 — a REAL single tool call, rendered as protocol cards (with the real JSON)
# ==================================================================================================
def fig_call_trace(model: ToolCallingModel) -> None:
    """Render one REAL single-tool trace: User -> structured tool_call (the actual JSON) -> result -> answer."""
    result = run_agent(model, "What is 481 multiplied by 32, then plus 19?")
    rows: list[tuple[str, str]] = [("User", result.query)]
    for turn in result.turns:
        for call in turn.calls:
            rows.append(("tool_call", f'{{"name": "{call.name}", "arguments": {json.dumps(call.arguments)}}}'))
        for call, res in zip(turn.calls, turn.results):
            rows.append(("Tool result", f"{call.name} -> {res}"))
    if result.answer is not None:
        rows.append(("Assistant", result.answer))
    _render_cards(
        rows,
        "A real function call: JSON schema -> structured tool_call -> execute -> tool result -> answer",
        "call_trace",
    )


# ==================================================================================================
# Figure 2 — a REAL parallel multi-tool trace (two tool_calls in ONE turn)
# ==================================================================================================
def fig_parallel_trace(model: ToolCallingModel) -> None:
    """Render a REAL parallel trace: two independent tool_calls in one turn, two results, then the answer."""
    result = run_agent(model, "Convert 42 kilometres to miles, and separately convert 5 kilograms to pounds.")
    rows: list[tuple[str, str]] = [("User", result.query)]
    for turn in result.turns:
        for call in turn.calls:
            rows.append(("tool_call", f'{{"name": "{call.name}", "arguments": {json.dumps(call.arguments)}}}'))
        for call, res in zip(turn.calls, turn.results):
            rows.append(("Tool result", f"{call.name} -> {res}"))
    if result.answer is not None:
        rows.append(("Assistant", result.answer))
    _render_cards(
        rows,
        "A real PARALLEL trace: two independent tool_calls in one turn, one tool result each",
        "parallel_trace",
    )


# ==================================================================================================
# Figure 3 — the REAL structured-vs-text reliability comparison
# ==================================================================================================
def fig_reliability(model: ToolCallingModel) -> None:
    """Bar chart of the REAL 'dispatchable call' rate: structured function-calling vs text-parsing."""
    rows = compare_structured_vs_text(model)
    n = len(rows)
    s_ok = sum(r.structured_ok for r in rows)
    t_ok = sum(r.text_ok for r in rows)

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    methods = ["Structured\nfunction-calling\n(JSON, schema-validated)", "Text protocol\n(TOOL: prose,\nregex-parsed)"]
    rates = [s_ok / n, t_ok / n]
    bars = ax.bar(methods, rates, color=[GREEN, RED], width=0.56, edgecolor="white", linewidth=1.5)
    for bar, ok in zip(bars, (s_ok, t_ok)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.0%}\n({ok}/{n})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share of queries yielding a dispatchable call")
    ax.set_title(
        f"Structured calling parses reliably; text-parsing does not\n({n} real single-tool queries, same model)",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    _style_axis(ax)
    _save(fig, "reliability")


def main() -> None:
    print("Loading the real model to regenerate figures from the real function-calling run ...")
    model = ToolCallingModel()
    print(f"  model {model.model_id} on {model.device}")
    fig_call_trace(model)
    fig_parallel_trace(model)
    fig_reliability(model)
    print("Done — all figures regenerated from the real function-calling run.")


if __name__ == "__main__":
    main()

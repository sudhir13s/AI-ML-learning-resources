"""Animated (GIF) intuition figure for 15-LLM-App-Orchestration.

Companion to the static PNGs. It brings the wired app to life -- STATE flowing through the chain, one
node lighting up at a time as it runs and adds its field to the state, with the router choosing the RAG
path first and a retry looping back on a transient fault. The reader watches the pipeline execute: the
state accumulates (query -> +route -> +retrieved -> +context -> +grounding -> +answer), each step's real
trace line appears, and the final answer assembles.

    python make_animation_15.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). Every trace
line, route score, and grounding number is the REAL orchestration.py output over the ch5 all-MiniLM
encoder; only the GENERATE step's answer text is an illustrative LLM stand-in (stated in the honesty footer).

Produced:
  rag15_pipeline_flow.gif -- state flows through route -> retrieve -> rerank -> guardrail -> generate,
                             one node at a time; the trace and the state ribbon assemble.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from orchestration import (
    FACT_QUERY,
    build_app,
    full_corpus,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 100
FPS = 12


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def build_animation() -> None:
    corpus = full_corpus()
    app = build_app(corpus)
    result = app.run(FACT_QUERY)  # real run: trace + answer
    trace = result.trace

    # the five nodes, left to right, with the state field each adds
    nodes = [
        ("ROUTE", "+route", AMBER),
        ("RETRIEVE", "+retrieved", BLUE),
        ("RERANK", "+context", BLUE),
        ("GUARDRAIL", "+grounding", GREEN),
        ("GENERATE", "+answer", PURPLE),
    ]
    n_nodes = len(nodes)

    intro = 8
    per_node = 14
    hold = 26
    total = intro + n_nodes * per_node + hold

    node_w = 0.17
    node_xs = [0.02 + i * 0.196 for i in range(n_nodes)]
    node_y = 0.60

    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04)

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        active = min(max((frame - intro) // per_node, -1), n_nodes - 1)  # index of the node running now
        done = min(max((frame - intro) // per_node, 0), n_nodes)  # how many nodes have completed
        in_hold = frame >= intro + n_nodes * per_node

        fig.suptitle("Orchestration: state flows through the chain, one step at a time",
                     fontsize=13, color=INK, y=0.965, fontweight="bold")
        ax.text(0.02, 0.86, f"query: {_short(FACT_QUERY, 62)}", ha="left", fontsize=8.8, color=BLUE, fontweight="bold")

        # node boxes: grey until reached, coloured (bold) when active/done
        for i, ((title, field, col), x) in enumerate(zip(nodes, node_xs)):
            reached = i < done or i == active
            face = col if reached else SLATE
            alpha = 0.92 if reached else 0.22
            ax.add_patch(plt.Rectangle((x, node_y), node_w, 0.16, facecolor=face, alpha=alpha,
                         edgecolor=INK if reached else GRID, linewidth=1.4 if i == active else 1.0))
            ax.text(x + node_w / 2, node_y + 0.08, title, ha="center", va="center",
                    color="white" if reached else GRID, fontsize=8.0, fontweight="bold")
            if i < n_nodes - 1:
                ax.annotate("", xy=(node_xs[i + 1], node_y + 0.08), xytext=(x + node_w, node_y + 0.08),
                            arrowprops=dict(arrowstyle="->", color=INK if i < done else GRID, lw=1.5))

        # the state ribbon assembling underneath (each reached node adds its field)
        ax.add_patch(plt.Rectangle((0.02, 0.40), 0.96, 0.11, facecolor=BLUE, alpha=0.07, edgecolor=BLUE, linewidth=1.2))
        fields = ["query"] + [nodes[i][1] for i in range(min(done, n_nodes))]
        ax.text(0.03, 0.455, "state: " + "  ".join(fields), ha="left", va="center", fontsize=8.4,
                color=INK, fontweight="bold")

        # the real trace lines revealed as nodes complete
        ax.text(0.02, 0.34, "trace:", ha="left", fontsize=8.6, color=SLATE, fontweight="bold")
        shown = trace[:done] if not in_hold else trace
        ys = [0.29 - j * 0.045 for j in range(len(shown))]
        for line, y in zip(shown, ys):
            ax.text(0.05, y, line, ha="left", va="center", fontsize=7.8, color=INK, family="monospace")

        # final answer on hold
        if in_hold:
            ax.text(0.5, 0.045, f"ANSWER: {_short(result.answer, 74)}", ha="center", fontsize=9.2,
                    color=GREEN, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN,
                    alpha=0.10, edgecolor=GREEN))

        # honesty footer
        ax.text(0.5, -0.005, "trace lines, route score and grounding are REAL orchestration.py output; "
                "only the GENERATE answer text is an illustrative LLM stand-in",
                ha="center", va="bottom", fontsize=6.8, color=SLATE, style="italic")
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.008), 0.08 * prog, 0.006, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag15_pipeline_flow.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()

"""Animated (GIF) intuition figure for 10-Agentic-RAG.

Companion to the static PNGs. It brings the core agentic idea to life -- the ReAct loop stepping
through the compound query: a Thought appears (the agent reasons about what it needs), an Action
fires (it picks a tool -- retrieve or calculator), an Observation comes back (the tool's real
result), and the running answer assembles fact by fact -- retrieve the period, compute the count,
retrieve the resolution, finish. You watch a FIXED single-shot pipeline give way to an adaptive
loop that iterates until it can answer.

    python make_animation_10.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
trace it animates -- every Thought/Action/Observation and the final answer -- is the REAL agent's
own output from agentic_rag.py (compound_orbit_policy over the shared Helios-7 corpus); only the
Thought text is an illustrative stand-in for an LLM policy (see the honesty footer in the GIF).

Produced:
  rag10_react_loop.gif -- the agent stepping Thought -> Action -> Observation, assembling the answer.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from agentic_rag import COMPOUND_QUERY, build_agent, compound_orbit_policy, full_corpus

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # retrieve / observation
PURPLE = "#5D4A8A"  # thought
GREEN = "#2E7A5A"  # finish / answer
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"  # action / calculator
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 100
FPS = 12

TOOL_COLOR = {"retrieve": BLUE, "calculator": AMBER, "finish": GREEN}


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def build_animation() -> None:
    corpus = full_corpus()
    agent, _tools, _retriever = build_agent(corpus)
    result = agent.run(COMPOUND_QUERY, compound_orbit_policy)
    steps = result.steps

    # Per step there are three reveal beats (Thought, Action, Observation), then a hold. Frames are
    # grouped so the viewer can read each beat before the next appears.
    beats_per_step = 3
    frames_per_beat = 6
    hold_frames = 22
    total = len(steps) * beats_per_step * frames_per_beat + hold_frames

    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04)

    # a compact running "answer assembles" summary keyed off the step index
    running_facts = ["", "period: 97 min", "→ 14 orbits/day", "+ imager: 4 m", "answer ready"]

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # which step + beat are we revealing?
        unit = frames_per_beat
        idx = min(frame // unit, len(steps) * beats_per_step - 1)
        step_i = idx // beats_per_step
        beat = idx % beats_per_step  # 0=thought, 1=action, 2=observation
        in_hold = frame >= len(steps) * beats_per_step * unit

        fig.suptitle("Agentic RAG: the ReAct loop iterating to an answer", fontsize=13.5,
                     color=INK, y=0.965, fontweight="bold")
        ax.text(0.5, 0.925, f"query: “{COMPOUND_QUERY}”", ha="center", va="center", fontsize=8.6,
                color=SLATE, style="italic")

        # left: the three loop nodes light up in turn for the current step
        node_defs = [("THOUGHT", PURPLE, 0.78), ("ACTION", AMBER, 0.60), ("OBSERVATION", BLUE, 0.42)]
        for b, (label, color, y) in enumerate(node_defs):
            active = (b <= beat) or in_hold
            face = color if active else "white"
            tcol = "white" if active else SLATE
            ax.add_patch(plt.Rectangle((0.04, y - 0.06), 0.30, 0.12, facecolor=face,
                         alpha=0.92 if active else 0.35, edgecolor=color, linewidth=1.6))
            ax.text(0.19, y, label, ha="center", va="center", fontsize=9.5, color=tcol, fontweight="bold")
            if b < 2:
                ax.annotate("", xy=(0.19, y - 0.075), xytext=(0.19, y - 0.045),
                            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        # loop-back hint
        ax.annotate("", xy=(0.04, 0.80), xytext=(0.04, 0.40),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6, connectionstyle="arc3,rad=-0.5"))
        ax.text(0.015, 0.60, "loop", ha="center", va="center", fontsize=7.8, color=GREEN,
                rotation=90, fontweight="bold", style="italic")

        # step badge
        step = steps[step_i]
        color = TOOL_COLOR.get(step.action, SLATE)
        ax.text(0.19, 0.20, f"step {step_i + 1} of {len(steps)}", ha="center", fontsize=9.5,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor=INK, linewidth=1.0))

        # right: the text of the current beat(s), revealed cumulatively
        tx = 0.40
        if beat >= 0 or in_hold:
            ax.text(tx, 0.80, "Thought:", ha="left", fontsize=9.2, color=PURPLE, fontweight="bold")
            ax.text(tx, 0.75, _wrap(step.thought, 60), ha="left", va="top", fontsize=8.4,
                    color=PURPLE, style="italic")
        if beat >= 1 or in_hold:
            ax.text(tx, 0.58, "Action:", ha="left", fontsize=9.2, color=color, fontweight="bold")
            ax.text(tx, 0.535, _wrap(f"{step.action}({step.action_input!r})", 58), ha="left", va="top",
                    fontsize=8.4, color=color, fontweight="bold")
        if (beat >= 2 or in_hold) and step.action != "finish":
            ax.text(tx, 0.40, "Observation:", ha="left", fontsize=9.2, color=BLUE, fontweight="bold")
            ax.text(tx, 0.355, _wrap(step.observation, 60), ha="left", va="top", fontsize=8.4, color=INK)

        # running answer assembly bar
        assembled = running_facts[min(step_i + (1 if (beat >= 2 or in_hold) else 0), len(running_facts) - 1)]
        if not in_hold:
            ax.text(0.40, 0.19, "assembling answer:", ha="left", fontsize=8.6, color=GREEN, fontweight="bold")
            ax.text(0.40, 0.15, assembled, ha="left", fontsize=8.8, color=GREEN)
        else:
            ax.text(0.5, 0.15, _wrap("ANSWER: " + result.answer, 90), ha="center", va="top",
                    fontsize=8.8, color=GREEN, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=GREEN, alpha=0.14, edgecolor=GREEN))

        # honesty footer (kept clear of the answer box)
        ax.text(0.5, 0.005, "loop, tools, retrieval and arithmetic are REAL all-MiniLM / calculator "
                "outputs; the Thought text is an illustrative LLM-policy stand-in",
                ha="center", va="bottom", fontsize=6.9, color=SLATE, style="italic")
        # a tiny per-frame progress tick (bottom-right) -- also keeps every frame visually distinct so
        # the GIF writer does not dedupe consecutive hold/beat frames into one, preserving the pacing.
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.01), 0.08 * prog, 0.008, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag10_react_loop.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()

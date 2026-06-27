"""Animated (GIF) figure generator for 17-Chain-of-Thought-Reasoning.

Companion to make_figures_17.py. Animates the chapter's intuition figure -- the same two-panel
prompt comparison, brought to life so the reader watches the chain of thought *unfold* line by
line and land the right answer, while the direct prompt blurts a wrong one.

    python make_animations_17.py

Writes ../../images/cot_direct_vs_cot_prompt.gif via Pillow (per-frame durations so each reasoning
step holds long enough to read; final answer held longest). Palette imported from make_figures_17
so it matches the static figure exactly; the worked problem and numbers are identical.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from make_figures_17 import GREEN, INK, RED, SLATE

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 95

# The right panel's chain of thought, revealed one line per frame.
COT_LINES = [
    "A: Let's think step by step.",
    "  Start: 12 cups.",
    "  Pour out 1/3 of 12 = 4, leaving 8.",
    "  Add 5: 8 + 5 = 13.",
]


def _render(reveal: int, show_wrong: bool, show_right: bool) -> Image.Image:
    """reveal = how many CoT lines are shown on the right; show_wrong/right toggle the verdicts."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Left panel: direct prompt -> blurts wrong.
    ax_l.add_patch(plt.Rectangle((0.04, 0.06), 0.92, 0.88, facecolor="white", edgecolor=RED, linewidth=2))
    ax_l.text(0.5, 0.88, "DIRECT  (answer only)", ha="center", fontsize=13, fontweight="bold", color=RED)
    ax_l.text(0.08, 0.74,
              "Q: A jug holds 12 cups. You pour out\n"
              "    1/3, then add 5 cups. How many\n"
              "    cups are in the jug now?",
              ha="left", va="top", fontsize=10.5, color=INK)
    if show_wrong:
        ax_l.text(0.08, 0.40, "A: 9", ha="left", va="top", fontsize=12, color=INK, fontfamily="monospace")
        ax_l.text(0.5, 0.18, "✗  wrong  (blurted)", ha="center", fontsize=12, fontweight="bold", color=RED)

    # Right panel: CoT prompt -> reasons, then lands right.
    ax_r.add_patch(plt.Rectangle((0.04, 0.06), 0.92, 0.88, facecolor="white", edgecolor=GREEN, linewidth=2))
    ax_r.text(0.5, 0.88, "CHAIN-OF-THOUGHT  (step by step)", ha="center", fontsize=13, fontweight="bold", color=GREEN)
    ax_r.text(0.08, 0.80, "Q: ... (same problem)", ha="left", va="top", fontsize=10.5, color=INK)
    if reveal >= 1:
        ax_r.text(0.08, 0.70, COT_LINES[0], ha="left", va="top", fontsize=10.5, color=INK)
    body = "\n".join(COT_LINES[1:reveal]) if reveal > 1 else ""
    if body:
        ax_r.text(0.08, 0.58, body, ha="left", va="top", fontsize=10, color=SLATE, fontfamily="monospace")
    if show_right:
        ax_r.text(0.08, 0.24, "  Answer: 13", ha="left", va="top", fontsize=12, color=INK, fontfamily="monospace")
        ax_r.text(0.5, 0.11, "✓  right  (worked)", ha="center", fontsize=12, fontweight="bold", color=GREEN)

    fig.suptitle("Same problem, two prompt structures: room to work changes the answer",
                 fontsize=13, color=INK, y=0.97)
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.97)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_cot_prompt_gif() -> None:
    # (reveal, show_wrong, show_right, duration_ms)
    schedule = [
        (0, False, False, 900),   # both questions posed
        (0, True, False, 1100),   # left blurts the wrong answer
        (1, True, False, 1000),   # right: "let's think step by step"
        (2, True, False, 1000),   # right: start 12
        (3, True, False, 1100),   # right: pour out -> 8
        (4, True, False, 1100),   # right: add 5 -> 13
        (4, True, True, 2600),    # right: lands 13, verdicts (hold)
    ]
    frames = [_render(r, w, rt) for (r, w, rt, _d) in schedule]
    durations = [d for (*_x, d) in schedule]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cot_direct_vs_cot_prompt.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_cot_prompt_gif()

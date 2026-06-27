"""Animated (GIF) figure generator for 07-Mixture-of-Experts.

Brings the MoE routing intuition to life: a stream of tokens, each sent by the router to its top-2
experts (arrows weighted by the gate values), one token at a time, so the reader watches sparse
routing happen and the per-expert load build up. Complements the static moe_routing.png schematic.

    python make_animations_07.py

Writes ../../images/moe_routing_flow.gif via Pillow (per-frame durations). Illustrative routing
(a randomly-initialised router produces arbitrary assignments anyway); the palette matches the
chapter's Mermaid classDefs. Self-contained.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
MUTE = "#C7CDD4"
BG = "#FBFCFD"

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 95
N_EXPERTS = 8

# Illustrative top-2 routing per token: (expert_index, gate) pairs, gates sum to 1.
ROUTING = [
    [(0, 0.70), (2, 0.30)],
    [(4, 0.60), (1, 0.40)],
    [(7, 0.80), (3, 0.20)],
    [(1, 0.60), (5, 0.40)],
    [(6, 0.70), (0, 0.30)],
    [(3, 0.60), (7, 0.40)],
]
N_TOKENS = len(ROUTING)

EXPERT_X = [0.6 + i * 1.05 for i in range(N_EXPERTS)]   # experts spread across the top
EXPERT_Y = 3.15
TOKEN_X = [1.4 + i * 1.15 for i in range(N_TOKENS)]      # tokens along the bottom
TOKEN_Y = 0.35
ROUTER_X, ROUTER_Y = 4.55, 1.75


def _expert_box(ax, i: int, load: int, active: bool) -> None:
    x = EXPERT_X[i]
    color = GREEN if active else SLATE
    ax.add_patch(Rectangle((x - 0.42, EXPERT_Y - 0.32), 0.84, 0.64, facecolor=color,
                           alpha=0.16 if not active else 0.26, edgecolor=color, linewidth=2))
    ax.text(x, EXPERT_Y + 0.02, f"E{i + 1}", ha="center", va="center", color=INK,
            fontsize=9.5, fontweight="bold")
    # a small load meter under each expert
    ax.text(x, EXPERT_Y - 0.55, "▮" * load if load else "·", ha="center", va="center",
            color=AMBER, fontsize=9)


def _render(step: int) -> Image.Image:
    """step = number of tokens routed so far; the (step-1)-th is the 'current' highlighted one."""
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.2, 3.9)
    ax.axis("off")
    ax.set_title("Sparse routing: each token goes to its top-2 experts",
                 fontsize=13, fontweight="bold", color=INK)

    loads = [0] * N_EXPERTS
    for ti in range(step):
        for (ei, _g) in ROUTING[ti]:
            loads[ei] += 1
    current = step - 1

    # Router hub.
    ax.add_patch(Rectangle((ROUTER_X - 0.9, ROUTER_Y - 0.3), 1.8, 0.6, facecolor=PURPLE,
                           alpha=0.22, edgecolor=PURPLE, linewidth=2))
    ax.text(ROUTER_X, ROUTER_Y, "router (top-2)", ha="center", va="center", color=INK,
            fontsize=10, fontweight="bold")

    # Experts.
    active_experts = {ei for (ei, _g) in ROUTING[current]} if 0 <= current < N_TOKENS else set()
    for i in range(N_EXPERTS):
        _expert_box(ax, i, loads[i], i in active_experts)

    # Tokens + routing arrows.
    for ti in range(N_TOKENS):
        is_done = ti < step
        is_cur = ti == current
        tcolor = BLUE if is_cur else (SLATE if is_done else MUTE)
        ax.add_patch(Rectangle((TOKEN_X[ti] - 0.32, TOKEN_Y - 0.22), 0.64, 0.44,
                               facecolor=tcolor, alpha=0.85 if is_cur else (0.4 if is_done else 0.2),
                               edgecolor=tcolor, linewidth=1.5))
        ax.text(TOKEN_X[ti], TOKEN_Y, f"t{ti + 1}", ha="center", va="center",
                color="white" if is_cur else INK, fontsize=9, fontweight="bold")
        if not is_done:
            continue
        # token -> router
        ax.add_patch(FancyArrowPatch((TOKEN_X[ti], TOKEN_Y + 0.24), (ROUTER_X, ROUTER_Y - 0.32),
                                     arrowstyle="-", color=SLATE,
                                     alpha=0.5 if is_cur else 0.12, linewidth=1.0))
        # router -> top-2 experts (thickness ~ gate)
        for (ei, g) in ROUTING[ti]:
            col = GREEN if is_cur else SLATE
            ax.add_patch(FancyArrowPatch((ROUTER_X, ROUTER_Y + 0.32),
                                         (EXPERT_X[ei], EXPERT_Y - 0.34),
                                         arrowstyle="-|>", mutation_scale=12, color=col,
                                         alpha=0.9 if is_cur else 0.18,
                                         linewidth=1.0 + 3.5 * g))
            if is_cur:
                mx = (ROUTER_X + EXPERT_X[ei]) / 2
                my = (ROUTER_Y + 0.32 + EXPERT_Y - 0.34) / 2
                ax.text(mx, my, f"{g:.1f}", ha="center", va="center", color=GREEN,
                        fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none"))

    if 0 <= current < N_TOKENS:
        combine = " + ".join(f"{g:.1f}·E{ei + 1}(x)" for (ei, g) in ROUTING[current])
        ax.text(4.6, -0.05, f"t{current + 1}:  y = {combine}", ha="center", va="center",
                color=INK, fontsize=10, fontweight="bold")
    else:
        ax.text(4.6, -0.05, "every token used only 2 of 8 experts — that's the sparsity win",
                ha="center", va="center", color=INK, fontsize=10, fontweight="bold")

    fig.subplots_adjust(top=0.9, bottom=0.04, left=0.02, right=0.98)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_routing_gif() -> None:
    schedule = [(0, 900)] + [(s, 1250) for s in range(1, N_TOKENS + 1)]
    schedule[-1] = (N_TOKENS, 2600)  # hold the final routed state
    frames = [_render(s) for (s, _d) in schedule]
    durations = [d for (_s, d) in schedule]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "moe_routing_flow.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_routing_gif()

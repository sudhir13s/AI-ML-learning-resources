"""Animated (GIF) figure generator for 06-Efficient-Attention-FlashAttention.

The killer FlashAttention intuition is the TILED/BLOCKWISE sweep: a block of keys/values steps
across the sequence, contributing to the running softmax (running max m + running denominator l)
WITHOUT the full N x N score matrix ever existing in one place. This GIF makes that visible -- you
watch one tile move along the key axis, the "sticky note" of (m, l) update block by block, and the
full matrix stay un-materialized (greyed) the whole time.

    python make_animation_06.py

The (m, l) values stepped through are the EXACT seeded trace flash_attention() emits for query 0
(via flash_attention.online_softmax_trace) -- no invented numbers. Written to ../../images/ with
matplotlib PillowWriter (no ffmpeg) and quantised to a 64-colour palette so the file stays small.

Verified on Python 3.12 / matplotlib 3.10 / Pillow.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_attention as fa  # seeded (m, l) trace -- the same numbers the .py prints

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"
GRID = "#D4D9DF"
IDLE = "#E7EAEE"  # un-materialized cells of the matrix

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 96
FPS = 2  # slow -- the reader needs to read each block's bookkeeping
HOLD_FRAMES = 3  # repeat the final frame so the GIF rests on the finished state


def _quantise_save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Save a list of PIL frames as a 64-colour GIF (small file) with a final hold."""
    held = frames + [frames[-1]] * HOLD_FRAMES
    quantised = [f.convert("RGB").quantize(colors=64, method=Image.MEDIANCUT) for f in held]
    duration_ms = int(1000 / fps)
    quantised[0].save(
        path, save_all=True, append_images=quantised[1:], duration=duration_ms,
        loop=0, optimize=True, disposal=2,
    )
    print(f"wrote {path.name} ({len(quantised)} frames, {path.stat().st_size // 1024} KB)")


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()


def make_tiling_gif() -> None:
    """Step a K/V tile across the key axis; update the running (m, l) sticky note each block.

    Left panel: the N x N score grid. The full matrix stays IDLE-grey (never materialized);
    only the current tile's column-band lights up GREEN as it sweeps, and the query-0 row is
    outlined so the reader tracks whose softmax is being accumulated. Right panel: the sticky
    note carrying the running max m, running denominator l, and the correction factor applied
    when m jumps -- the exact seeded trace for query 0.
    """
    n, block = 8, 2
    idx, m_trace, l_trace, corr_trace = fa.online_softmax_trace(
        seq_len=n, head_dim=fa.HEAD_DIM, block_size=block
    )
    n_blocks = len(idx)
    q_row = 0  # the query whose running softmax we visualise

    frames: list[Image.Image] = []
    for step in range(n_blocks):
        col_start = step * block
        fig, (ax_grid, ax_note) = plt.subplots(
            1, 2, figsize=(10.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
        )

        # ---- Left: the score grid, tile sweeping across keys -------------------------
        for i in range(n):
            for j in range(n):
                in_tile = (col_start <= j < col_start + block)
                done = j < col_start
                if in_tile:
                    color, alpha = GREEN, 0.92
                elif done:
                    color, alpha = BLUE, 0.30  # already streamed (contribution folded in)
                else:
                    color, alpha = IDLE, 0.55  # not yet seen -- never materialized in full
                ax_grid.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=color,
                                            edgecolor="white", linewidth=1.0, alpha=alpha))
        # outline the query row whose softmax we are accumulating
        ax_grid.add_patch(Rectangle((0, n - 1 - q_row), n, 1, fill=False,
                                    edgecolor=PURPLE, linewidth=2.6))
        ax_grid.text(-0.15, n - 1 - q_row + 0.5, "query 0", rotation=0, va="center",
                     ha="right", color=PURPLE, fontsize=9, fontweight="bold")
        # block bracket under the active tile
        ax_grid.add_patch(Rectangle((col_start, -0.55), block, 0.32, facecolor=GREEN,
                                    edgecolor="none"))
        ax_grid.text(col_start + block / 2, -0.85, f"K/V tile {step}", ha="center",
                     color=GREEN, fontsize=9.5, fontweight="bold")

        ax_grid.set_xlim(-0.4, n + 0.3)
        ax_grid.set_ylim(-1.3, n + 0.9)
        ax_grid.set_aspect("equal")
        ax_grid.axis("off")
        ax_grid.text(n / 2, n + 0.45, "keys (streamed one tile at a time →)", ha="center",
                     color=SLATE, fontsize=9.5)
        ax_grid.set_title("Score grid — the full $N{\\times}N$ matrix is never assembled",
                          color=INK, fontsize=11, fontweight="bold")

        # ---- Right: the running-softmax sticky note ----------------------------------
        ax_note.set_xlim(0, 1)
        ax_note.set_ylim(0, 1)
        ax_note.axis("off")
        ax_note.add_patch(FancyBboxPatch((0.04, 0.06), 0.92, 0.88,
                                         boxstyle="round,pad=0.02,rounding_size=0.03",
                                         facecolor="#FBF7E9", edgecolor=AMBER, linewidth=2.0))
        ax_note.text(0.5, 0.88, "sticky note (per query)", ha="center", color=AMBER,
                     fontsize=11, fontweight="bold")
        ax_note.text(0.5, 0.80, "two scalars + an output accumulator — all of O(1)",
                     ha="center", color=SLATE, fontsize=8.5, style="italic")

        m_now, l_now, corr_now = m_trace[step], l_trace[step], corr_trace[step]
        ax_note.text(0.10, 0.64, "running max  $m$", color=PURPLE, fontsize=12, fontweight="bold")
        ax_note.text(0.90, 0.64, f"{m_now:+.4f}", color=PURPLE, fontsize=13,
                     fontweight="bold", ha="right")
        ax_note.text(0.10, 0.50, "running denom  $\\ell$", color=AMBER, fontsize=12,
                     fontweight="bold")
        ax_note.text(0.90, 0.50, f"{l_now:.4f}", color=AMBER, fontsize=13,
                     fontweight="bold", ha="right")

        # the rescale message: on the first block corr is 0 (inf seed); show the action.
        if step == 0:
            msg = "block 0: seed $m,\\ell$ from this tile"
            msg_color = SLATE
        elif corr_now >= 0.999:
            msg = f"$m$ unchanged → corr = {corr_now:.3f}\n(no rescale; just add this tile)"
            msg_color = SLATE
        else:
            msg = (f"$m$ jumped → corr = $e^{{m_{{old}}-m_{{new}}}}$ = {corr_now:.3f}\n"
                   f"old $\\ell$, acc rescaled by {corr_now:.3f} before adding")
            msg_color = RED
        ax_note.text(0.5, 0.32, msg, ha="center", va="center", color=msg_color,
                     fontsize=9.5, fontweight="bold")
        ax_note.text(0.5, 0.14, f"after block {step} of {n_blocks - 1}", ha="center",
                     color=INK, fontsize=10)
        ax_note.set_title("Online softmax — the running statistics (query 0)",
                          color=INK, fontsize=11, fontweight="bold")

        fig.suptitle("FlashAttention: stream one tile, update the running softmax, never "
                     "materialize the matrix", color=INK, fontsize=12.5, fontweight="bold",
                     y=0.97)
        fig.subplots_adjust(top=0.82, wspace=0.18)
        frames.append(_fig_to_image(fig))
        plt.close(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _quantise_save_gif(frames, OUT_DIR / "fa_tiling.gif", FPS)


def main() -> None:
    print(f"writing animation to {OUT_DIR}")
    make_tiling_gif()
    print("done.")


if __name__ == "__main__":
    main()

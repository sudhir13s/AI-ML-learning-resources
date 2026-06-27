"""Animated (GIF) figure generator for 02-Pretraining-at-Scale.

Companion to ``make_figures_02.py``. Where the static PNGs show the final loss curve, this brings
it to life so a reader *watches* the loss fall as tokens stream through the model -- the single
clearest moving intuition for pretraining. Run:

    python make_animation_02.py

The GIF is written to ../../images/ via matplotlib's PillowWriter (no ffmpeg required) and reduced
to a 64-colour palette so it stays small. The numbers are NOT invented for the animation: the loss
and LR traces come from ``training_loop_trace()`` in ``pretraining_at_scale.py`` -- the exact seeded
real-recipe run (AdamW + warmup->cosine + grad clip) the page and notebook report, start ~2.84 ->
end ~0.75 over 300 steps. The running "tokens seen" counter is steps x CONTEXT_LEN x batch (=1).

Animation produced:
  pt_training_loss.gif -- the loss curve drawn point by point as the LR warms up then cosine-decays,
                          with a live tokens-streamed / loss / LR readout; final frame == the static
                          pt_training_loss.png.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x / Pillow (PillowWriter + Image).
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from PIL import Image

# Reuse the EXACT palette/styling and the SAME seeded trace as the static figures (one source of
# truth) so the animation cannot drift from the chapter's PNG or prose.
from make_figures_02 import AMBER, BLUE, GREEN, INK, RED, _style_axis
from pretraining_at_scale import CONTEXT_LEN, LR_PEAK, training_loop_trace

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
DPI = 80
GIF_COLORS = 48  # quantize to a small palette (<=64 colours) -> small GIF


def make_training_loss_gif() -> None:
    """Draw the real-recipe loss curve point by point as tokens stream in; the LR schedule rides a
    twin axis. Frames are assembled with Pillow (one growing-curve frame every few steps, plus a
    hold on the final state) and quantized to GIF_COLORS so the file stays small."""
    steps, losses, lrs, _ = training_loop_trace()
    n = len(steps)
    tokens_seen = [(s + 1) * CONTEXT_LEN for s in steps]  # batch=1: CONTEXT_LEN new tokens per step

    # Reveal every few steps for a watchable pace; one frame per reveal point.
    reveal = list(range(2, n + 1, 8))
    if reveal[-1] != n:
        reveal.append(n)

    def render(k: int) -> Image.Image:
        """Render the curve up to step index k (1-based count of revealed points)."""
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        _style_axis(ax)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, losses[0] * 1.08)
        ax.set_xlabel("optimizer step  (tokens stream in →)")
        ax.set_ylabel("next-token cross-entropy loss (nats)", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)

        ax_lr = ax.twinx()
        ax_lr.set_ylim(0, LR_PEAK * 1.15)
        ax_lr.set_ylabel("learning rate (warmup → cosine)", color=AMBER)
        ax_lr.tick_params(axis="y", colors=AMBER)
        ax_lr.spines["top"].set_visible(False)

        x = steps[:k]
        ax_lr.plot(x, lrs[:k], color=AMBER, linewidth=1.7, linestyle="--", zorder=3)
        ax.plot(x, losses[:k], color=BLUE, linewidth=2.6, zorder=4)
        j = k - 1
        ax.scatter([steps[j]], [losses[j]], color=GREEN if k == n else BLUE, s=55, zorder=5)
        if k == n:
            ax.scatter([steps[0]], [losses[0]], color=RED, s=55, zorder=5)

        ax.set_title("Pretraining: the loss falls as tokens stream through the model",
                     fontsize=12.5, fontweight="bold", color=INK)
        fig.text(0.5, 0.012,
                 f"tokens seen {tokens_seen[j]:>4,}    step {steps[j]:>3}/{n - 1}    "
                 f"loss {losses[j]:.3f}    lr {lrs[j]:.2e}",
                 ha="center", color=INK, fontsize=10, fontweight="bold")

        fig.tight_layout(rect=(0, 0.05, 1, 1))
        buf = io.BytesIO()
        # Fixed figsize, NO bbox_inches="tight" -> every frame is the same pixel size (a GIF
        # requirement); white facecolor so frames composite cleanly.
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    frames = [render(k) for k in reveal]
    palette = frames[-1].quantize(colors=GIF_COLORS)  # one stable 64-colour palette for all frames
    frames_p = [f.quantize(colors=GIF_COLORS, palette=palette) for f in frames]
    durations = [150] * len(frames_p)
    durations[-1] = 2600  # hold the final frame (= the static figure) so it's readable

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pt_training_loss.gif"
    frames_p[0].save(out, save_all=True, append_images=frames_p[1:], duration=durations,
                     loop=0, disposal=2, optimize=True)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KiB, {len(frames_p)} frames)")


def main() -> None:
    make_training_loss_gif()
    print("animation written to", OUT_DIR)


if __name__ == "__main__":
    main()

"""Animated (GIF) intuition figure for 01-Language-Modeling-Objectives.

The intuition the page opens with -- a model that *generates* by predicting one token at a time,
each new token conditioned on every token before it -- is fundamentally dynamic, so it gets an
animation rather than a static panel. This GIF steps through autoregressive generation token by
token: the context so far is shown as a row, the next-token distribution lights up over the toy
vocabulary, the arg-max token is emitted and appended to the context, and the loop repeats. That
"see only the past, predict forward" loop is exactly the causal objective made visible.

    python make_animation_01.py

The GIF is written to ../../images/lm_autoregressive.gif using Pillow per-frame durations (no
ffmpeg required), kept small with a 64-colour palette and modest dimensions.

The vocabulary and the emitted sequence ("the cat sat on mat") are the SAME toy setup the page,
notebook, and the static figures use -- the per-step next-token distributions are deterministic
(a fixed peaked distribution centred on the true next token), so the animation is reproducible.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a buffer, never open a window
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from language_modeling_objectives import VOCAB

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

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 92
MAX_COLORS = 64  # keep the GIF small

# The toy sentence the model "generates" autoregressively (token ids into VOCAB).
SEQUENCE_IDS = [0, 1, 2, 3, 4]  # the cat sat on mat


def _peaked_distribution(true_id: int) -> np.ndarray:
    """A fixed, deterministic peaked next-token distribution centred on the true next token.

    The model is confident-but-not-certain: it puts the bulk of the mass on the true token and
    spreads a little over the rest. Deterministic so every frame (and every run) is identical.
    """
    probs = np.full(len(VOCAB), 0.04)
    probs[true_id] = 1.0  # peak on the true next token
    probs = probs / probs.sum()
    return probs


def render_step(emitted: int) -> Image.Image:
    """Render the frame after `emitted` tokens have been generated (emitted in 1..len-1).

    Top: the context-so-far as a row of tokens (the past the model conditions on, blue) with the
    just-emitted token highlighted (green). Bottom: the next-token distribution the model uses to
    pick the following token, with the chosen (arg-max) bar in green.
    """
    n = len(SEQUENCE_IDS)
    fig, (ax_seq, ax_dist) = plt.subplots(2, 1, figsize=(8.6, 5.2),
                                          gridspec_kw={"height_ratios": [1, 1.5]})

    # ---- Top: the generated sequence so far -------------------------------------------
    ax_seq.set_xlim(-0.3, n + 0.3)
    ax_seq.set_ylim(-0.2, 1.2)
    ax_seq.axis("off")
    ax_seq.set_title("Autoregressive generation: one token at a time, conditioned on the past",
                     fontsize=13, color=INK, fontweight="bold", pad=12)
    for i in range(n):
        if i < emitted:
            fc, txt_color, tok = BLUE, "#fff", VOCAB[SEQUENCE_IDS[i]]
            if i == emitted - 1:
                fc = GREEN  # the token emitted on THIS step
        else:
            fc, txt_color, tok = "#EAECEF", "#9aa3ad", "?"  # not generated yet
        ax_seq.add_patch(plt.Rectangle((i + 0.06, 0.25), 0.88, 0.6, facecolor=fc,
                                       edgecolor="white", linewidth=1.2, zorder=3))
        ax_seq.text(i + 0.5, 0.55, tok, ha="center", va="center", color=txt_color,
                    fontsize=13, fontweight="bold", zorder=4)
    # bracket the context the next prediction conditions on (tokens 0..emitted-1)
    if emitted < n:
        ax_seq.annotate("", xy=(emitted, 0.12), xytext=(0.06, 0.12),
                        arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=1.4))
        ax_seq.text(emitted / 2.0, 0.0, "context so far (the past) — see only this",
                    ha="center", va="top", fontsize=9.5, color=SLATE)

    # ---- Bottom: the next-token distribution -------------------------------------------
    x = np.arange(len(VOCAB))
    if emitted < n:
        true_next = SEQUENCE_IDS[emitted]
        probs = _peaked_distribution(true_next)
        colors = [GREEN if i == true_next else SLATE for i in range(len(VOCAB))]
        alphas = [1.0 if i == true_next else 0.4 for i in range(len(VOCAB))]
        for i in range(len(VOCAB)):
            ax_dist.bar(x[i], probs[i], color=colors[i], alpha=alphas[i], edgecolor="white",
                        linewidth=0.8, zorder=3)
        ax_dist.set_title(f"step {emitted + 1}: predict next token  →  emit '{VOCAB[true_next]}'  "
                          f"(p = {probs[true_next]:.2f})", fontsize=11.5, color=GREEN,
                          fontweight="bold")
    else:
        # final frame: the whole sentence is generated, nothing left to predict
        ax_dist.bar(x, np.zeros(len(VOCAB)), color=SLATE, alpha=0.3, zorder=3)
        ax_dist.set_title("sequence complete — 'the cat sat on mat' generated left-to-right",
                          fontsize=11.5, color=BLUE, fontweight="bold")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels([f"'{t}'" for t in VOCAB], fontsize=10)
    ax_dist.set_ylabel("p(next token)")
    ax_dist.set_ylim(0, 1.05)
    ax_dist.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax_dist.set_axisbelow(True)
    for side in ("top", "right"):
        ax_dist.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_dist.spines[side].set_color(SLATE)
    ax_dist.tick_params(colors=INK)
    ax_dist.yaxis.label.set_color(INK)

    fig.tight_layout()
    buf = io.BytesIO()
    # Fixed figsize, NO bbox_inches="tight" -> every frame is the same pixel size (a GIF
    # requirement); facecolor white so frames composite cleanly.
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_autoregressive_gif() -> None:
    """Assemble the per-step frames into lm_autoregressive.gif with Pillow per-frame durations."""
    n = len(SEQUENCE_IDS)
    # one frame per emitted-token count: 1, 2, ..., n (the last frame is the completed sentence)
    images = [render_step(emitted) for emitted in range(1, n + 1)]
    # quantize to a 64-colour palette to keep the GIF small
    images = [img.convert("P", palette=Image.ADAPTIVE, colors=MAX_COLORS) for img in images]
    durations = [1100] * n
    durations[-1] = 2400  # hold the completed sentence so it's readable before the loop restarts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lm_autoregressive.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    print(f"wrote {out}")


def main() -> None:
    make_autoregressive_gif()
    print("animation written to", OUT_DIR)


if __name__ == "__main__":
    main()

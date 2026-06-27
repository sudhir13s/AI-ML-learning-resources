"""Animated (GIF) figure generator for 13-Supervised-Fine-Tuning.

Companion to make_figures_13.py. Animates the chapter's most important visual (sft_prompt_mask):
the same (prompt, response) token stream, but now the reader watches the mask *switch on* -- you
start as if computing loss on every token (all green), then SFT masks the prompt tokens to -100 one
by one, until loss flows only over the response. The final frame is the static figure.

    python make_animations_13.py

Writes ../../images/sft_prompt_mask.gif via Pillow (per-frame durations). The exact token stream,
prompt/response split, tokenizer, and palette are imported from the chapter's own modules.

Verified on Python 3.12 / matplotlib 3.x / Pillow.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from make_figures_13 import AMBER, GREEN, INK, RED, SLATE
from supervised_fine_tuning import DEMOS, build_example, build_tokenizer

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95

INSTRUCTION, RESPONSE = DEMOS[0]
_STOI, _ITOS = build_tokenizer(DEMOS)
_INPUT_IDS, _LABELS, N_PROMPT = build_example(INSTRUCTION, RESPONSE, _STOI, "cpu")
TOKS = [_ITOS[i] for i in _INPUT_IDS.tolist()]
N = len(TOKS)


def _render(masked: int, show_headers: bool) -> Image.Image:
    """masked = how many leading PROMPT tokens have been masked to -100 so far."""
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for idx, tok in enumerate(TOKS):
        is_prompt = idx < N_PROMPT
        masked_now = is_prompt and idx < masked
        colour = SLATE if masked_now else GREEN  # green = loss; slate = masked (-100)
        rect = mpatches.FancyBboxPatch(
            (idx + 0.06, 0.30), 0.88, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=colour, edgecolor="white", linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(idx + 0.5, 0.51, tok, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=3)
        label_txt = "-100" if masked_now else "loss"
        ax.text(idx + 0.5, 0.16, label_txt, ha="center", va="center",
                color=RED if masked_now else GREEN, fontsize=8.5, fontweight="bold")
        ax.text(idx + 0.5, 0.80, str(idx), ha="center", va="center", color=INK, fontsize=8)

    if show_headers:
        ax.text(N_PROMPT / 2, 0.93, "PROMPT  (masked, label = -100)", ha="center",
                color=SLATE, fontsize=10.5, fontweight="bold")
        ax.text(N_PROMPT + (N - N_PROMPT) / 2 + 0.15, 0.93, "RESPONSE  (loss here only)",
                ha="center", color=GREEN, fontsize=10.5, fontweight="bold")
        ax.axvline(N_PROMPT, color=AMBER, linewidth=2.0, linestyle="--", zorder=1)

    title = ("Naive: loss on every token" if masked == 0
             else "SFT prompt masking: loss is computed ONLY on response tokens")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.02, right=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_prompt_mask_gif() -> None:
    # 0 masked (all green) -> mask prompt tokens one by one -> headers + final hold.
    schedule = [(0, False, 1200)]
    for m in range(1, N_PROMPT + 1):
        schedule.append((m, False, 600))
    schedule.append((N_PROMPT, True, 2600))  # reveal headers + boundary, hold
    frames = [_render(m, h) for (m, h, _d) in schedule]
    durations = [d for (*_x, d) in schedule]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sft_prompt_mask.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_prompt_mask_gif()

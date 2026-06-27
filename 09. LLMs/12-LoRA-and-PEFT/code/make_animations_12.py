"""Animated (GIF) figure generator for 12-LoRA-and-PEFT.

Companion to ``make_figures_12.py``. Where that renders the static PNGs, this brings two of them to
life so a reader *watches* the behaviour rather than reading the final state off a finished plot.

    python make_animations_12.py

GIFs are written to ../../images/ via matplotlib's PillowWriter (no ffmpeg required). The palette,
axis styling, and constants are imported from ``make_figures_12`` so the animations cannot drift
from the static figures or the prose -- the delta-growth run uses the SAME seed and construction
order as ``fig_delta_growth`` / ``lora_peft.py``, and the rank sweep uses the same measured losses.

Animations produced:
  lora_delta_growth.gif -- max|ΔW| over training: exactly 0 at step 0 (B = 0, so the adapted layer
                           IS the pretrained layer), then growing as the adapter learns. The
                           animated twin of lora_delta_growth.png.
  lora_rank_vs_fit.gif  -- the rank sweep revealed point by point: trainable params rise linearly
                           with r while the fit loss falls off a cliff exactly at the true update
                           rank (4) and is flat after. The animated twin of lora_rank_vs_fit.png.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter

# Reuse the EXACT palette, axis styling, and constants from the static generator (one source of
# truth) so the animation matches the chapter's figures and prose.
from make_figures_12 import (
    GREEN,
    INK,
    IN_FEATURES,
    LORA_RANK,
    OUT_FEATURES,
    PURPLE,
    RANK_SWEEP,
    RANK_SWEEP_LOSS,
    RED,
    SLATE,
    TRUE_RANK,
    _style_axis,
)

torch.set_num_threads(1)  # single-threaded -> bit-reproducible, matches make_figures_12 / lora_peft.py
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 16


def _delta_growth_history(steps: int = 300) -> list[float]:
    """Reproduce fig_delta_growth's run EXACTLY (same seed, constants, init order) and return the
    max|ΔW| trace, step 0 .. step `steps`. ΔW = (alpha/r)·B·A; B starts at 0 so ΔW starts at 0."""
    seed = 0
    rank, alpha = LORA_RANK, 16
    scaling = alpha / rank
    n_samples, true_rank = 256, TRUE_RANK

    torch.manual_seed(seed)
    _ = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.02  # frozen W consumes RNG first (init order)
    lora_a = torch.empty(rank, IN_FEATURES)
    lora_b = torch.zeros(OUT_FEATURES, rank)  # B = 0 -> ΔW = 0 at init
    torch.nn.init.kaiming_uniform_(lora_a, a=5**0.5)
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)

    torch.manual_seed(seed)  # make_synthetic_task re-seeds, exactly as in lora_peft.py
    base_weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.02
    u = torch.randn(OUT_FEATURES, true_rank) * 0.1
    v = torch.randn(true_rank, IN_FEATURES) * 0.1
    x = torch.randn(n_samples, IN_FEATURES)
    y = F.linear(x, base_weight + u @ v)
    weight = base_weight

    optimizer = torch.optim.Adam([lora_a, lora_b], lr=1e-2)
    delta_max: list[float] = []
    for _ in range(steps + 1):  # record step 0 (pre-update) through step `steps`
        delta_max.append(((lora_b @ lora_a) * scaling).abs().max().item())
        optimizer.zero_grad()
        update = F.linear(F.linear(x, lora_a), lora_b) * scaling
        F.mse_loss(F.linear(x, weight) + update, y).backward()
        optimizer.step()
    return delta_max


def make_delta_growth_gif() -> None:
    """Animate max|ΔW| climbing from exactly 0 (B = 0 init) as the adapter learns."""
    delta = np.asarray(_delta_growth_history(), dtype=float)
    steps = np.arange(len(delta))

    stride = 4
    reveal = list(range(1, len(steps) + 1, stride))
    if reveal[-1] != len(steps):
        reveal.append(len(steps))
    frames = reveal + [len(steps)] * 16

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, delta.max() * 1.08)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"max$|\Delta W| = $ max$|(\alpha/r)\,B\!\cdot\!A|$")
    _style_axis(ax)
    ax.set_title("ΔW starts at exactly 0 (B = 0), then the adapter walks away",
                 fontweight="bold")

    (line,) = ax.plot([], [], color=PURPLE, linewidth=2.4, zorder=4)
    (dot,) = ax.plot([], [], "o", color=PURPLE, markersize=7, zorder=6)
    start = ax.scatter([0], [delta[0]], color=GREEN, s=80, zorder=7)
    ax.annotate("step 0: max|ΔW| = 0\n(B = 0 → the adapted layer\nIS the pretrained layer)",
                xy=(0, delta[0]), xytext=(steps[-1] * 0.12, delta.max() * 0.55),
                color=INK, arrowprops=dict(arrowstyle="->", color=SLATE), fontsize=9.5)
    counter = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=10, fontweight="bold")

    def update(k: int):
        x = steps[:k]
        line.set_data(x, delta[:k])
        j = k - 1
        dot.set_data([steps[j]], [delta[j]])
        counter.set_text(f"step {int(steps[j]):>3}/{int(steps[-1])}    max|ΔW| = {delta[j]:.1f}")
        return line, dot, start

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lora_delta_growth.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_rank_vs_fit_gif() -> None:
    """Reveal the rank sweep point by point: params rise linearly with r, fit loss falls off a cliff
    at the true update rank (4) and is flat after. Same numbers as lora_rank_vs_fit.png."""
    ranks = list(RANK_SWEEP)
    d = IN_FEATURES
    params = [2 * r * d for r in ranks]
    losses = [RANK_SWEEP_LOSS[r] for r in ranks]

    # Reveal one rank at a time, holding a beat on each -- and a longer beat on the cliff (r=4).
    frames: list[int] = []
    for k in range(1, len(ranks) + 1):
        hold = 14 if ranks[k - 1] == TRUE_RANK else 7
        frames += [k] * hold
    frames += [len(ranks)] * 16

    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.set_xlabel("LoRA rank $r$  (log scale)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(ranks[0] * 0.8, ranks[-1] * 1.25)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels([str(r) for r in ranks])
    ax1.set_ylim(0, max(params) * 1.1)
    ax1.set_ylabel("trainable params ($2rd$)", color=GREEN)
    ax1.tick_params(axis="y", colors=GREEN)
    _style_axis(ax1)
    ax1.set_title("Rank vs (params, fit): the cliff sits at the true intrinsic rank",
                 fontweight="bold")

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(min(losses) * 0.3, max(losses) * 3)
    ax2.set_ylabel("final fit loss (log)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["top"].set_visible(False)

    (lp,) = ax1.plot([], [], color=GREEN, marker="o", linewidth=2.2, zorder=4, label="params")
    (ll,) = ax2.plot([], [], color=RED, marker="s", linewidth=2.2, zorder=5, label="loss")
    cliff = ax1.axvline(TRUE_RANK, color=SLATE, linestyle="--", linewidth=1.2, zorder=3, alpha=0.0)
    note = ax1.text(TRUE_RANK * 1.06, max(params) * 0.5, "", color=INK, fontsize=9.5)
    counter = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=10, fontweight="bold")

    def update(k: int):
        lp.set_data(ranks[:k], params[:k])
        ll.set_data(ranks[:k], losses[:k])
        r_now = ranks[k - 1]
        reached_cliff = k >= ranks.index(TRUE_RANK) + 1
        cliff.set_alpha(1.0 if reached_cliff else 0.0)
        if reached_cliff:
            note.set_text(f"true update rank = {TRUE_RANK}\nfit collapses here;\n"
                          f"more rank only adds params")
        if r_now < TRUE_RANK:
            msg = f"r = {r_now}:  under the true rank → underfits (loss {losses[k - 1]:.1e})"
        elif r_now == TRUE_RANK:
            msg = f"r = {r_now}:  hits the true rank → loss falls off a cliff ({losses[k - 1]:.1e})"
        else:
            msg = f"r = {r_now}:  past the true rank → {params[k - 1]:,} params, no better fit"
        counter.set_text(msg)
        return lp, ll, cliff, note

    # Combined legend (both axes).
    lines = [lp, ll]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center left", frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lora_rank_vs_fit.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    make_delta_growth_gif()
    make_rank_vs_fit_gif()
    print("all animations written to", OUT_DIR)


if __name__ == "__main__":
    main()

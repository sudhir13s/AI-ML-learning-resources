"""Reproducible figure generator for 02-Pretraining-at-Scale.

Produces every embedded PNG for the chapter from the SAME seeded numbers used on the page, in the
notebook, and in ``pretraining_at_scale.py`` -- so the figures cannot silently drift from the prose.
Run:

    python make_figures_02.py

Each figure is written to ../../images/ at 150 dpi, prefixed ``pt_``. The palette matches the
chapter's Mermaid diagrams (muted, ink text on light grid). Every number is recomputed here by
importing the seeded helpers from ``pretraining_at_scale`` -- never hardcoded from memory:
  - the warmup->cosine LR schedule (peak 3e-4 at end of warmup, floor 3e-5 at the end);
  - the real-recipe training-loss curve (2.84 -> 0.75 over 300 steps) with the LR overlaid;
  - gradient-accumulation equivalence (|big| == |accumulated| == 1.42, the buggy /K-dropped 8x);
  - the gradient-norm-vs-clip stability trace (the seatbelt capping the step at 1.0);
  - the 6ND production table (GPT-3 3.15e23 ... Llama-3-70B 6.3e24) by tokens/param;
  - the loss-vs-compute power-law (more 6ND -> lower loss);
  - the Adam optimizer-state memory tax (~16 bytes/param) and the ZeRO/FSDP memory shrink.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

# Reuse the EXACT seeded machinery so figures and the script can never disagree.
from pretraining_at_scale import (
    GRAD_CLIP_NORM,
    LR_FLOOR,
    LR_PEAK,
    WARMUP_STEPS,
    adam_memory_tax,
    grad_accum_norms,
    lr_schedule_trace,
    production_table,
    training_loop_trace,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.name}")


# =====================================================================================
# Figure 1 -- the real-recipe training-loss curve (the headline visual of pretraining)
# =====================================================================================
def fig_training_loss() -> None:
    """The toy-but-real recipe loss curve: a fast early drop, then a long slow descent -- the
    recognisable pretraining-loss profile, with the LR schedule overlaid on a twin axis."""
    steps, losses, lrs, _ = training_loop_trace()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style_axis(ax)
    ax.plot(steps, losses, color=BLUE, linewidth=2.4, zorder=4, label="training loss")
    ax.scatter([steps[0], steps[-1]], [losses[0], losses[-1]],
               color=[RED, GREEN], s=60, zorder=5)
    ax.annotate(f"start {losses[0]:.2f}\n(≈ log 16, random init)", (steps[0], losses[0]),
                textcoords="offset points", xytext=(26, -30), color=RED,
                fontsize=9.5, fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.0))
    ax.annotate(f"end {losses[-1]:.2f}", (steps[-1], losses[-1]),
                textcoords="offset points", xytext=(-8, 24), color=GREEN,
                fontsize=10, fontweight="bold", ha="right")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("next-token cross-entropy loss (nats)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0, losses[0] * 1.08)

    ax_lr = ax.twinx()
    ax_lr.plot(steps, lrs, color=AMBER, linewidth=1.8, linestyle="--", zorder=3,
               label="learning rate")
    ax_lr.set_ylabel("learning rate (warmup → cosine)", color=AMBER)
    ax_lr.tick_params(axis="y", colors=AMBER)
    ax_lr.spines["top"].set_visible(False)
    ax_lr.set_ylim(0, LR_PEAK * 1.15)

    lines = ax.get_lines()[:1] + ax_lr.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", frameon=False,
              fontsize=9.5)
    ax.set_title("Pretraining loss descends as the LR warms up then cosine-decays",
                 fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "pt_training_loss.png")


# =====================================================================================
# Figure 2 -- the warmup + cosine learning-rate schedule (the proven endpoints)
# =====================================================================================
def fig_lr_schedule() -> None:
    """The standalone schedule: a linear ramp to exactly the peak at the end of warmup, then a
    smooth half-cosine to exactly the floor at the end. The two phases shaded and labelled."""
    steps, lrs = lr_schedule_trace()
    total = steps[-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _style_axis(ax)
    ax.axvspan(0, WARMUP_STEPS, color=BLUE, alpha=0.10, zorder=0)
    ax.axvspan(WARMUP_STEPS, total, color=PURPLE, alpha=0.07, zorder=0)
    ax.plot(steps[: WARMUP_STEPS + 1], lrs[: WARMUP_STEPS + 1], color=BLUE,
            linewidth=2.6, zorder=4, label="linear warmup")
    ax.plot(steps[WARMUP_STEPS:], lrs[WARMUP_STEPS:], color=PURPLE,
            linewidth=2.6, zorder=4, label="cosine decay")

    ax.scatter([WARMUP_STEPS, total], [LR_PEAK, LR_FLOOR], color=[GREEN, RED], s=60, zorder=5)
    ax.annotate(f"peak = {LR_PEAK:.0e}\nexactly at end of warmup", (WARMUP_STEPS, LR_PEAK),
                textcoords="offset points", xytext=(14, -2), color=GREEN,
                fontsize=9.5, fontweight="bold", va="top")
    ax.annotate(f"floor = {LR_FLOOR:.0e}\nexactly at the end", (total, LR_FLOOR),
                textcoords="offset points", xytext=(-10, 30), color=RED,
                fontsize=9.5, fontweight="bold", ha="right")

    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_xlim(0, total)
    ax.set_ylim(0, LR_PEAK * 1.18)
    ax.legend(loc="lower center", frameon=False, fontsize=10)
    ax.set_title("Warmup → cosine: ramp in gently, then anneal to a small floor",
                 fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "pt_lr_schedule.png")


# =====================================================================================
# Figure 3 -- gradient-accumulation equivalence (and the missing-/K bug)
# =====================================================================================
def fig_grad_accum() -> None:
    """K micro-batches == one big batch for the gradient (same norm to float noise); drop the /K
    and the accumulated gradient is K=8x too large -- the classic silent large-batch bug."""
    big_norm, accum_norm, buggy_norm, max_diff = grad_accum_norms()
    labels = [
        "one big batch\n(size m·K = 32)",
        "K=8 micro-batches\n(correct: ÷K)",
        "K=8 micro-batches\n(BUG: dropped ÷K)",
    ]
    norms = [big_norm, accum_norm, buggy_norm]
    colours = [SLATE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _style_axis(ax)
    bars = ax.bar(labels, norms, color=colours, edgecolor="white", linewidth=1.5,
                  width=0.62, zorder=3)
    for bar, val in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.18, f"|grad| = {val:.3f}",
                ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax.axhline(big_norm, color=SLATE, linewidth=1.1, linestyle=":", zorder=2)
    ax.annotate(
        f"correct path matches the big batch\nto float noise: max|diff| = {max_diff:.1e}",
        xy=(1, accum_norm), xytext=(0.55, big_norm + 4.2), textcoords="data",
        fontsize=9, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN),
    )
    ax.annotate(
        f"buggy path is {buggy_norm / big_norm:.0f}× too large\n(summed K means, not averaged)",
        xy=(1.66, buggy_norm), xytext=(0.30, buggy_norm * 0.80), textcoords="data",
        fontsize=9, color=RED, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=RED),
    )
    ax.set_ylabel("gradient L2 norm")
    ax.set_ylim(0, buggy_norm * 1.18)
    ax.set_title("Gradient accumulation = a big batch — if you keep the ÷K",
                 fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "pt_grad_accum.png")


# =====================================================================================
# Figure 4 -- gradient-norm stability: clipping is the seatbelt
# =====================================================================================
def fig_grad_norm_stability() -> None:
    """The per-step total gradient norm (measured before clipping) vs the clip threshold. Spikes
    above the line are exactly what clipping caps so one bad batch can't blow up the run."""
    steps, _, _, grad_norms = training_loop_trace()
    arr = np.array(grad_norms)
    steps_arr = np.array(steps)
    clipped = arr > GRAD_CLIP_NORM
    n_clipped = int(clipped.sum())

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _style_axis(ax)
    # Shade the excess (everything above the clip line) -- this is the step size clipping removes.
    ax.fill_between(steps_arr, GRAD_CLIP_NORM, arr, where=arr > GRAD_CLIP_NORM,
                    color=RED, alpha=0.16, zorder=2, label="excess removed by clipping")
    # The effective (post-clip) norm: min(raw, threshold) -- what the optimizer actually steps on.
    ax.plot(steps_arr, np.minimum(arr, GRAD_CLIP_NORM), color=GREEN, linewidth=2.4, zorder=4,
            label=f"effective norm after clip = {GRAD_CLIP_NORM:g}")
    ax.plot(steps_arr, arr, color=SLATE, linewidth=2.0, zorder=5,
            label="raw total grad norm (pre-clip)")
    ax.axhline(GRAD_CLIP_NORM, color=AMBER, linewidth=2.0, linestyle="--", zorder=3)
    ax.text(steps[-1], GRAD_CLIP_NORM - 0.06,
            f"clip threshold = {GRAD_CLIP_NORM:g}  —  all {n_clipped} steps clipped here",
            ha="right", va="top", color=AMBER, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient L2 norm")
    ax.set_ylim(0, arr.max() * 1.14)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Gradient clipping caps the step — the guard against loss spikes",
                 fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "pt_grad_norm_stability.png")


# =====================================================================================
# Figure 5 -- the 6ND production table as a compute/tokens-per-param landscape
# =====================================================================================
def fig_compute_mix() -> None:
    """Real models placed by tokens/param (the strategic axis) with their 6ND compute as bar
    height -- GPT-3 undertrained at ~1.7, Chinchilla-optimal ~20, Llama-3-8B wildly over-trained."""
    table = production_table()
    labels = [r[0] for r in table]
    tok_per_param = [r[3] for r in table]
    compute = [r[4] for r in table]
    # colour by regime: undertrained (GPT-3), near-optimal (Chinchilla), over-trained (the rest)
    colours = [RED, GREEN, BLUE, AMBER, PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # Left: tokens/param on a log scale -- the regime axis
    _style_axis(ax1)
    bars = ax1.bar(labels, tok_per_param, color=colours, edgecolor="white",
                   linewidth=1.5, width=0.62, zorder=3)
    ax1.set_yscale("log")
    for bar, val in zip(bars, tok_per_param):
        # one decimal below 10 (so GPT-3 reads ~1.7, not a rounded 2), integer above
        txt = f"{val:.1f}" if val < 10 else f"{val:,.0f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.18, txt,
                 ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
    ax1.axhline(20, color=GREEN, linewidth=1.4, linestyle="--", zorder=2)
    ax1.text(len(labels) - 0.5, 23, "Chinchilla-optimal ≈ 20", ha="right", va="bottom",
             color=GREEN, fontsize=9, fontweight="bold")
    ax1.set_ylabel("training tokens per parameter (D/N, log scale)")
    ax1.tick_params(axis="x", labelrotation=20)
    ax1.set_title("Where each model sits: under- vs over-trained", fontsize=11.5,
                  fontweight="bold", color=INK)

    # Right: the 6ND compute budget (the same formula budgets them all)
    _style_axis(ax2)
    bars2 = ax2.bar(labels, compute, color=colours, edgecolor="white",
                    linewidth=1.5, width=0.62, zorder=3)
    ax2.set_yscale("log")
    for bar, val in zip(bars2, compute):
        # 2 sig-figs so GPT-3 reads 3.15e+23 (the full 6ND value), matching the prose and table
        # exactly -- 1 sig-fig would round it to 3.2e+23 and contradict the page's 3.15e23.
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.25, f"{val:.2e}",
                 ha="center", va="bottom", color=INK, fontsize=8.2, fontweight="bold")
    ax2.set_ylabel("training compute  C = 6·N·D  (FLOPs, log scale)")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylim(top=compute[-1] * 4)
    ax2.set_title("...the same 6ND formula budgets every run", fontsize=11.5,
                  fontweight="bold", color=INK)
    fig.suptitle("Data × parameters: one 6ND budget, very different allocations",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    _save(fig, "pt_compute_mix.png")


# =====================================================================================
# Figure 6 -- loss vs compute: the scaling (power-law) relationship
# =====================================================================================
def fig_scaling_compute() -> None:
    """The empirical regularity that motivates spending 6ND: held-out loss falls as a power law in
    training compute. We anchor the curve at the real models' 6ND budgets (x-positions from
    production_table) and draw the illustrative L(C) = E + a·C^(-b) trend through them."""
    table = production_table()
    compute = np.array([r[4] for r in table])  # real 6ND budgets, the x-axis anchors

    # Illustrative compute-loss law (Kaplan/Hoffmann form): L(C) = E + a * C^(-b). Constants chosen
    # so the curve passes through realistic held-out losses (~1.7-2.0 nats) at these 6ND budgets;
    # the SHAPE (a straight line on log-log, flattening toward the irreducible floor E) is the
    # teaching point. Labelled illustrative since exact L(C) depends on the model/data.
    irreducible = 1.69
    a, b = 12.5, 0.057
    c_grid = np.logspace(22.6, 25.0, 200)
    loss_grid = irreducible + a * c_grid**(-b)
    anchor_loss = irreducible + a * compute**(-b)

    # Place each model's label with a manual (dx, dy) point offset so the clustered points
    # (Chinchilla and Llama-3-8B nearly coincide in 6ND) don't collide.
    label_off = {"GPT-3": (-2, 16), "Chinchilla": (-44, -18), "Llama-2-70B": (8, 14),
                 "Llama-3-8B": (10, -20), "Llama-3-70B": (8, 12)}

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    _style_axis(ax)
    ax.plot(c_grid, loss_grid, color=BLUE, linewidth=2.4, zorder=3,
            label=r"$L(C) = E + a\,C^{-b}$  (illustrative power law)")
    ax.axhline(irreducible, color=SLATE, linewidth=1.3, linestyle=":", zorder=2)
    ax.text(c_grid[-1], irreducible + 0.012, "irreducible loss E (entropy of language)",
            ha="right", va="bottom", color=SLATE, fontsize=9)
    ax.set_xscale("log")
    colours = [RED, GREEN, BLUE, AMBER, PURPLE]
    ax.scatter(compute, anchor_loss, color=colours, s=70, zorder=5, edgecolor="white")
    for (label, *_), c, lo, col in zip(table, compute, anchor_loss, colours):
        ax.annotate(label, (c, lo), textcoords="offset points", xytext=label_off[label],
                    color=col, fontsize=8.8, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, linewidth=0.7, alpha=0.6))
    ax.set_xlabel("training compute  C = 6·N·D  (FLOPs, log scale)")
    ax.set_ylabel("held-out loss in nats (illustrative)")
    ax.set_ylim(irreducible - 0.05, anchor_loss.max() + 0.12)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("More compute → lower loss, as a power law (the reason to scale)",
                 fontsize=12.5, fontweight="bold", color=INK)
    _save(fig, "pt_scaling_compute.png")


# =====================================================================================
# Figure 7 -- the optimizer-state memory tax (~16 bytes/param) and the ZeRO/FSDP shrink
# =====================================================================================
def fig_memory_tax() -> None:
    """Why even a 7B model must shard: Adam mixed-precision state is ~16 bytes/param (left, the
    stacked breakdown), and ZeRO/FSDP cuts the per-GPU footprint as the cluster grows (right)."""
    n_params = 7e9  # the 7B model the page uses for the optimizer-state-tax note
    breakdown, total_gib = adam_memory_tax(n_params)
    gib = 1024**3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # Left: stacked per-param byte breakdown -> total GiB for 7B (one colour per segment; the
    # list MUST be at least as long as the breakdown so no segment is silently dropped).
    _style_axis(ax1)
    seg_colours = [BLUE, GREEN, NAVY, PURPLE, AMBER]
    assert len(seg_colours) >= len(breakdown), "need a colour per memory segment"
    bottom = 0.0
    for (name, byts), col in zip(breakdown, seg_colours):
        seg_gib = n_params * byts / gib
        ax1.bar(["7B model"], [seg_gib], bottom=bottom, color=col, edgecolor="white",
                linewidth=1.5, width=0.5, zorder=3, label=f"{name} ({byts} B)")
        ax1.text(0, bottom + seg_gib / 2, f"{seg_gib:.0f}", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
        bottom += seg_gib
    ax1.axhline(80, color=RED, linewidth=2.0, linestyle="--", zorder=4)
    ax1.text(0.42, 82, "one H100 = 80 GiB", ha="right", va="bottom", color=RED,
             fontsize=9.5, fontweight="bold")
    ax1.annotate(f"total ≈ {total_gib:.0f} GiB  (~16 B/param)", xy=(0.25, total_gib),
                 xytext=(0.30, total_gib + 18), textcoords="data", ha="left",
                 color=INK, fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=INK))
    ax1.set_ylabel("memory for 7B optimizer + master state (GiB)")
    ax1.set_xlim(-0.6, 0.9)
    ax1.set_ylim(0, total_gib * 1.34)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax1.set_title("Adam's optimizer-state tax: ~16 bytes/param", fontsize=11.5,
                  fontweight="bold", color=INK)

    # Right: ZeRO/FSDP -- per-GPU state shrinks as 1/num_GPUs
    _style_axis(ax2)
    n_gpus = np.array([1, 2, 4, 8, 16, 32, 64])
    per_gpu = total_gib / n_gpus  # Stage-3 / FSDP: state sharded across the DP group
    ax2.plot(n_gpus, per_gpu, color=GREEN, linewidth=2.4, marker="o", markersize=6, zorder=4,
             label="per-GPU state (ZeRO-3 / FSDP)")
    ax2.axhline(total_gib, color=SLATE, linewidth=1.4, linestyle=":", zorder=2)
    ax2.text(n_gpus[-1], total_gib * 0.92, "plain DP: full copy on every GPU", ha="right",
             va="top", color=SLATE, fontsize=9)
    ax2.axhline(80, color=RED, linewidth=2.0, linestyle="--", zorder=3)
    ax2.text(n_gpus[1], 84, "fits one H100 below here", ha="left", va="bottom", color=RED,
             fontsize=9.5, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(n_gpus)
    ax2.set_xticklabels([str(g) for g in n_gpus])
    ax2.set_xlabel("number of data-parallel GPUs (sharding the state)")
    ax2.set_ylabel("per-GPU memory (GiB)")
    ax2.set_ylim(0, total_gib * 1.1)
    ax2.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax2.set_title("ZeRO/FSDP: per-GPU memory shrinks with the cluster", fontsize=11.5,
                  fontweight="bold", color=INK)
    fig.suptitle("The memory wall — and how sharding the optimizer state climbs over it",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    _save(fig, "pt_memory_tax.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_training_loss()
    fig_lr_schedule()
    fig_grad_accum()
    fig_grad_norm_stability()
    fig_compute_mix()
    fig_scaling_compute()
    fig_memory_tax()
    print("done.")


if __name__ == "__main__":
    main()

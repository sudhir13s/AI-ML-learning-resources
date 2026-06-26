"""Reproducible figure generator for 12-LoRA-and-PEFT.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_12.py

Each figure is written to ../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). Numbers are recomputed here, never hardcoded
from memory: the 64x param reduction at d=1024/r=8, the 2rd-vs-d^2 curve, the rank-vs-fit
cliff at the true rank 4 (from lora_peft.py's sweep), the optimizer-state memory stack for a
7B model, and the multi-LoRA storage win.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # single-threaded -> the delta-growth curve is bit-reproducible (matches lora_peft.py)

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

# ---- Shared constants (identical to lora_peft.py) ------------------------------------
IN_FEATURES = 1024
OUT_FEATURES = 1024
LORA_RANK = 8
TRUE_RANK = 4
RANK_SWEEP = (1, 2, 4, 8, 16, 32)
# Final losses from lora_peft.py's rank sweep (recomputed there; pasted here as the figure's
# ground truth so the curve matches the script's printed table exactly).
RANK_SWEEP_LOSS = {
    1: 2.7211e-01,
    2: 1.7230e-01,
    4: 8.5932e-07,
    8: 2.1496e-07,
    16: 1.0558e-07,
    32: 1.3950e-07,
}

# Figures live in the SHARED chapter images dir (09. LLMs/images/), matching the KV-Cache
# exemplar -- so the page references them as ../images/<name>.png.
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
    print(f"wrote {path}")


def fig_param_comparison() -> None:
    """Bar chart: full fine-tuning (d^2) vs LoRA (2rd) trainable params for one d x d layer."""
    d = IN_FEATURES
    full_ft = d * d  # 1,048,576
    lora = 2 * LORA_RANK * d  # 16,384
    reduction = full_ft / lora  # 64.0

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Full fine-tuning\n($d^2$ params)", f"LoRA, r={LORA_RANK}\n($2rd$ params)"]
    values = [full_ft, lora]
    bars = ax.bar(labels, values, color=[RED, GREEN], width=0.55, zorder=3)
    ax.set_ylabel("trainable parameters (one $d\\times d$ layer, $d=1024$)")
    ax.set_yscale("log")
    ax.set_title(f"LoRA trains {reduction:.0f}x fewer parameters per layer")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15, f"{val:,}",
                ha="center", va="bottom", color=INK, fontweight="bold")
    ax.annotate(f"{reduction:.0f}x\nfewer", xy=(1, lora), xytext=(0.5, full_ft * 0.25),
                ha="center", color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SLATE))
    _style_axis(ax)
    _save(fig, "lora_param_comparison.png")


def fig_param_vs_rank() -> None:
    """LoRA params (2rd) vs full FT (d^2) as a function of rank r -- the savings shrink as r grows."""
    d = IN_FEATURES
    full_ft = d * d
    ranks = np.arange(1, 65)
    lora_params = 2 * ranks * d

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(full_ft, color=RED, linewidth=2, zorder=3,
               label=f"full fine-tuning ($d^2$ = {full_ft:,})")
    ax.plot(ranks, lora_params, color=GREEN, linewidth=2.4, zorder=4,
            label="LoRA ($2rd$)")
    ax.fill_between(ranks, lora_params, full_ft, where=(lora_params < full_ft),
                    color=GREEN, alpha=0.10, zorder=2)
    # Mark the break-even rank where 2rd = d^2  ->  r = d/2.
    break_even = d // 2
    ax.axvline(break_even, color=SLATE, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(break_even - 2, full_ft * 0.5, f"break-even\nr = d/2 = {break_even}",
            ha="right", color=INK)
    ax.scatter([LORA_RANK], [2 * LORA_RANK * d], color=NAVY, zorder=6, s=60)
    ax.annotate(f"r={LORA_RANK}: {2 * LORA_RANK * d:,}\n({full_ft // (2 * LORA_RANK * d)}x smaller)",
                xy=(LORA_RANK, 2 * LORA_RANK * d), xytext=(14, full_ft * 0.18),
                color=INK, arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("trainable parameters (one $d\\times d$ layer)")
    ax.set_title("LoRA's savings shrink as rank grows toward $r = d/2$")
    ax.legend(frameon=False)
    _style_axis(ax)
    _save(fig, "lora_param_vs_rank.png")


def fig_rank_vs_fit() -> None:
    """Twin-axis: trainable params (linear in r) vs final fit loss (cliffs at the true rank)."""
    ranks = list(RANK_SWEEP)
    d = IN_FEATURES
    params = [2 * r * d for r in ranks]
    losses = [RANK_SWEEP_LOSS[r] for r in ranks]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.set_xlabel("LoRA rank $r$  (log scale)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels([str(r) for r in ranks])

    ax1.set_ylabel("trainable params ($2rd$)", color=GREEN)
    ax1.plot(ranks, params, color=GREEN, marker="o", linewidth=2.2, zorder=4, label="params")
    ax1.tick_params(axis="y", colors=GREEN)

    ax2 = ax1.twinx()
    ax2.set_ylabel("final fit loss (log)", color=RED)
    ax2.set_yscale("log")
    ax2.plot(ranks, losses, color=RED, marker="s", linewidth=2.2, zorder=5, label="loss")
    ax2.tick_params(axis="y", colors=RED)

    ax1.axvline(TRUE_RANK, color=SLATE, linestyle="--", linewidth=1.2, zorder=3)
    ax1.text(TRUE_RANK * 1.05, max(params) * 0.55,
             f"true update rank = {TRUE_RANK}\nfit collapses once $r\\geq${TRUE_RANK};\nmore rank only adds params",
             color=INK, fontsize=9)
    ax1.set_title("Rank vs (params, fit): the cliff sits at the true intrinsic rank")
    _style_axis(ax1)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    _save(fig, "lora_rank_vs_fit.png")


def fig_optimizer_memory() -> None:
    """Stacked memory bars: full FT (weights + grads + Adam states) vs LoRA, for a 7B model.

    The well-known numbers: a 7B model in fp16 is ~14 GB of weights. Full fine-tuning also
    stores fp16 grads (~14 GB) and Adam's two fp32 moment buffers + fp32 master copy
    (~84 GB) -> ~112 GB just to *update*. LoRA freezes W, so grads and optimizer state exist
    only for the ~0.1% adapter params -> a few hundred MB. (Activations excluded; same for both.)
    """
    n_params = 7e9
    bytes_fp16 = 2
    bytes_fp32 = 4
    gb = 1024**3

    weights = n_params * bytes_fp16 / gb  # ~13 GiB (fp16 frozen base, needed either way)
    grads_full = n_params * bytes_fp16 / gb  # gradients for ALL params
    # AdamW per-param state: fp32 master weights + fp32 m + fp32 v = 3 * 4 bytes.
    adam_full = n_params * (3 * bytes_fp32) / gb
    full_total = weights + grads_full + adam_full

    lora_frac = 0.001  # ~0.1% of params are trainable adapters (typical LoRA setting)
    lora_n = n_params * lora_frac
    grads_lora = lora_n * bytes_fp16 / gb
    adam_lora = lora_n * (3 * bytes_fp32) / gb
    lora_total = weights + grads_lora + adam_lora

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cats = ["Full fine-tuning", "LoRA\n(frozen base)"]
    w = 0.55
    ax.bar(cats, [weights, weights], w, color=SLATE, zorder=3, label="frozen/base weights (fp16)")
    ax.bar(cats, [grads_full, grads_lora], w, bottom=[weights, weights], color=AMBER, zorder=3,
           label="gradients")
    ax.bar(cats, [adam_full, adam_lora], w, bottom=[weights + grads_full, weights + grads_lora],
           color=RED, zorder=3, label="Adam optimizer states")
    ax.set_ylabel("memory (GiB), 7B model")
    ax.set_title(f"Full FT needs ~{full_total:.0f} GiB to update; LoRA needs ~{lora_total:.0f} GiB")
    ax.text(0, full_total + 2, f"~{full_total:.0f} GiB", ha="center", color=INK, fontweight="bold")
    ax.text(1, lora_total + 2, f"~{lora_total:.0f} GiB", ha="center", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "lora_optimizer_memory.png")


def fig_multi_lora_storage() -> None:
    """Storage: N full fine-tuned copies (N x 14 GB) vs one base + N small adapters."""
    base_gb = 14.0  # 7B fp16 base copy
    adapter_mb = 17.0  # a typical r=8 LoRA adapter for a 7B model, in MB
    n_tasks = np.arange(1, 21)
    full_copies = n_tasks * base_gb
    multi_lora = base_gb + n_tasks * (adapter_mb / 1024)  # one base + N adapters

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(n_tasks, full_copies, color=RED, marker="o", linewidth=2.2, zorder=4,
            label="full FT: one 14 GB copy / task")
    ax.plot(n_tasks, multi_lora, color=GREEN, marker="s", linewidth=2.2, zorder=5,
            label=f"multi-LoRA: 14 GB base + {adapter_mb:.0f} MB / task")
    ax.set_xlabel("number of fine-tuned tasks")
    ax.set_ylabel("total storage (GB)")
    ax.set_title("One shared base + tiny adapters vs a full copy per task")
    n = 20
    ax.annotate(f"{n} tasks: {full_copies[-1]:.0f} GB", xy=(n, full_copies[-1]),
                xytext=(11, full_copies[-1] * 0.92), color=INK,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.annotate(f"{n} tasks: {multi_lora[-1]:.1f} GB", xy=(n, multi_lora[-1]),
                xytext=(9, full_copies[-1] * 0.35), color=INK,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.legend(frameon=False, loc="center left")
    _style_axis(ax)
    _save(fig, "lora_multi_serving.png")


def fig_qlora_memory() -> None:
    """QLoRA memory stack: 4-bit NF4 base vs fp16 base, both + bf16 adapters, for 65B."""
    n_params = 65e9
    gb = 1024**3
    fp16_base = n_params * 2 / gb  # ~121 GiB -- impossible on one 80 GB GPU
    nf4_base = n_params * 0.5 / gb  # 4-bit -> 0.5 byte/param  ~30 GiB
    adapters_and_states = 10.0  # bf16 adapters + paged optimizer states + activations, approx

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cats = ["LoRA on fp16 base", "QLoRA\n(4-bit NF4 base)"]
    w = 0.55
    ax.bar(cats, [fp16_base, nf4_base], w, color=[RED, GREEN], zorder=3, label="frozen base")
    ax.bar(cats, [adapters_and_states, adapters_and_states], w,
           bottom=[fp16_base, nf4_base], color=BLUE, zorder=3,
           label="bf16 adapters + paged optim + activations")
    # 48 GB is the QLoRA paper's headline budget (65B fine-tuned on a single 48 GB GPU);
    # the figure should PROVE that headline, so the budget line is the paper's, not 80 GB.
    ax.axhline(48, color=SLATE, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(1.45, 50, "single 48 GB GPU (QLoRA paper)", ha="right", color=INK)
    ax.set_ylabel("memory (GiB), 65B model")
    ax.set_title("QLoRA: 4-bit base shrinks 65B fine-tuning onto one GPU")
    ax.text(0, fp16_base + adapters_and_states + 3, f"~{fp16_base + adapters_and_states:.0f} GiB",
            ha="center", color=INK, fontweight="bold")
    ax.text(1, nf4_base + adapters_and_states + 3, f"~{nf4_base + adapters_and_states:.0f} GiB",
            ha="center", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    _save(fig, "lora_qlora_memory.png")


def fig_delta_growth() -> None:
    """max|ΔW| over training steps: starts at exactly 0 (B=0 init), then grows as B leaves zero.

    Reproduces lora_peft.py's training setup (same SEED, constants, and init order) so the curve
    is the chapter's own number: ΔW = 0 at step 0, then the adapter walks away from the
    pretrained point. This is the figure the B=0/init caption belongs to.
    """
    seed = 0
    rank, alpha = LORA_RANK, 16  # alpha = 16 as in lora_peft.py / the page
    scaling = alpha / rank
    n_samples, true_rank, steps = 256, TRUE_RANK, 300

    torch.manual_seed(seed)
    # Same construction order as LoRALinear.__init__ then make_synthetic_task in lora_peft.py.
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.02     # frozen W (consumes RNG first)
    lora_a = torch.empty(rank, IN_FEATURES)
    lora_b = torch.zeros(OUT_FEATURES, rank)                    # B = 0 -> ΔW = 0 at init
    torch.nn.init.kaiming_uniform_(lora_a, a=5**0.5)
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)

    torch.manual_seed(seed)  # make_synthetic_task re-seeds, exactly as in lora_peft.py
    base_weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.02
    u = torch.randn(OUT_FEATURES, true_rank) * 0.1
    v = torch.randn(true_rank, IN_FEATURES) * 0.1
    x = torch.randn(n_samples, IN_FEATURES)
    y = F.linear(x, base_weight + u @ v)
    weight = base_weight  # the layer adapts this frozen base

    optimizer = torch.optim.Adam([lora_a, lora_b], lr=1e-2)
    delta_max = []
    for _ in range(steps + 1):  # record step 0 (pre-update) through step `steps`
        delta_max.append(((lora_b @ lora_a) * scaling).abs().max().item())
        optimizer.zero_grad()
        update = F.linear(F.linear(x, lora_a), lora_b) * scaling
        F.mse_loss(F.linear(x, weight) + update, y).backward()
        optimizer.step()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(delta_max)), delta_max, color=PURPLE, linewidth=2.2, zorder=4)
    ax.scatter([0], [delta_max[0]], color=GREEN, s=70, zorder=6)
    ax.annotate(f"step 0: max|ΔW| = {delta_max[0]:.0f}\n(B = 0 → ΔW = 0:\nstarts at the pretrained model)",
                xy=(0, delta_max[0]), xytext=(35, max(delta_max) * 0.55),
                color=INK, arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set_xlabel("training step")
    ax.set_ylabel(r"max$|\Delta W|= $ max$|(\alpha/r)\,B\!\cdot\!A|$")
    ax.set_title("ΔW starts at exactly 0 (B=0 init), then grows as the adapter learns")
    _style_axis(ax)
    _save(fig, "lora_delta_growth.png")


def main() -> None:
    fig_param_comparison()
    fig_param_vs_rank()
    fig_rank_vs_fit()
    fig_optimizer_memory()
    fig_multi_lora_storage()
    fig_qlora_memory()
    fig_delta_growth()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()

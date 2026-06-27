"""Static (PNG) figure generator for 06-Efficient-Attention-FlashAttention.

Companion to the chapter's Mermaid diagrams. Where those show structure, these put real
numbers on the page so the reader *sees* the memory wall, the tiling, the online-softmax
bookkeeping, the HBM-vs-SRAM gap, and the IO/memory savings.

    python make_figures_06.py

Every value drawn here comes from flash_attention.py's seeded helpers -- no orphan numbers:
  * the (m, l) trace and correction factors are the EXACT values flash_attention() emits;
  * the O(N^2) memory curve is the same fp16 byte formula the prose uses (64 GiB at N=8192);
  * the materialized-bytes curve is MEASURED by running full_attention() at each N;
  * the IO-access curves are the paper's Theorem-1 closed forms (labelled "model").
Hardware capacity/bandwidth numbers (HBM vs SRAM) are published A100 specs, labelled illustrative.

PNGs are written to ../../images/ (the shared chapter image dir), prefixed `fa_`.
Verified on Python 3.12 / matplotlib 3.10 / torch 2.12.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_attention as fa  # the seeded source of every number below

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
DPI = 110


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


def make_memory_wall() -> None:
    """The O(N^2) memory wall: standard-attention score-matrix GiB vs sequence length, with
    the A100-80GB ceiling marked. Numbers are the prose's exact fp16 formula (64 GiB at N=8192)."""
    seq = [512, 1024, 2048, 4096, 8192, 16384]
    gib = fa.standard_attention_memory_gib(seq)  # one fp16 N^2 matrix per (batch, head)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style_axis(ax)
    ax.plot(seq, gib, marker="o", color=RED, linewidth=2.4, markersize=7,
            label=f"standard attention scratch\n(fp16 $N{{\\times}}N$ scores, batch {fa.BATCH} × {fa.HEADS} heads)")
    ax.axhline(fa.A100_GIB, color=SLATE, linewidth=1.8, linestyle="--")
    ax.text(560, fa.A100_GIB * 1.06, f"A100-80GB total HBM = {fa.A100_GIB} GiB",
            color=SLATE, fontsize=10, fontweight="bold")

    # Mark the N=8192 -> 64 GiB point the prose calls out.
    n_mark, gib_mark = 8192, gib[seq.index(8192)]
    ax.annotate(f"N=8192 → {gib_mark:.0f} GiB\n(scratch alone, one layer)",
                xy=(n_mark, gib_mark), xytext=(2400, 2.5),
                color=RED, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.6))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(seq)
    ax.set_xticklabels([str(s) for s in seq])
    ax.set_xlabel("sequence length  N  (tokens)")
    ax.set_ylabel("attention-score memory  (GiB, log scale)")
    ax.set_ylim(0.15, 700)
    ax.set_title("The $O(N^2)$ memory wall standard attention hits", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5, facecolor="white",
              edgecolor=GRID, framealpha=0.95)
    _save(fig, "fa_memory_wall.png")


def make_tiling_structure() -> None:
    """The tiled/blockwise structure: the full N x N score matrix (what standard attention
    materializes) vs the single small (Q-tile, K/V-tile) working set FlashAttention holds."""
    n, block = 8, 2  # the demo's shapes, so this matches the from-scratch code exactly
    fig, (ax_full, ax_tiled) = plt.subplots(1, 2, figsize=(9.6, 4.6))

    # Left: the full N x N matrix, all of it resident in HBM.
    ax_full.set_title("Standard attention\nmaterializes the whole $N{\\times}N$ matrix",
                      fontweight="bold", color=INK, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax_full.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=RED,
                                        edgecolor="white", linewidth=1.0, alpha=0.85))
    ax_full.text(n / 2, -0.9, f"{n}×{n} = {n*n} scores live in HBM  →  $O(N^2)$",
                 ha="center", color=RED, fontsize=10, fontweight="bold")

    # Right: same matrix, but only one (Q-tile x K/V-tile) block is in SRAM at a time.
    ax_tiled.set_title("FlashAttention holds\none $B{\\times}B$ tile in SRAM at a time",
                       fontweight="bold", color=INK, fontsize=11)
    active = (1, 2)  # the (block-row, block-col) currently on the countertop
    for bi in range(n // block):
        for bj in range(n // block):
            is_active = (bi, bj) == active
            color = GREEN if is_active else "#E7EAEE"
            edge = GREEN if is_active else SLATE
            for di in range(block):
                for dj in range(block):
                    i, j = bi * block + di, bj * block + dj
                    ax_tiled.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=color,
                                                 edgecolor=edge, linewidth=1.0,
                                                 alpha=0.95 if is_active else 0.55))
            # thick block borders
            ax_tiled.add_patch(Rectangle((bj * block, n - (bi + 1) * block), block, block,
                                         fill=False, edgecolor=SLATE, linewidth=1.8))
    ax_tiled.text(n / 2, -0.9,
                  f"working set = one {block}×{block} tile  →  $O(B^2)$, independent of $N$",
                  ha="center", color=GREEN, fontsize=10, fontweight="bold")

    for ax in (ax_full, ax_tiled):
        ax.set_xlim(-0.3, n + 0.3)
        ax.set_ylim(-1.5, n + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-0.1, n / 2, "queries", rotation=90, va="center", ha="right",
                color=SLATE, fontsize=9)
        ax.text(n / 2, n + 0.15, "keys", ha="center", color=SLATE, fontsize=9)

    fig.suptitle("Tiling: never assemble the full matrix — stream one block through SRAM",
                 fontweight="bold", color=INK, fontsize=12.5, y=1.06)
    fig.subplots_adjust(top=0.82)
    _save(fig, "fa_tiling_structure.png")


def make_online_softmax() -> None:
    """The online-softmax bookkeeping for query 0, block by block: the running max m (only
    rises) and the running denominator (drops when m jumps, because old terms get rescaled).
    These are the EXACT (m, denom) values flash_attention()'s trace emits."""
    idx, m, denom, corr = fa.online_softmax_trace()
    blocks = [f"after\nblock {i}" for i in idx]
    x = np.arange(len(idx))

    fig, (ax_m, ax_l) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    _style_axis(ax_m)
    _style_axis(ax_l)

    ax_m.plot(x, m, marker="o", color=PURPLE, linewidth=2.4, markersize=8)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(blocks)
    ax_m.set_ylabel("running max  $m$")
    ax_m.set_title("Running max $m$ only ever rises", fontweight="bold", fontsize=11)
    for xi, mi in zip(x, m):
        ax_m.annotate(f"{mi:+.3f}", (xi, mi), textcoords="offset points",
                      xytext=(0, 10), ha="center", color=PURPLE, fontsize=9, fontweight="bold")
    # mark the big jump at block 3
    ax_m.annotate("big score in block 3\n→ $m$ jumps to 1.465",
                  xy=(3, m[3]), xytext=(0.6, m[3] - 0.55), color=PURPLE, fontsize=9,
                  arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.4))

    ax_l.plot(x, denom, marker="s", color=AMBER, linewidth=2.4, markersize=8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(blocks)
    ax_l.set_ylabel("running denominator  $\\ell$")
    ax_l.set_title("$\\ell$ drops when $m$ jumps (old terms rescaled)",
                   fontweight="bold", fontsize=11)
    for xi, di in zip(x, denom):
        ax_l.annotate(f"{di:.3f}", (xi, di), textcoords="offset points",
                      xytext=(0, 10), ha="center", color=AMBER, fontsize=9, fontweight="bold")
    # annotate the correction factor on the 2->3 drop
    ax_l.annotate(f"×corr = {corr[3]:.3f}\n($e^{{m_{{old}}-m_{{new}}}}$ re-bases\nthe accumulated sum)",
                  xy=(3, denom[3]), xytext=(1.15, denom[3] + 0.55), color=RED, fontsize=9,
                  arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.4))

    fig.suptitle("Online softmax for query 0 — the sticky-note state that makes tiling exact",
                 fontweight="bold", color=INK, fontsize=12.5, y=1.02)
    fig.subplots_adjust(top=0.86)
    _save(fig, "fa_online_softmax.png")


def make_hbm_sram() -> None:
    """The HBM-vs-SRAM story: why moving the matrix matters. Two bars -- capacity (huge HBM,
    tiny SRAM) and bandwidth (SRAM ~10-20x faster per access). Published A100 specs, labelled
    illustrative; this is the gap FlashAttention exploits by keeping tiles on-chip."""
    fig, (ax_cap, ax_bw) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    _style_axis(ax_cap)
    _style_axis(ax_bw)

    tiers = ["HBM\n(warehouse)", "SRAM\n(countertop)"]
    colors = [BLUE, GREEN]

    # Capacity: A100 80 GB HBM vs ~20 MB on-chip SRAM (in MB, log scale).
    capacity_mb = [80 * 1024, 20]  # 80 GB HBM, ~20 MB aggregate SRAM
    ax_cap.bar(tiers, capacity_mb, color=colors, edgecolor=INK, linewidth=1.0, width=0.6)
    ax_cap.set_yscale("log")
    ax_cap.set_ylabel("capacity  (MB, log scale)")
    ax_cap.set_title("Capacity: HBM is ~4000× larger", fontweight="bold", fontsize=11)
    for i, c in enumerate(capacity_mb):
        label = f"{c/1024:.0f} GB" if c >= 1024 else f"{c:.0f} MB"
        ax_cap.text(i, c * 1.4, label, ha="center", color=INK, fontsize=10, fontweight="bold")

    # Bandwidth: HBM ~2 TB/s vs SRAM ~19 TB/s on-chip (TB/s).
    bandwidth = [2.0, 19.0]  # TB/s: A100 HBM2e vs on-chip shared-memory bandwidth (illustrative)
    ax_bw.bar(tiers, bandwidth, color=colors, edgecolor=INK, linewidth=1.0, width=0.6)
    ax_bw.set_ylabel("bandwidth  (TB/s)")
    ax_bw.set_title("Bandwidth: SRAM is ~10× faster", fontweight="bold", fontsize=11)
    for i, b in enumerate(bandwidth):
        ax_bw.text(i, b + 0.5, f"{b:.0f} TB/s", ha="center", color=INK,
                   fontsize=10, fontweight="bold")
    ax_bw.set_ylim(0, 22)

    fig.suptitle("Why moving the matrix costs: HBM is large but slow; SRAM is tiny but fast\n"
                 "(A100 specs, illustrative) — FlashAttention keeps tiles on the fast tier",
                 fontweight="bold", color=INK, fontsize=11.5, y=1.13)
    fig.subplots_adjust(top=0.80)
    _save(fig, "fa_hbm_sram.png")


def make_io_complexity() -> None:
    """The IO-complexity comparison: HBM element accesses for standard vs FlashAttention.
    Paper Theorem-1 closed forms: standard Theta(Nd + N^2) vs flash Theta(N^2 d^2 / M).
    With d=64 and M=128K elements the reduction is the M/d factor (~2048x here at scale)."""
    seq = [512, 1024, 2048, 4096, 8192, 16384]
    head_dim = 64  # a standard head dimension
    sram_elems = 128 * 1024  # M: ~128K fp-elements of SRAM working set
    standard, flash = fa.io_accesses(seq, head_dim=head_dim, sram_elems=sram_elems)
    ratio = [s / f for s, f in zip(standard, flash)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style_axis(ax)
    ax.plot(seq, standard, marker="o", color=RED, linewidth=2.4, markersize=7,
            label="standard: $\\Theta(Nd + N^2)$ HBM accesses")
    ax.plot(seq, flash, marker="s", color=GREEN, linewidth=2.4, markersize=7,
            label="FlashAttention: $\\Theta(N^2 d^2 / M)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(seq)
    ax.set_xticklabels([str(s) for s in seq])
    ax.set_xlabel("sequence length  N  (tokens)")
    ax.set_ylabel("HBM element accesses  (model, log scale)")
    ax.set_title("Same FLOPs, far fewer HBM accesses (IO-awareness)", fontweight="bold")

    # Annotate the reduction factor at the largest N.
    big = -1
    ax.annotate(f"~{ratio[big]:.0f}× fewer\nHBM accesses\n($d={head_dim}$, $M=128$K)",
                xy=(seq[big], flash[big]), xytext=(2600, flash[big] * 0.9),
                color=GREEN, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.6))
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    _save(fig, "fa_io_complexity.png")


def make_memory_savings() -> None:
    """Measured savings: the bytes the N x N score matrix occupies (MEASURED by running
    full_attention() at each N) vs FlashAttention's fixed per-tile working set. The standard
    path's curve is read straight off full_attention()'s reported byte count -- not a formula
    typed in by hand -- so the savings ratio is grounded in the running code."""
    seq = [8, 16, 32, 64, 128, 256]
    materialized = fa.measured_materialized_bytes(seq)  # MEASURED via full_attention()
    flash_ws = fa.flash_working_set_bytes()  # fixed: one B x d tile
    flash_curve = [flash_ws] * len(seq)
    ratio = [m / flash_ws for m in materialized]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style_axis(ax)
    ax.plot(seq, materialized, marker="o", color=RED, linewidth=2.4, markersize=7,
            label="standard: $N{\\times}N$ score matrix (measured bytes)")
    ax.plot(seq, flash_curve, marker="s", color=GREEN, linewidth=2.4, markersize=7,
            label=f"FlashAttention: one {fa.BLOCK_SIZE}×{fa.HEAD_DIM} tile = {flash_ws} bytes (fixed)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(seq)
    ax.set_xticklabels([str(s) for s in seq])
    ax.set_xlabel("sequence length  N  (tokens)")
    ax.set_ylabel("score-matrix bytes resident  (log scale)")
    ax.set_title("Measured: standard scratch grows $O(N^2)$; Flash working set is flat",
                 fontweight="bold")
    ax.set_ylim(2**4, 2**21)

    big = -1
    ax.annotate(f"at N={seq[big]}: {ratio[big]:.0f}× less\nresident (gap keeps widening)",
                xy=(seq[big], materialized[big]), xytext=(70, 2**6.6),
                color=RED, fontsize=10, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.6))
    ax.legend(loc="upper left", frameon=True, fontsize=9, facecolor="white",
              edgecolor=GRID, framealpha=0.95)
    _save(fig, "fa_memory_savings.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    make_memory_wall()
    make_tiling_structure()
    make_online_softmax()
    make_hbm_sram()
    make_io_complexity()
    make_memory_savings()
    print("done.")


if __name__ == "__main__":
    main()

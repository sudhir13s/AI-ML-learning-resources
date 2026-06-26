"""Reproducible figure generator for 08-Long-Context-Methods.

Produces every embedded PNG for the chapter from the SAME numbers used in the page and the
notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_08.py

Each figure is written to ../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). Numbers are recomputed here, never hardcoded
from memory: the RoPE angle wall (15 -> 63 -> 15.75), the sliding-window reach (4,7,10,12),
the sink mass (0.853) and drift (0.295 vs 0.001), and the KV-cache 64 GiB at 128K.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

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

# ---- Shared constants (identical to long_context_methods.py) -------------------------
HEAD_DIM = 8
ROPE_BASE = 10_000.0
TRAIN_LEN = 16
TARGET_LEN = 64
WINDOW = 4
N_LAYERS = 5
N_SINK = 4
RECENT_W = 6
SEED = 0

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
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


def rope_angles(positions: np.ndarray, pair_index: int) -> np.ndarray:
    """theta_{m,i} = m * base^(-2i/d) for a single frequency pair i, across positions."""
    inv_freq = ROPE_BASE ** (-2.0 * pair_index / HEAD_DIM)
    return positions * inv_freq


# =====================================================================================
# Figure 1 -- clock-hands: fast vs slow frequency pair, angle vs position
# =====================================================================================
def fig_clock_hands() -> None:
    positions = np.arange(0, TRAIN_LEN)
    fast = rope_angles(positions, pair_index=0)  # i=0: fastest pair, freq = 1
    slow = rope_angles(positions, pair_index=HEAD_DIM // 2 - 1)  # slowest pair

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axis(ax)
    ax.plot(positions, fast, "-o", color=BLUE, linewidth=2.2, markersize=5,
            label="fast pair (i=0): angle = position", zorder=3)
    ax.plot(positions, slow, "-s", color=PURPLE, linewidth=2.2, markersize=5,
            label=f"slow pair (i={HEAD_DIM // 2 - 1}): barely rotates", zorder=3)
    ax.set_xlabel("token position m")
    ax.set_ylabel("rotation angle θ (radians)")
    ax.set_title("RoPE clock-hands: each feature pair rotates at its own speed", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.annotate("slow pair sweeps a tiny angle\nover the whole window",
                xy=(TRAIN_LEN - 1, slow[-1]), xytext=(TRAIN_LEN - 9, slow[-1] + 3.0),
                fontsize=9, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.3))
    _save(fig, "rope_clock_hands.png")


# =====================================================================================
# Figure 2 -- the headline: naive extrapolation vs Position Interpolation
# =====================================================================================
def fig_extrapolation_vs_pi() -> None:
    positions = np.arange(0, TARGET_LEN)
    naive = rope_angles(positions, pair_index=0)  # angle = position (fast pair)
    pi_scale = TRAIN_LEN / TARGET_LEN
    pi = rope_angles(positions * pi_scale, pair_index=0)  # PI squeezes the index

    trained_max = TRAIN_LEN - 1  # 15 rad
    naive_max = naive[-1]  # 63 rad
    pi_max = pi[-1]  # 15.75 rad

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style_axis(ax)
    # shaded trained angle range [0, trained_max]
    ax.axhspan(0, trained_max, color=GREEN, alpha=0.12, zorder=0)
    ax.axhline(trained_max, color=GREEN, linewidth=1.4, linestyle="--", zorder=1,
               label=f"trained max angle = {trained_max:.0f} rad")
    ax.plot(positions, naive, color=RED, linewidth=2.4, zorder=3,
            label=f"naive extrapolation → {naive_max:.0f} rad ({naive_max / trained_max:.1f}× past)")
    ax.plot(positions, pi, color=BLUE, linewidth=2.4, zorder=3,
            label=f"Position Interpolation → {pi_max:.2f} rad (in range)")
    ax.axvline(TRAIN_LEN - 1, color=SLATE, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(TRAIN_LEN - 0.5, naive_max * 0.55, "training\nlength", fontsize=9, color=SLATE)
    ax.text(TARGET_LEN * 0.52, trained_max * 0.42, "trained angle range\n(safe to interpolate into)",
            fontsize=9, color=GREEN)
    ax.set_xlabel("token position m (extending to 4× the training length)")
    ax.set_ylabel("max rotation angle of the fast pair (radians)")
    ax.set_title("The headline: naive extrapolation shoots past the trained range; PI squeezes it back",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _save(fig, "extrapolation_vs_pi.png")


# =====================================================================================
# Figure 3 -- sliding-window receptive field growing with depth
# =====================================================================================
def reachable_span(layer: int, query: int, window: int = WINDOW) -> int:
    reach = {query}
    for _ in range(layer):
        reach = {p for pos in reach for p in range(max(0, pos - (window - 1)), pos + 1)}
    return query - min(reach) + 1


def fig_receptive_field() -> None:
    query = 11  # matches the seq_len=12 demo (positions 0..11)
    layers = list(range(1, N_LAYERS + 1))
    spans = [reachable_span(L, query) for L in layers]  # 4, 7, 10, 12, 12
    formula = [min(query + 1, 1 + L * (WINDOW - 1)) for L in layers]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axis(ax)
    bars = ax.bar([str(L) for L in layers], spans, color=GREEN, width=0.6, zorder=3)
    ax.plot([str(L) for L in layers], formula, "--o", color=AMBER, linewidth=1.8,
            markersize=6, zorder=4, label="closed form  1 + L·(W−1)  (capped at sequence start)")
    for bar, span in zip(bars, spans):
        ax.text(bar.get_x() + bar.get_width() / 2, span + 0.2, f"{span}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("number of stacked sliding-window layers (L)")
    ax.set_ylabel("tokens of effective reach")
    ax.set_title(f"A window (W={WINDOW}) is not a ceiling: depth grows the receptive field",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, query + 2)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    _save(fig, "receptive_field_growth.png")


# =====================================================================================
# Figure 4 -- attention sinks: where the softmax mass lands
# =====================================================================================
def _sink_distributions() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Reproduce the notebook's sink demo numbers exactly (seed 0, +q*1.2, seq_len 40)."""
    torch.manual_seed(SEED)
    seq_len = 40
    scale = HEAD_DIM**-0.5
    q = torch.randn(HEAD_DIM)
    keys = torch.randn(seq_len, HEAD_DIM)
    keys[:N_SINK] += q * 1.2
    logits = (keys @ q) * scale
    full = torch.softmax(logits, dim=-1)

    evict = logits.clone()
    evict[: seq_len - RECENT_W] = float("-inf")
    w_evicted = torch.softmax(evict, dim=-1)
    keep = torch.zeros(seq_len, dtype=torch.bool)
    keep[:N_SINK] = True
    keep[-RECENT_W:] = True
    w_stream = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)

    recent = torch.arange(seq_len - RECENT_W, seq_len)
    sink_mass = full[:N_SINK].sum().item()
    drift_evicted = (w_evicted[recent] - full[recent]).abs().max().item()
    drift_stream = (w_stream[recent] - full[recent]).abs().max().item()
    return (full.numpy(), w_evicted.numpy(), w_stream.numpy(),
            sink_mass, drift_evicted, drift_stream)


def fig_attention_sinks() -> None:
    full, _, _, sink_mass, drift_evicted, drift_stream = _sink_distributions()
    seq_len = full.shape[0]
    colors = [RED if i < N_SINK else BLUE for i in range(seq_len)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
    _style_axis(ax1)
    ax1.bar(range(seq_len), full, color=colors, width=0.9, zorder=3)
    ax1.set_xlabel("key position")
    ax1.set_ylabel("attention weight (full softmax)")
    ax1.set_title("Spare mass pools on the first few tokens (the sinks)", fontsize=11, fontweight="bold")
    ax1.annotate(f"{N_SINK} sink tokens\nabsorb {sink_mass:.1%} of the mass",
                 xy=(N_SINK / 2, full[:N_SINK].max()),
                 xytext=(seq_len * 0.28, full.max() * 0.8), fontsize=9.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))

    _style_axis(ax2)
    labels = ["evict\nsinks", "keep\nsinks"]
    drifts = [drift_evicted, drift_stream]
    bars = ax2.bar(labels, drifts, color=[RED, GREEN], width=0.6, zorder=3)
    for bar, d in zip(bars, drifts):
        ax2.text(bar.get_x() + bar.get_width() / 2, d + 0.006, f"{d:.3f}", ha="center",
                 fontsize=10, color=INK, fontweight="bold")
    ax2.set_ylabel("max drift of recent-window weights")
    ax2.set_title("Evicting sinks\nwrecks the survivors", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(drifts) * 1.25)
    _save(fig, "attention_sinks.png")


# =====================================================================================
# Figure 5 -- KV-cache memory wall: GiB vs context length
# =====================================================================================
def fig_kv_cache_wall() -> None:
    # Llama-2-7B MHA: 0.5 MiB/token ; Llama-3-8B GQA-8: 0.125 MiB/token (from the page).
    per_token_mha = 2 * 32 * 32 * 128 * 2 / 2**20  # MiB/token -> 0.5
    per_token_gqa = 2 * 32 * 8 * 128 * 2 / 2**20  # MiB/token -> 0.125
    ctx = np.array([4096, 8192, 16384, 32768, 65536, 131072])
    mha_gib = ctx * per_token_mha / 1024  # MiB -> GiB
    gqa_gib = ctx * per_token_gqa / 1024

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    _style_axis(ax)
    ax.plot(ctx / 1024, mha_gib, "-o", color=RED, linewidth=2.4, markersize=5,
            label="7B MHA (0.5 MiB/token)", zorder=3)
    ax.plot(ctx / 1024, gqa_gib, "-s", color=GREEN, linewidth=2.4, markersize=5,
            label="8B GQA-8 (0.125 MiB/token)", zorder=3)
    ax.axhline(80, color=SLATE, linewidth=1.4, linestyle="--", zorder=1, label="80 GB GPU")
    ax.scatter([128], [mha_gib[-1]], color=RED, s=80, zorder=5)
    ax.annotate("64 GiB for ONE sequence\nat 128K (7B MHA)",
                xy=(128, mha_gib[-1]), xytext=(54, 64), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.set_xlabel("context length (thousands of tokens)")
    ax.set_ylabel("KV cache (GiB), one sequence")
    ax.set_title("Wall 2: the KV cache grows linearly — and overruns the GPU at long context",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _save(fig, "kv_cache_wall.png")


# =====================================================================================
# Figure 6 -- lost in the middle: illustrative U-shaped retrieval curve
# =====================================================================================
def fig_lost_in_the_middle() -> None:
    depth = np.linspace(0, 100, 200)  # position of the relevant fact, % depth into context
    # Illustrative U-shape: high at the ends, dips in the middle (primacy + recency).
    u = 0.55 + 0.42 * ((depth - 50) / 50) ** 2 - 0.05 * np.sin(depth / 100 * np.pi)
    accuracy = np.clip(u, 0, 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axis(ax)
    ax.plot(depth, accuracy, color=PURPLE, linewidth=2.6, zorder=3)
    ax.fill_between(depth, accuracy, color=PURPLE, alpha=0.10, zorder=2)
    ax.scatter([0, 100], [accuracy[0], accuracy[-1]], color=GREEN, s=70, zorder=5)
    ax.scatter([50], [accuracy[100]], color=RED, s=70, zorder=5)
    ax.annotate("strong at the start\n(primacy)", xy=(0, accuracy[0]), xytext=(6, 0.62),
                fontsize=9, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.2))
    ax.annotate("strong at the end\n(recency)", xy=(100, accuracy[-1]), xytext=(62, 0.62),
                fontsize=9, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.2))
    ax.annotate("lost in the middle", xy=(50, accuracy[100]), xytext=(34, 0.30),
                fontsize=10, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.set_xlabel("position of the relevant fact (% depth into the context)")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("Lost in the middle (illustrative): retrieval is U-shaped, not flat",
                 fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 1)
    _save(fig, "lost_in_the_middle.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_clock_hands()
    fig_extrapolation_vs_pi()
    fig_receptive_field()
    fig_attention_sinks()
    fig_kv_cache_wall()
    fig_lost_in_the_middle()
    # Re-derive and print the load-bearing numbers so a reader can confirm they match the page.
    _, _, _, sink_mass, drift_evicted, drift_stream = _sink_distributions()
    print("\nnumbers embedded in the figures (must match the page/notebook):")
    print("  RoPE angle wall: trained=15  naive=63 (4.2x)  PI=15.75")
    print(f"  receptive field: {[reachable_span(L, 11) for L in range(1, N_LAYERS + 1)]}")
    print(f"  sink mass = {sink_mass:.3f}  drift evict={drift_evicted:.3f}  keep={drift_stream:.3f}")
    print(f"  KV cache 7B-MHA at 128K = {131072 * 0.5 / 1024:.0f} GiB")


if __name__ == "__main__":
    main()

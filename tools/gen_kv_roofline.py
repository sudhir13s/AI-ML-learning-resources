"""Roofline figure for 09. LLMs/05-KV-Cache/05-KV-Cache.md (muted palette).

One visual:
  kv_roofline.png -- WHY DECODE IS MEMORY-BOUND: log-log roofline (arithmetic
  intensity vs achievable FLOP/s) for an A100 (~312 TFLOP/s, ~2 TB/s, ridge at
  ~156 FLOP/byte); batch-1 decode sits at ~1 FLOP/byte deep in the
  bandwidth-bound region, prefill sits compute-bound past the ridge, and
  batching walks decode rightward toward the ridge.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "09. LLMs", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

PEAK_TFLOPS = 312.0          # A100 dense FP16 tensor-core peak
BANDWIDTH_TBS = 2.0          # A100 80GB HBM2e
RIDGE = PEAK_TFLOPS / BANDWIDTH_TBS  # = 156 FLOP/byte


def roofline():
    ai = np.logspace(-1, 3.2, 400)  # arithmetic intensity, FLOP/byte
    achievable = np.minimum(PEAK_TFLOPS, BANDWIDTH_TBS * ai)  # TFLOP/s

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ai, achievable, color=SLATE, lw=2.8, label="A100 roofline")
    ax.fill_between(ai, 0.01, achievable, where=ai <= RIDGE, color=RED, alpha=0.07)
    ax.fill_between(ai, 0.01, achievable, where=ai >= RIDGE, color=GREEN, alpha=0.07)

    ax.axvline(RIDGE, color=SLATE, ls="--", lw=1.4)
    ax.text(RIDGE * 1.12, 0.075, f"ridge ≈ {RIDGE:.0f} FLOP/byte\n(312 TFLOP/s ÷ 2 TB/s)",
            color=SLATE, fontsize=9, fontweight="bold")

    # batch-1 decode: intensity ~1, achievable = 2 TFLOP/s (0.6% of peak)
    ax.scatter([1], [BANDWIDTH_TBS * 1], color=RED, s=90, zorder=5, edgecolor="white")
    ax.annotate("decode, batch 1\n≈ 1 FLOP/byte → 2 TFLOP/s\n(~99% of compute idle)",
                (1, 2), textcoords="offset points", xytext=(-72, 52),
                fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))

    # batching walks decode toward the ridge: B = 8, 64
    for b in (8, 64):
        ax.scatter([b], [BANDWIDTH_TBS * b], color=AMBER, s=70, zorder=5, edgecolor="white")
        ax.annotate(f"batch {b}", (b, BANDWIDTH_TBS * b), textcoords="offset points",
                    xytext=(10, -14), fontsize=9, color=AMBER, fontweight="bold")
    ax.text(2.6, 0.35, "batching multiplies intensity ×B\n(read weights once, do B× the math)",
            color=AMBER, fontsize=9.5, fontweight="bold")

    # prefill: long prompt, high intensity, compute-bound
    ax.scatter([600], [PEAK_TFLOPS], color=GREEN, s=90, zorder=5, edgecolor="white")
    ax.annotate("prefill (whole prompt\nat once): compute-bound",
                (600, PEAK_TFLOPS), textcoords="offset points", xytext=(-52, -66),
                fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))

    ax.text(0.14, 90, "MEMORY-BOUND\n(slope = bandwidth)", color=RED,
            fontsize=10, fontweight="bold", alpha=0.8)
    ax.text(230, 2.2, "COMPUTE-BOUND\n(flat = peak FLOP/s)", color=GREEN,
            fontsize=10, fontweight="bold", alpha=0.8)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 1600); ax.set_ylim(0.05, 700)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte moved)")
    ax.set_ylabel("Achievable throughput (TFLOP/s)")
    ax.set_title("Roofline: batch-1 decode is ~156× below the compute knee",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kv_roofline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote kv_roofline.png")


if __name__ == "__main__":
    roofline()

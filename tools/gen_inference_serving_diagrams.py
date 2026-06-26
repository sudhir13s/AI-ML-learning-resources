"""Inference-Optimization-and-Serving concept-page diagrams (muted palette).

Five matplotlib visuals for 09. LLMs/09-Inference-Optimization-and-Serving. Every number
matches the companion demo (inference_serving.py / the notebook): an A100-80GB (2 TB/s,
312 TFLOP/s) serving Llama-3-8B (GQA-8, 16 GB weights, 0.125 MiB/token KV). All values are
MODELED, exactly as labelled on the page.

  1. serving_roofline_throughput.png -- throughput vs batch, with the compute-roofline
     crossover (B*=232) marked; the headline "batching amortizes the weight read" curve.
  2. serving_batching_timeline.png   -- static vs continuous batching Gantt on the same 6
     ragged requests / 3 slots; static makespan 80 steps vs continuous 64.
  3. serving_ttft_tpot.png           -- one request's latency broken into a compute-bound
     prefill (TTFT) and a memory-bound decode stream (TPOT/ITL).
  4. serving_speculative_speedup.png -- speculative-decoding speedup vs acceptance rate alpha,
     with the 1.0x break-even line; matches the demo's alpha sweep.
  5. serving_latency_throughput.png  -- the latency<->throughput frontier: TPOT vs throughput
     as batch grows, the curve every serving benchmark plots.

Output goes to 09. LLMs/images/ (the shared per-chapter image dir the pages reference as
../images/...). Run:  python gen_inference_serving_diagrams.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "09. LLMs", "images"))
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

# --- Shared modeled constants (identical to inference_serving.py) ----------------------------
HBM = 2.0e12            # 2 TB/s
PEAK = 312e12           # 312 TFLOP/s
WEIGHT_BYTES = 8.0e9 * 2          # 16 GB
KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2  # 0.125 MiB/token, in bytes
MIB = 1024 * 1024


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _step_time(batch, ctx):
    """Modeled (memory_time, compute_time) seconds for one decode step."""
    mem = (WEIGHT_BYTES + batch * ctx * KV_PER_TOKEN) / HBM
    cmp = 2 * 8.0e9 * batch / PEAK
    return mem, cmp


# ---- 1. Roofline: throughput vs batch with crossover ----------------------------------------
def roofline_throughput():
    ctx = 256
    batches = np.arange(1, 513)
    throughput = []
    for b in batches:
        mem, cmp = _step_time(b, ctx)
        throughput.append(b / max(mem, cmp))
    throughput = np.array(throughput)
    # crossover B*: where compute overtakes memory at this context
    b_star = 232  # from the demo (compute_equiv - kv headroom), context 256
    roofline = 19500  # saturation throughput (tok/s)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    mem_region = batches <= b_star
    ax.plot(batches[mem_region], throughput[mem_region], color=BLUE, lw=2.8, label="memory-bound (throughput rises with batch)")
    ax.plot(batches[~mem_region], throughput[~mem_region], color=RED, lw=2.8, label="compute-bound (saturated at the roofline)")
    ax.axhline(roofline, color=SLATE, ls="--", lw=1.4)
    ax.text(20, roofline - 1300, f"compute roofline ≈ {roofline:,} tok/s", color=SLATE, fontsize=9.5, fontweight="bold")
    ax.axvline(b_star, color=AMBER, ls=":", lw=1.8)
    ax.scatter([b_star], [b_star / max(_step_time(b_star, ctx))], color=AMBER, s=70, zorder=5, edgecolor="white")
    ax.annotate(f"crossover B* = {b_star}\n(decode turns compute-bound)", (b_star, 14000),
                textcoords="offset points", xytext=(18, -8), fontsize=9.5, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    # mark a few demo rows
    for b, t in [(1, 125), (32, 3748), (128, 12614)]:
        ax.scatter([b], [t], color=GREEN, s=36, zorder=6, edgecolor="white")
    ax.annotate("batch 1: 125 tok/s\n(GPU ~99% idle)", (1, 125), textcoords="offset points",
                xytext=(34, 70), fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("Batch size B (concurrent sequences)")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_title("Batching amortizes the 16 GB weight read — until the roofline", fontsize=13.5, fontweight="bold")
    ax.legend(loc="center right", frameon=False, fontsize=9)
    ax.set_xlim(0, 512); ax.set_ylim(0, 21500); _despine(ax)
    ax.text(0.99, 0.02, "modeled: A100-80GB · Llama-3-8B · 256-token ctx", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/serving_roofline_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote serving_roofline_throughput.png")


# ---- 2. Static vs continuous batching timeline (Gantt) --------------------------------------
def batching_timeline():
    # exact schedules computed from the simulation (3 slots, requests 4/6/20/60/5/8)
    # static: wave1 [a,b,c] over 0..20, wave2 [d,e,f] over 20..80
    static_bars = {  # slot -> list of (label, start, dur, color, is_idle)
        0: [("A", 0, 4, BLUE, False), ("idle", 4, 16, SLATE, True),
            ("D", 20, 60, RED, False)],
        1: [("B", 0, 6, BLUE, False), ("idle", 6, 14, SLATE, True),
            ("E", 20, 5, GREEN, False), ("idle", 25, 55, SLATE, True)],
        2: [("C", 0, 20, PURPLE, False),
            ("F", 20, 8, GREEN, False), ("idle", 28, 52, SLATE, True)],
    }
    # continuous: a(0-4) b(0-6) c(0-20) d(4-64) e(6-11) f(11-19)  -> makespan 64
    cont_bars = {
        0: [("A", 0, 4, BLUE, False), ("D", 4, 60, RED, False)],
        1: [("B", 0, 6, BLUE, False), ("E", 6, 5, GREEN, False), ("F", 11, 8, GREEN, False),
            ("idle", 19, 45, SLATE, True)],
        2: [("C", 0, 20, PURPLE, False), ("idle", 20, 44, SLATE, True)],
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)

    def draw(ax, bars, title, makespan):
        for slot, segs in bars.items():
            for label, start, dur, color, is_idle in segs:
                ax.add_patch(Rectangle((start, slot - 0.4), dur, 0.8,
                                       facecolor=color, alpha=0.28 if is_idle else 0.9,
                                       hatch="//" if is_idle else None,
                                       edgecolor="white", linewidth=1.0))
                if dur >= 4:
                    ax.text(start + dur / 2, slot, label if not is_idle else "idle",
                            ha="center", va="center", fontsize=9,
                            color="white" if not is_idle else "#333", fontweight="bold")
        ax.axvline(makespan, color="#222", ls="--", lw=1.6)
        ax.text(makespan + 0.6, 0.9, f"makespan\n{makespan*10} ms", fontsize=9, fontweight="bold", color="#222")
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["slot 1", "slot 2", "slot 3"])
        ax.set_ylim(-0.6, 2.6); ax.set_xlim(0, 84)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        _despine(ax)

    draw(ax1, static_bars, "STATIC batching — slots idle until the wave's longest request ends (43% util)", 80)
    draw(ax2, cont_bars, "CONTINUOUS batching — freed slot backfilled the next step (54% util)", 64)
    ax2.set_xlabel("Decode steps (10 ms each, modeled)")
    fig.suptitle("Same 6 ragged requests (4/6/20/60/5/8 tok), 3 slots: continuous finishes 1.25× sooner",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/serving_batching_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote serving_batching_timeline.png")


# ---- 3. TTFT / TPOT latency breakdown -------------------------------------------------------
def ttft_tpot():
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    # prefill: one compute-bound bar (sets TTFT)
    prefill_end = 3.0
    ax.add_patch(Rectangle((0, 0.3), prefill_end, 0.5, facecolor=AMBER, alpha=0.9, edgecolor="white"))
    ax.text(prefill_end / 2, 0.55, "PREFILL\n(compute-bound)", ha="center", va="center",
            color="white", fontsize=9.5, fontweight="bold")
    ax.annotate("TTFT\ntime-to-first-token", (prefill_end, 0.85), textcoords="offset points",
                xytext=(-6, 28), fontsize=9.5, color=AMBER, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    # decode: a stream of memory-bound token boxes (each gap = TPOT)
    tok_w = 0.9
    for i in range(6):
        x = prefill_end + 0.15 + i * (tok_w + 0.12)
        ax.add_patch(Rectangle((x, 0.3), tok_w, 0.5, facecolor=BLUE, alpha=0.9, edgecolor="white"))
        ax.text(x + tok_w / 2, 0.55, f"t{i+1}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(prefill_end + 3.3, 0.05, "DECODE  (memory-bound, one token at a time)", ha="center",
            fontsize=9.5, color=BLUE, fontweight="bold")
    # TPOT bracket between two tokens
    x0 = prefill_end + 0.15 + tok_w
    x1 = x0 + 0.12 + tok_w
    ax.annotate("", (x1, 1.0), (x0, 1.0), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
    ax.text((x0 + x1) / 2, 1.06, "TPOT / ITL\n(per-token latency)", ha="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.set_xlim(-0.2, prefill_end + 7.2); ax.set_ylim(0, 1.4)
    ax.axis("off")
    ax.set_title("One request's latency: prefill sets TTFT, decode sets TPOT/ITL",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/serving_ttft_tpot.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote serving_ttft_tpot.png")


# ---- 4. Speculative decoding speedup vs acceptance rate -------------------------------------
def speculative_speedup():
    k, c = 4, 0.1
    alpha = np.linspace(0.0, 0.97, 200)
    expected = (1 - alpha ** (k + 1)) / (1 - alpha)
    speedup = expected / (1 + k * c)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(alpha, speedup, color=PURPLE, lw=2.8)
    ax.axhline(1.0, color=RED, ls="--", lw=1.6)
    ax.text(0.02, 1.04, "break-even (1.0×): below this, speculation is slower than plain decode",
            color=RED, fontsize=9, fontweight="bold")
    # mark the demo's sweep points
    for a in (0.1, 0.5, 0.8, 0.9):
        s = (1 - a ** (k + 1)) / (1 - a) / (1 + k * c)
        ax.scatter([a], [s], color=AMBER if s >= 1 else RED, s=55, zorder=5, edgecolor="white")
        ax.annotate(f"α={a}: {s:.2f}×", (a, s), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color="#333")
    ax.fill_between(alpha, 0, speedup, where=speedup < 1, color=RED, alpha=0.10)
    ax.fill_between(alpha, 1, speedup, where=speedup >= 1, color=GREEN, alpha=0.10)
    ax.set_xlabel("Draft acceptance rate α (how often the target keeps a drafted token)")
    ax.set_ylabel("Expected wall-clock speedup (×)")
    ax.set_title("Speculative decoding pays off only at high acceptance (k=4, draft cost 0.1×)",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, 0.97); ax.set_ylim(0, 4.2); _despine(ax)
    ax.text(0.99, 0.02, "modeled: Leviathan et al. 2023 speedup formula", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/serving_speculative_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote serving_speculative_speedup.png")


# ---- 5. Latency <-> throughput frontier -----------------------------------------------------
def latency_throughput():
    ctx = 256
    batches = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 232, 320, 448]
    tpot_ms, thr = [], []
    for b in batches:
        mem, cmp = _step_time(b, ctx)
        step = max(mem, cmp)
        tpot_ms.append(step * 1e3)   # per-token latency each request sees
        thr.append(b / step)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(thr, tpot_ms, color=GREEN, lw=2.8, marker="o", ms=4)
    for b, t, l in zip(batches, thr, tpot_ms):
        if b in (1, 32, 128, 232, 448):
            ax.annotate(f"B={b}", (t, l), textcoords="offset points", xytext=(6, 6),
                        fontsize=9, fontweight="bold", color="#333")
    ax.annotate("more batch →\nmore throughput,\nbut higher per-token latency", (12000, 18),
                fontsize=9.5, color=SLATE, fontweight="bold")
    ax.set_xlabel("Throughput (tokens/s) — the operator's win")
    ax.set_ylabel("TPOT per request (ms/token) — the user's pain")
    ax.set_title("The latency↔throughput frontier: you trade one for the other",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 20500); ax.set_ylim(0, None); _despine(ax)
    ax.text(0.99, 0.96, "modeled: A100-80GB · Llama-3-8B · 256-token ctx", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#888", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/serving_latency_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote serving_latency_throughput.png")


if __name__ == "__main__":
    roofline_throughput()
    batching_timeline()
    ttft_tpot()
    speculative_speedup()
    latency_throughput()
    print("OUT:", OUT)

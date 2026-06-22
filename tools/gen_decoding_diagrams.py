"""Decoding-strategies concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 06. NLP/concepts/17-Decoding-Strategies.md:
  1. dec_temperature_softmax.png -- TEMPERATURE: the same 4-logit vector pushed
     through softmax at T=0.5 / 1.0 / 2.0 (grouped bars), showing T<1 sharpens
     toward greedy and T>1 flattens toward uniform.
  2. dec_topk_vs_topp.png        -- TRUNCATION: the same sorted 8-token
     distribution truncated by top-k=2 (fixed count) vs top-p=0.9 (dynamic mass);
     kept bars solid, cut bars faded, with the kept set annotated.
  3. dec_beam_tree.png           -- SEARCH: a 2-step tree where greedy (argmax at
     each step) takes AX (p=0.22) but beam B=2 keeps both A and B and recovers the
     true best BP (p=0.4275).
  4. dec_quality_diversity.png   -- TRADE-OFF: quality vs diversity for the main
     methods, with the "human" target band and the degeneration / incoherence
     corners marked.

All numbers match the worked examples in the page (verified in Python 3.12).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---- 1. Temperature reshapes the softmax ------------------------------------
def temperature_softmax():
    logits = np.array([3.0, 2.0, 1.0, 0.0])      # tokens A, B, C, D
    labels = ["A", "B", "C", "D"]
    temps = [0.5, 1.0, 2.0]
    colors = [GREEN, BLUE, RED]
    names = ["T = 0.5 (sharper → greedy)", "T = 1.0 (the raw model)", "T = 2.0 (flatter → uniform)"]
    probs = [_softmax(logits / T) for T in temps]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(labels))
    w = 0.26
    for i, (p, c, nm) in enumerate(zip(probs, colors, names)):
        bars = ax.bar(x + (i - 1) * w, p, w, color=c, edgecolor="white", linewidth=0.7, label=nm)
        for xi, v in zip(x + (i - 1) * w, p):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.2, color="#333")
    ax.axhline(0.25, color=SLATE, ls=":", lw=1.2)
    ax.text(3.35, 0.255, "uniform = 0.25", color=SLATE, fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"token {l}\nlogit {z:.0f}" for l, z in zip(labels, logits)])
    ax.set_ylabel("Probability after softmax")
    ax.set_title("Temperature rescales logits before softmax (same 4-logit vector)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, 1.0); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dec_temperature_softmax.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dec_temperature_softmax.png")


# ---- 2. top-k vs top-p truncation -------------------------------------------
def topk_vs_topp():
    p = np.array([0.40, 0.25, 0.13, 0.08, 0.06, 0.04, 0.025, 0.015])  # sorted desc
    labels = [f"t{i+1}" for i in range(len(p))]
    x = np.arange(len(p))
    cum = np.cumsum(p)
    keep_k = 2                       # top-k = 2
    keep_p = int(np.searchsorted(cum, 0.9) + 1)   # top-p = 0.9 -> 5 tokens

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

    # top-k
    ax = axes[0]
    for i, v in enumerate(p):
        kept = i < keep_k
        ax.bar(i, v, 0.72, color=BLUE if kept else SLATE,
               alpha=1.0 if kept else 0.28, edgecolor="white", linewidth=0.7)
    ax.axvline(keep_k - 0.5, color=RED, ls="--", lw=1.8)
    ax.text(keep_k - 0.35, 0.385, "cut here", color=RED, fontsize=9.5, fontweight="bold")
    ax.set_title(f"top-k = 2  →  keep 2 tokens (mass {p[:keep_k].sum():.2f})",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Probability")
    ax.text(0.5, 0.30, "fixed count,\nignores the shape", color=RED, fontsize=9, fontweight="bold")
    _despine(ax)

    # top-p
    ax = axes[1]
    for i, v in enumerate(p):
        kept = i < keep_p
        ax.bar(i, v, 0.72, color=GREEN if kept else SLATE,
               alpha=1.0 if kept else 0.28, edgecolor="white", linewidth=0.7)
    ax.axvline(keep_p - 0.5, color=RED, ls="--", lw=1.8)
    ax.text(keep_p - 0.35, 0.385, "cut here", color=RED, fontsize=9.5, fontweight="bold")
    ax.set_title(f"top-p = 0.9  →  keep {keep_p} tokens (mass {p[:keep_p].sum():.2f})",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.text(2.4, 0.30, "smallest set whose\ncumulative mass ≥ 0.9", color=GREEN, fontsize=9, fontweight="bold")
    _despine(ax)

    fig.suptitle("Same distribution, two truncations: fixed count vs adaptive mass",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/dec_topk_vs_topp.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dec_topk_vs_topp.png")


# ---- 3. Beam tree: greedy misses, beam recovers -----------------------------
def beam_tree():
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    # node positions
    root = (0, 2.0)
    A = (1.6, 3.0); B = (1.6, 1.0)
    AX = (3.4, 3.7); AY = (3.4, 3.0); AZ = (3.4, 2.3)
    BP = (3.4, 1.3); BQ = (3.4, 0.6)

    def node(xy, txt, color, big=False):
        w, h = (1.05, 0.5) if big else (0.92, 0.44)
        ax.add_patch(Rectangle((xy[0] - w / 2, xy[1] - h / 2), w, h,
                     facecolor=color, edgecolor="white", lw=1.0, zorder=3))
        ax.text(xy[0], xy[1], txt, ha="center", va="center", color="#fff",
                fontsize=9.5, fontweight="bold", zorder=4)

    def edge(a, b, label, color=SLATE, lw=1.6, ls="-"):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-", color=color, lw=lw, ls=ls), zorder=1)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", fontsize=8.3, color=color, fontweight="bold")

    # edges step 1
    edge(root, A, "0.55", color=AMBER, lw=2.4)         # greedy goes here
    edge(root, B, "0.45", color=GREEN, lw=2.4)         # beam keeps this too
    # edges step 2 from A
    edge(A, AX, "0.40", color=AMBER, lw=2.2)
    edge(A, AY, "0.35")
    edge(A, AZ, "0.25")
    # edges step 2 from B
    edge(B, BP, "0.95", color=GREEN, lw=2.6)
    edge(B, BQ, "0.05")

    node(root, "start", NAVY)
    node(A, "A  .55", AMBER); node(B, "B  .45", GREEN)
    node(AX, "AX  .22", AMBER); node(AY, "AY  .19", SLATE); node(AZ, "AZ  .14", SLATE)
    node(BP, "BP  .43", GREEN, big=True); node(BQ, "BQ  .02", SLATE)

    ax.text(4.15, 3.7, "← greedy ends here\n(p = 0.22)", color=AMBER, fontsize=9.5, fontweight="bold", va="center")
    ax.text(4.25, 1.3, "← beam B=2 winner\n(p = 0.4275, the true best)",
            color=GREEN, fontsize=9.5, fontweight="bold", va="center")

    ax.set_xlim(-0.6, 7.2); ax.set_ylim(0.0, 4.4)
    ax.set_title("Greedy is myopic: it commits to A (0.55) and never sees that B→P (0.95) wins overall",
                 fontsize=12.5, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/dec_beam_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dec_beam_tree.png")


# ---- 4. Quality vs diversity trade-off --------------------------------------
def quality_diversity():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    # (diversity, quality) heuristic coordinates on a 0..1 scale
    pts = [
        ("Greedy / Beam", 0.12, 0.55, RED),
        ("Low temp + top-p\n(T≈0.3)", 0.30, 0.82, BLUE),
        ("Nucleus top-p≈0.9", 0.62, 0.80, GREEN),
        ("High temp\n(T≈1.5)", 0.85, 0.50, AMBER),
        ("Pure sampling\n(T=1, no truncation)", 0.95, 0.30, PURPLE),
    ]
    # human target band
    ax.add_patch(Rectangle((0.50, 0.70), 0.28, 0.20, facecolor=GREEN, alpha=0.12, edgecolor=GREEN, lw=1.2, ls="--"))
    ax.text(0.64, 0.92, "human-like band", color=GREEN, fontsize=9.5, fontweight="bold", ha="center")
    for name, d, q, c in pts:
        ax.scatter([d], [q], s=180, color=c, edgecolor="white", lw=1.3, zorder=4)
        ax.annotate(name, (d, q), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=c, fontweight="bold")
    # corner labels
    ax.text(0.04, 0.10, "repetitive,\ndull (degeneration)", color=RED, fontsize=9, fontweight="bold")
    ax.text(0.72, 0.10, "incoherent,\noff-topic", color=PURPLE, fontsize=9, fontweight="bold")
    ax.set_xlabel("Diversity  (distinct n-grams, surprise)  →")
    ax.set_ylabel("Quality  (coherence, factuality)  →")
    ax.set_title("The quality–diversity trade-off every decoder navigates",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.0); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/dec_quality_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dec_quality_diversity.png")


if __name__ == "__main__":
    temperature_softmax()
    topk_vs_topp()
    beam_tree()
    quality_diversity()
    print("OUT:", OUT)

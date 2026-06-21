"""Bias-variance concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 03. Supervised_Learning/concepts/12-Bias-Variance-Tradeoff.md:
  1. bv_dartboard.png -- the classic 2x2 analogy: bias (off-centre) x variance
     (scattered), shown as dart patterns on a target.
  2. bv_ucurve.png -- a REAL bootstrap bias-variance decomposition: sweep
     polynomial degree, measure bias^2, variance, and total test error -> the
     U-shaped curve with the sweet spot.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bv_dartboard():
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    configs = [
        (axes[0, 0], (0, 0), 0.10, "low bias · low variance", GREEN, "the goal"),
        (axes[0, 1], (0, 0), 0.45, "low bias · high variance", AMBER, "overfit: right on average, scattered"),
        (axes[1, 0], (0.7, 0.4), 0.10, "high bias · low variance", BLUE, "underfit: tight but off-target"),
        (axes[1, 1], (0.7, 0.4), 0.45, "high bias · high variance", RED, "worst of both"),
    ]
    for ax, center, spread, title, col, sub in configs:
        for r, c in [(1.0, "#e9ecf0"), (0.66, "#dde2e8"), (0.33, "#cfd6de")]:
            ax.add_patch(Circle((0, 0), r, facecolor=c, edgecolor="white", zorder=1))
        ax.add_patch(Circle((0, 0), 0.08, facecolor=SLATE, zorder=2))
        pts = rng.normal(center, spread, (12, 2))
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=55, zorder=4, edgecolor="white")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=col)
        ax.text(0, -1.28, sub, ha="center", fontsize=8.8, color=SLATE)
    fig.suptitle("Bias vs variance: the dartboard (bullseye = the true answer)",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"{OUT}/bv_dartboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bv_dartboard.png")


def bv_ucurve():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(1.5 * x)                       # true function
    noise = 0.25
    xt = np.linspace(-3, 3, 60)                          # test points (full range: edges overfit)
    degrees = list(range(1, 13))
    bias2, var, total = [], [], []
    for d in degrees:
        preds = []
        for _ in range(150):                            # bootstrap training sets
            xtr = rng.uniform(-3, 3, 18); ytr = f(xtr) + rng.normal(0, noise, 18)  # few points -> high-degree overfits
            coef = np.polyfit(xtr, ytr, d)
            preds.append(np.polyval(coef, xt))
        preds = np.array(preds)                          # (runs, test)
        mean_pred = preds.mean(0)
        bias2.append(((mean_pred - f(xt)) ** 2).mean())
        var.append(preds.var(0).mean())
        total.append(bias2[-1] + var[-1] + noise ** 2)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    YTOP = 1.2
    ax.set_ylim(0, YTOP)                                # set limits BEFORE annotations
    ax.plot(degrees, np.clip(bias2, 0, YTOP), color=BLUE, lw=2.4, marker="o", ms=4, label="bias²  (down with complexity)")
    ax.plot(degrees, np.clip(var, 0, YTOP), color=RED, lw=2.4, marker="s", ms=4, label="variance  (up with complexity)")
    ax.plot(degrees, np.clip(total, 0, YTOP), color=PURPLE, lw=2.8, marker="^", ms=4, label="total test error (= bias² + var + noise)")
    ax.axhline(noise ** 2, color=SLATE, ls=":", lw=1.3)
    ax.text(8.2, noise**2 + 0.02, "irreducible noise σ²", fontsize=8.5, color=SLATE)
    best = degrees[int(np.argmin(total))]
    ax.axvline(best, color=GREEN, ls="--", lw=2)
    ax.text(best + 0.15, YTOP * 0.82, f"sweet spot\n(degree {best})", color=GREEN, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("error")
    ax.set_title("The bias–variance trade-off, measured by bootstrap",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bv_ucurve.png", dpi=150)
    plt.close(fig); print("wrote bv_ucurve.png")


if __name__ == "__main__":
    bv_dartboard()
    bv_ucurve()
    print("OUT:", OUT)

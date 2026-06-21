"""Gradient-descent concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 01. Foundations/concepts/13-Gradient-Descent-Theory.md:
  1. gd_lr.png    -- learning-rate effect on f(x)=x^2: too small (slow crawl),
     well-chosen (fast, smooth), too large (oscillates / diverges).
  2. gd_conditioning.png -- GD on a 2D quadratic: well-conditioned (near-circular
     contours -> straight path) vs ill-conditioned (elongated -> zig-zag).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "01. Foundations", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gd_lr():
    xs = np.linspace(-2.6, 2.6, 400)
    f = xs ** 2
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    configs = [("η = 0.05  too small", 0.05, BLUE), ("η = 0.4  well-chosen", 0.4, GREEN),
               ("η = 1.02  too large", 1.02, RED)]
    for ax, (title, lr, col) in zip(axes, configs):
        ax.plot(xs, f, color=SLATE, lw=1.8, alpha=0.6)
        x = 2.4; path = [x]
        for _ in range(12):
            x = x - lr * (2 * x)            # grad of x^2 is 2x
            path.append(x)
        path = np.array(path)
        ax.plot(path, path ** 2, "o-", color=col, lw=1.6, ms=5, alpha=0.9)
        ax.scatter([0], [0], color=AMBER, s=70, zorder=6, marker="*")
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=col)
        ax.set_xlabel("parameter x"); ax.set_ylim(-0.6, 7); ax.set_xlim(-2.8, 2.8); _despine(ax)
    axes[0].set_ylabel("loss  f(x) = x²")
    axes[1].annotate("minimum", (0, 0), textcoords="offset points", xytext=(6, 22), fontsize=9, color=AMBER)
    fig.suptitle("The learning rate controls everything: crawl, converge, or diverge",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/gd_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gd_lr.png")


def _run_gd(a, b, lr, steps=40, start=(4.5, 4.5)):
    # quadratic f = 0.5(a x^2 + b y^2); grad = (a x, b y)
    p = np.array(start, float); path = [p.copy()]
    for _ in range(steps):
        p = p - lr * np.array([a * p[0], b * p[1]])
        path.append(p.copy())
    return np.array(path)


def gd_conditioning():
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    gx, gy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-5, 5, 300))
    for ax, (a, b, name, kappa) in zip(
            axes, [(1.0, 1.0, "well-conditioned  (κ = 1)", 1), (1.0, 12.0, "ill-conditioned  (κ = 12)", 12)]):
        Z = 0.5 * (a * gx ** 2 + b * gy ** 2)
        ax.contour(gx, gy, Z, levels=18, colors=SLATE, linewidths=0.6, alpha=0.5)
        lr = 1.0 / max(a, b) * 0.9            # stable step ~ 1/L
        path = _run_gd(a, b, lr)
        ax.plot(path[:, 0], path[:, 1], "o-", color=RED if kappa > 1 else GREEN,
                lw=1.5, ms=3.5, alpha=0.9)
        ax.scatter([0], [0], color=AMBER, s=90, marker="*", zorder=6)
        ax.set_title(name, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂"); ax.set_aspect("equal")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); _despine(ax)
    axes[1].annotate("zig-zag: slow down\nthe narrow valley", (0.6, 2.4),
                     fontsize=9.5, color=RED, fontweight="bold")
    fig.suptitle("Conditioning sets the speed: round bowls go straight, narrow valleys zig-zag",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/gd_conditioning.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gd_conditioning.png")


if __name__ == "__main__":
    gd_lr()
    gd_conditioning()
    print("OUT:", OUT)

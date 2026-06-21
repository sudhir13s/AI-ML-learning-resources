"""Regularization concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/09-Regularization.md:
  1. reg_l1_l2.png  -- the iconic geometry: elliptical loss contours meeting the
     L2 (circle) vs L1 (diamond) constraint. L1's corner lands ON an axis -> a
     weight is exactly 0 (sparsity); L2's tangent keeps both weights small-but-nonzero.
  2. reg_earlystop.png -- train loss falling monotonically while val loss is
     U-shaped; the early-stopping point sits at the val minimum.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def reg_l1_l2():
    # unconstrained optimum (off-axis), elliptical contours of the data loss
    w_star = np.array([2.4, 1.7])
    A = np.array([[1.0, 0.5], [0.5, 1.3]])      # contour shape
    g1, g2 = np.meshgrid(np.linspace(-0.6, 3.4, 400), np.linspace(-0.6, 3.0, 400))
    d0 = g1 - w_star[0]; d1 = g2 - w_star[1]
    Z = A[0, 0]*d0**2 + 2*A[0, 1]*d0*d1 + A[1, 1]*d1**2

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    for ax, kind in zip(axes, ["L2", "L1"]):
        ax.contour(g1, g2, Z, levels=10, colors=SLATE, linewidths=0.7, alpha=0.55)
        ax.scatter(*w_star, color=RED, s=55, zorder=6)
        ax.annotate("unconstrained\noptimum", w_star, textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=RED, fontweight="bold")
        t = 1.25
        if kind == "L2":
            ax.add_patch(Circle((0, 0), t, fill=True, facecolor=BLUE, alpha=0.16,
                                edgecolor=BLUE, lw=2.2))
            sol = w_star / np.linalg.norm(w_star) * t            # approx tangent on circle
            sub = "L2 (Ridge): ‖w‖₂ ≤ t — round\ntangent lands OFF the axes →\nboth weights small but non-zero"
            scol = BLUE
        else:
            t1 = 1.45
            ax.add_patch(Polygon([(t1, 0), (0, t1), (-t1, 0), (0, -t1)], closed=True,
                                 facecolor=PURPLE, alpha=0.16, edgecolor=PURPLE, lw=2.2))
            sol = np.array([t1, 0.0])                            # corner on the w1 axis
            sub = "L1 (Lasso): ‖w‖₁ ≤ t — diamond\ncorner lands ON an axis →\nw₂ = 0 exactly (sparsity)"
            scol = PURPLE
        ax.scatter(*sol, color=GREEN, s=80, zorder=7, edgecolor="white")
        ax.annotate("solution", sol, textcoords="offset points", xytext=(8, -14),
                    fontsize=9.5, color=GREEN, fontweight="bold")
        ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(sub, fontsize=10.5, fontweight="bold", color=scol)
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
        ax.set_xlim(-0.6, 3.4); ax.set_ylim(-0.6, 3.0); ax.set_aspect("equal"); _despine(ax)
    fig.suptitle("Why L1 gives sparsity and L2 doesn't: where loss contours meet the constraint",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/reg_l1_l2.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote reg_l1_l2.png")


def reg_earlystop():
    e = np.linspace(1, 60, 400)
    train = 0.15 + 1.6*np.exp(-e/12)                       # falls monotonically
    val = 0.30 + 1.5*np.exp(-e/10) + 0.010*np.clip(e-22, 0, None)   # U-shaped
    stop = e[np.argmin(val)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(e, train, color=BLUE, lw=2.4, label="training loss (keeps dropping)")
    ax.plot(e, val, color=RED, lw=2.4, label="validation loss (U-shaped)")
    ax.axvline(stop, color=GREEN, ls="--", lw=2)
    ax.scatter([stop], [val.min()], color=GREEN, s=70, zorder=6)
    ax.annotate("early stop here\n(val minimum)", (stop, val.min()), textcoords="offset points",
                xytext=(12, 18), fontsize=10, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.axvspan(stop, 60, color=RED, alpha=0.06)
    ax.text((stop+60)/2, 1.5, "overfitting\nregion", ha="center", fontsize=9.5, color=RED)
    ax.set_xlabel("training epoch"); ax.set_ylabel("loss")
    ax.set_title("Early stopping: halt at the validation-loss minimum",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, 2.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/reg_earlystop.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote reg_earlystop.png")


if __name__ == "__main__":
    reg_l1_l2()
    reg_earlystop()
    print("OUT:", OUT)

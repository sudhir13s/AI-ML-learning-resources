"""Linear-regression concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/01-Linear-Regression.md:
  1. lr_fit.png  -- the least-squares line through scattered points, with the
     vertical RESIDUALS whose squares are what's being minimized.
  2. lr_surface.png -- the MSE loss surface over (slope, intercept) is a convex
     bowl; the normal equation lands at its minimum in one shot, gradient descent
     descends to the same point.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def lr_fit():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 24); y = 1.2 * x + 2 + rng.normal(0, 2.2, len(x))
    w, b = np.polyfit(x, y, 1)                            # least-squares fit
    yhat = w * x + b
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for xi, yi, yh in zip(x, y, yhat):                    # residual segments
        ax.plot([xi, xi], [yi, yh], color=RED, lw=1.4, alpha=0.7, zorder=1)
    ax.scatter(x, y, color=BLUE, s=42, edgecolor="white", lw=0.5, zorder=3, label="data points")
    ax.plot(x, yhat, color=NAVY, lw=2.6, zorder=2, label=f"least-squares line  ŷ = {w:.2f}x + {b:.2f}")
    ax.plot([], [], color=RED, lw=1.5, label="residuals (y − ŷ)")
    ax.text(0.3, 14.2, "least squares chooses the line that\nminimizes the SUM of squared residuals",
            fontsize=9.5, color=RED, fontweight="bold")
    ax.set_xlabel("feature x"); ax.set_ylabel("target y")
    ax.set_title("Linear regression: the line that minimizes squared residuals",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_fit.png")


def lr_surface():
    rng = np.random.default_rng(0)
    x0 = np.linspace(0, 10, 40); y = 1.2 * x0 + 2 + rng.normal(0, 1.5, len(x0))
    x = (x0 - x0.mean()) / x0.std()                       # standardize -> well-conditioned surface
    w_opt, b_opt = np.polyfit(x, y, 1)
    ws = np.linspace(w_opt - 7, w_opt + 7, 200); bs = np.linspace(b_opt - 6, b_opt + 6, 200)
    W, B = np.meshgrid(ws, bs)
    MSE = np.array([[np.mean((wv * x + bv - y) ** 2) for wv in ws] for bv in bs])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    cs = ax.contour(W, B, MSE, levels=18, colors=SLATE, linewidths=0.7, alpha=0.6)
    # gradient descent path on the MSE surface (standardized -> converges cleanly)
    w, b = w_opt - 6, b_opt + 5; eta = 0.25; path = [(w, b)]
    for _ in range(40):
        yh = w * x + b
        gw = np.mean(2 * (yh - y) * x); gb = np.mean(2 * (yh - y))
        w -= eta * gw; b -= eta * gb; path.append((w, b))
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=BLUE, lw=1.6, ms=3, alpha=0.85, label="gradient-descent path")
    ax.scatter([w_opt], [b_opt], color=AMBER, s=180, marker="*", zorder=6,
               edgecolor="white", label="normal-equation optimum (one shot)")
    ax.set_xlabel("slope  w"); ax.set_ylabel("intercept  b")
    ax.set_title("The MSE surface is a convex bowl — both methods reach the same minimum",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_surface.png")


if __name__ == "__main__":
    lr_fit()
    lr_surface()
    print("OUT:", OUT)

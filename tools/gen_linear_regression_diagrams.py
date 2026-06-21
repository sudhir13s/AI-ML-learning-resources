"""Linear-regression concept-page diagrams (muted palette, parallel scale).

Figures for 03. Supervised_Learning/concepts/01-Linear-Regression.md:
  1. lr_fit.png      -- the least-squares line through scattered points, with the
     vertical RESIDUALS whose squares are what's being minimized.
  2. lr_surface.png  -- the MSE loss surface over (slope, intercept) is a convex
     bowl; the normal equation lands at its minimum in one shot, gradient descent
     descends to the same point.
  3. lr_projection.png -- the GEOMETRY of least squares: y is projected
     orthogonally onto the column space of X; the residual is perpendicular to it.
  4. lr_r2.png       -- R^2 as variance decomposition: total variation (about the
     mean) splits into explained (by the fit) + residual; R^2 is the explained share.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

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
    ax.contour(W, B, MSE, levels=18, colors=SLATE, linewidths=0.7, alpha=0.6)
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


def lr_projection():
    """Geometry of least squares: y projects orthogonally onto col(X). The fitted
    yhat lives in the column space (a plane); the residual y - yhat is perpendicular
    to that plane. n=3 observations so the picture is a real 3D vector geometry."""
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    # The target vector y sticks up out of the z=0 plane (= col space of X here).
    y = np.array([2.4, 1.7, 1.9])
    yhat = np.array([y[0], y[1], 0.0])                    # orthogonal projection onto z=0 plane
    # draw the column-space plane (a translucent patch)
    gx, gy = np.meshgrid(np.linspace(-0.4, 3.0, 2), np.linspace(-0.4, 2.6, 2))
    gz = np.zeros_like(gx)
    ax.plot_surface(gx, gy, gz, alpha=0.12, color=BLUE, edgecolor="none")
    # basis columns of X spanning the plane
    for c in [np.array([2.4, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]:
        ax.quiver(0, 0, 0, c[0], c[1], c[2], color=BLUE, lw=2.2, arrow_length_ratio=0.08)
    ax.text(2.5, 0.0, -0.18, "col(X): all  Xw", color=BLUE, fontsize=10, fontweight="bold")
    # y vector (out of plane), yhat (in plane), residual (perpendicular)
    ax.quiver(0, 0, 0, y[0], y[1], y[2], color=NAVY, lw=2.8, arrow_length_ratio=0.07)
    ax.quiver(0, 0, 0, yhat[0], yhat[1], yhat[2], color=GREEN, lw=2.8, arrow_length_ratio=0.07)
    ax.quiver(yhat[0], yhat[1], yhat[2], 0, 0, y[2], color=RED, lw=2.6, arrow_length_ratio=0.12)
    ax.text(y[0], y[1], y[2] + 0.12, "y  (target)", color=NAVY, fontsize=10, fontweight="bold")
    ax.text(yhat[0] + 0.05, yhat[1], 0.05, "ŷ = Xw  (fit)", color=GREEN, fontsize=10, fontweight="bold")
    ax.text(yhat[0] + 0.08, yhat[1] + 0.05, y[2] / 2, "residual  y − ŷ\nperpendicular to col(X)",
            color=RED, fontsize=9.5, fontweight="bold")
    # right-angle marker at the foot of the residual
    s = 0.18
    ax.plot([yhat[0], yhat[0]], [yhat[1], yhat[1]], [0, s], color=RED, lw=1.0)
    ax.plot([yhat[0], yhat[0]-s], [yhat[1], yhat[1]], [s, s], color=RED, lw=1.0)
    ax.set_title("Least squares = orthogonal projection of y onto the column space of X",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("obs 1"); ax.set_ylabel("obs 2"); ax.set_zlabel("obs 3")
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.6); ax.set_zlim(0, 2.6)
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((1.2, 1.0, 1.0))
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_projection.png")


def lr_r2():
    """R^2 as a variance decomposition. For one point, show that total deviation
    (y - ybar) splits into explained (yhat - ybar) + residual (y - yhat); R^2 is the
    explained share of the TOTAL sum of squares: TSS = ESS + RSS, R^2 = ESS/TSS."""
    rng = np.random.default_rng(1)
    x = np.linspace(1, 9, 12); y = 1.1 * x + 2 + rng.normal(0, 1.6, len(x))
    w, b = np.polyfit(x, y, 1); yhat = w * x + b; ybar = y.mean()
    tss = np.sum((y - ybar)**2); rss = np.sum((y - yhat)**2); ess = tss - rss
    r2 = 1 - rss / tss
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.2),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    # left: the decomposition for one highlighted point
    ax.axhline(ybar, color=SLATE, ls="--", lw=1.4, label="mean baseline  ȳ")
    ax.plot(x, yhat, color=NAVY, lw=2.4, label=f"fit  ŷ = {w:.2f}x + {b:.2f}")
    ax.scatter(x, y, color=BLUE, s=40, edgecolor="white", lw=0.5, zorder=4, label="data")
    j = 9  # highlight one point with a big total deviation
    xi = x[j]
    ax.plot([xi, xi], [ybar, yhat[j]], color=GREEN, lw=5, alpha=0.8, solid_capstyle="butt")
    ax.plot([xi, xi], [yhat[j], y[j]], color=RED, lw=5, alpha=0.8, solid_capstyle="butt")
    ax.annotate("explained\n(ŷ − ȳ)", (xi, (ybar + yhat[j]) / 2), xytext=(xi - 3.4, ybar - 1.4),
                color=GREEN, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("residual\n(y − ŷ)", (xi, (yhat[j] + y[j]) / 2), xytext=(xi - 3.4, y[j] + 0.4),
                color=RED, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("feature x"); ax.set_ylabel("target y")
    ax.set_title("Each total deviation (y − ȳ) = explained + residual", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); _despine(ax)
    # right: the sum-of-squares bar decomposition
    ax2.bar(["TSS\n(about ȳ)"], [tss], color=SLATE, width=0.55, label="total (TSS)")
    ax2.bar(["ESS + RSS"], [ess], color=GREEN, width=0.55, label="explained (ESS)")
    ax2.bar(["ESS + RSS"], [rss], bottom=[ess], color=RED, width=0.55, label="residual (RSS)")
    ax2.set_ylabel("sum of squares")
    ax2.set_title(f"R² = ESS / TSS = {r2:.2f}\n(TSS = ESS + RSS)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9, loc="upper center"); _despine(ax2)
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_r2.png  (R2 = %.3f)" % r2)


if __name__ == "__main__":
    lr_fit()
    lr_surface()
    lr_projection()
    lr_r2()
    print("OUT:", OUT)

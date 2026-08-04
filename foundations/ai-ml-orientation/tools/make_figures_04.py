"""Figure generator for 04-How-Models-Learn — every number from the REAL learning loop.

All figures come from the same real datasets and the same from-scratch gradient descent the chapter
and notebook use (``how_models_learn.py``): the real California-housing regression and the real
breast-cancer classifier, both trained by the module's own GD. Nothing is hand-typed.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``basics04_``:

  basics04_loss_curve.png     -- REAL MSE of the linear model FALLING over gradient-descent epochs;
                                 the single picture of "a model getting less wrong."
  basics04_fit_evolution.png  -- the fitted line rotating into place on the REAL income->price
                                 scatter, at epochs 0, 2, 5, 20, 200.
  basics04_gd_surface.png     -- the REAL loss bowl over (slope, intercept) with the GD path
                                 stepping downhill to the minimum: "the ball rolling downhill."
  basics04_lr_sweep.png       -- the SAME real regression at three learning rates: too small
                                 crawls, well-chosen converges, too large diverges (measured).
  basics04_logreg_boundary.png-- the REAL breast-cancer decision boundary sharpening across
                                 training, with the log-loss falling alongside.

    python make_figures_04.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``00. Basics/tools/`` folder, while the chapter module it
# demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``how_models_learn`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "04-How-Models-Learn" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from how_models_learn import (  # noqa: E402  (resolved via the sys.path insert above)
    LR_SWEEP,
    REG_EPOCHS,
    linear_gd_path,
    load_income_price,
    load_tumor_2d,
    lr_sweep,
    mse_loss,
    predict_proba,
    sklearn_linear,
    train_linear_gd,
    train_logistic_gd,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / input
PURPLE = "#5D4A8A"  # process / model
GREEN = "#2E7A5A"  # good / converged / lower loss
RED = "#8B3B4A"  # cost / high loss / divergence
SLATE = "#4A5B6E"  # neutral / crawl
AMBER = "#7A6528"  # highlight
NAVY = "#2A5B80"  # secondary
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

# From ``00. Basics/tools/`` the shared image dir is one level up: ``00. Basics/images``.
OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "basics04_"


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ============================ Fig 1: the loss curve =============================================
def fig_loss_curve() -> None:
    """REAL MSE of the linear model falling over GD epochs — 'a model getting less wrong.'"""
    reg = load_income_price()
    fit = train_linear_gd(reg.x, reg.y)
    curve = fit.loss_curve

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(curve.size), curve, color=RED, linewidth=2.4, label="training loss (MSE)")
    ax.scatter([0], [curve[0]], color=SLATE, zorder=5)
    ax.scatter([curve.size - 1], [curve[-1]], color=GREEN, zorder=5)
    ax.annotate(
        f"start: {curve[0]:.2f}\n(flat line, knows nothing)",
        xy=(0, curve[0]),
        xytext=(curve.size * 0.16, curve[0] * 0.82),
        fontsize=9.5,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax.annotate(
        f"converged: {curve[-1]:.3f}",
        xy=(curve.size - 1, curve[-1]),
        xytext=(curve.size * 0.5, curve[-1] + 1.2),
        fontsize=9.5,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax.set_xlabel("gradient-descent epoch")
    ax.set_ylabel("loss  =  mean squared error")
    ax.set_title(
        "Learning is loss going down\n"
        "(linear regression on real California housing; each epoch = one downhill step)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9.5)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}loss_curve.png")


# ============================ Fig 2: the fit evolving ===========================================
def fig_fit_evolution() -> None:
    """The fitted line rotating into place on the REAL income->price scatter across epochs."""
    reg = load_income_price()
    fit = train_linear_gd(reg.x, reg.y)
    xs = np.linspace(reg.x.min(), reg.x.max(), 100)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(reg.x, reg.y, s=8, color=BLUE, alpha=0.18, label="real districts")
    # colour the snapshot lines pale -> dark green as training progresses
    epochs_sorted = sorted(fit.snapshots)
    shades = plt.cm.Greens(np.linspace(0.35, 0.95, len(epochs_sorted)))
    for shade, epoch in zip(shades, epochs_sorted):
        w, b = fit.snapshots[epoch]
        ax.plot(xs, w[0] * xs + b, color=shade, linewidth=2.2,
                label=f"epoch {epoch}  (loss {mse_loss(reg.x[:, None], reg.y, w, b):.2f})")
    ax.set_xlabel(reg.feature_name)
    ax.set_ylabel(reg.target_name)
    ax.set_title(
        "The model fitting itself to the data\n"
        "(the line starts flat at 0 and rotates into the best fit as the loss falls)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}fit_evolution.png")


# ============================ Fig 3: the loss surface + descent path ============================
def fig_gd_surface() -> None:
    """The REAL loss bowl over (slope, intercept) with the GD path stepping to the minimum."""
    reg = load_income_price()
    w_hist, b_hist, loss_hist = linear_gd_path(reg.x, reg.y)
    sk_w, sk_b = sklearn_linear(reg.x, reg.y)

    # build the loss surface on a grid around the descent path
    w_grid = np.linspace(-0.3, 1.3, 120)
    b_grid = np.linspace(-0.2, 2.6, 120)
    ww, bb = np.meshgrid(w_grid, b_grid)
    surface = np.empty_like(ww)
    for i in range(ww.shape[0]):
        for j in range(ww.shape[1]):
            surface[i, j] = mse_loss(reg.x[:, None], reg.y, np.array([ww[i, j]]), bb[i, j])

    fig, ax = plt.subplots(figsize=(8.4, 6))
    cs = ax.contourf(ww, bb, surface, levels=25, cmap="Blues_r", alpha=0.9)
    ax.contour(ww, bb, surface, levels=12, colors="white", linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, ax=ax, label="loss (MSE)")
    ax.plot(w_hist, b_hist, color=RED, linewidth=1.8, marker="o", markersize=2.5,
            label="gradient-descent path")
    ax.scatter([w_hist[0]], [b_hist[0]], color=SLATE, s=70, zorder=6, label="start (0, 0)")
    ax.scatter([sk_w[0]], [sk_b], color=GREEN, marker="*", s=260, zorder=6,
               edgecolor="white", label="least-squares minimum")
    ax.set_xlabel("slope  w")
    ax.set_ylabel("intercept  b")
    ax.set_title(
        "Gradient descent is a ball rolling downhill\n"
        f"(real loss surface; {REG_EPOCHS} steps from (0,0) to the true minimum)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}gd_surface.png")


# ============================ Fig 4: learning-rate sweep ========================================
def fig_lr_sweep() -> None:
    """The SAME real regression at three learning rates: crawl / converge / diverge (measured)."""
    reg = load_income_price()
    curves = lr_sweep(reg.x, reg.y)
    styles = {
        LR_SWEEP[0]: (SLATE, "too small: crawls"),
        LR_SWEEP[1]: (GREEN, "well-chosen: converges"),
        LR_SWEEP[2]: (RED, "too large: diverges"),
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for rate, curve in curves.items():
        color, label = styles[rate]
        # clip the diverging curve so the log axis stays readable; the trend is the point
        plotted = np.clip(np.nan_to_num(curve, nan=1e6, posinf=1e6), 1e-3, 1e6)
        ax.plot(range(plotted.size), plotted, color=color, linewidth=2.3,
                label=f"lr = {rate}  ({label})")
    ax.set_yscale("log")
    ax.set_xlabel("gradient-descent epoch")
    ax.set_ylabel("loss (MSE, log scale)")
    ax.set_title(
        "The learning rate decides everything\n"
        "(identical model & data; only the step size changes)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9.5)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}lr_sweep.png")


# ============================ Fig 5: logistic decision boundary =================================
def _boundary_points(w: NDArray[np.float64], b: float, xlim: tuple[float, float]) -> tuple:
    """The line w0*x + w1*y + b = 0 solved for y across xlim (the 0.5-probability contour)."""
    xs = np.array(xlim)
    ys = -(w[0] * xs + b) / w[1]
    return xs, ys


def fig_logreg_boundary() -> None:
    """The REAL breast-cancer decision boundary sharpening across training + the log-loss falling."""
    cls = load_tumor_2d()
    fit = train_logistic_gd(cls.x, cls.y)
    xlim = (cls.x[:, 0].min() - 0.3, cls.x[:, 0].max() + 0.3)
    ylim = (cls.x[:, 1].min() - 0.3, cls.x[:, 1].max() + 0.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: scatter + boundary at each snapshot (pale -> dark as it sharpens)
    benign = cls.y == 1
    ax1.scatter(cls.x[benign, 0], cls.x[benign, 1], s=16, color=GREEN, alpha=0.6, label="benign")
    ax1.scatter(cls.x[~benign, 0], cls.x[~benign, 1], s=16, color=RED, alpha=0.6, label="malignant")
    epochs_sorted = sorted(fit.snapshots)
    shades = plt.cm.Purples(np.linspace(0.4, 0.95, len(epochs_sorted)))
    for shade, epoch in zip(shades, epochs_sorted):
        w, b = fit.snapshots[epoch]
        if abs(w[1]) < 1e-6:
            continue  # epoch-0 line is undefined (w=0); skip the degenerate start
        bx, by = _boundary_points(w, b, xlim)
        ax1.plot(bx, by, color=shade, linewidth=2.2, label=f"boundary @ epoch {epoch}")
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel(cls.feature_names[0])
    ax1.set_ylabel(cls.feature_names[1])
    ax1.set_title("The decision boundary sharpening\n(real tumours; the line learns to separate)", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    _style_axis(ax1)

    # right: the log-loss falling (same loop, classification loss)
    curve = fit.loss_curve
    ax2.plot(range(curve.size), curve, color=PURPLE, linewidth=2.4)
    ax2.axhline(np.log(2), color=SLATE, linestyle="--", linewidth=1.2, label=f"coin flip: ln 2 = {np.log(2):.3f}")
    ax2.annotate(
        f"converged: {curve[-1]:.3f}",
        xy=(curve.size - 1, curve[-1]),
        xytext=(curve.size * 0.4, curve[-1] + 0.16),
        fontsize=9.5,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax2.set_xlabel("gradient-descent epoch")
    ax2.set_ylabel("loss  =  log-loss (cross-entropy)")
    ax2.set_title("The same loop, a different loss\n(log-loss falls exactly like MSE did)", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    _style_axis(ax2)

    acc = float(np.mean((predict_proba(cls.x, fit.w, fit.b) >= 0.5).astype(int) == cls.y))
    fig.suptitle(
        f"Logistic regression learns the SAME way — predict, measure loss, step downhill "
        f"(final training accuracy {acc:.0%})",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, f"{IMG_PREFIX}logreg_boundary.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_loss_curve()
    fig_fit_evolution()
    fig_gd_surface()
    fig_lr_sweep()
    fig_logreg_boundary()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()

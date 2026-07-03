"""Figure generator for 05-Overfitting-and-Underfitting — every number from the REAL measured study.

All figures come from the same real pipeline the chapter and notebook use
(``overfitting_underfitting.py``): the controlled ``cos(1.5*pi*x)`` signal-plus-noise study, fit by
the module's own from-scratch (and sklearn-verified) polynomial least squares. Nothing is hand-typed;
every curve, bar, and annotation is read off an executed function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``basics05_``:

  basics05_ucurve.png        -- the classic U: TRAIN error falling monotonically with capacity while
                                held-out VALIDATION error falls, bottoms at the sweet spot, then rises.
  basics05_three_fits.png    -- the three regimes as fitted curves over the real data: degree 1
                                underfits (misses the bend), degree 4 fits well, degree 15 overfits
                                (wiggles through every noisy point). The money shot.
  basics05_bias_variance.png -- the measured decomposition: bias^2 falling and variance rising with
                                capacity, summing (with the noise floor) to the U. The math, measured.
  basics05_ridge.png         -- regularization as the fix: sweeping the L2 penalty on the overfit
                                degree-15 model pulls validation error back down to the sweet spot.
  basics05_learning_curve.png-- the other fix: growing the training set shrinks the generalisation
                                gap (validation minus training error) toward zero.

    python make_figures_05.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``00. Basics/tools/`` folder, while the chapter module it
# demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``overfitting_underfitting`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "05-Overfitting-and-Underfitting" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from overfitting_underfitting import (  # noqa: E402  (resolved via the sys.path insert above)
    GOOD_DEGREE,
    LC_DEGREE,
    N_TRAIN,
    N_VAL,
    NOISE_SIGMA,
    OVERFIT_DEGREE,
    TRAIN_SEED,
    UNDERFIT_DEGREE,
    VAL_SEED,
    bias_variance_decomposition,
    complexity_sweep,
    fit_poly,
    learning_curve,
    make_dataset,
    predict_poly,
    ridge_lambda_sweep,
    true_function,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / input
PURPLE = "#5D4A8A"  # process / model
GREEN = "#2E7A5A"  # good / sweet spot / lower error
RED = "#8B3B4A"  # cost / high error / overfit
SLATE = "#4A5B6E"  # neutral / underfit
AMBER = "#7A6528"  # highlight / noise floor
NAVY = "#2A5B80"  # secondary
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

# From ``00. Basics/tools/`` the shared image dir is one level up: ``00. Basics/images``.
OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "basics05_"


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


# ============================ Fig 1: the U-curve ================================================
def fig_ucurve() -> None:
    """TRAIN error falling monotonically vs VALIDATION error's U — the definition of over/under-fit."""
    train = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    val = make_dataset(N_VAL, seed=VAL_SEED)
    sweep = complexity_sweep(train, val)
    d = sweep.degrees
    best = sweep.best_degree
    best_i = int(np.where(d == best)[0][0])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(d, sweep.train_mse, color=BLUE, linewidth=2.4, marker="o", markersize=4, label="training error")
    ax.plot(d, sweep.val_mse, color=RED, linewidth=2.4, marker="s", markersize=4, label="validation error (held out)")
    ax.axhline(NOISE_SIGMA**2, color=AMBER, linestyle=":", linewidth=1.6, label=f"noise floor  $\\sigma^2$ = {NOISE_SIGMA**2:.3f}")
    ax.scatter([best], [sweep.val_mse[best_i]], color=GREEN, s=150, zorder=6, marker="*", edgecolor="white",
               label=f"sweet spot (degree {best})")
    ax.annotate("underfit\n(too simple)", xy=(d[0], sweep.val_mse[0]), xytext=(1.4, sweep.val_mse[0] - 0.045),
                fontsize=9.5, color=SLATE, ha="left")
    ax.annotate("overfit\n(too complex)", xy=(d[-1], sweep.val_mse[-1]), xytext=(11.2, sweep.val_mse[-1] + 0.03),
                fontsize=9.5, color=RED, ha="left")
    ax.set_xlabel("model complexity  =  polynomial degree")
    ax.set_ylabel("error  =  mean squared error")
    ax.set_xticks(d)
    ax.set_title(
        "The U-curve: more capacity always lowers TRAINING error,\n"
        "but VALIDATION error falls, bottoms out, then rises",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper center")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}ucurve.png")


# ============================ Fig 2: the three fits (the money shot) ============================
def fig_three_fits() -> None:
    """Underfit / good / overfit as fitted curves over the real data — the same data, only capacity."""
    train = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    val = make_dataset(N_VAL, seed=VAL_SEED)
    xs = np.linspace(0.0, 1.0, 400)
    f_true = true_function(xs)
    regimes = [
        (UNDERFIT_DEGREE, "Underfit", "too simple — misses the bend (high bias)", SLATE),
        (GOOD_DEGREE, "Good fit", "about right — tracks the true curve", GREEN),
        (OVERFIT_DEGREE, "Overfit", "too complex — chases the noise (high variance)", RED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for ax, (deg, name, sub, colour) in zip(axes, regimes):
        fit = fit_poly(train.x, train.y, deg)
        val_mse = float(np.mean((val.y - predict_poly(fit, val.x)) ** 2))
        ax.scatter(train.x, train.y, s=26, color=BLUE, alpha=0.55, zorder=3, label="training points")
        ax.plot(xs, f_true, color=INK, linestyle="--", linewidth=1.8, alpha=0.7, label="true function")
        ax.plot(xs, predict_poly(fit, xs), color=colour, linewidth=2.6, zorder=4, label=f"degree-{deg} fit")
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("x")
        ax.set_title(f"{name}  (degree {deg})\n{sub}\nvalidation MSE = {val_mse:.3f}", fontsize=10.5, color=colour)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        _style_axis(ax)
    axes[0].set_ylabel("y")
    fig.suptitle(
        "The same 40 noisy points, three model capacities — underfit, just-right, overfit",
        fontsize=12.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, f"{IMG_PREFIX}three_fits.png")


# ============================ Fig 3: the bias-variance decomposition ============================
def fig_bias_variance() -> None:
    """The measured decomposition: bias^2 down + variance up + noise = the U (log scale for range)."""
    bv = bias_variance_decomposition()
    d = bv.degrees
    total = bv.bias2 + bv.variance + bv.noise

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(d, bv.bias2, color=BLUE, linewidth=2.4, marker="o", markersize=4, label="bias$^2$ (systematic error)")
    ax.plot(d, bv.variance, color=RED, linewidth=2.4, marker="s", markersize=4, label="variance (sensitivity to the sample)")
    ax.plot(d, total, color=PURPLE, linewidth=2.8, marker="D", markersize=4, label="total = bias$^2$ + variance + $\\sigma^2$")
    ax.axhline(bv.noise, color=AMBER, linestyle=":", linewidth=1.6, label=f"irreducible noise  $\\sigma^2$ = {bv.noise:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("model complexity  =  polynomial degree")
    ax.set_ylabel("error (log scale)")
    ax.set_xticks(d)
    ax.annotate("bias falls\n(model can bend to the truth)", xy=(2, bv.bias2[1]), xytext=(2.6, 0.0016),
                fontsize=9, color=BLUE, ha="left",
                arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.annotate("variance rises\n(chases the sample's noise)", xy=(8, bv.variance[7]), xytext=(4.0, 0.32),
                fontsize=9, color=RED, ha="left",
                arrowprops={"arrowstyle": "->", "color": RED})
    ax.set_title(
        "Bias-variance decomposition, measured over 600 resampled fits\n"
        "their sum is exactly the U — that is the whole tradeoff",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}bias_variance.png")


# ============================ Fig 4: the ridge lambda sweep =====================================
def fig_ridge() -> None:
    """Regularization as the fix: the L2 penalty pulls the overfit degree-15 model back to the sweet spot."""
    train = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    val = make_dataset(N_VAL, seed=VAL_SEED)
    rs = ridge_lambda_sweep(train, val)
    sweep = complexity_sweep(train, val)
    best_i = int(np.argmin(rs.val_mse))
    sweet_val = float(sweep.val_mse[int(np.where(sweep.degrees == sweep.best_degree)[0][0])])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(rs.lambdas, rs.val_mse, color=RED, linewidth=2.4, marker="s", markersize=4, label="validation error")
    ax.plot(rs.lambdas, rs.train_mse, color=BLUE, linewidth=2.0, marker="o", markersize=3, alpha=0.8, label="training error")
    ax.axhline(rs.unpenalised_val_mse, color=SLATE, linestyle="--", linewidth=1.4,
               label=f"overfit ($\\lambda$=0) val = {rs.unpenalised_val_mse:.3f}")
    ax.axhline(sweet_val, color=GREEN, linestyle=":", linewidth=1.6,
               label=f"sweet-spot degree-{sweep.best_degree} val = {sweet_val:.3f}")
    ax.scatter([rs.lambdas[best_i]], [rs.val_mse[best_i]], color=GREEN, s=150, zorder=6, marker="*",
               edgecolor="white", label=f"best $\\lambda$ = {rs.best_lambda:.2g}")
    ax.set_xscale("log")
    ax.set_xlabel("L2 penalty strength  $\\lambda$  (log scale)")
    ax.set_ylabel("error  =  mean squared error")
    ax.set_title(
        f"Regularization cures overfitting: an L2 penalty on the wild degree-{OVERFIT_DEGREE} model\n"
        "shrinks its weights and drops validation error back to the sweet spot",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}ridge.png")


# ============================ Fig 5: the learning curve ========================================
def fig_learning_curve() -> None:
    """More data is the other cure: growing the training set shrinks the generalisation gap."""
    val = make_dataset(N_VAL, seed=VAL_SEED)
    lc = learning_curve(val)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(lc.sizes, lc.train_mse, color=BLUE, linewidth=2.4, marker="o", markersize=5, label="training error")
    ax.plot(lc.sizes, lc.val_mse, color=RED, linewidth=2.4, marker="s", markersize=5, label="validation error")
    ax.fill_between(lc.sizes, lc.train_mse, lc.val_mse, color=RED, alpha=0.10, label="generalisation gap")
    ax.axhline(NOISE_SIGMA**2, color=AMBER, linestyle=":", linewidth=1.6, label=f"noise floor  $\\sigma^2$ = {NOISE_SIGMA**2:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training-set size  n  (log scale)")
    ax.set_ylabel("error (log scale)")
    ax.annotate(f"gap = {lc.gap[0]:.2f}\n(overfitting)", xy=(lc.sizes[0], lc.val_mse[0]),
                xytext=(lc.sizes[0] * 1.25, lc.val_mse[0] * 0.62), fontsize=9, color=RED)
    ax.annotate(f"gap -> {lc.gap[-1]:.3f}\n(generalises)", xy=(lc.sizes[-1], lc.val_mse[-1]),
                xytext=(lc.sizes[-1] * 0.32, lc.val_mse[-1] * 1.9), fontsize=9, color=GREEN)
    ax.set_title(
        f"More data cures overfitting: fixed degree-{LC_DEGREE} model, growing training set\n"
        "training and validation error converge — the gap closes",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}learning_curve.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_ucurve()
    fig_three_fits()
    fig_bias_variance()
    fig_ridge()
    fig_learning_curve()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()

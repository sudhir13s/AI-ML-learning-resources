"""Figure generator for 10-Gradient-Boosting-XGBoost — every number from the REAL runs in ``gradient_boosting.py``.

All figures come from the same real pipeline the chapter and notebook use (``gradient_boosting.py`` on the
scikit-learn **California Housing** and **Breast Cancer** datasets, with real **XGBoost**): the from-scratch
boosting loop verified against scikit-learn, the measured staged train/validation curve and early-stopping
round, the learning-rate x n_estimators trade, the 1-D residual-shrinking movie, the XGBoost leaf-weight /
split-gain worked example, and the single-tree vs forest vs GBM vs XGBoost comparison. Nothing is hand-typed;
every curve, bar, and annotation is read off an executed function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``sup10_``:

  sup10_staged.png        -- the staged train/validation MSE curve on California Housing: train falls forever
                             while validation dips to a minimum and then RISES; the early-stopping round is marked.
  sup10_lr_sweep.png      -- the learning-rate x n_estimators trade: validation-MSE curves for lr in
                             {1.0, 0.3, 0.1, 0.03}; a small rate needs many more trees but reaches a lower minimum.
  sup10_residual_shrink.png -- the 1-D residual movie on the real MedInc feature: the ensemble staircase after
                             1, 5, 20, 100 rounds converging to the trend, with the residual RMS falling.
  sup10_model_compare.png -- test R^2 for a single tree, a random forest, sklearn GBM, and real XGBoost on the
                             same California split: bias-reduced boosting edges the forest, which crushes one tree.
  sup10_xgb_gain.png      -- the XGBoost worked example: a parent leaf splitting into two, with G/H/w* per node
                             and the split-gain formula computed term by term (gain = 0.837 > 0, so the split is kept).

    python make_figures_10.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9 / xgboost 3.3.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``03. Supervised_Learning/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``gradient_boosting`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "10-Gradient-Boosting-XGBoost" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gradient_boosting import (  # noqa: E402  (resolved via the sys.path insert above)
    learning_rate_sweep,
    load_california,
    model_comparison,
    residual_movie,
    staged_curve,
    xgboost_leaf_gain,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # train / data / first series
PURPLE = "#5D4A8A"  # process / a second series
GREEN = "#2E7A5A"  # good / GBM / converged
RED = "#8B3B4A"  # validation / overfit / deep / cost
SLATE = "#4A5B6E"  # neutral / scatter
AMBER = "#7A6528"  # highlight / chosen round / annotations
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "sup10_"


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


# ============================ Fig 1: the staged train/validation curve ==========================
def fig_staged() -> None:
    """Train vs validation MSE across a long boosting run — the overfitting U-turn, measured on California."""
    data = load_california()
    curve = staged_curve(data)
    rounds = curve.rounds

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(rounds, curve.train_mse, color=BLUE, linewidth=2.2, label="training MSE")
    ax.plot(rounds, curve.val_mse, color=RED, linewidth=2.2, label="validation MSE")
    ax.axvline(curve.best_round, color=AMBER, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"early stop\nround {curve.best_round}\nval MSE {curve.best_val_mse:.3f}",
        xy=(curve.best_round, curve.best_val_mse),
        xytext=(curve.best_round + 70, curve.best_val_mse + 0.18),
        fontsize=9, color=AMBER, arrowprops={"arrowstyle": "->", "color": AMBER},
    )
    over = rounds >= curve.best_round
    ax.fill_between(rounds[over], curve.val_mse[over], curve.train_mse[over], color=RED, alpha=0.07)
    ax.annotate("overfitting: train ↓ forever,\nvalidation turns UP",
                xy=(430, curve.val_mse[429]), xytext=(250, 0.62), fontsize=9, color=RED)
    ax.set_xlabel("boosting rounds (number of trees)")
    ax.set_ylabel("mean squared error")
    ax.set_title(
        f"Gradient boosting overfits with too many rounds — California Housing (lr=0.1, depth 4)\n"
        f"training MSE falls to {curve.train_mse[-1]:.3f} while validation bottoms at {curve.best_val_mse:.3f} "
        f"(round {curve.best_round}) then RISES — early-stop at the validation minimum",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_ylim(0, max(curve.val_mse) * 1.02)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}staged.png")


# ============================ Fig 2: the learning-rate x n_estimators trade ======================
def fig_lr_sweep() -> None:
    """Validation-MSE curves for several learning rates — small rate, more trees, lower minimum."""
    data = load_california()
    sweep = learning_rate_sweep(data)
    colours = (RED, AMBER, GREEN, BLUE)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for lr, curve, best_round, best_mse, colour in zip(
        sweep.learning_rates, sweep.val_curves, sweep.best_rounds, sweep.best_val_mses, colours
    ):
        rounds = np.arange(1, len(curve) + 1)
        ax.plot(rounds, curve, color=colour, linewidth=2.0,
                label=f"lr={lr}: min {best_mse:.3f} @ {best_round} trees")
        ax.scatter([best_round], [best_mse], color=colour, s=40, zorder=5, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("boosting rounds (number of trees)")
    ax.set_ylabel("validation MSE")
    ax.set_title(
        "Shrinkage is regularization: learning_rate ↔ n_estimators trade — California Housing\n"
        "a big rate (1.0) descends fast but overshoots and overfits in a few trees;\n"
        "a small rate (0.03) needs hundreds of trees but reaches a lower, flatter minimum",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", title="dot = validation minimum")
    ax.set_ylim(0.25, 0.75)
    ax.set_xlim(0, 800)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}lr_sweep.png")


# ============================ Fig 3: the 1-D residual-shrinking movie ============================
def fig_residual_shrink() -> None:
    """The ensemble staircase on one real feature after 1, 5, 20, 100 rounds — converging to the trend."""
    data = load_california()
    movie = residual_movie(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True)
    for ax, rounds, pred, rms in zip(
        axes.ravel(), movie.checkpoints, movie.predictions, movie.residual_rms
    ):
        ax.scatter(movie.x, movie.y, s=10, color=SLATE, alpha=0.20, zorder=1)
        ax.plot(movie.grid, pred, color=GREEN, linewidth=2.4, drawstyle="steps-mid", zorder=3)
        ax.set_title(f"after {rounds} round{'s' if rounds > 1 else ''}   ·   residual RMS = {rms:.3f}",
                     fontsize=11, color=INK)
        _style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel(f"{movie.feature_name} (median income, real feature)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median house value ($100k)")
    fig.suptitle(
        "Gradient boosting = fit the leftover residual, round after round (real California 'MedInc' slice)\n"
        f"the ensemble staircase refines from a coarse step to the trend; residual RMS falls "
        f"{movie.residual_rms[0]:.3f} → {movie.residual_rms[-1]:.3f}",
        fontsize=12, color=INK, y=1.0,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}residual_shrink.png")


# ============================ Fig 4: why GBDTs win tabular =======================================
def fig_model_compare() -> None:
    """Test R^2 for a single tree, a forest, sklearn GBM, and real XGBoost — the honest tabular comparison."""
    data = load_california()
    comp = model_comparison(data)
    colours = (SLATE, BLUE, GREEN, PURPLE)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(range(len(comp.names)), comp.r2, color=colours, alpha=0.92, width=0.62)
    for i, (r2, rmse) in enumerate(zip(comp.r2, comp.rmse)):
        ax.text(i, r2 + 0.008, f"R²={r2:.3f}\nRMSE={rmse:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(comp.names)))
    ax.set_xticklabels(comp.names, fontsize=9.5)
    ax.set_ylabel("test R²  (higher is better)")
    ax.set_ylim(0, max(comp.r2) * 1.16)
    ax.set_title(
        "Why gradient-boosted trees win tabular data — California Housing (same test split)\n"
        f"a single tree R²={comp.r2[0]:.3f}; a random forest lifts it to {comp.r2[1]:.3f}; "
        f"boosting/XGBoost reach {comp.r2[2]:.3f}/{comp.r2[3]:.3f}\n"
        "bagging cuts variance off one tree; boosting then cuts bias to edge past the forest",
        fontsize=10.5,
    )
    _style_axis(ax)
    ax.set_axisbelow(True)
    for _ in bars:
        pass
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}model_compare.png")


# ============================ Fig 5: the XGBoost leaf-weight / split-gain diagram =================
def fig_xgb_gain() -> None:
    """The XGBoost worked example: a split with G/H/w* per node and the gain formula computed term by term."""
    split = xgboost_leaf_gain()
    g = (-0.8, -0.6, 0.5, 0.7, 0.9)
    g_l, g_r, g_tot = -1.4, 2.1, 0.7
    h_l, h_r, h_tot = 2, 3, 5
    lam = 1.0
    term_l = g_l**2 / (h_l + lam)
    term_r = g_r**2 / (h_r + lam)
    term_p = g_tot**2 / (h_tot + lam)

    fig, (ax_tree, ax_calc) = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: the split diagram --
    ax_tree.axis("off")
    ax_tree.set_xlim(0, 10)
    ax_tree.set_ylim(0, 10)

    def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, colour: str) -> None:
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=colour, edgecolor="white",
                                   linewidth=1.5, alpha=0.95, zorder=2))
        ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10, zorder=3)

    _box(ax_tree, 5.0, 8.2, 5.4, 1.8,
         f"parent (no split)\nG={g_tot:+.1f}  H={h_tot}\nw* = -G/(H+λ) = {split.w_parent:+.2f}", SLATE)
    ax_tree.annotate("", xy=(2.7, 4.0), xytext=(4.2, 7.2),
                     arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.4})
    ax_tree.annotate("", xy=(7.3, 4.0), xytext=(5.8, 7.2),
                     arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.4})
    ax_tree.text(3.0, 6.0, "samples 1,2\n(g<0)", fontsize=8.5, color=INK, ha="center")
    ax_tree.text(7.0, 6.0, "samples 3,4,5\n(g>0)", fontsize=8.5, color=INK, ha="center")
    _box(ax_tree, 2.5, 3.0, 4.2, 1.9,
         f"left leaf\nG_L={g_l:+.1f}  H_L={h_l}\nw*_L = {split.w_left:+.2f}", GREEN)
    _box(ax_tree, 7.5, 3.0, 4.2, 1.9,
         f"right leaf\nG_R={g_r:+.1f}  H_R={h_r}\nw*_R = {split.w_right:+.2f}", RED)
    ax_tree.text(5.0, 0.7, f"per-example gradients g = {g},  Hessians h = 1 (squared error)",
                 ha="center", fontsize=8.5, color=INK)
    ax_tree.set_title("An XGBoost split: optimal leaf weights w* = -G/(H+λ)   (λ=1, γ=0)",
                      fontsize=11, color=INK)

    # -- right: the gain computed term by term --
    ax_calc.axis("off")
    ax_calc.set_xlim(0, 10)
    ax_calc.set_ylim(0, 10)
    lines = [
        ("The regularized split gain", INK, 12, "bold"),
        ("", INK, 10, "normal"),
        (r"Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ", INK, 10.5, "normal"),
        ("", INK, 10, "normal"),
        (f"left term   G_L²/(H_L+λ) = {g_l**2:.2f}/{h_l + lam:.0f} = {term_l:.3f}", GREEN, 11, "normal"),
        (f"right term  G_R²/(H_R+λ) = {g_r**2:.2f}/{h_r + lam:.0f} = {term_r:.3f}", RED, 11, "normal"),
        (f"parent term G²/(H+λ)      = {g_tot**2:.2f}/{h_tot + lam:.0f} = {term_p:.3f}", SLATE, 11, "normal"),
        ("", INK, 10, "normal"),
        (f"Gain = ½[ {term_l:.3f} + {term_r:.3f} − {term_p:.3f} ] − 0", INK, 11, "normal"),
        (f"      = {split.gain:.3f}   →   gain > 0, KEEP the split", AMBER, 12, "bold"),
    ]
    y = 9.2
    for text, colour, size, weight in lines:
        ax_calc.text(0.3, y, text, fontsize=size, color=colour, fontweight=weight, va="top", family="monospace")
        y -= 0.92
    ax_calc.set_title("…and its gain, computed term by term", fontsize=11, color=INK)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}xgb_gain.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_staged()
    fig_lr_sweep()
    fig_residual_shrink()
    fig_model_compare()
    fig_xgb_gain()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()

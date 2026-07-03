"""Figure generator for 11-Data-Leakage — every number from the REAL leakage experiments.

All figures come from the same real pipelines the chapter and notebook use (``data_leakage.py``): the
feature-selection preprocessing leak on controlled noise (honest truth = chance), the target leak on the
real Breast Cancer dataset, and the temporal leak on a realistic daily series. Nothing is hand-typed;
every bar, curve, and annotation is read off an executed function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``dataprep11_``:

  dataprep11_leaky_vs_honest.png    -- the MONEY figure: the leaky CV score vs the honest Pipeline score
                                       vs an untouched hold-out, on pure-noise data whose true accuracy
                                       is chance. The inflation gap, measured.
  dataprep11_where_leakage_enters.png -- a schematic of WHERE leakage enters cross-validation: the leaky
                                       path fits a transform on ALL rows (including the validation fold);
                                       the correct path fits it inside each training fold only.
  dataprep11_selection_k_sweep.png  -- the leak as a function of how much you let in: leaky CV climbs
                                       toward 1.0 as more features are cherry-picked from all the data,
                                       while the honest curve stays pinned at chance.
  dataprep11_target_leak.png        -- target leakage on REAL Breast Cancer data: a proxy-of-the-label
                                       column takes accuracy to ~1.00 (and predicts the label almost
                                       perfectly on its own); drop it and the honest ~0.96 returns.
  dataprep11_temporal_leak.png      -- temporal leakage: a random (shuffled) split trains on the future
                                       and inflates R^2; a forward TimeSeriesSplit reports the honest R^2.

    python make_figures_11.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``02. Data_Preprocessing/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``data_leakage`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "11-Data-Leakage" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from data_leakage import (  # noqa: E402  (resolved via the sys.path insert above)
    CHANCE,
    LEAK_COL_NAME,
    CV_FOLDS,
    make_breast_cancer_leak,
    make_noise_data,
    make_time_series,
    selection_leak,
    selection_leak_sweep,
    target_leak,
    temporal_leak,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # honest / train / data
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # correct / honest / good
RED = "#8B3B4A"  # leaky / inflated / contamination
SLATE = "#4A5B6E"  # neutral / baseline
AMBER = "#7A6528"  # validation fold / highlight / chance line
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "dataprep11_"


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


# ============================ Fig 1: the money figure (leaky vs honest) =========================
def fig_leaky_vs_honest() -> None:
    """The headline: a leaky protocol inflates a score that is really chance; the Pipeline fix collapses it."""
    sel = selection_leak(make_noise_data())
    labels = [
        "LEAKY CV\n(select on all data,\nthen cross-validate)",
        "HONEST CV\n(select inside a\nPipeline, per fold)",
        "HONEST hold-out\n(untouched\ntest set)",
    ]
    values = [sel.leaky_cv, sel.honest_cv, sel.honest_holdout]
    colours = [RED, GREEN, BLUE]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bars = ax.bar(labels, values, color=colours, alpha=0.9, width=0.62)
    ax.axhline(CHANCE, color=AMBER, linestyle="--", linewidth=1.8, zorder=1,
               label=f"honest truth = chance ({CHANCE:.2f})  —  the data is pure noise")
    # value labels drawn AFTER the chance line, each on a white patch so the dashed line never
    # strikes through a digit (the 0.48 / 0.53 bars sit right at the 0.50 line)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.2f}", ha="center", fontsize=13,
                fontweight="bold", color=INK, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title(
        "The same model, two protocols: leakage inflates a score that is really chance\n"
        f"pure-noise features, random label — any accuracy above {CHANCE:.2f} is leakage, measured",
        fontsize=12.5,
    )
    # the inflation arrow
    ax.annotate(
        "", xy=(0, sel.leaky_cv), xytext=(0, sel.honest_cv),
        arrowprops={"arrowstyle": "<->", "color": RED, "linewidth": 2.0},
    )
    ax.text(0.42, (sel.leaky_cv + sel.honest_cv) / 2,
            f"inflation\n+{sel.gap * 100:.0f} points\nof pure fiction", color=RED, fontsize=11,
            fontweight="bold", va="center", ha="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}leaky_vs_honest.png")


# ============================ Fig 2: where leakage enters cross-validation =======================
def _draw_folds(ax: plt.Axes, *, leaky: bool) -> None:
    """Draw K stacked CV folds; shade the fold's validation block amber, train blocks blue.

    In the leaky panel a red 'fit on ALL rows' band spans the full width (including every validation
    block) — the transform has seen the data it will be scored on. In the correct panel a green 'fit'
    marker sits over each fold's TRAIN blocks only.
    """
    n = CV_FOLDS
    for i in range(n):
        y = n - 1 - i
        val_start = i / n
        # train blocks (blue) and the validation block (amber)
        ax.add_patch(Rectangle((0, y + 0.12), 1.0, 0.5, facecolor=BLUE, alpha=0.35, edgecolor="white"))
        ax.add_patch(Rectangle((val_start, y + 0.12), 1 / n, 0.5, facecolor=AMBER, alpha=0.95,
                               edgecolor="white"))
        ax.text(-0.02, y + 0.37, f"fold {i + 1}", ha="right", va="center", fontsize=9, color=INK)
    ax.text(1 / (2 * n), n + 0.02, "val", ha="center", fontsize=8.5, color=AMBER, fontweight="bold")
    ax.text(0.62, n + 0.02, "train", ha="center", fontsize=8.5, color=BLUE, fontweight="bold")

    if leaky:
        ax.add_patch(Rectangle((0, -0.55), 1.0, 0.4, facecolor=RED, alpha=0.85, edgecolor="none"))
        ax.text(0.5, -0.35, "transform / feature-selection FIT on ALL rows (incl. every 'val' block)",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        for i in range(n):
            y = n - 1 - i
            xc = i / n + 1 / (2 * n)
            ax.annotate("", xy=(xc, y + 0.14), xytext=(xc, -0.15),
                        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.3, "alpha": 0.8})
        ax.set_title("LEAKY: fit the transform BEFORE the split\nit sees the validation rows → optimistic",
                     fontsize=11, color=RED)
    else:
        for i in range(n):
            y = n - 1 - i
            # a green 'fit here' marker over the train blocks of this fold (everything except val)
            ax.add_patch(Rectangle((0, y + 0.66), 1.0, 0.14, facecolor="white", edgecolor="none"))
            val_start = i / n
            if val_start > 0:
                ax.add_patch(Rectangle((0, y + 0.66), val_start, 0.12, facecolor=GREEN, alpha=0.85))
            if val_start + 1 / n < 1:
                ax.add_patch(Rectangle((val_start + 1 / n, y + 0.66), 1 - val_start - 1 / n, 0.12,
                                       facecolor=GREEN, alpha=0.85))
        ax.text(0.5, n + 0.62, "green = transform fit on THIS fold's train only", ha="center",
                fontsize=8.5, color=GREEN, fontweight="bold")
        ax.set_title("CORRECT: fit the transform INSIDE each fold\nit never sees 'val' → honest",
                     fontsize=11, color=GREEN)

    ax.set_xlim(-0.14, 1.02)
    ax.set_ylim(-0.65, n + 0.95)
    ax.axis("off")


def fig_where_leakage_enters() -> None:
    """Schematic: the transform fit before the split (leaky) vs inside each fold (correct)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    _draw_folds(ax1, leaky=True)
    _draw_folds(ax2, leaky=False)
    fig.suptitle(
        "Where leakage enters cross-validation — and how a Pipeline shuts it out",
        fontsize=13, color=INK,
    )
    handles = [
        Patch(facecolor=BLUE, alpha=0.35, label="training rows"),
        Patch(facecolor=AMBER, alpha=0.95, label="validation rows (scored)"),
        Patch(facecolor=RED, alpha=0.85, label="leaky fit (sees everything)"),
        Patch(facecolor=GREEN, alpha=0.85, label="correct fit (train only)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    _save(fig, f"{IMG_PREFIX}where_leakage_enters.png")


# ============================ Fig 3: the leak grows with k =======================================
def fig_selection_k_sweep() -> None:
    """Leaky CV climbs toward 1.0 as more features are cherry-picked; honest CV stays at chance."""
    ks, leaky, honest = selection_leak_sweep(make_noise_data())
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(ks, leaky, "o-", color=RED, linewidth=2.4, markersize=7,
            label="LEAKY: select on all data, then CV")
    ax.plot(ks, honest, "s-", color=GREEN, linewidth=2.4, markersize=7,
            label="HONEST: select inside the Pipeline (per fold)")
    ax.axhline(CHANCE, color=AMBER, linestyle="--", linewidth=1.6, label=f"chance ({CHANCE:.2f})")
    ax.fill_between(ks, honest, leaky, color=RED, alpha=0.08)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k  =  number of noise features cherry-picked (log scale)")
    ax.set_ylabel("cross-validated accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(
        "The more you let the selector peek, the bigger the lie\n"
        "leaky accuracy climbs toward 1.0 on PURE NOISE; the honest curve never leaves chance",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=10, loc="center left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}selection_k_sweep.png")


# ============================ Fig 4: target leakage on real data =================================
def fig_target_leak() -> None:
    """A proxy-of-the-label column inflates a real-data model to ~1.00; the leaked column IS the answer."""
    bundle = make_breast_cancer_leak()
    tgt = target_leak(bundle)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: the accuracy bars --
    labels = ["30 REAL\nfeatures only\n(honest)", "REAL + leaked\ncolumn\n(inflated)",
              "LEAKED column\nALONE"]
    values = [tgt.acc_without_leak, tgt.acc_with_leak, tgt.acc_leak_only]
    colours = [GREEN, RED, SLATE]
    bars = ax1.bar(labels, values, color=colours, alpha=0.9, width=0.62)
    for b, v in zip(bars, values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=12,
                 fontweight="bold", color=INK)
    ax1.set_ylim(0.5, 1.03)
    ax1.set_ylabel("cross-validated accuracy")
    ax1.set_title(
        "Target leakage on REAL Breast Cancer data\n"
        f"one post-diagnosis proxy column takes accuracy to {tgt.acc_with_leak:.2f} "
        f"and predicts the label {tgt.acc_leak_only:.2f} on its own",
        fontsize=11,
    )
    _style_axis(ax1)

    # -- right: the leaked column is the label in disguise (two separated class humps) --
    leak_vals = bundle.x_leaky[:, bundle.leak_col]
    for cls, colour, name in [(0, RED, "class 0"), (1, BLUE, "class 1")]:
        ax2.hist(leak_vals[bundle.y == cls], bins=30, color=colour, alpha=0.65, edgecolor="white",
                 linewidth=0.4, label=name)
    ax2.set_xlabel(f"value of '{LEAK_COL_NAME}'")
    ax2.set_ylabel("count (patients)")
    ax2.set_title(
        f"The leaked column split by label\n|corr with label| = {tgt.leak_corr:.2f} — it is a near-copy of y",
        fontsize=11,
    )
    ax2.legend(frameon=False, fontsize=10)
    _style_axis(ax2)

    fig.suptitle("A feature that encodes the answer: great in the notebook, absent in production",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, f"{IMG_PREFIX}target_leak.png")


# ============================ Fig 5: temporal leakage ===========================================
def fig_temporal_leak() -> None:
    """A random split trains on the future (inflated R^2); a forward TimeSeriesSplit is honest."""
    ts = make_time_series()
    tmp = temporal_leak(ts)
    n = ts.series.shape[0]
    cut = int(n * 0.8)
    rng = np.random.default_rng(0)
    shuffled_test = rng.choice(n, size=int(n * 0.2), replace=False)  # illustrate interleaving

    fig = plt.figure(figsize=(14, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1, 1], hspace=0.55, wspace=0.28,
                          top=0.82, bottom=0.10, left=0.06, right=0.98)
    ax_series = fig.add_subplot(gs[0, 0])
    ax_shuf = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[:, 1])

    # -- top-left: the forward (honest) split --
    ax_series.plot(np.arange(n), ts.series, color=INK, linewidth=0.9)
    ax_series.axvspan(0, cut, color=BLUE, alpha=0.14)
    ax_series.axvspan(cut, n, color=AMBER, alpha=0.20)
    ax_series.axvline(cut, color=INK, linestyle=":", linewidth=1.2)
    # add headroom above the series so the captions sit ABOVE the line, not over it
    _ymin, _ymax = ts.series.min(), ts.series.max()
    ax_series.set_ylim(_ymin - 0.05 * (_ymax - _ymin), _ymax + 0.30 * (_ymax - _ymin))
    _label_y = _ymax + 0.14 * (_ymax - _ymin)
    ax_series.text(cut / 2, _label_y, "train = the PAST", ha="center", va="center", color=BLUE,
                   fontsize=10, fontweight="bold")
    ax_series.text((cut + n) / 2, _label_y, "test = the FUTURE", ha="center", va="center",
                   color=AMBER, fontsize=10, fontweight="bold")
    ax_series.set_title("Forward TimeSeriesSplit: train on the past, predict the future (HONEST)",
                        fontsize=10.5, color=GREEN)
    ax_series.set_ylabel("value")
    _style_axis(ax_series)

    # -- bottom-left: the shuffled (leaky) split --
    ax_shuf.plot(np.arange(n), ts.series, color=INK, linewidth=0.6, alpha=0.5)
    mask = np.zeros(n, dtype=bool)
    mask[shuffled_test] = True
    ax_shuf.scatter(np.arange(n)[~mask], ts.series[~mask], s=4, color=BLUE, alpha=0.5, label="train")
    ax_shuf.scatter(np.arange(n)[mask], ts.series[mask], s=10, color=RED, alpha=0.9, label="test")
    ax_shuf.set_title("Shuffled split: test days sit BETWEEN training days — the model reads the future (LEAKY)",
                      fontsize=10.5, color=RED)
    ax_shuf.set_xlabel("day")
    ax_shuf.set_ylabel("value")
    ax_shuf.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    _style_axis(ax_shuf)

    # -- right: the R^2 bars --
    labels = ["shuffled\nK-fold\n(leaky)", "forward\nTimeSeriesSplit\n(honest)"]
    values = [tmp.shuffled_r2, tmp.forward_r2]
    colours = [RED, GREEN]
    bars = ax_bar.bar(labels, values, color=colours, alpha=0.9, width=0.6)
    for b, v in zip(bars, values):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=13,
                    fontweight="bold", color=INK)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("cross-validated R²")
    ax_bar.set_title(f"Same model, same data, only the split differs\ninflation: +{tmp.gap:.2f} R²",
                     fontsize=10.5)
    _style_axis(ax_bar)

    fig.suptitle(f"Temporal leakage: shuffling {n} time-ordered days inflates R² from "
                 f"{tmp.forward_r2:.2f} to {tmp.shuffled_r2:.2f}", fontsize=12.5, color=INK, y=0.97)
    _save(fig, f"{IMG_PREFIX}temporal_leak.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_leaky_vs_honest()
    fig_where_leakage_enters()
    fig_selection_k_sweep()
    fig_target_leak()
    fig_temporal_leak()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()

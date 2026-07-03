"""Figure generator for 02-Feature-Scaling-and-Normalization — every number from the REAL Wine study.

All figures come from the same real pipeline the chapter and notebook use (``feature_scaling.py`` on
the scikit-learn **Wine** dataset): the measured distance decomposition, the from-scratch (and
sklearn-verified) scalers, the measured model-accuracy table, and the from-scratch gradient-descent
conditioning demo. Nothing is hand-typed; every bar, curve, and annotation is read off an executed
function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``dataprep02_``:

  dataprep02_scale_disparity.png  -- the PROBLEM: the 13 feature ranges spanning ~2,500x, and the share
                                     of Euclidean distance each feature owns — proline ~99.7% raw, ~1/13
                                     after standardizing.
  dataprep02_scalers_on_feature.png -- a real right-skewed feature (magnesium) under standard / minmax /
                                     robust: what each does to the distribution and its outliers.
  dataprep02_model_scores.png     -- the MONEY figure: test accuracy of KNN, RBF-SVM, LogReg, and a
                                     random forest without vs with each scaler. The distance/gradient
                                     models leap; the forest doesn't move.
  dataprep02_knn_neighborhood.png -- two real features: an unscaled nearest-neighbour query grabs a
                                     horizontal slab (wrong-class neighbours); scaled, it grabs a proper
                                     local circle (right-class neighbours).
  dataprep02_gd_convergence.png   -- from-scratch gradient descent: the same step size diverges on raw
                                     features (condition number ~10^5) and converges on scaled ones (~3).

    python make_figures_02.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``02. Data_Preprocessing/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``feature_scaling`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "02-Feature-Scaling-and-Normalization" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from feature_scaling import (  # noqa: E402  (resolved via the sys.path insert above)
    GD_FEATURES,
    SKEWED_FEATURE,
    MinMaxScalerScratch,
    RobustScalerScratch,
    StandardScalerScratch,
    distance_share,
    evaluate_models,
    feature_ranges,
    gd_conditioning,
    load_wine_split,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # standardization / data
PURPLE = "#5D4A8A"  # robust / process
GREEN = "#2E7A5A"  # minmax / good / correct
RED = "#8B3B4A"  # unscaled / cost / diverges
SLATE = "#4A5B6E"  # neutral / baseline (no scaling)
AMBER = "#7A6528"  # highlight / floor
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "dataprep02_"
SCALER_COLOURS = {"none": SLATE, "standard": BLUE, "minmax": GREEN, "robust": PURPLE}


def _skewness(v: np.ndarray) -> float:
    """Fisher-Pearson skewness of a 1-D array (positive = a long right tail) — no scipy needed."""
    c = v - v.mean()
    m2 = float(np.mean(c**2))
    m3 = float(np.mean(c**3))
    return m3 / m2**1.5 if m2 > 0 else 0.0


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


# ============================ Fig 1: the problem (scale disparity + distance share) ==============
def fig_scale_disparity() -> None:
    """The two faces of the problem: feature ranges span ~2,500x, and one feature owns the distance."""
    split = load_wine_split()
    ranges = feature_ranges(split.x_train)
    share = distance_share(split.x_train, split.feature_names)
    order = np.argsort(ranges)
    names = [split.feature_names[i] for i in order]
    ratio = ranges.max() / ranges.min()
    top = int(np.argmax(share.raw_share))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    # -- left: feature ranges on a log axis --
    colours = [RED if i == top else SLATE for i in order]
    ax1.barh(range(len(names)), ranges[order], color=colours, alpha=0.9)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("feature range  (max - min, log scale)")
    ax1.set_title(
        f"The 13 Wine features live on wildly different scales\n"
        f"largest range is {ratio:,.0f}x the smallest — '{split.feature_names[top]}' towers over all",
        fontsize=11,
    )
    ax1.annotate(f"{split.feature_names[top]}\nrange ≈ {ranges[top]:,.0f}",
                 xy=(ranges[top], len(names) - 1), xytext=(ranges[top] * 0.12, len(names) - 3.4),
                 fontsize=9, color=RED, arrowprops={"arrowstyle": "->", "color": RED})
    _style_axis(ax1)

    # -- right: share of Euclidean distance, raw vs standardized --
    idx = np.argsort(share.raw_share)[::-1]
    labels = [split.feature_names[i] for i in idx]
    x = np.arange(len(labels))
    w = 0.4
    ax2.bar(x - w / 2, share.raw_share[idx] * 100, w, color=RED, alpha=0.9, label="raw features")
    ax2.bar(x + w / 2, share.scaled_share[idx] * 100, w, color=BLUE, alpha=0.9, label="after standardizing")
    ax2.axhline(100 / len(labels), color=AMBER, linestyle=":", linewidth=1.6,
                label=f"a fair share = 1/13 = {100 / len(labels):.1f}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
    ax2.set_ylabel("share of average squared Euclidean distance (%)")
    ax2.set_title(
        f"Unscaled, '{split.feature_names[top]}' alone is {share.raw_share[top] * 100:.1f}% of the distance\n"
        f"after standardizing it is {share.scaled_share[top] * 100:.1f}% — every feature gets a fair vote",
        fontsize=11,
    )
    ax2.legend(frameon=False, fontsize=9)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}scale_disparity.png")


# ============================ Fig 2: the three scalers on a real skewed feature =================
def fig_scalers_on_feature() -> None:
    """A real right-skewed feature under standard / minmax / robust — what each does to shape & outliers."""
    split = load_wine_split()
    j = split.feature_names.index(SKEWED_FEATURE)
    col = split.x_train[:, j : j + 1]

    views = [
        ("original", col.ravel(), SLATE, "raw units"),
        ("standardized  (x−μ)/σ", StandardScalerScratch.fit(col).transform(col).ravel(), BLUE, "mean 0, std 1"),
        ("min-max  (x−min)/(max−min)", MinMaxScalerScratch.fit(col).transform(col).ravel(), GREEN, "bounded to [0, 1]"),
        ("robust  (x−median)/IQR", RobustScalerScratch.fit(col).transform(col).ravel(), PURPLE, "median 0, IQR 1"),
    ]
    sk = _skewness(col.ravel())

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (title, vals, colour, sub) in zip(axes, views):
        ax.hist(vals, bins=22, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
        hi = vals.max()
        ax.axvline(hi, color=RED, linestyle="--", linewidth=1.4)
        ax.annotate("outlier", xy=(hi, 1), xytext=(hi, ax.get_ylim()[1] * 0.7),
                    fontsize=8, color=RED, ha="right", rotation=90)
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, color=colour)
        ax.set_xlabel("value")
        _style_axis(ax)
    axes[0].set_ylabel("count (training wines)")
    fig.suptitle(
        f"The same real feature ('{SKEWED_FEATURE}', right-skewed, skew ≈ {sk:.2f}) under three scalers — "
        "min-max lets the outlier squash the bulk; robust resists it",
        fontsize=12, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, f"{IMG_PREFIX}scalers_on_feature.png")


# ============================ Fig 3: the measured model-accuracy effect (money) =================
def fig_model_scores() -> None:
    """Test accuracy of four models without vs with each scaler — the distance/gradient models leap."""
    split = load_wine_split()
    scores = evaluate_models(split)
    models = scores.model_names
    scalers = scores.scaler_names
    x = np.arange(len(models))
    w = 0.2

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for k, scaler in enumerate(scalers):
        heights = [scores.accuracy[m][k] for m in models]
        bars = ax.bar(x + (k - 1.5) * w, heights, w, color=SCALER_COLOURS[scaler], alpha=0.9,
                      label=f"{scaler}" + (" (no scaling)" if scaler == "none" else ""))
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.006, f"{h:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.55, 1.04)
    knn = scores.accuracy["KNN"]
    svm = scores.accuracy["SVM-RBF"]
    ax.set_title(
        "Scaling's measured effect on real Wine models\n"
        f"KNN {knn[0]:.2f} → {max(knn):.2f}  and  SVM-RBF {svm[0]:.2f} → {max(svm):.2f} with scaling; "
        "the random forest never moves",
        fontsize=11.5,
    )
    ax.annotate("random forest:\nsame every column\n(threshold splits are\nscale-invariant)",
                xy=(3, 1.0), xytext=(2.55, 0.66), fontsize=8.5, color=GREEN,
                arrowprops={"arrowstyle": "->", "color": GREEN})
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="lower center")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}model_scores.png")


# ============================ Fig 4: the kNN neighbourhood, distorted by scale ===================
def _knn_indices(points: np.ndarray, query: int, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of ``points[query]`` by Euclidean distance (excluding itself)."""
    d = np.sqrt(((points - points[query]) ** 2).sum(axis=1))
    return np.argsort(d)[1 : k + 1]


def fig_knn_neighborhood() -> None:
    """The same query point's 5 nearest neighbours: a horizontal slab on raw axes, a circle when scaled."""
    split = load_wine_split()
    x = np.vstack([split.x_train, split.x_test])
    y = np.concatenate([split.y_train, split.y_test])
    fx, fy = GD_FEATURES  # ('proline', 'flavanoids')
    jx = split.feature_names.index(fx)
    jy = split.feature_names.index(fy)
    pts_raw = x[:, [jy, jx]]  # plot flavanoids (x) vs proline (y): proline is the large axis
    z = StandardScalerScratch.fit(x).transform(x)
    pts_scaled = z[:, [jy, jx]]

    k = 5
    query = int(np.argmin(np.abs(x[:, jx] - np.median(x[:, jx]))))  # a central wine
    nn_raw = _knn_indices(pts_raw, query, k)
    nn_scaled = _knn_indices(pts_scaled, query, k)
    class_colours = [BLUE, GREEN, PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, pts, nn, title, sub in [
        (ax1, pts_raw, nn_raw, "RAW features: nearest neighbours span a horizontal slab",
         "proline (~100s–1000s) dwarfs flavanoids (~0–5): 'nearest' ≈ 'nearest in proline'"),
        (ax2, pts_scaled, nn_scaled, "STANDARDIZED: neighbours form a proper local circle",
         "both features on a common ruler: 'nearest' is genuinely nearest"),
    ]:
        for c in range(3):
            m = y == c
            ax.scatter(pts[m, 0], pts[m, 1], s=26, color=class_colours[c], alpha=0.35, label=f"cultivar {c}")
        ax.scatter(pts[nn, 0], pts[nn, 1], s=150, facecolors="none", edgecolors=RED, linewidths=2.0, zorder=5,
                   label="5 nearest neighbours")
        ax.scatter(pts[query, 0], pts[query, 1], s=180, color=AMBER, marker="*", edgecolor="black",
                   zorder=6, label="query wine")
        n_wrong = int(np.sum(y[nn] != y[query]))
        ax.set_title(f"{title}\n{sub}\nneighbours from a different cultivar than the query: {n_wrong} of {k}",
                     fontsize=9.8)
        ax.set_xlabel(fy)
        ax.set_ylabel(fx)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style_axis(ax)
    fig.suptitle("Unscaled axes distort who is 'nearest' — the root cause of KNN's accuracy jump",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, f"{IMG_PREFIX}knn_neighborhood.png")


# ============================ Fig 5: gradient-descent conditioning ===============================
def fig_gd_convergence() -> None:
    """Same learning rate: raw features diverge (condition ~10^5), scaled converge (condition ~3)."""
    split = load_wine_split()
    demo = gd_conditioning(split)
    iters = np.arange(len(demo.loss_raw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})

    # -- left: loss vs iteration --
    ax1.plot(iters, demo.loss_raw, color=RED, linewidth=2.4, label=f"raw features (cond ≈ {demo.cond_raw:,.0f})")
    ax1.plot(iters, demo.loss_scaled, color=GREEN, linewidth=2.4, label=f"standardized (cond ≈ {demo.cond_scaled:.1f})")
    ax1.set_yscale("log")
    ax1.set_xlabel("gradient-descent iteration")
    ax1.set_ylabel("training loss (log scale)")
    ax1.set_title(
        f"Same optimizer, same learning rate ({demo.lr}), two features\n"
        "raw: the step overshoots the razor-thin valley and diverges; scaled: smooth descent",
        fontsize=10.5,
    )
    ax1.legend(frameon=False, fontsize=9.5, loc="center right")
    _style_axis(ax1)

    # -- right: the curvature ellipses (contours of the quadratic loss approximation) --
    for label, cov, colour in [
        (f"scaled\ncond ≈ {demo.cond_scaled:.1f}", np.cov(demo.x_scaled.T), GREEN),
        (f"raw\ncond ≈ {demo.cond_raw:,.0f}", np.cov(demo.x_raw.T), RED),
    ]:
        vals, vecs = np.linalg.eigh(cov)
        vals = vals / vals.max()  # normalise longest axis to 1 so shape (not size) is the message
        theta = np.linspace(0, 2 * np.pi, 200)
        # ellipse whose axis lengths are 1/sqrt(curvature) ~ sqrt(eigenvalue): elongated = ill-conditioned
        radii = np.sqrt(vals)
        unit = np.vstack([np.cos(theta), np.sin(theta)])
        ell = (vecs @ (radii[:, None] * unit))
        ax2.plot(ell[0], ell[1], color=colour, linewidth=2.4, label=label)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("weight direction 1")
    ax2.set_ylabel("weight direction 2")
    ax2.set_title("loss contours\n(circular = easy, thin = hard)", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}gd_convergence.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_scale_disparity()
    fig_scalers_on_feature()
    fig_model_scores()
    fig_knn_neighborhood()
    fig_gd_convergence()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()

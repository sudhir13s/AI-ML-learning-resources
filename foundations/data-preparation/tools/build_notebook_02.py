"""Generate the step-by-step teaching notebook (02-Feature-Scaling-and-Normalization.ipynb).

The notebook mirrors ``feature_scaling.py`` one measurement at a time, so a learner can open it, run
every cell live, and *see* scaling change real models on the real Wine dataset — the scale disparity,
the distance decomposition, the three scalers (verified against scikit-learn), the measured
model-accuracy table, the KNN neighbourhood distortion, and the gradient-descent conditioning story.
Each numbered step has a short markdown lead-in (the intuition) followed by ONE focused code cell with
real output. This generator writes the .ipynb; a separate nbconvert pass executes it headless so the
outputs are embedded.

    python build_notebook_02.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../02-Feature-Scaling-and-Normalization/code/02-Feature-Scaling-and-Normalization.ipynb"

This generator lives in the domain-level ``02. Data_Preprocessing/tools/`` folder; the notebook it
writes (and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a
hand-edited .ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "02-Feature-Scaling-and-Normalization" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

from feature_scaling import SKEWED_FEATURE  # noqa: E402  (resolved via the sys.path insert above)

NB_PATH = _CHAPTER_CODE / "02-Feature-Scaling-and-Normalization.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Feature Scaling & Normalization — a step-by-step, runnable notebook\n"
    "\n"
    "A model that measures *distance* or learns by *gradient descent* reads your features in whatever "
    "**units** you happen to give it. If one feature is numbered in the thousands and another in the "
    "tenths, the big-numbered one silently dominates — not because it matters more, but because its units "
    "are bigger. **Feature scaling** puts every feature on a common ruler so the model judges them on "
    "merit, not magnitude.\n"
    "\n"
    "This notebook shows that, one measurement at a time, on the **real Wine dataset** (178 wines, 13 "
    "chemical measurements, 3 cultivars — it ships with scikit-learn, no download). It is the executable "
    "companion to the chapter and to `feature_scaling.py`; every function used here lives in that module, "
    "imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **measured**, not just been told:\n"
    "\n"
    "1. that one feature (`proline`) owns **~99.7%** of the raw Euclidean distance — and a fair ~1/13 after scaling;\n"
    "2. the three scalers (**standard**, **min-max**, **robust**) built from scratch and **matched to scikit-learn**;\n"
    "3. scaling lifting **KNN 0.72 → 0.96** and **SVM 0.67 → 1.00**, while a random forest **never moves**;\n"
    "4. *why* — an unscaled nearest-neighbour query grabs the wrong neighbours, and gradient descent diverges;\n"
    "5. why you **fit the scaler on the training split only**.\n"
    "\n"
    "Everything runs on CPU in a couple of seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real functions from the chapter module (so this notebook uses the *exact same code* the "
    "figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "\n"
    "from feature_scaling import (\n"
    "    load_wine_split, feature_ranges, distance_share,\n"
    "    StandardScalerScratch, MinMaxScalerScratch, RobustScalerScratch,\n"
    "    scaler_match_report, evaluate_models, gd_conditioning, logreg_iterations,\n"
    "    leakage_probe, SKEWED_FEATURE, MATCH_TOL,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__}')\n"
    "split = load_wine_split()\n"
    "print(f'Wine: {split.x_train.shape[0]} train + {split.x_test.shape[0]} test wines, '\n"
    "      f'{len(split.feature_names)} features, {len(np.unique(split.y_train))} cultivars')"
)

# ---- Step 1: the scale disparity ----
add_md(
    "## Step 1 — The problem: features on wildly different scales\n"
    "\n"
    "Look at the **range** (max − min) of each Wine feature. `proline` (a concentration, in the hundreds "
    "to thousands) spans thousands of units; `nonflavanoid_phenols` spans about half a unit. That is a "
    "**~2,500×** gap — and a distance-based model measures closeness in exactly these raw units."
)
add_code(
    "ranges = feature_ranges(split.x_train)\n"
    "order = np.argsort(ranges)[::-1]\n"
    "print(f'{\"feature\":<26}{\"range (max-min)\":>16}')\n"
    "for i in order:\n"
    "    print(f'{split.feature_names[i]:<26}{ranges[i]:>16.3f}')\n"
    "print(f'\\nlargest range / smallest range = {ranges.max() / ranges.min():,.0f}x')"
)

# ---- Step 2: distance decomposition ----
add_md(
    "## Step 2 — Quantify it: one feature owns the distance\n"
    "\n"
    "Euclidean distance² is a **sum over features** of $(x_i - x'_i)^2$. Averaged over all pairs of wines, "
    "each feature's contribution is proportional to its **variance**. So we can compute exactly what "
    "fraction of 'how far apart two wines are' each feature owns — before and after standardizing."
)
add_code(
    "share = distance_share(split.x_train, split.feature_names)\n"
    "top = int(np.argmax(share.raw_share))\n"
    "print(f'{\"feature\":<26}{\"raw share\":>12}{\"scaled share\":>14}')\n"
    "idx = np.argsort(share.raw_share)[::-1]\n"
    "for i in idx[:5]:\n"
    "    print(f'{split.feature_names[i]:<26}{share.raw_share[i]*100:>11.2f}%{share.scaled_share[i]*100:>13.2f}%')\n"
    "print(f'\\n{split.feature_names[top]!r}: {share.raw_share[top]*100:.1f}% of the raw distance '\n"
    "      f'-> {share.scaled_share[top]*100:.1f}% after standardizing (a fair 1/13 = {100/13:.1f}%).')\n"
    "print('Unscaled, \"nearest neighbour\" essentially means \"nearest in proline\". That is the disease.')"
)

# ---- Step 3: the three scalers from scratch ----
add_md(
    "## Step 3 — The three scalers, from scratch\n"
    "\n"
    "Three ways to put a feature on a common ruler — each fit on the **training split only**:\n"
    "\n"
    "* **standardize**: $z = (x-\\mu)/\\sigma$ → mean 0, std 1 (preserves shape; does not resist outliers);\n"
    "* **min-max**: $(x-\\min)/(\\max-\\min)$ → bounded to $[0,1]$ (an outlier fixes an endpoint);\n"
    "* **robust**: $(x-\\text{median})/\\text{IQR}$ → median 0, IQR 1 (outlier-resistant).\n"
    "\n"
    "Here are all three on the first few test values of `proline`."
)
add_code(
    "j = split.feature_names.index('proline')\n"
    "col_tr = split.x_train[:, j:j+1]\n"
    "col_te = split.x_test[:, j:j+1]\n"
    "std = StandardScalerScratch.fit(col_tr)\n"
    "mm = MinMaxScalerScratch.fit(col_tr)\n"
    "rb = RobustScalerScratch.fit(col_tr)\n"
    "print(f'proline fit stats (train): mean={std.mean_[0]:.1f} std={std.std_[0]:.1f} '\n"
    "      f'min={mm.min_[0]:.1f} range={mm.range_[0]:.1f} median={rb.median_[0]:.1f} IQR={rb.iqr_[0]:.1f}')\n"
    "print(f'{\"raw\":>10}{\"standard\":>12}{\"minmax\":>10}{\"robust\":>10}')\n"
    "for r, s, m, b in zip(col_te[:5,0], std.transform(col_te)[:5,0], mm.transform(col_te)[:5,0], rb.transform(col_te)[:5,0]):\n"
    "    print(f'{r:>10.1f}{s:>12.3f}{m:>10.3f}{b:>10.3f}')"
)

# ---- Step 4: match sklearn ----
add_md(
    "## Step 4 — Are they correct? Match against scikit-learn\n"
    "\n"
    "Our from-scratch scalers should reproduce scikit-learn's `StandardScaler` / `MinMaxScaler` / "
    "`RobustScaler` to machine precision when fed the identical data. If they do, the transforms above are "
    "the genuine article — not a lookalike."
)
add_code(
    "report = scaler_match_report(split)\n"
    "for name, diff in report.items():\n"
    "    print(f'{name:>9}: max|ours - sklearn| = {diff:.2e}  match: {bool(diff < MATCH_TOL)}')\n"
    "assert all(d < MATCH_TOL for d in report.values()), 'scalers must match scikit-learn'\n"
    "print('\\n=> all three match to 1e-9. The from-scratch transforms are exact.')"
)

# ---- Step 5: scalers on a skewed feature ----
add_md(
    "## Step 5 — See how they treat outliers: a real skewed feature\n"
    "\n"
    f"`{SKEWED_FEATURE}` is a real right-skewed Wine feature with high outliers. Watch what each scaler "
    "does — the *shape* is identical (scaling is an affine map), but **read the x-axis**: min-max lets the "
    "outlier pin the max at 1 and squashes the bulk into a narrow band near 0; robust centres the median "
    "at 0 with the bulk well-spread, leaving the outlier far out where it belongs."
)
add_code(
    "j = split.feature_names.index(SKEWED_FEATURE)\n"
    "col = split.x_train[:, j:j+1]\n"
    "views = [('original', col.ravel(), '#4A5B6E'),\n"
    "         ('standardized', StandardScalerScratch.fit(col).transform(col).ravel(), '#3A6B96'),\n"
    "         ('min-max', MinMaxScalerScratch.fit(col).transform(col).ravel(), '#2E7A5A'),\n"
    "         ('robust', RobustScalerScratch.fit(col).transform(col).ravel(), '#5D4A8A')]\n"
    "fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))\n"
    "for ax, (title, vals, colour) in zip(axes, views):\n"
    "    ax.hist(vals, bins=20, color=colour, alpha=0.85, edgecolor='white', lw=0.4)\n"
    "    ax.axvline(vals.max(), color='#8B3B4A', ls='--', lw=1.3)\n"
    "    ax.set_title(f'{title}\\n[{vals.min():.2f}, {vals.max():.2f}]', color=colour, fontsize=10)\n"
    "    ax.set_xlabel('value')\n"
    "    ax.grid(alpha=0.3)\n"
    "axes[0].set_ylabel('count')\n"
    "plt.suptitle(f'{SKEWED_FEATURE}: same shape, different rulers (dashed = the outlier)', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 6: the model table (money) ----
add_md(
    "## Step 6 — The money result: scaling changes real model accuracy\n"
    "\n"
    "Train four models on the same split, **without** scaling and **with** each scaler (fit on train "
    "only). Three are scale-sensitive (KNN and the RBF-SVM use distance; logistic regression is "
    "gradient-trained); the **random forest** splits on thresholds and should be immune. Watch the "
    "distance-based models leap."
)
add_code(
    "scores = evaluate_models(split)\n"
    "print(f'{\"model\":<13}' + ''.join(f'{s:>10}' for s in scores.scaler_names))\n"
    "for m in scores.model_names:\n"
    "    row = ''.join(f'{a:>10.3f}' for a in scores.accuracy[m])\n"
    "    tag = '   <- INVARIANT' if m == 'RandomForest' else ''\n"
    "    print(f'{m:<13}{row}{tag}')\n"
    "knn, svm = scores.accuracy['KNN'], scores.accuracy['SVM-RBF']\n"
    "print(f'\\nKNN:     {knn[0]:.3f} (none) -> {max(knn):.3f} (best scaler)  = +{(max(knn)-knn[0])*100:.0f} points')\n"
    "print(f'SVM-RBF: {svm[0]:.3f} (none) -> {max(svm):.3f} (best scaler)  = +{(max(svm)-svm[0])*100:.0f} points')"
)

# ---- Step 7: plot the money bars ----
add_md(
    "## Step 7 — The same result, as a picture\n"
    "\n"
    "One bar chart makes the story unmissable: the distance/gradient models jump when scaled; the random "
    "forest is a flat wall of equal bars."
)
add_code(
    "colours = {'none': '#4A5B6E', 'standard': '#3A6B96', 'minmax': '#2E7A5A', 'robust': '#5D4A8A'}\n"
    "x = np.arange(len(scores.model_names))\n"
    "w = 0.2\n"
    "plt.figure(figsize=(10, 5))\n"
    "for k, s in enumerate(scores.scaler_names):\n"
    "    h = [scores.accuracy[m][k] for m in scores.model_names]\n"
    "    plt.bar(x + (k-1.5)*w, h, w, color=colours[s], label=s)\n"
    "plt.xticks(x, scores.model_names)\n"
    "plt.ylim(0.55, 1.03)\n"
    "plt.ylabel('test accuracy')\n"
    "plt.title('scaling helps distance/gradient models; the forest never moves')\n"
    "plt.legend(ncol=4, fontsize=9)\n"
    "plt.grid(alpha=0.3, axis='y')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: RF invariance ----
add_md(
    "## Step 8 — Why the forest is immune: monotone-invariant splits\n"
    "\n"
    "A tree asks yes/no questions like `proline <= 755`. Any scaling is *monotone* — it preserves order — "
    "so the very same rows fall on each side of every threshold; only the number in the question changes. "
    "The tree structure, and therefore the predictions, are **identical**. We can check that directly: the "
    "forest's predictions are bit-for-bit the same under every scaler."
)
add_code(
    "print(f'random-forest predictions identical across all scalers: {scores.rf_predictions_identical}')\n"
    "print('scaling is monotone -> thresholds map 1:1 -> same splits -> same tree -> same predictions.')\n"
    "print('If your whole pipeline is trees/GBMs, scaling is optional. For everything else, it is not.')"
)

# ---- Step 9: GD conditioning (measure) ----
add_md(
    "## Step 9 — Why gradient descent needs it: conditioning\n"
    "\n"
    "Run the **same** from-scratch logistic-regression gradient descent (same learning rate) on two Wine "
    "features, raw vs standardized. The **condition number** of the feature covariance measures how "
    "*elongated* the loss surface's contours are. Raw, it is ~$10^5$: a razor-thin valley, where a step "
    "big enough to move along the valley overshoots across it and the loss **diverges**. Standardized, it "
    "is ~3, and the same step converges."
)
add_code(
    "demo = gd_conditioning(split)\n"
    "print(f'condition number of the loss surface: raw {demo.cond_raw:,.0f}   scaled {demo.cond_scaled:.1f}')\n"
    "print(f'from-scratch GD (lr={demo.lr}) final loss: raw {demo.loss_raw[-1]:.3f}   scaled {demo.loss_scaled[-1]:.3f}')\n"
    "print('same optimizer, same step size, same data -> only the feature scale changed the geometry.')"
)

# ---- Step 10: plot GD ----
add_md(
    "## Step 10 — The loss curves, plotted\n"
    "\n"
    "Raw features (red) send the loss oscillating high — the step overshoots the thin valley every "
    "iteration. Standardized features (green) descend smoothly. This is the optimization cost of skipping "
    "the scaler, made visible."
)
add_code(
    "it = np.arange(len(demo.loss_raw))\n"
    "plt.figure(figsize=(8.5, 4.6))\n"
    "plt.plot(it, demo.loss_raw, color='#8B3B4A', lw=2.2, label=f'raw (cond {demo.cond_raw:,.0f})')\n"
    "plt.plot(it, demo.loss_scaled, color='#2E7A5A', lw=2.2, label=f'standardized (cond {demo.cond_scaled:.1f})')\n"
    "plt.yscale('log')\n"
    "plt.xlabel('gradient-descent iteration')\n"
    "plt.ylabel('training loss (log)')\n"
    "plt.title('same learning rate: raw diverges, scaled converges')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 11: LogReg iterations ----
add_md(
    "## Step 11 — Even scikit-learn's solver pays the price\n"
    "\n"
    "This is not a from-scratch artefact. scikit-learn's own `LogisticRegression` (a good quasi-Newton "
    "solver) needs **thousands** of iterations to converge on unscaled Wine, and a **handful** once it is "
    "standardized. Scaling doesn't just help hand-rolled gradient descent — it conditions the problem for "
    "every gradient-based method."
)
add_code(
    "n_raw, n_scaled = logreg_iterations(split)\n"
    "print(f'LogisticRegression iterations to converge:  unscaled {n_raw:>6}   standardized {n_scaled:>4}')\n"
    "print(f'-> standardizing cut the iteration count by ~{n_raw / max(n_scaled, 1):,.0f}x.')"
)

# ---- Step 12: leakage preview ----
add_md(
    "## Step 12 — Fit on TRAIN only (a data-leakage preview)\n"
    "\n"
    "One rule ties scaling to honest evaluation: **fit the scaler on the training split only**, then apply "
    "it to test. If you fit on the whole dataset before splitting, the test set's own statistics leak into "
    "its transform — the model gets a peek at data it won't have in production. Here we *measure* the leak: "
    "how far the fitted statistics move, and how much a transformed test value changes. It is small for "
    "scaling — but the habit is non-negotiable, and the dramatic cases are the Data Leakage chapter."
)
add_code(
    "probe = leakage_probe(split)\n"
    "print(f'fitting on all data (vs train-only) shifts the mean by up to {probe.mean_shift.max():.3f}')\n"
    "print(f'and the std by up to {probe.std_shift.max():.3f} per feature')\n"
    "print(f'largest change in a transformed TEST value: {probe.max_test_transform_diff:.4f}')\n"
    "print('\\nSmall here, but always: split first, fit the scaler on train, then transform test.')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "On the real Wine dataset, one runnable notebook showed the whole of feature scaling:\n"
    "\n"
    "| Step | What we measured | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–2 | feature ranges; distance decomposition | one feature (`proline`) owns **~99.7%** of raw distance |\n"
    "| 3–4 | the three scalers, from scratch | standard / min-max / robust, **matched to scikit-learn** |\n"
    "| 5 | scalers on a skewed feature | min-max is outlier-sensitive; **robust resists outliers** |\n"
    "| 6–7 | model accuracy without vs with | **KNN 0.72→0.96, SVM 0.67→1.00**; the forest never moves |\n"
    "| 8 | random-forest invariance | monotone splits ⇒ **identical** predictions under any scaler |\n"
    "| 9–11 | gradient-descent conditioning | raw cond ~$10^5$ diverges; scaled ~3 converges (thousands→tens of iters) |\n"
    "| 12 | fit on train only | test statistics must not leak into the transform |\n"
    "\n"
    "**Scaling changes the ruler, not the information.** Distance-based models (KNN, SVM-RBF, k-means, "
    "PCA) and gradient-based models (linear/logistic regression, neural nets) *need* it; tree ensembles "
    "don't. Fit on the training split, apply everywhere, and put every feature on an even footing."
)


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NB_PATH} with {len(CELLS)} cells "
          f"({sum(c['cell_type'] == 'code' for c in CELLS)} code, "
          f"{sum(c['cell_type'] == 'markdown' for c in CELLS)} markdown)")


if __name__ == "__main__":
    main()

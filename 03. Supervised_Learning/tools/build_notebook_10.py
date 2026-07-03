"""Generate the step-by-step teaching notebook (10-Gradient-Boosting-XGBoost.ipynb).

The notebook mirrors ``gradient_boosting.py`` one measurement at a time, so a learner can open it, run every
cell live, and *see* how gradient boosting is grown on real data — the pseudo-residual = negative-gradient
identity, a from-scratch ensemble whose loss falls each round and matches scikit-learn, the staged
train/validation curve and its early-stopping round, the learning-rate x n_estimators trade, the 1-D
residual-shrinking movie, the XGBoost leaf-weight / split-gain worked example, a from-scratch log-loss
classifier that matches scikit-learn, the single-tree vs forest vs GBM vs XGBoost comparison, and a real
XGBoost model with early stopping. Each numbered step has a short markdown lead-in (the intuition) followed by
ONE focused code cell with real output. This generator writes the .ipynb; a separate nbconvert pass executes
it headless so the outputs are embedded.

    python build_notebook_10.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../10-Gradient-Boosting-XGBoost/code/10-Gradient-Boosting-XGBoost.ipynb"

This generator lives in the domain-level ``03. Supervised_Learning/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "10-Gradient-Boosting-XGBoost" / "code"
NB_PATH = _CHAPTER_CODE / "10-Gradient-Boosting-XGBoost.ipynb"

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
    "# Gradient Boosting (XGBoost) — a step-by-step, runnable notebook\n"
    "\n"
    "Gradient boosting builds shallow trees **sequentially**: each new tree fits the **residual error** the "
    "current ensemble still makes, and is added on with a small learning rate. Do that carefully and you get "
    "**XGBoost / LightGBM / CatBoost** — the reigning champions of tabular data. The deep idea, which we will "
    "*measure*, is that 'fit the residuals' is exactly **gradient descent in function space**: each tree "
    "approximates the negative gradient of the loss with respect to the current predictions.\n"
    "\n"
    "This notebook builds that, one measurement at a time, on **real scikit-learn datasets** (no manual "
    "download): **California Housing** (20,640 districts, seeded-subsampled for a fast from-scratch loop) for "
    "the regression story, and **Breast Cancer** (569 tumours) for the log-loss classification loop. It is the "
    "executable companion to the chapter and to `gradient_boosting.py`; every function used here lives in that "
    "module, imported so the notebook and the module can never drift apart. Real **XGBoost** is used where the "
    "chapter cites it.\n"
    "\n"
    "By the end you will have **measured**, not just been told:\n"
    "\n"
    "1. that the **pseudo-residual is the negative gradient** — `y - F` for squared error, `y - p` for log-loss;\n"
    "2. a gradient-boosting ensemble **grown from scratch** whose loss falls every round and **matches** scikit-learn;\n"
    "3. the **staged train/validation curve** — train falls forever while validation dips then RISES (early stopping);\n"
    "4. the **learning-rate ↔ n_estimators trade** — a small rate needs more trees but generalizes better;\n"
    "5. the **residual-shrinking movie** — the ensemble staircase converging to the trend on one real feature;\n"
    "6. XGBoost's **regularized leaf weight** `w* = -G/(H+λ)` and **split gain**, on real numbers;\n"
    "7. why **gradient-boosted trees win tabular data**, measured against a tree and a forest.\n"
    "\n"
    "Everything runs on CPU in a few seconds, seeded for reproducibility."
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
    "import xgboost\n"
    "\n"
    "from gradient_boosting import (\n"
    "    verify_gradient_identity, GradientBoostingScratch, verify_against_sklearn,\n"
    "    load_california, load_cancer, staged_curve, learning_rate_sweep,\n"
    "    residual_movie, xgboost_leaf_gain, boost_classification, model_comparison,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__} | xgboost {xgboost.__version__}')\n"
    "cal = load_california()\n"
    "print(f'California Housing: {cal.x_train.shape[0]} train + {cal.x_test.shape[0]} test districts, '\n"
    "      f'{cal.x_train.shape[1]} features {cal.feature_names}')"
)

# ---- Step 1: the gradient identity ----
add_md(
    "## Step 1 — The pseudo-residual IS the negative gradient\n"
    "\n"
    "Boosting fits each tree to the **negative gradient of the loss** with respect to the current prediction "
    "$F(x_i)$ — the *pseudo-residual*. The name 'gradient' comes from here. Two cases that matter, derived:\n"
    "\n"
    "- **squared error** $L=\\tfrac12(y-F)^2$: $\\;\\partial L/\\partial F = -(y-F)$, so $-\\text{grad}=y-F$ — "
    "the **ordinary residual**;\n"
    "- **log-loss** with $p=\\sigma(F)$: $\\;\\partial L/\\partial F = p-y$, so $-\\text{grad}=y-p$.\n"
    "\n"
    "'Fit the residuals' *is* 'fit the negative gradient'. Let's confirm both to machine zero."
)
add_code(
    "ident = verify_gradient_identity()\n"
    "print(f'MSE      : -grad = {ident.mse_neg_grad:+.4f}   y-F = {ident.mse_residual:+.4f}   '\n"
    "      f'match={ident.mse_match}')\n"
    "print(f'log-loss : -grad = {ident.logloss_neg_grad:+.4f}   y-p = {ident.logloss_residual:+.4f}   '\n"
    "      f'match={ident.logloss_match}')\n"
    "print('\\nThe residual and the negative gradient are literally the same number — that is why it is')\n"
    "print('called GRADIENT boosting.')"
)

# ---- Step 2: boost from scratch, loss falls each round ----
add_md(
    "## Step 2 — Boost from scratch: the loss falls every round\n"
    "\n"
    "The whole algorithm: start at the constant $F_0=\\bar y$ (the mean, which minimizes squared error), then "
    "each round compute the residual $r=y-F$, fit a shallow tree to it, and add a shrunken step "
    "$F\\leftarrow F+\\eta\\,h$. `GradientBoostingScratch` does exactly this. Watch the training MSE fall as the "
    "ensemble chips away at the leftover error."
)
add_code(
    "gb = GradientBoostingScratch(n_estimators=60, learning_rate=0.3, max_depth=3).fit(cal.x_train, cal.y_train)\n"
    "staged = gb.staged_predict(cal.x_train)   # predictions after 0, 1, 2, ... rounds\n"
    "for m in (1, 5, 10, 30, 60):\n"
    "    mse = np.mean((cal.y_train - staged[m]) ** 2)\n"
    "    print(f'after {m:>2} rounds: train MSE = {mse:.4f}')\n"
    "print('\\nEach shallow tree fits the residual the ensemble still gets wrong — bias reduction in action.')"
)

# ---- Step 3: verify vs sklearn ----
add_md(
    "## Step 3 — Is it the real thing? Verify against scikit-learn\n"
    "\n"
    "Grown with the **same** `learning_rate`, `max_depth`, and `n_estimators`, our from-scratch ensemble should "
    "trace the **same validation-loss curve** as scikit-learn's `GradientBoostingRegressor`, round for round. "
    "(We compare loss *curves* rather than demanding identical per-point predictions: when two candidate splits "
    "tie on gain, each implementation breaks the tie with its own RNG, so a different-but-equally-good tree is "
    "legitimate — same subtlety as a single decision tree.)"
)
add_code(
    "match = verify_against_sklearn(cal)\n"
    "print(f'from-scratch test MSE : {match.scratch_test_mse:.5f}')\n"
    "print(f'scikit-learn test MSE : {match.sklearn_test_mse:.5f}')\n"
    "print(f'worst per-round validation-loss gap over {match.n_estimators} rounds: {match.max_staged_loss_gap:.5f}')\n"
    "print('\\n=> same loss curve round for round: the from-scratch loop IS scikit-learn\\'s GBM.')"
)

# ---- Step 4: staged curve table ----
add_md(
    "## Step 4 — Why boosting overfits: the staged train/validation curve\n"
    "\n"
    "Unlike a random forest, **more trees CAN overfit** a boosted model — each tree keeps reducing *training* "
    "error, so eventually the ensemble fits the noise. Run a long boosting run and watch training MSE fall "
    "forever while **validation** MSE dips to a minimum and then turns back **up**. That minimum is exactly the "
    "round an early-stopping rule would keep."
)
add_code(
    "curve = staged_curve(cal)\n"
    "for r in (1, 10, 50, curve.best_round, 300, 500):\n"
    "    i = r - 1\n"
    "    mark = '   <- best val (early stop)' if r == curve.best_round else ''\n"
    "    print(f'round {r:>4}: train MSE = {curve.train_mse[i]:.4f}   val MSE = {curve.val_mse[i]:.4f}{mark}')\n"
    "print(f'\\nbest validation MSE {curve.best_val_mse:.4f} at round {curve.best_round}; '\n"
    "      f'train keeps falling to {curve.train_mse[-1]:.4f}.')"
)

# ---- Step 5: plot staged curve ----
add_md(
    "## Step 5 — The overfitting U-turn, plotted\n"
    "\n"
    "The growing **gap** between train and validation is overfitting made visible. The dashed line marks the "
    "validation minimum — the number of trees you should actually keep."
)
add_code(
    "r = curve.rounds\n"
    "plt.figure(figsize=(9, 5))\n"
    "plt.plot(r, curve.train_mse, color='#3A6B96', lw=2.0, label='training MSE')\n"
    "plt.plot(r, curve.val_mse, color='#8B3B4A', lw=2.0, label='validation MSE')\n"
    "plt.axvline(curve.best_round, color='#7A6528', ls='--', lw=1.5,\n"
    "            label=f'early stop @ {curve.best_round} trees')\n"
    "plt.xlabel('boosting rounds (number of trees)')\n"
    "plt.ylabel('mean squared error')\n"
    "plt.title('Gradient boosting overfits with too many rounds (California Housing)')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 6: LR sweep table ----
add_md(
    "## Step 6 — The learning-rate ↔ n_estimators trade\n"
    "\n"
    "`learning_rate` and `n_estimators` trade off. A big rate descends fast but overshoots and overfits within "
    "a few trees; a small rate takes many more trees but reaches a **lower** minimum — shrinkage acts as "
    "regularization. Measure the validation minimum and the rounds it takes for four rates."
)
add_code(
    "sweep = learning_rate_sweep(cal)\n"
    "print(f'{\"learning_rate\":>14}{\"best val MSE\":>14}{\"rounds to best\":>16}')\n"
    "for lr, best_mse, best_round in zip(sweep.learning_rates, sweep.best_val_mses, sweep.best_rounds):\n"
    "    print(f'{lr:>14}{best_mse:>14.4f}{best_round:>16}')\n"
    "print('\\nSmaller rate -> many more trees -> lower/flatter minimum. The standard recipe: small lr,')\n"
    "print('many trees, and early stopping.')"
)

# ---- Step 7: LR sweep plot ----
add_md(
    "## Step 7 — The shrinkage trade, plotted\n"
    "\n"
    "Each curve is one learning rate's validation MSE vs number of trees; the dot is its minimum. See the big "
    "rate bottom out early and high, the small rates descend slowly to a lower floor."
)
add_code(
    "colours = ['#8B3B4A', '#7A6528', '#2E7A5A', '#3A6B96']\n"
    "plt.figure(figsize=(9, 5))\n"
    "for lr, vcurve, br, bm, c in zip(sweep.learning_rates, sweep.val_curves, sweep.best_rounds,\n"
    "                                 sweep.best_val_mses, colours):\n"
    "    rr = np.arange(1, len(vcurve) + 1)\n"
    "    plt.plot(rr, vcurve, color=c, lw=1.8, label=f'lr={lr}: min {bm:.3f} @ {br} trees')\n"
    "    plt.scatter([br], [bm], color=c, s=40, zorder=5, edgecolor='white')\n"
    "plt.ylim(0.25, 0.75)\n"
    "plt.xlabel('boosting rounds (number of trees)')\n"
    "plt.ylabel('validation MSE')\n"
    "plt.title('learning_rate ↔ n_estimators: shrinkage is regularization')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: residual movie ----
add_md(
    "## Step 8 — The residual-shrinking movie (one real feature)\n"
    "\n"
    "Boost on a **single real feature** (`MedInc`, median income) so the ensemble prediction is a 1-D staircase "
    "you can literally watch refine. After 1 round it is a coarse step; after 100 it hugs the income→value "
    "trend. The residual RMS falls at every checkpoint."
)
add_code(
    "movie = residual_movie(cal)\n"
    "fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)\n"
    "for ax, rounds, pred, rms in zip(axes.ravel(), movie.checkpoints, movie.predictions, movie.residual_rms):\n"
    "    ax.scatter(movie.x, movie.y, s=8, color='#4A5B6E', alpha=0.18)\n"
    "    ax.plot(movie.grid, pred, color='#2E7A5A', lw=2.2, drawstyle='steps-mid')\n"
    "    ax.set_title(f'after {rounds} round(s)  ·  residual RMS = {rms:.3f}')\n"
    "for ax in axes[-1]:\n"
    "    ax.set_xlabel('MedInc (median income)')\n"
    "for ax in axes[:, 0]:\n"
    "    ax.set_ylabel('house value ($100k)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print('residual RMS:', [f'{r:.3f}' for r in movie.residual_rms], '<- falls every checkpoint')"
)

# ---- Step 9: XGBoost leaf weight & gain ----
add_md(
    "## Step 9 — XGBoost's regularized leaf weight and split gain\n"
    "\n"
    "XGBoost sharpens plain boosting with a **second-order** objective (gradient *and* Hessian) plus explicit "
    "regularization. Minimizing it gives the closed-form optimal leaf weight $w^*=-G/(H+\\lambda)$ and the split "
    "gain $\\tfrac12[G_L^2/(H_L+\\lambda)+G_R^2/(H_R+\\lambda)-G^2/(H+\\lambda)]-\\gamma$. Compute them on the "
    "chapter's five-sample worked example (squared error, so each Hessian $h_i=1$)."
)
add_code(
    "split = xgboost_leaf_gain()   # g=(-0.8,-0.6,0.5,0.7,0.9), left={1,2}, right={3,4,5}, lambda=1, gamma=0\n"
    "print(f'optimal leaf weights: w_left={split.w_left:+.2f}  w_right={split.w_right:+.2f}  '\n"
    "      f'w_parent={split.w_parent:+.2f}')\n"
    "print(f'split gain = {split.gain:.3f}  ->  {\"KEEP the split\" if split.kept else \"PRUNE the split\"}')\n"
    "print('\\nThe under-predicted samples (g<0) go left with a positive weight; the over-predicted (g>0) go')\n"
    "print('right with a negative weight. gain>0, so the regularized loss drops and XGBoost keeps the split.')"
)

# ---- Step 10: classification ----
add_md(
    "## Step 10 — Boosting for classification (log-loss), from scratch\n"
    "\n"
    "The same machine does classification: accumulate **log-odds** across trees, and read the output through a "
    "sigmoid. Init at the base-rate log-odds; each round the pseudo-residual is $y-p$, and each leaf's value is "
    "the **Newton step** $\\sum(y-p)/\\sum p(1-p)$. We store each round's Newton leaf values and score on a "
    "**held-out test split**, so both log-losses are clearly above zero — a real equality check, not two "
    "memorized-to-zero numbers. The from-scratch log-loss should match `GradientBoostingClassifier` on the "
    "unseen Breast Cancer tumours."
)
add_code(
    "cancer = load_cancer()\n"
    "cls = boost_classification(cancer)\n"
    "print(f'from-scratch test log-loss : {cls.scratch_test_log_loss:.5f}')\n"
    "print(f'scikit-learn test log-loss : {cls.sklearn_test_log_loss:.5f}')\n"
    "gap = abs(cls.scratch_test_log_loss - cls.sklearn_test_log_loss)\n"
    "print(f'gap = {gap:.5f}   (both non-trivial, matched on held-out data)')\n"
    "assert gap < 0.05\n"
    "print('\\ny-p residuals + Newton leaf values reproduce scikit-learn\\'s classifier on UNSEEN data.')"
)

# ---- Step 11: model comparison ----
add_md(
    "## Step 11 — Why gradient-boosted trees win tabular data\n"
    "\n"
    "The honest, measured comparison on one California test split: a single tree (high variance), a random "
    "forest (bagging → variance down), scikit-learn gradient boosting and real **XGBoost** (boosting → bias "
    "down, regularized). Boosting edges out the forest, which crushes the single tree."
)
add_code(
    "comp = model_comparison(cal)\n"
    "print(f'{\"model\":<26}{\"test R^2\":>10}{\"test RMSE\":>12}')\n"
    "for name, r2, rmse in zip(comp.names, comp.r2, comp.rmse):\n"
    "    print(f'{name.replace(chr(10), \" \"):<26}{r2:>10.3f}{rmse:>12.4f}')\n"
    "print('\\nsingle tree < random forest < gradient boosting <= XGBoost: bias-reduced ensembles win tabular.')"
)

# ---- Step 12: real XGBoost early stopping ----
add_md(
    "## Step 12 — Real XGBoost with early stopping (the production workflow)\n"
    "\n"
    "Finally, the real-world recipe with the actual library: set `n_estimators` to a large **upper bound**, use "
    "a **small** `learning_rate`, pass an `eval_set`, and let `early_stopping_rounds` pick the tree count for "
    "you. The model stops when validation stops improving and keeps the best iteration — never the last."
)
add_code(
    "from xgboost import XGBRegressor\n"
    "from sklearn.metrics import r2_score\n"
    "\n"
    "model = XGBRegressor(\n"
    "    n_estimators=2000,          # an UPPER BOUND — early stopping picks the real count\n"
    "    learning_rate=0.05,         # small lr + many trees + early stopping = the recipe\n"
    "    max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,\n"
    "    early_stopping_rounds=50, eval_metric='rmse', random_state=0, n_jobs=-1)\n"
    "model.fit(cal.x_train, cal.y_train, eval_set=[(cal.x_test, cal.y_test)], verbose=False)\n"
    "\n"
    "print(f'best_iteration (early-stopped) = {model.best_iteration}   (<< 2000)')\n"
    "print(f'test R^2 at best_iteration      = {r2_score(cal.y_test, model.predict(cal.x_test)):.3f}')\n"
    "imp = model.get_booster().get_score(importance_type='gain')\n"
    "top = sorted(imp.items(), key=lambda kv: -kv[1])[:3]\n"
    "print('top-3 features by gain:', [f'{cal.feature_names[int(k[1:])]}={v:.0f}' for k, v in top])"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "On real scikit-learn datasets (plus real XGBoost), one runnable notebook built the whole of gradient boosting:\n"
    "\n"
    "| Step | What we measured | The takeaway |\n"
    "|---|---|---|\n"
    "| 1 | pseudo-residual = negative gradient | `y - F` (MSE) and `y - p` (log-loss), to machine zero |\n"
    "| 2–3 | boost from scratch; verify vs sklearn | loss falls every round; **same loss curve** as scikit-learn |\n"
    "| 4–5 | the staged curve | train ↓ forever, validation dips then **RISES** — early-stop at the minimum |\n"
    "| 6–7 | learning-rate ↔ n_estimators | small rate, more trees, lower minimum — shrinkage as regularization |\n"
    "| 8 | the residual movie | the ensemble staircase **converges** to the trend; residual RMS falls |\n"
    "| 9 | XGBoost leaf weight & gain | $w^*=-G/(H+\\lambda)$; gain>0 keeps the split, gain<0 prunes it |\n"
    "| 10 | log-loss boosting from scratch | `y-p` residuals + Newton leaf values reproduce scikit-learn |\n"
    "| 11–12 | model comparison; real XGBoost | boosting > forest > tree; early stopping picks the tree count |\n"
    "\n"
    "**Gradient boosting adds shallow trees sequentially, each fitting the negative gradient of the loss, with a "
    "small learning rate — gradient descent in function space.** It reduces **bias** (vs a forest's variance), it "
    "**can overfit** with too many trees (use early stopping), and its modern implementations — XGBoost, "
    "LightGBM, CatBoost — dominate tabular machine learning."
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

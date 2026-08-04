"""Generate the step-by-step teaching notebook (05-Overfitting-and-Underfitting.ipynb).

The notebook mirrors ``overfitting_underfitting.py`` one measurement at a time, so a beginner can open
it, run every cell live, and *see* over- and under-fitting on real data — the three fits, the U-curve,
the measured bias-variance decomposition, and both cures. Each numbered step has a short markdown
lead-in (the intuition) followed by ONE focused code cell with real output. This generator writes the
.ipynb; a separate nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_05.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../05-Overfitting-and-Underfitting/code/05-Overfitting-and-Underfitting.ipynb"

This generator lives in the domain-level ``00. Basics/tools/`` folder; the notebook it writes (and the
module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep: the same experiments, typed once in the
module, demonstrated step-by-step here.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "05-Overfitting-and-Underfitting"
    / "code"
    / "05-Overfitting-and-Underfitting.ipynb"
)

_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """A code cell (outputs filled in by the nbconvert execute pass)."""
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
    "# Overfitting & Underfitting — a step-by-step, runnable notebook\n"
    "\n"
    "A model can score **perfectly** on the data it trained on and still fail on new data — because it "
    "**memorised the noise** instead of learning the signal. This notebook shows that, one measurement at "
    "a time, on **real data** (a known curve plus real Gaussian noise, so we can measure what's normally "
    "invisible). It is the executable companion to the chapter and to `overfitting_underfitting.py`; every "
    "function used here lives in that module, imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **measured**, not just been told:\n"
    "\n"
    "1. the **three regimes** as fitted curves — underfit, just-right, overfit — on the same 40 points;\n"
    "2. the **U-curve**: training error only falls with capacity, validation error falls then rises;\n"
    "3. the **generalisation gap** (validation − training) as your overfitting diagnostic;\n"
    "4. the **bias-variance decomposition**, measured over hundreds of resamples, summing to the U;\n"
    "5. the two **cures** — L2 regularization and more data — working on measured numbers;\n"
    "6. every fit **cross-checked against scikit-learn**.\n"
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
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "\n"
    "from overfitting_underfitting import (\n"
    "    make_dataset, true_function, fit_poly, predict_poly, mse,\n"
    "    complexity_sweep, sklearn_poly_predict, bias_variance_decomposition,\n"
    "    ridge_lambda_sweep, learning_curve,\n"
    "    N_TRAIN, N_VAL, NOISE_SIGMA, TRAIN_SEED, VAL_SEED,\n"
    "    UNDERFIT_DEGREE, GOOD_DEGREE, OVERFIT_DEGREE, LC_DEGREE,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__} '\n"
    "      f'| matplotlib {matplotlib.__version__}')"
)

# ---- Step 1: the real data ----
add_md(
    "## Step 1 — The real data: a known curve seen through noise\n"
    "\n"
    "We learn a gentle true curve, $f(x) = \\cos(1.5\\pi x)$ on $x\\in[0,1]$, but we only ever observe it "
    "through **noisy measurements**: $y = f(x) + \\varepsilon$, with $\\varepsilon$ Gaussian, standard "
    "deviation $\\sigma = 0.25$. That noise is the part no model can ever predict — the *irreducible* "
    "error. We draw **40** training points. (Using a known curve is what lets us later *measure* bias, "
    "variance, and noise separately — impossible on a real dataset where the truth is unknown.)"
)
add_code(
    "train = make_dataset(N_TRAIN, seed=TRAIN_SEED)\n"
    "print(f'{train.x.size} training points | noise sigma = {NOISE_SIGMA} | irreducible floor = sigma^2 = {NOISE_SIGMA**2:.4f}')\n"
    "\n"
    "xs = np.linspace(0, 1, 400)\n"
    "plt.figure(figsize=(7.5, 4.2))\n"
    "plt.plot(xs, true_function(xs), '--', color='#1C2530', lw=2, label='true signal  cos(1.5 pi x)')\n"
    "plt.scatter(train.x, train.y, s=28, color='#3A6B96', alpha=0.7, label='40 noisy training points')\n"
    "plt.xlabel('x')\n"
    "plt.ylabel('y')\n"
    "plt.title('the true curve, and the noisy points we actually get to see')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 2: capacity, train vs val ----
add_md(
    "## Step 2 — Two numbers per model: training error and validation error\n"
    "\n"
    "**Capacity** = how wiggly a function a model can represent; for a polynomial it is simply the "
    "**degree**. We judge a model by **two** errors: on the training points it learned from, and on a "
    "large **held-out validation set** (4,000 fresh points) it never saw. Watch the difference: a "
    "degree-1 line and a degree-15 polynomial have very different *training* errors — but the number that "
    "matters is the *validation* one."
)
add_code(
    "val = make_dataset(N_VAL, seed=VAL_SEED)\n"
    "for d in (1, 4, 15):\n"
    "    fit = fit_poly(train.x, train.y, d)\n"
    "    tr = mse(train.y, predict_poly(fit, train.x))\n"
    "    va = mse(val.y, predict_poly(fit, val.x))\n"
    "    print(f'degree {d:2d}: train MSE = {tr:.3f}   validation MSE = {va:.3f}')\n"
    "print('\\nnote: degree 15 has the LOWEST training error but a HIGH validation error — that is overfitting.')"
)

# ---- Step 3: the three fits (money shot) ----
add_md(
    "## Step 3 — See it: underfit / good fit / overfit\n"
    "\n"
    "The same 40 points, three capacities. **Degree 1** is too rigid — it misses the bend (underfit, high "
    "bias). **Degree 4** tracks the true curve (good). **Degree 15** wiggles through nearly every noisy "
    "point and swings wildly at the edges (overfit, high variance). This one picture is the whole subject."
)
add_code(
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)\n"
    "regimes = [(UNDERFIT_DEGREE, 'underfit', '#4A5B6E'),\n"
    "           (GOOD_DEGREE, 'good fit', '#2E7A5A'),\n"
    "           (OVERFIT_DEGREE, 'overfit', '#8B3B4A')]\n"
    "for ax, (deg, name, colour) in zip(axes, regimes):\n"
    "    fit = fit_poly(train.x, train.y, deg)\n"
    "    va = mse(val.y, predict_poly(fit, val.x))\n"
    "    ax.scatter(train.x, train.y, s=22, color='#3A6B96', alpha=0.5, label='training points')\n"
    "    ax.plot(xs, true_function(xs), '--', color='#1C2530', lw=1.6, alpha=0.7, label='true function')\n"
    "    ax.plot(xs, predict_poly(fit, xs), color=colour, lw=2.4, label=f'degree-{deg} fit')\n"
    "    ax.set_ylim(-2, 2)\n"
    "    ax.set_xlabel('x')\n"
    "    ax.set_title(f'{name} (deg {deg}) — val MSE {va:.3f}', color=colour)\n"
    "    ax.legend(fontsize=8, loc='lower left')\n"
    "    ax.grid(alpha=0.3)\n"
    "axes[0].set_ylabel('y')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 4: the complexity sweep / U-curve ----
add_md(
    "## Step 4 — The U-curve: sweep every capacity\n"
    "\n"
    "Now fit *every* degree from 1 to 15 and plot both errors against capacity. The signatures to watch: "
    "**training error only ever falls** (more capacity always hugs the training points harder), while "
    "**validation error falls, bottoms out at the sweet spot, then rises** — the classic U. The bottom of "
    "the red curve is the best model."
)
add_code(
    "sweep = complexity_sweep(train, val)\n"
    "print(f'sweet-spot degree (lowest validation error) = {sweep.best_degree}')\n"
    "\n"
    "plt.figure(figsize=(8, 4.6))\n"
    "plt.plot(sweep.degrees, sweep.train_mse, 'o-', color='#3A6B96', lw=2, label='training error')\n"
    "plt.plot(sweep.degrees, sweep.val_mse, 's-', color='#8B3B4A', lw=2, label='validation error')\n"
    "plt.axhline(NOISE_SIGMA**2, ls=':', color='#7A6528', label=f'noise floor sigma^2 = {NOISE_SIGMA**2:.3f}')\n"
    "plt.scatter([sweep.best_degree], [sweep.val_mse[sweep.best_degree - 1]], color='#2E7A5A',\n"
    "            marker='*', s=200, zorder=5, edgecolor='white', label=f'sweet spot (deg {sweep.best_degree})')\n"
    "plt.xlabel('polynomial degree (capacity)')\n"
    "plt.ylabel('mean squared error')\n"
    "plt.title('training error only falls; validation error is a U')\n"
    "plt.xticks(sweep.degrees)\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 5: the generalization gap ----
add_md(
    "## Step 5 — The generalisation gap: your one-number diagnostic\n"
    "\n"
    "In the wild you can't see bias and variance directly — but you can always compute the "
    "**generalisation gap** = validation error − training error. It is how you diagnose a model:\n"
    "\n"
    "* both errors **high**, small gap → **underfit** (too simple);\n"
    "* both errors **low**, small gap → **good**;\n"
    "* training **low**, validation **high**, **large** gap → **overfit** (memorising)."
)
add_code(
    "print(f'{\"regime\":<10}{\"degree\":>7}{\"train\":>9}{\"val\":>9}{\"gap\":>9}   diagnosis')\n"
    "for deg, name, diag in [(UNDERFIT_DEGREE, 'underfit', 'both high -> too simple'),\n"
    "                        (GOOD_DEGREE, 'good', 'both low -> just right'),\n"
    "                        (OVERFIT_DEGREE, 'overfit', 'big gap -> memorising')]:\n"
    "    fit = fit_poly(train.x, train.y, deg)\n"
    "    tr = mse(train.y, predict_poly(fit, train.x))\n"
    "    va = mse(val.y, predict_poly(fit, val.x))\n"
    "    print(f'{name:<10}{deg:>7}{tr:>9.3f}{va:>9.3f}{va - tr:>+9.3f}   {diag}')\n"
    "print('\\nthe overfit model has the LOWEST training error yet the LARGEST gap. watch the gap, not train error.')"
)

# ---- Step 6: match sklearn ----
add_md(
    "## Step 6 — We measured the real thing: it matches scikit-learn\n"
    "\n"
    "Are our from-scratch least-squares fits *correct*, or just close? We hand scikit-learn's own solver "
    "the identical features and compare predictions. If they match to machine precision, the curves above "
    "are the genuine estimator, not a lookalike."
)
add_code(
    "for d in (UNDERFIT_DEGREE, GOOD_DEGREE, OVERFIT_DEGREE):\n"
    "    ours = predict_poly(fit_poly(train.x, train.y, d), val.x)\n"
    "    theirs = sklearn_poly_predict(train.x, train.y, val.x, d)\n"
    "    print(f'degree {d:2d}: max|ours - sklearn| = {np.max(np.abs(ours - theirs)):.2e}  '\n"
    "          f'match: {bool(np.allclose(ours, theirs, atol=1e-6))}')\n"
    "print('\\n=> our from-scratch least squares reproduces scikit-learn to 1e-6. The measurements are real.')"
)

# ---- Step 7: bias-variance decomposition (measure) ----
add_md(
    "## Step 7 — Measuring the cause: the bias-variance decomposition\n"
    "\n"
    "The U-curve is the *symptom*. Its *cause* is the split $\\mathbb{E}[(y-\\hat f)^2] = "
    "\\text{bias}^2 + \\text{variance} + \\sigma^2$. Because we know the true curve and the noise, we can "
    "**measure** each term: resample many training sets, fit each, and see how far the *average* fit is "
    "from the truth (bias$^2$) and how much fits *bounce* around that average (variance). Watch bias fall "
    "and variance rise with degree."
)
add_code(
    "bv = bias_variance_decomposition()\n"
    "print(f'{\"degree\":>6}{\"bias^2\":>10}{\"variance\":>10}{\"noise\":>8}{\"sum\":>10}{\"measured\":>10}')\n"
    "for d, b2, var, tot in zip(bv.degrees, bv.bias2, bv.variance, bv.total_measured):\n"
    "    print(f'{int(d):>6}{b2:>10.4f}{var:>10.4f}{bv.noise:>8.4f}{b2 + var + bv.noise:>10.4f}{tot:>10.4f}')\n"
    "print(f'\\nbias^2 collapses {bv.bias2[0]:.3f} -> {bv.bias2[3]:.4f} (deg 1 -> 4); '\n"
    "      f'variance explodes {bv.variance[0]:.3f} -> {bv.variance[-1]:.3f} (deg 1 -> {int(bv.degrees[-1])}).')"
)

# ---- Step 8: bias-variance plot ----
add_md(
    "## Step 8 — The decomposition, plotted: their sum is the U\n"
    "\n"
    "Plotting the three measured curves on a log scale makes the tradeoff visible: bias$^2$ (blue) falls, "
    "variance (red) rises, and their sum plus the noise floor (purple) is the **U** — bottoming at the same "
    "sweet spot the validation sweep found. This *is* the U-curve, split into its causes."
)
add_code(
    "total = bv.bias2 + bv.variance + bv.noise\n"
    "plt.figure(figsize=(8, 4.6))\n"
    "plt.plot(bv.degrees, bv.bias2, 'o-', color='#3A6B96', lw=2, label='bias^2 (systematic error)')\n"
    "plt.plot(bv.degrees, bv.variance, 's-', color='#8B3B4A', lw=2, label='variance (sample sensitivity)')\n"
    "plt.plot(bv.degrees, total, 'D-', color='#5D4A8A', lw=2.4, label='total = bias^2 + variance + sigma^2')\n"
    "plt.axhline(bv.noise, ls=':', color='#7A6528', label=f'irreducible noise sigma^2 = {bv.noise:.3f}')\n"
    "plt.yscale('log')\n"
    "plt.xlabel('polynomial degree (capacity)')\n"
    "plt.ylabel('error (log scale)')\n"
    "plt.title('bias down + variance up = the U')\n"
    "plt.xticks(bv.degrees)\n"
    "plt.legend(fontsize=8, loc='lower left')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: the identity check ----
add_md(
    "## Step 9 — The decomposition is an equation, and we check it\n"
    "\n"
    "The theorem is not a metaphor: bias$^2 +$ variance $+ \\sigma^2$ should **equal** the independently-"
    "measured test error at every degree. We measured the total a second way (scoring the fits against "
    "fresh noisy targets) — here is the equation holding, degree by degree."
)
add_code(
    "worst = 0.0\n"
    "for d, b2, var, tot in zip(bv.degrees, bv.bias2, bv.variance, bv.total_measured):\n"
    "    s = b2 + var + bv.noise\n"
    "    rel = abs(s - tot) / tot\n"
    "    worst = max(worst, rel)\n"
    "    print(f'degree {int(d):>2}: bias^2+var+noise = {s:8.4f}   measured = {tot:8.4f}   rel. error {rel:.1%}')\n"
    "print(f'\\nlargest relative mismatch across all degrees: {worst:.1%}  (pure Monte-Carlo noise). The identity holds.')"
)

# ---- Step 10: cure 1 - ridge ----
add_md(
    "## Step 10 — Cure 1: regularization (an L2 penalty) tames the overfit model\n"
    "\n"
    "You can leave a model's capacity in place but **discourage it from using it recklessly**. **L2 "
    "regularization** adds $\\lambda\\lVert w\\rVert^2$ to the loss, penalising large weights. Since the "
    "wild overfit wiggles *need* large weights, shrinking the weights kills the wiggles — variance drops "
    "for a small bias cost. We take the wildly-overfit **degree-15** model and sweep $\\lambda$: watch "
    "validation error fall back toward the sweet spot, then rise again if we over-penalise (a U in "
    "$\\lambda$ too)."
)
add_code(
    "rs = ridge_lambda_sweep(train, val)\n"
    "best_val = float(rs.val_mse.min())\n"
    "sweet_val = sweep.val_mse[sweep.best_degree - 1]\n"
    "print(f'overfit degree-15 model, lambda=0 : validation MSE = {rs.unpenalised_val_mse:.3f}')\n"
    "print(f'best-penalised model, lambda={rs.best_lambda:.3g}: validation MSE = {best_val:.3f}')\n"
    "print(f'sweet-spot degree-{sweep.best_degree} model         : validation MSE = {sweet_val:.3f}  (for comparison)')\n"
    "print(f'-> ridge cut the overfit error by {100 * (rs.unpenalised_val_mse - best_val) / rs.unpenalised_val_mse:.0f}%, without dropping any capacity.')\n"
    "\n"
    "plt.figure(figsize=(8, 4.6))\n"
    "plt.plot(rs.lambdas, rs.val_mse, 's-', color='#8B3B4A', lw=2, label='validation error')\n"
    "plt.plot(rs.lambdas, rs.train_mse, 'o-', color='#3A6B96', lw=1.8, alpha=0.8, label='training error')\n"
    "plt.axhline(rs.unpenalised_val_mse, ls='--', color='#4A5B6E', label=f'overfit (lambda=0) = {rs.unpenalised_val_mse:.3f}')\n"
    "plt.axhline(sweet_val, ls=':', color='#2E7A5A', label=f'sweet-spot deg-{sweep.best_degree} = {sweet_val:.3f}')\n"
    "plt.xscale('log')\n"
    "plt.xlabel('L2 penalty strength lambda (log)')\n"
    "plt.ylabel('mean squared error')\n"
    "plt.title('regularization pulls the overfit model back to the sweet spot')\n"
    "plt.legend(fontsize=8)\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 11: ridge matches sklearn ----
add_md(
    "## Step 11 — And our ridge matches scikit-learn too\n"
    "\n"
    "Same correctness check as Step 6, for the regularized fit: our from-scratch ridge should reproduce "
    "scikit-learn's `Ridge` at the best $\\lambda$."
)
add_code(
    "ours = predict_poly(fit_poly(train.x, train.y, OVERFIT_DEGREE, l2=rs.best_lambda), val.x)\n"
    "theirs = sklearn_poly_predict(train.x, train.y, val.x, OVERFIT_DEGREE, l2=rs.best_lambda)\n"
    "print(f'ridge (lambda={rs.best_lambda:.3g}) max|ours - sklearn| = {np.max(np.abs(ours - theirs)):.2e}')\n"
    "print(f'match: {bool(np.allclose(ours, theirs, atol=1e-6))}')"
)

# ---- Step 12: cure 2 - more data ----
add_md(
    "## Step 12 — Cure 2: more data (noise can't be memorised in bulk)\n"
    "\n"
    "The other cure needs no cleverness. An overfit model works by memorising the *specific* noise in its "
    "points — but you can only memorise so much. Fix a mildly-overfitting **degree-6** model and grow the "
    "training set from 20 to 600 points. The **generalisation gap collapses**: with more data the model is "
    "forced to fit the signal (the thing all points agree on) instead of the noise."
)
add_code(
    "lc = learning_curve(val)\n"
    "print(f'{\"n_train\":>8}{\"train\":>9}{\"val\":>9}{\"gap\":>9}')\n"
    "for n, tr, va, g in zip(lc.sizes, lc.train_mse, lc.val_mse, lc.gap):\n"
    "    print(f'{int(n):>8}{tr:>9.3f}{va:>9.3f}{g:>9.3f}')\n"
    "print(f'-> the gap shrank from {lc.gap[0]:.3f} (n={int(lc.sizes[0])}) to {lc.gap[-1]:.3f} (n={int(lc.sizes[-1])}).')\n"
    "\n"
    "plt.figure(figsize=(8, 4.6))\n"
    "plt.plot(lc.sizes, lc.train_mse, 'o-', color='#3A6B96', lw=2, label='training error')\n"
    "plt.plot(lc.sizes, lc.val_mse, 's-', color='#8B3B4A', lw=2, label='validation error')\n"
    "plt.fill_between(lc.sizes, lc.train_mse, lc.val_mse, color='#8B3B4A', alpha=0.1, label='generalisation gap')\n"
    "plt.axhline(NOISE_SIGMA**2, ls=':', color='#7A6528', label=f'noise floor sigma^2 = {NOISE_SIGMA**2:.3f}')\n"
    "plt.xscale('log')\n"
    "plt.yscale('log')\n"
    "plt.xlabel('training-set size n (log)')\n"
    "plt.ylabel('error (log)')\n"
    "plt.title(f'more data closes the gap (fixed degree-{LC_DEGREE} model)')\n"
    "plt.legend(fontsize=8)\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "In one runnable notebook, on real measured data, we saw the whole of over- and under-fitting:\n"
    "\n"
    "| Step | What we did | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–2 | a known curve seen through noise; train vs val error | judge on **held-out** data, not training error |\n"
    "| 3 | the three fits | underfit (high bias) / good / overfit (high variance) — *seen* |\n"
    "| 4 | the complexity sweep | training error only falls; validation error is a **U** |\n"
    "| 5 | the generalisation gap | val − train diagnoses under/over-fitting in one number |\n"
    "| 6, 11 | match scikit-learn | the fits are the **genuine** estimator, to 1e-6 |\n"
    "| 7–9 | the bias-variance decomposition | bias↓ + variance↑ + noise = the U, as an **equation** |\n"
    "| 10 | ridge (L2) regularization | penalise weights → less variance → overfit **cured**, capacity kept |\n"
    "| 12 | more data | the gap collapses as n grows — noise can't be memorised in bulk |\n"
    "\n"
    "A model's error is **bias² + variance + irreducible noise**. Underfitting is high bias; overfitting is "
    "high variance; the sweet spot minimises their sum. You find it — and the right regularization — by "
    "**measuring on data the model never saw**. That is the diagnostic under every model in the rest of the "
    "repo."
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

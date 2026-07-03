"""Generate the step-by-step teaching notebook (11-Data-Leakage.ipynb).

The notebook mirrors ``data_leakage.py`` one measurement at a time, so a learner can open it, run every
cell live, and *watch* a leaky protocol inflate a score and the ``Pipeline`` fix collapse it back to the
truth — on pure noise (where the truth is known to be chance), on the real Breast Cancer dataset (a
proxy-of-the-label column), and on a realistic time series (a random split that trains on the future).
Each numbered step has a short markdown lead-in (the intuition) followed by ONE focused code cell with
real output. This generator writes the .ipynb; a separate nbconvert pass executes it headless so the
outputs are embedded.

    python build_notebook_11.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../11-Data-Leakage/code/11-Data-Leakage.ipynb"

This generator lives in the domain-level ``02. Data_Preprocessing/tools/`` folder; the notebook it
writes (and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a
hand-edited .ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "11-Data-Leakage" / "code"

NB_PATH = _CHAPTER_CODE / "11-Data-Leakage.ipynb"

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
    "# Data Leakage — a step-by-step, runnable notebook\n"
    "\n"
    "A model reports **95%** in your notebook and **70%** in the real world. That gap is the single most "
    "expensive bug in applied machine learning, and its most common cause is **data leakage**: at training "
    "time the model saw information it will *not* have at prediction time. The score was never real.\n"
    "\n"
    "Leakage is invisible in a single number — a great score looks like a great model. The only way to *see* "
    "it is to measure the **same problem two ways**: a leaky protocol and an honest one, side by side. This "
    "notebook does exactly that, one measurement at a time, for the three leaks you will actually meet:\n"
    "\n"
    "1. **Preprocessing leakage** — selecting features on the whole dataset before cross-validating. We use "
    "pure **noise** with a random label, so the honest answer is *provably* chance (0.50) — and any accuracy "
    "above it is leakage, measured. Leaky CV says **0.86**; the `Pipeline` fix says **0.48**.\n"
    "2. **Target leakage** — a column that is a proxy for the label, on the **real Breast Cancer** dataset. "
    "It takes accuracy to **1.00**; drop it and the honest **0.96** returns.\n"
    "3. **Temporal leakage** — a random split of a time series trains on the future. Shuffled R² **0.97**; "
    "a forward `TimeSeriesSplit` says **0.69**.\n"
    "\n"
    "Everything is the executable companion to the chapter and to `data_leakage.py`; every function used "
    "here lives in that module, imported so the notebook and the module can never drift apart. It all runs "
    "on CPU in a few seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "Import the real functions from the chapter module (so this notebook uses the *exact same code* the "
    "figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "\n"
    "from data_leakage import (\n"
    "    make_noise_data, selection_leak, selection_leak_sweep,\n"
    "    make_breast_cancer_leak, target_leak,\n"
    "    make_time_series, temporal_leak,\n"
    "    CHANCE, CV_FOLDS, LEAK_COL_NAME,\n"
    ")\n"
    "from sklearn.feature_selection import SelectKBest, f_classif\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "from sklearn.pipeline import Pipeline\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__}')\n"
    "print(f'cross-validation folds used throughout: {CV_FOLDS}')"
)

# ---- Step 1: the noise dataset (truth = chance) ----
add_md(
    "## Step 1 — A dataset where we KNOW the honest answer\n"
    "\n"
    "To turn leakage into a *measurement* we need a dataset whose honest accuracy we already know. So we "
    "build one with **no signal at all**: 200 rows of 5,000 pure-Gaussian-noise features and a **random** "
    "0/1 label, drawn without ever looking at the features. The best any honest model can possibly do "
    "out-of-sample is **chance, 0.50**. Anything above that is not skill — it is information leaking."
)
add_code(
    "noise = make_noise_data()\n"
    "print(f'X: {noise.x.shape[0]} rows x {noise.x.shape[1]} pure-noise features')\n"
    "print(f'y: balanced random label, class counts = {np.bincount(noise.y)}')\n"
    "# a sanity check: no single feature is meaningfully correlated with the label\n"
    "corrs = np.abs([np.corrcoef(noise.x[:, j], noise.y)[0, 1] for j in range(200)])\n"
    "print(f'largest |corr(feature, label)| among the first 200 columns: {corrs.max():.3f}  (pure chance)')\n"
    "print(f'=> the honest generalization accuracy is provably {CHANCE:.2f}. Remember that number.')"
)

# ---- Step 2: the leaky protocol ----
add_md(
    "## Step 2 — The leaky protocol: select features on ALL the data\n"
    "\n"
    "Here is the mistake, and it looks completely reasonable. We pick the 50 features most correlated with "
    "the label using **`SelectKBest` on the whole dataset**, and *then* cross-validate a classifier on those "
    "columns. The bug: the selector looked at every row — including the rows each fold will later be scored "
    "on. It has already peeked at the answer key."
)
add_code(
    "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
    "cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)\n"
    "\n"
    "# LEAKY: select on all of X, y first, then cross-validate on the pre-selected columns\n"
    "x_selected = SelectKBest(f_classif, k=50).fit_transform(noise.x, noise.y)\n"
    "leaky_cv = np.mean(cross_val_score(clf, x_selected, noise.y, cv=cv))\n"
    "print(f'LEAKY cross-validated accuracy: {leaky_cv:.3f}')\n"
    "print(f'...on data with NO signal, whose honest accuracy is {CHANCE:.2f}.')\n"
    "print('That 0.86 is pure fiction — the selection saw the test folds.')"
)

# ---- Step 3: the honest protocol ----
add_md(
    "## Step 3 — The fix: put selection INSIDE a Pipeline\n"
    "\n"
    "The cure is structural. Wrap the *same* `SelectKBest` and classifier in a `Pipeline` and hand the whole "
    "thing to `cross_val_score`. Now scikit-learn re-fits the selector on **each training fold only**, so "
    "the columns it picks are correlated with the label on the training rows but not on the held-out fold. "
    "The score falls straight back to chance — the truth."
)
add_code(
    "# HONEST: selection lives inside the pipeline, re-fit on each training fold only\n"
    "pipe = Pipeline([('select', SelectKBest(f_classif, k=50)), ('clf', clf)])\n"
    "honest_cv = np.mean(cross_val_score(pipe, noise.x, noise.y, cv=cv))\n"
    "print(f'LEAKY  CV accuracy (select on all data) : {leaky_cv:.3f}')\n"
    "print(f'HONEST CV accuracy (select in Pipeline)  : {honest_cv:.3f}  (~chance)')\n"
    "print(f'measured inflation gap                   : {leaky_cv - honest_cv:+.3f}')\n"
    "assert leaky_cv > honest_cv + 0.15, 'the leak must materially inflate the score'\n"
    "assert abs(honest_cv - CHANCE) < 0.1, 'the honest score must sit near chance'\n"
    "print('\\nSame data, same model, same k — only WHERE the selector was fit changed the answer.')"
)

# ---- Step 4: hold-out confirmation ----
add_md(
    "## Step 4 — Confirm it with an untouched hold-out\n"
    "\n"
    "Cross-validation can feel abstract, so let's confirm the honest number a second, independent way: fit "
    "the honest pipeline on a training split and score it **once** on a test set it has never seen. If it "
    "also lands near chance, we can trust that the honest ~0.48 — not the leaky 0.86 — is the real "
    "generalization accuracy. The `selection_leak` helper runs all three at once."
)
add_code(
    "sel = selection_leak(noise)\n"
    "print(f'leaky CV        : {sel.leaky_cv:.3f}   <- inflated fiction')\n"
    "print(f'honest CV       : {sel.honest_cv:.3f}   <- the truth (~chance)')\n"
    "print(f'honest hold-out : {sel.honest_holdout:.3f}   <- an untouched test set agrees')\n"
    "print(f'\\ninflation gap  : {sel.gap:+.3f}  ({sel.gap*100:.0f} points of pure leakage)')"
)

# ---- Step 5: the leak grows with k ----
add_md(
    "## Step 5 — The more you let it peek, the bigger the lie\n"
    "\n"
    "Leakage is not all-or-nothing — it scales with *how much* information you let in. Sweep the number of "
    "cherry-picked features `k`: the **leaky** curve climbs toward 1.0 (more noise columns to fit the test "
    "folds with), while the **honest** curve never leaves chance for any `k`. The widening gap is the leak, "
    "drawn as a dial you control."
)
add_code(
    "ks, leaky, honest = selection_leak_sweep(noise)\n"
    "plt.figure(figsize=(9, 5))\n"
    "plt.plot(ks, leaky, 'o-', color='#8B3B4A', lw=2.2, label='LEAKY: select on all data, then CV')\n"
    "plt.plot(ks, honest, 's-', color='#2E7A5A', lw=2.2, label='HONEST: select inside the Pipeline')\n"
    "plt.axhline(CHANCE, color='#7A6528', ls='--', lw=1.5, label=f'chance ({CHANCE:.2f})')\n"
    "plt.fill_between(ks, honest, leaky, color='#8B3B4A', alpha=0.08)\n"
    "plt.xscale('log')\n"
    "plt.xticks(ks, [str(k) for k in ks])\n"
    "plt.xlabel('k = number of noise features cherry-picked (log scale)')\n"
    "plt.ylabel('cross-validated accuracy')\n"
    "plt.ylim(0.3, 1.02)\n"
    "plt.title('leaky accuracy climbs toward 1.0 on pure noise; honest never leaves chance')\n"
    "plt.legend(fontsize=9)\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "for k, leaky_k, honest_k in zip(ks, leaky, honest):\n"
    "    print(f'k={k:>4}  leaky={leaky_k:.3f}  honest={honest_k:.3f}  gap={leaky_k-honest_k:+.3f}')"
)

# ---- Step 6: target leakage — build the dataset ----
add_md(
    "## Step 6 — Target leakage on REAL data: a proxy for the label\n"
    "\n"
    "The second leak needs no synthetic noise — it happens on real datasets constantly. We load the real "
    "**Breast Cancer Wisconsin** data (569 patients, 30 genuine measurements) and append one realistic "
    "**leaky** column: a *confirmatory marker recorded after the biopsy*. In the database it looks like just "
    "another feature; in truth it is a near-copy of the diagnosis — and it will not exist when we predict on "
    "a new patient."
)
add_code(
    "bundle = make_breast_cancer_leak()\n"
    "print(f'real features: {bundle.x_clean.shape[1]}   (+1 injected leak = {bundle.x_leaky.shape[1]} total)')\n"
    "leak = bundle.x_leaky[:, bundle.leak_col]\n"
    "corr = np.abs(np.corrcoef(leak, bundle.y)[0, 1])\n"
    "print(f'the leaked column: {LEAK_COL_NAME!r}')\n"
    "print(f'|corr(leaked column, label)| = {corr:.3f}   (a feature that is basically the answer)')"
)

# ---- Step 7: measure target leakage ----
add_md(
    "## Step 7 — Measure it: with the leak, without it, and the leak alone\n"
    "\n"
    "Cross-validate a random forest three ways: with the leaked column, on the 30 real features only, and on "
    "the **leaked column alone**. That last one is the smoking gun — if a *single* column predicts the "
    "diagnosis almost perfectly, it is not a feature, it is the label in disguise. The cure is not a clever "
    "transform; it is *deleting information you won't have at prediction time*."
)
add_code(
    "tgt = target_leak(bundle)\n"
    "print(f'accuracy WITH the leaked column       : {tgt.acc_with_leak:.3f}   <- inflated')\n"
    "print(f'accuracy on the 30 REAL features only : {tgt.acc_without_leak:.3f}   <- honest')\n"
    "print(f'accuracy using ONLY the leaked column : {tgt.acc_leak_only:.3f}   <- it IS the answer')\n"
    "print(f'\\ninflation gap: {tgt.gap:+.3f}. The 1.00 is a mirage — one post-diagnosis column carries it.')\n"
    "assert tgt.acc_with_leak > tgt.acc_without_leak\n"
    "assert tgt.acc_leak_only > 0.9"
)

# ---- Step 8: plot the leaked column ----
add_md(
    "## Step 8 — See why: the leaked column separates the classes on its own\n"
    "\n"
    "Plot the leaked column split by diagnosis. Two cleanly separated humps mean the column all but *is* the "
    "label — exactly the fingerprint of target leakage. A legitimate feature overlaps heavily across classes; "
    "a leak looks like this."
)
add_code(
    "plt.figure(figsize=(8.5, 4.6))\n"
    "for cls, colour, name in [(0, '#8B3B4A', 'class 0'), (1, '#3A6B96', 'class 1')]:\n"
    "    plt.hist(leak[bundle.y == cls], bins=30, color=colour, alpha=0.65, edgecolor='white', lw=0.4, label=name)\n"
    "plt.xlabel(f'value of {LEAK_COL_NAME!r}')\n"
    "plt.ylabel('count (patients)')\n"
    "plt.title(f'the leaked column split by label (|corr| = {corr:.2f}): two separated humps = a leak')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: temporal leakage — build ----
add_md(
    "## Step 9 — Temporal leakage: build a realistic time series\n"
    "\n"
    "The third leak is about *time*. We simulate two years of daily observations with the three ingredients "
    "real series have: an upward **trend**, a **weekly** cycle, and **autocorrelated** noise (today looks "
    "like yesterday). All three make adjacent days similar — which is exactly what a careless split will "
    "exploit. The task: predict each day from its previous 7 days (lag features)."
)
add_code(
    "ts = make_time_series()\n"
    "print(f'series: {ts.series.shape[0]} daily observations')\n"
    "print(f'supervised rows: {ts.x.shape[0]} x {ts.x.shape[1]} lag features (y[t-1..t-7]) -> y[t]')\n"
    "plt.figure(figsize=(11, 3.6))\n"
    "plt.plot(ts.series, color='#1C2530', lw=0.8)\n"
    "plt.xlabel('day')\n"
    "plt.ylabel('value')\n"
    "plt.title('a realistic daily series: trend + weekly season + autocorrelated noise')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: measure temporal leakage ----
add_md(
    "## Step 10 — Random split vs forward TimeSeriesSplit\n"
    "\n"
    "Same model (ridge on the lag features), same data — only the **splitting** differs. A shuffled K-fold "
    "puts *future* days into the training set for a given test day; since neighbours are similar, the model "
    "practically reads the answer off them, and R² looks great. `TimeSeriesSplit` always trains on an earlier "
    "window and tests on a later one — the real forecasting task — and reports the honest, lower R²."
)
add_code(
    "tmp = temporal_leak(ts)\n"
    "print(f'shuffled K-fold R^2 (trains on the FUTURE)   : {tmp.shuffled_r2:.3f}   <- inflated')\n"
    "print(f'forward TimeSeriesSplit R^2 (past -> future) : {tmp.forward_r2:.3f}   <- honest')\n"
    "print(f'\\ninflation gap: {tmp.gap:+.3f} R^2. Shuffling time-ordered data lets you see tomorrow.')\n"
    "assert tmp.shuffled_r2 > tmp.forward_r2"
)

# ---- Step 11: plot the temporal split ----
add_md(
    "## Step 11 — Why: a shuffled split scatters the future into training\n"
    "\n"
    "The picture makes it obvious. A **forward** split cuts the timeline once: train on the past (left), "
    "predict the future (right). A **shuffled** split sprinkles test days (red) *between* training days "
    "(blue) — so for almost every test day, both its neighbours are in the training set, and the model can "
    "interpolate the answer it is supposed to forecast."
)
add_code(
    "n = ts.series.shape[0]\n"
    "cut = int(n * 0.8)\n"
    "rng = np.random.default_rng(0)\n"
    "test_idx = rng.choice(n, size=int(n * 0.2), replace=False)\n"
    "mask = np.zeros(n, dtype=bool)\n"
    "mask[test_idx] = True\n"
    "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)\n"
    "a1.plot(ts.series, color='#1C2530', lw=0.8)\n"
    "a1.axvspan(0, cut, color='#3A6B96', alpha=0.14)\n"
    "a1.axvspan(cut, n, color='#7A6528', alpha=0.20)\n"
    "a1.set_title('forward split: train on the past (blue), predict the future (amber) — HONEST')\n"
    "a1.set_ylabel('value')\n"
    "a2.scatter(np.arange(n)[~mask], ts.series[~mask], s=5, color='#3A6B96', alpha=0.5, label='train')\n"
    "a2.scatter(np.arange(n)[mask], ts.series[mask], s=11, color='#8B3B4A', alpha=0.9, label='test')\n"
    "a2.set_title('shuffled split: test days sit BETWEEN training days — LEAKY')\n"
    "a2.set_ylabel('value')\n"
    "a2.set_xlabel('day')\n"
    "a2.legend(ncol=2, fontsize=9)\n"
    "a1.grid(alpha=0.3)\n"
    "a2.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 12: the unifying fix ----
add_md(
    "## Step 12 — The one habit that prevents all three\n"
    "\n"
    "Three different leaks, one discipline: **every data-dependent step must be fit using only the training "
    "portion of each split — and the split must respect time.** In scikit-learn that is two rules:\n"
    "\n"
    "1. Put **every** transform (scaling, imputation, feature selection, encoding, target statistics) inside "
    "a `Pipeline`, and cross-validate the *pipeline* — never fit a transform on all the data first.\n"
    "2. For time-ordered data, split by **time** (`TimeSeriesSplit`), never shuffle.\n"
    "\n"
    "And drop any feature that won't exist at prediction time. Below is the honest pattern in full — the "
    "template you should reach for every time."
)
add_code(
    "from sklearn.preprocessing import StandardScaler\n"
    "\n"
    "# the honest, leak-proof template: transforms fit per-fold, inside cross-validation\n"
    "honest_pipeline = Pipeline([\n"
    "    ('scale', StandardScaler()),               # fit on each training fold only\n"
    "    ('select', SelectKBest(f_classif, k=50)),  # fit on each training fold only\n"
    "    ('clf', LogisticRegression(max_iter=1000, random_state=42)),\n"
    "])\n"
    "score = np.mean(cross_val_score(honest_pipeline, noise.x, noise.y, cv=cv))\n"
    "print(f'honest pipeline on the noise data: {score:.3f}  (~chance {CHANCE:.2f}, as it must be)')\n"
    "print('Every transform saw only training rows. No number here is a lie.')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "Three leaks, each measured as the gap between a leaky protocol and the honest truth:\n"
    "\n"
    "| Step | Leak | Leaky score | Honest score | The fix |\n"
    "|---|---|---|---|---|\n"
    "| 1–5 | **Preprocessing** — select features on all data | **0.86** | **0.48** (≈ chance) | selection inside a `Pipeline`, re-fit per fold |\n"
    "| 6–8 | **Target** — a proxy-of-the-label column (real data) | **1.00** | **0.96** | drop features unavailable at prediction time |\n"
    "| 9–11 | **Temporal** — a shuffled split of a time series | **0.97** R² | **0.69** R² | split by time (`TimeSeriesSplit`) |\n"
    "| 12 | — | — | — | fit every transform on train only; respect time |\n"
    "\n"
    "**Leakage is about the protocol, not the model.** A score you cannot reproduce out-of-sample is the red "
    "flag — and the only way to catch a leak is to build the honest protocol and watch the inflated number "
    "collapse. Fit every data-dependent step inside the cross-validation loop, split time-ordered data by "
    "time, and never train on information the future will not have."
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

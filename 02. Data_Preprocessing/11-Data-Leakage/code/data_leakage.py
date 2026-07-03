"""Data leakage MEASURED on real code — the chapter module.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from
real pipelines (``numpy`` + ``scikit-learn``): a leaky protocol that reports an **inflated** score, the
correct protocol (``sklearn.pipeline.Pipeline`` fit inside cross-validation) that reports the **honest**
score, and the measured gap between them. The whole point of the chapter is that gap — a model that
looks great in your notebook and then collapses in the real world, because at training time it saw
information it will not have at prediction time.

Three distinct leakage types, each with the inflated score, the honest score, and the fix:

  * **Preprocessing leakage (the headline).** The classic "wrong way to cross-validate": pick the
    ``k`` features most correlated with the label using the *whole* dataset, then cross-validate a
    classifier on those features. On a controlled dataset of pure **noise** with a **random** label —
    where the honest generalization accuracy is provably **chance (0.50)** — this leaky protocol reports
    a wildly inflated CV accuracy, because the feature selection peeked at the very rows it is later
    scored on. Do the identical steps inside a ``Pipeline`` (so selection is re-fit on each training
    fold only) and the score collapses to ~0.50 — the truth. Using noise is the strongest possible
    control: any accuracy above chance is *pure leakage*, measured. This is the experiment from Hastie,
    Tibshirani & Friedman, *The Elements of Statistical Learning*, S 7.10.2 ("The Wrong and Right Way to
    Do Cross-validation").

  * **Target leakage (a proxy for the label) on a REAL dataset.** On the real Breast Cancer Wisconsin
    dataset that ships with scikit-learn (569 patients, 30 real measurements), we add one realistic
    **leaky** column: a "confirmatory marker" that is really recorded *after* the diagnosis and is
    therefore a near-copy of the label. The model's accuracy jumps toward 1.00 and the leaked column
    alone predicts the diagnosis almost perfectly — proof it *is* the answer in disguise. Drop the one
    column and the honest accuracy returns. At prediction time that column does not exist, so the
    inflated score is a lie.

  * **Temporal leakage on a realistic time series.** For time-ordered data, a *random* train/test split
    lets the model train on the future and predict the past. On a realistic daily series (trend + weekly
    seasonality + autocorrelated noise), shuffled K-fold cross-validation reports an inflated score,
    while a forward-chronological ``TimeSeriesSplit`` (train only on the past) reports the honest, lower
    score you would actually get in production.

Everything is seeded and CPU-only; runs standalone in a few seconds::

    python data_leakage.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RANDOM_STATE = 42  # the one seed that fixes every dataset, split, and model
CV_FOLDS = 5  # folds for cross-validation throughout
CHANCE = 0.5  # honest accuracy of a balanced two-class problem with NO signal

# -- Demo 1: preprocessing leakage (feature selection before the split) --
NOISE_SAMPLES = 200  # rows in the controlled pure-noise dataset
NOISE_FEATURES = 5000  # columns of pure Gaussian noise (p >> n makes the leak vivid)
SELECT_K = 50  # features the leaky selector cherry-picks (correlated with y BY CHANCE)
K_SWEEP = (1, 2, 5, 10, 20, 50, 100, 200)  # k values for the "leak grows with k" figure

# -- Demo 2: target leakage on the real Breast Cancer dataset --
LEAK_NOISE_STD = 0.12  # noise on the injected proxy column (small => a near-perfect leak, ~100%)
LEAK_COL_NAME = "confirmatory_marker (LEAK)"  # the injected post-diagnosis proxy feature

# -- Demo 3: temporal leakage on a realistic time series --
SERIES_LEN = 730  # two years of daily observations
TREND_PER_STEP = 0.03  # upward drift per day (a real trend the random split can interpolate)
SEASON_PERIOD = 7  # weekly seasonality
SEASON_AMP = 2.0  # amplitude of the weekly cycle
AR_PHI = 0.6  # autocorrelation of the noise (neighbours are similar => shuffling leaks)
NOISE_STD = 1.0  # innovation std of the AR(1) noise
N_LAGS = 7  # lag features y[t-1..t-7] used to predict y[t]


# ============================ 1. datasets ======================================================
@dataclass
class NoiseData:
    """A controlled dataset of pure noise with a random label — the honest accuracy is chance (0.5)."""

    x: NDArray[np.float64]  # (n, p) standard-normal noise, NO relationship to y
    y: NDArray[np.int64]  # (n,) balanced random 0/1 label


def make_noise_data(
    *, n: int = NOISE_SAMPLES, p: int = NOISE_FEATURES, seed: int = RANDOM_STATE
) -> NoiseData:
    """Pure Gaussian-noise features with a random balanced binary label.

    There is *no* signal: every column is independent standard normal, drawn without ever looking at
    ``y``. So the best any honest model can do out-of-sample is **chance (0.5)**. That is exactly why we
    use noise — it turns leakage into a measurement: any cross-validated accuracy above 0.5 is not skill,
    it is information about the test rows leaking into training.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = np.zeros(n, dtype=np.int64)
    y[: n // 2] = 1
    rng.shuffle(y)  # a coin-flip label, independent of x
    return NoiseData(x=x, y=y)


@dataclass
class BreastCancerLeak:
    """The real Breast Cancer dataset plus one injected 'leaky' proxy-of-the-label column."""

    x_clean: NDArray[np.float64]  # (n, 30) the real measurements only
    x_leaky: NDArray[np.float64]  # (n, 31) real measurements + the leaked column (last)
    y: NDArray[np.int64]  # (n,) diagnosis: 1 = malignant-encoding per sklearn
    feature_names: list[str]  # 30 real names + the injected leak's name
    leak_col: int  # index of the leaked column in x_leaky


def make_breast_cancer_leak(
    *, noise_std: float = LEAK_NOISE_STD, seed: int = RANDOM_STATE
) -> BreastCancerLeak:
    """Load real Breast Cancer data and append a realistic *target-leaking* column.

    The 30 real features (radius, texture, concavity, ...) are legitimate inputs available before a
    diagnosis. The injected column simulates a **confirmatory marker recorded *after* the biopsy** — the
    kind of field that sneaks into a training table because it was in the database, but that does not
    exist at prediction time. We build it as the label plus a little Gaussian noise, so it is a *near
    copy of the answer*. A model handed this column will lean on it and post a near-perfect score; the
    leaked column alone predicts the diagnosis almost perfectly. Both facts are measured below.
    """
    data = load_breast_cancer()
    x_clean = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    leak = y.astype(np.float64) + rng.normal(0.0, noise_std, size=y.shape)  # proxy for the label
    x_leaky = np.column_stack([x_clean, leak])
    names = list(data.feature_names) + [LEAK_COL_NAME]
    return BreastCancerLeak(
        x_clean=x_clean,
        x_leaky=x_leaky,
        y=y,
        feature_names=names,
        leak_col=x_leaky.shape[1] - 1,
    )


@dataclass
class TimeSeriesData:
    """A realistic daily time series turned into a supervised lag-feature regression problem."""

    x: NDArray[np.float64]  # (n_obs, N_LAGS) lagged values y[t-1..t-N_LAGS]
    y: NDArray[np.float64]  # (n_obs,) the value to predict, y[t]
    series: NDArray[np.float64]  # the raw series (for plotting)
    t: NDArray[np.int64]  # time index of each supervised row (for the forward-split figure)


def make_time_series(
    *,
    n: int = SERIES_LEN,
    n_lags: int = N_LAGS,
    seed: int = RANDOM_STATE,
) -> TimeSeriesData:
    """Simulate a realistic daily series (trend + weekly season + AR(1) noise) as a lag regression.

    The series has three real-world ingredients: an upward **trend**, a **weekly** cycle, and
    **autocorrelated** noise (today looks like yesterday). All three make *adjacent days similar* — which
    is precisely why a *random* split leaks: if a test day's neighbours are in the training set, the
    model can all but read the answer off them. A forward split (train on the past, test on the future)
    cannot, so it reports the honest error you would truly see tomorrow.
    """
    rng = np.random.default_rng(seed)
    trend = TREND_PER_STEP * np.arange(n)
    season = SEASON_AMP * np.sin(2 * np.pi * np.arange(n) / SEASON_PERIOD)
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, NOISE_STD)
    for i in range(1, n):
        noise[i] = AR_PHI * noise[i - 1] + rng.normal(0.0, NOISE_STD)  # AR(1): today ~ yesterday
    series = trend + season + noise

    rows_x, rows_y, rows_t = [], [], []
    for t in range(n_lags, n):
        rows_x.append(series[t - n_lags : t][::-1])  # [y[t-1], y[t-2], ..., y[t-n_lags]]
        rows_y.append(series[t])
        rows_t.append(t)
    return TimeSeriesData(
        x=np.array(rows_x),
        y=np.array(rows_y),
        series=series,
        t=np.array(rows_t, dtype=np.int64),
    )


# ============================ 2. demo 1: preprocessing leakage (feature selection) ===============
@dataclass
class SelectionLeak:
    """Feature-selection leakage: the inflated leaky CV score, the honest Pipeline score, and a held-out check."""

    leaky_cv: float  # CV accuracy when SelectKBest is fit on ALL data before cross-validation
    honest_cv: float  # CV accuracy when SelectKBest lives INSIDE the Pipeline (re-fit per fold)
    honest_holdout: float  # accuracy of the honest pipeline on a truly untouched hold-out set
    k: int  # number of features selected
    gap: float  # leaky_cv - honest_cv (the measured inflation)


def selection_leak(data: NoiseData, *, k: int = SELECT_K, seed: int = RANDOM_STATE) -> SelectionLeak:
    """The headline demo: select features on all data (leaky) vs inside a Pipeline (honest) vs a hold-out.

    * **Leaky.** ``SelectKBest`` is fit on the *entire* ``x, y`` — so the chosen columns are the ones that
      happen to correlate with ``y`` across *all* rows, including the rows each CV fold later scores on.
      The selector has peeked at the test folds. ``cross_val_score`` on the pre-selected columns then
      reports a badly optimistic number.

    * **Honest.** The identical ``SelectKBest`` + classifier live in a ``Pipeline`` handed whole to
      ``cross_val_score``. Now selection is re-fit on each *training* fold only, so the columns it picks
      are correlated with ``y`` on train but not on the held-out fold — and the score falls to chance.

    * **Hold-out.** As an independent check we also fit the honest pipeline on a training split and score
      it once on a never-touched test split. It agrees with the honest CV (~chance), confirming *that* is
      the trustworthy number.

    On pure noise the honest answer is provably 0.5, so ``gap = leaky_cv - honest_cv`` is leakage,
    measured to the point.
    """
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    # -- leaky: select on ALL the data, THEN cross-validate on the pre-selected columns --
    x_selected = SelectKBest(f_classif, k=k).fit_transform(data.x, data.y)
    leaky = float(np.mean(cross_val_score(clf, x_selected, data.y, cv=cv)))

    # -- honest: selection lives INSIDE the pipeline, re-fit on each training fold only --
    pipe = Pipeline([("select", SelectKBest(f_classif, k=k)), ("clf", clf)])
    honest = float(np.mean(cross_val_score(pipe, data.x, data.y, cv=cv)))

    # -- independent hold-out check of the honest pipeline --
    x_tr, x_te, y_tr, y_te = train_test_split(
        data.x, data.y, test_size=0.3, random_state=seed, stratify=data.y
    )
    holdout_pipe = Pipeline([("select", SelectKBest(f_classif, k=k)), ("clf", clf)])
    holdout_pipe.fit(x_tr, y_tr)
    holdout = float(holdout_pipe.score(x_te, y_te))

    return SelectionLeak(
        leaky_cv=leaky,
        honest_cv=honest,
        honest_holdout=holdout,
        k=k,
        gap=leaky - honest,
    )


def selection_leak_sweep(
    data: NoiseData, *, ks: tuple[int, ...] = K_SWEEP, seed: int = RANDOM_STATE
) -> tuple[list[int], list[float], list[float]]:
    """Sweep the number of selected features ``k``; return (ks, leaky CV, honest CV) for the figure.

    The leaky curve *climbs toward 1.0* as ``k`` grows — with more columns to cherry-pick from the whole
    dataset, the selector finds an ever-better spurious fit to the test folds. The honest curve stays
    pinned near chance for every ``k``, because per-fold selection can never exploit the held-out rows.
    The widening gap between the two curves is the leak, drawn as a function of how much you let it in.
    """
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    leaky_scores, honest_scores = [], []
    for k in ks:
        x_selected = SelectKBest(f_classif, k=k).fit_transform(data.x, data.y)
        leaky_scores.append(float(np.mean(cross_val_score(clf, x_selected, data.y, cv=cv))))
        pipe = Pipeline([("select", SelectKBest(f_classif, k=k)), ("clf", clf)])
        honest_scores.append(float(np.mean(cross_val_score(pipe, data.x, data.y, cv=cv))))
    return list(ks), leaky_scores, honest_scores


# ============================ 3. demo 2: target leakage on real data ============================
@dataclass
class TargetLeak:
    """Target leakage: accuracy with a leaked proxy column, without it, and the leaked column alone."""

    acc_with_leak: float  # CV accuracy of the model WITH the leaked column (inflated)
    acc_without_leak: float  # CV accuracy on the real features only (honest)
    acc_leak_only: float  # CV accuracy using ONLY the leaked column (it IS the label in disguise)
    gap: float  # acc_with_leak - acc_without_leak
    leak_corr: float  # |correlation| between the leaked column and the label


def target_leak(bundle: BreastCancerLeak, *, seed: int = RANDOM_STATE) -> TargetLeak:
    """Measure how a single target-derived column inflates a real-data model, then the fix (drop it).

    A random forest is cross-validated three ways on the real Breast Cancer data: with the leaked
    proxy column appended, on the 30 real features only, and on the leaked column *alone*. The last is
    the smoking gun — one column, near-perfect accuracy — proving the "feature" is really the answer.
    The cure is not a clever transform: it is *removing information that will not exist at prediction
    time*. Leakage is about the protocol and the data, not the model.
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    acc_with = float(np.mean(cross_val_score(clf, bundle.x_leaky, bundle.y, cv=cv)))
    acc_without = float(np.mean(cross_val_score(clf, bundle.x_clean, bundle.y, cv=cv)))
    leak_only = bundle.x_leaky[:, bundle.leak_col : bundle.leak_col + 1]
    acc_leak_only = float(np.mean(cross_val_score(clf, leak_only, bundle.y, cv=cv)))
    corr = float(np.abs(np.corrcoef(leak_only.ravel(), bundle.y)[0, 1]))
    return TargetLeak(
        acc_with_leak=acc_with,
        acc_without_leak=acc_without,
        acc_leak_only=acc_leak_only,
        gap=acc_with - acc_without,
        leak_corr=corr,
    )


# ============================ 4. demo 3: temporal leakage =======================================
@dataclass
class TemporalLeak:
    """Temporal leakage: inflated shuffled-CV score vs honest forward-chronological score."""

    shuffled_r2: float  # R^2 from a RANDOM (shuffled) K-fold split — trains on the future
    forward_r2: float  # R^2 from a forward TimeSeriesSplit — trains only on the past (honest)
    gap: float  # shuffled_r2 - forward_r2


def temporal_leak(data: TimeSeriesData, *, seed: int = RANDOM_STATE) -> TemporalLeak:
    """Compare a shuffled split (leaky) with a forward ``TimeSeriesSplit`` (honest) on time-ordered data.

    Same model (ridge regression on lag features), same data, only the *splitting* differs. Shuffled
    K-fold puts future days in the training set for a given test day; because neighbouring days are
    similar (trend + season + autocorrelation), the model effectively interpolates the answer, and R^2
    looks great. ``TimeSeriesSplit`` always trains on an earlier window and tests on a later one — the
    real forecasting task — so it reports the honest, lower R^2. The gap is the price of pretending you
    can see the future.
    """
    model = Ridge(alpha=1.0, random_state=seed)
    shuffled_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    forward_cv = TimeSeriesSplit(n_splits=CV_FOLDS)
    shuffled = float(np.mean(cross_val_score(model, data.x, data.y, cv=shuffled_cv, scoring="r2")))
    forward = float(np.mean(cross_val_score(model, data.x, data.y, cv=forward_cv, scoring="r2")))
    return TemporalLeak(shuffled_r2=shuffled, forward_r2=forward, gap=shuffled - forward)


# ============================ 5. run it all: the printed proof ==================================
def main() -> None:
    """Run every leakage demo and cross-check, printing the results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    # ---- Demo 1: preprocessing leakage (feature selection before the split) ----
    noise = make_noise_data()
    sel = selection_leak(noise)
    print("=== 1. Preprocessing leakage: feature selection BEFORE cross-validation ===")
    print(f"  dataset: {noise.x.shape[0]} rows x {noise.x.shape[1]} PURE-NOISE features, random label")
    print(f"  (there is NO signal, so the honest accuracy is provably chance = {CHANCE:.2f})")
    print(f"  leaky   CV accuracy (select on ALL data, then CV) : {sel.leaky_cv:.3f}   <- inflated FICTION")
    print(f"  honest  CV accuracy (select INSIDE the Pipeline)   : {sel.honest_cv:.3f}   <- the truth (~chance)")
    print(f"  honest  hold-out accuracy (untouched test set)     : {sel.honest_holdout:.3f}   <- confirms it")
    print(f"  measured inflation gap                             : {sel.gap:+.3f}")
    if not (sel.leaky_cv > sel.honest_cv + 0.15):
        raise AssertionError("the leaky protocol must materially inflate the score above the honest one")
    if not (abs(sel.honest_cv - CHANCE) < 0.1):
        raise AssertionError("on pure noise the honest CV score must sit near chance (0.5)")
    print("  -> the selector peeked at the test folds. The Pipeline fix re-fits it per fold.\n")

    # ---- Demo 2: target leakage on the real Breast Cancer dataset ----
    bundle = make_breast_cancer_leak()
    tgt = target_leak(bundle)
    print("=== 2. Target leakage: a proxy-of-the-label column on REAL Breast Cancer data ===")
    print(f"  dataset: {bundle.x_clean.shape[0]} patients x {bundle.x_clean.shape[1]} real features "
          f"(+1 injected leak)")
    print(f"  |corr(leaked column, label)|                 : {tgt.leak_corr:.3f}  (it is a near-copy of y)")
    print(f"  accuracy WITH the leaked column              : {tgt.acc_with_leak:.3f}   <- inflated")
    print(f"  accuracy on the 30 REAL features only        : {tgt.acc_without_leak:.3f}   <- honest")
    print(f"  accuracy using ONLY the leaked column        : {tgt.acc_leak_only:.3f}   <- it IS the answer")
    print(f"  measured inflation gap                       : {tgt.gap:+.3f}")
    if not (tgt.acc_with_leak > tgt.acc_without_leak):
        raise AssertionError("the leaked column must inflate accuracy above the honest baseline")
    if not (tgt.acc_leak_only > 0.9):
        raise AssertionError("the leaked column alone should predict the label almost perfectly")
    print("  -> the leak won't exist at prediction time; the honest number is the real one.\n")

    # ---- Demo 3: temporal leakage on a realistic time series ----
    ts = make_time_series()
    tmp = temporal_leak(ts)
    print("=== 3. Temporal leakage: shuffled split vs forward TimeSeriesSplit ===")
    print(f"  dataset: {ts.series.shape[0]} daily observations (trend + weekly season + AR(1) noise)")
    print(f"  shuffled K-fold R^2 (trains on the FUTURE)   : {tmp.shuffled_r2:.3f}   <- inflated")
    print(f"  forward TimeSeriesSplit R^2 (past -> future) : {tmp.forward_r2:.3f}   <- honest")
    print(f"  measured inflation gap                       : {tmp.gap:+.3f}")
    if not (tmp.shuffled_r2 > tmp.forward_r2):
        raise AssertionError("shuffling time-ordered data must inflate the score vs a forward split")
    print("  -> a random split lets the model read tomorrow off its neighbours. Split by time.\n")

    print("All three leaks reproduce: leaky > honest, every time, from the same seed.")


if __name__ == "__main__":
    main()

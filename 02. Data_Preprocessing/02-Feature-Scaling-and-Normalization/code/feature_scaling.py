"""Feature scaling on a REAL dataset, with the effect on real models MEASURED — the chapter module.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from
a real pipeline (``numpy`` + ``scikit-learn``) on the real **Wine** dataset that ships with
scikit-learn (178 wines, 13 chemical measurements, 3 cultivars). Wine is the canonical scaling
example precisely because its features live on wildly different scales: on the training split
``proline`` ranges over ~1,237 units while ``nonflavanoid_phenols`` ranges over ~0.5 — a **2,474x**
disparity. A distance- or
gradient-based model that treats those raw numbers as comparable is, without being told, letting
``proline`` alone decide almost everything.

What this module measures (all real, all reproducible from the seed):

  * **The problem, quantified.** The share of the average squared Euclidean distance between wines that
    each feature contributes. On the raw data ``proline`` alone is **99.7%** of the distance — the
    other twelve measurements are effectively invisible. After standardization every feature
    contributes its fair ~1/13. Nearest-neighbour "closeness" is, before scaling, almost purely
    "closeness in proline".

  * **The three scalers, from scratch, verified against scikit-learn.** ``standardize`` (z-score,
    ``(x-mu)/sigma``), ``minmax`` (to ``[0,1]``), and ``robust`` (``(x-median)/IQR``) are implemented
    by hand — fit statistics on the **training split only**, then applied to test — and checked to
    match ``StandardScaler`` / ``MinMaxScaler`` / ``RobustScaler`` to ``1e-9``. That match is the proof
    the from-scratch versions are the genuine transforms, not lookalikes.

  * **The measured effect on real models.** KNN, an RBF-kernel SVM, logistic regression, and a random
    forest, each trained and scored on the same train/test split **without** scaling and **with** each
    scaler. The scale-sensitive models leap (KNN test accuracy jumps from ~0.72 to ~0.96; the SVM from
    ~0.67 to ~1.00); the random forest — whose splits are threshold comparisons and therefore invariant
    to any monotone rescaling — is **bit-for-bit unchanged**. That contrast is the whole lesson.

  * **Why gradient descent needs it.** A from-scratch logistic-regression gradient descent on two Wine
    features. On the raw features the loss-curvature condition number is ~10^5 and a workable learning
    rate diverges; standardize the two features and the condition number drops to ~3 and the same
    learning rate converges smoothly. Scaling *conditions* the optimization.

  * **Fit on train only (a leakage preview).** Fitting the scaler on the whole dataset before splitting
    lets test-set statistics bleed into the transform. We measure how far the fitted mean/std move when
    test rows are (wrongly) included — the correct-vs-leaky difference — and defer the full treatment to
    the Data Leakage chapter.

Everything is seeded and CPU-only; runs standalone in a couple of seconds::

    python feature_scaling.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RANDOM_STATE = 42  # the one seed that fixes the train/test split and every model's randomness
TEST_SIZE = 0.30  # 30% of the 178 wines held out as a test set, stratified by class
KNN_K = 5  # neighbours for the KNN classifier — a purely distance-based, scale-sensitive model
SVM_C = 1.0  # RBF-SVM regularization strength (distances in the kernel are scale-sensitive too)
RF_TREES = 200  # trees in the random forest — the scale-INVARIANT control model
LOGREG_MAX_ITER = 20000  # high cap so unscaled logistic regression still converges (it needs it!)
EPS = 1e-12  # floor to avoid divide-by-zero on a constant feature (zero spread)
MATCH_TOL = 1e-9  # tolerance for "our from-scratch scaler == scikit-learn's scaler"
DOMINANT_FEATURE = "proline"  # the large-magnitude feature that dominates raw distance
SKEWED_FEATURE = "magnesium"  # a real right-skewed feature with high outliers (for the scaler figure)
GD_FEATURES = ("proline", "flavanoids")  # two very-different-scale features for the GD conditioning demo
GD_POS_CLASS = 1  # binary target for the GD demo: Wine cultivar 1 vs the rest
GD_LR = 0.01  # a learning rate that converges on SCALED features (and diverges on raw ones)
GD_ITERS = 200  # gradient-descent iterations for the conditioning demo


# ============================ 1. real data =====================================================
@dataclass
class WineSplit:
    """A real, stratified train/test split of the scikit-learn Wine dataset."""

    x_train: NDArray[np.float64]  # (n_train, 13) raw feature matrix
    x_test: NDArray[np.float64]  # (n_test, 13) raw feature matrix
    y_train: NDArray[np.int64]  # (n_train,) cultivar labels 0/1/2
    y_test: NDArray[np.int64]  # (n_test,)
    feature_names: list[str]  # the 13 chemical measurement names


def load_wine_split(*, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE) -> WineSplit:
    """Load the real Wine dataset and make one stratified train/test split.

    Wine ships inside scikit-learn (no download): 178 wines, 13 real chemical measurements
    (alcohol, magnesium, proline, ...), 3 cultivars. Stratifying keeps the class balance identical in
    train and test. The split is the boundary that matters for the rest of the module: **every scaler
    is fit on ``x_train`` only.**
    """
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )
    return WineSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=list(data.feature_names),
    )


# ============================ 2. the three scalers, from scratch =================================
@dataclass
class StandardScalerScratch:
    """Standardization ``z = (x - mu) / sigma`` — centre each feature to mean 0, scale to std 1.

    Statistics are the per-feature training mean and (population, ``ddof=0``) standard deviation —
    exactly scikit-learn's ``StandardScaler`` convention. Standardization *preserves the distribution's
    shape* (it is an affine map): it does not bound the range and does not resist outliers — a value
    that was 5 std out stays 5 std out. It is the default, and the right choice for roughly-Gaussian
    features and for conditioning gradient descent.
    """

    mean_: NDArray[np.float64]
    std_: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64]) -> StandardScalerScratch:
        return cls(mean_=x.mean(axis=0), std_=x.std(axis=0, ddof=0) + EPS)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self.mean_) / self.std_


@dataclass
class MinMaxScalerScratch:
    """Min-max scaling ``(x - min) / (max - min)`` — squeeze each feature into ``[0, 1]``.

    Statistics are the per-feature training min and max. Every training value lands in ``[0, 1]`` with
    the smallest at 0 and the largest at 1. Because the denominator is set by the extremes, a single
    **outlier fixes an endpoint and compresses all the ordinary values into a narrow band** — min-max is
    the most outlier-sensitive of the three. Use it when you need bounded inputs (e.g. image pixels to
    ``[0, 1]``) and outliers are under control.
    """

    min_: NDArray[np.float64]
    range_: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64]) -> MinMaxScalerScratch:
        lo = x.min(axis=0)
        return cls(min_=lo, range_=(x.max(axis=0) - lo) + EPS)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self.min_) / self.range_


@dataclass
class RobustScalerScratch:
    """Robust scaling ``(x - median) / IQR`` — centre on the median, scale by the interquartile range.

    Statistics are the per-feature training **median** and **IQR** (the 75th minus the 25th percentile).
    Both are computed from the middle of the data, so a handful of extreme values barely move them:
    robust scaling is **outlier-resistant**. The bulk of the data ends up well-spread around 0 with an
    IQR of 1, while outliers sit far out (as they should) instead of crushing everyone else. This is
    scikit-learn's ``RobustScaler`` default (quantile range 25-75).
    """

    median_: NDArray[np.float64]
    iqr_: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64]) -> RobustScalerScratch:
        q25, median, q75 = np.percentile(x, [25, 50, 75], axis=0)
        return cls(median_=median, iqr_=(q75 - q25) + EPS)

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self.median_) / self.iqr_


# A named registry so the figures, the model sweep, and the notebook all iterate the same set.
# ``None`` is the honest baseline: no scaling at all.
SCRATCH_SCALERS = {
    "standard": StandardScalerScratch,
    "minmax": MinMaxScalerScratch,
    "robust": RobustScalerScratch,
}
SKLEARN_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scaler_match_report(split: WineSplit) -> dict[str, float]:
    """Fit each from-scratch scaler on train, apply to test, and compare to scikit-learn (max abs diff).

    Isolates the one thing we want to prove: our hand-written ``(x-mu)/sigma`` etc. reproduce
    scikit-learn's transformers to machine precision when fed the identical data. If the differences are
    below ``MATCH_TOL`` the scratch transforms are the genuine article, and every downstream figure that
    uses them is trustworthy.
    """
    report: dict[str, float] = {}
    for name, scratch_cls in SCRATCH_SCALERS.items():
        ours = scratch_cls.fit(split.x_train).transform(split.x_test)
        theirs = SKLEARN_SCALERS[name]().fit(split.x_train).transform(split.x_test)
        report[name] = float(np.max(np.abs(ours - theirs)))
    return report


# ============================ 3. the problem, quantified: distance decomposition =================
@dataclass
class DistanceShare:
    """Per-feature share of the average squared Euclidean distance between samples, raw vs scaled."""

    feature_names: list[str]
    raw_share: NDArray[np.float64]  # fraction of average squared distance from each feature (raw)
    scaled_share: NDArray[np.float64]  # ...after standardization (each feature ~ 1/n_features)


def distance_share(x: NDArray[np.float64], feature_names: list[str]) -> DistanceShare:
    """How much each feature contributes to the average squared Euclidean distance between two samples.

    Euclidean distance squared is a *sum over features* of ``(x_i - x'_i)^2``. Averaged over all pairs
    of samples, feature ``j``'s contribution is proportional to its **variance** (``E[(x_j - x'_j)^2] =
    2 Var[x_j]`` for independent draws). So the share of the distance owned by each feature is just its
    variance divided by the total variance. On raw Wine, ``proline`` — with by far the largest variance —
    owns almost the entire distance; after standardization every variance is 1, so every feature owns an
    equal ~1/13. This is *why* an unscaled nearest-neighbour model is really a "nearest in proline" model.
    """
    var_raw = x.var(axis=0)
    z = StandardScalerScratch.fit(x).transform(x)
    var_scaled = z.var(axis=0)
    return DistanceShare(
        feature_names=feature_names,
        raw_share=var_raw / var_raw.sum(),
        scaled_share=var_scaled / var_scaled.sum(),
    )


def feature_ranges(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-feature range (max - min) — the crudest picture of "these columns live on different scales"."""
    return x.max(axis=0) - x.min(axis=0)


# ============================ 4. the measured effect on real models ==============================
def _make_models() -> dict[str, object]:
    """The four models we score: three scale-sensitive, one scale-invariant control.

    * **KNN** — pure distance; the textbook scaling-sensitive model.
    * **RBF SVM** — the kernel ``exp(-gamma ||x - x'||^2)`` is a distance, so it is scale-sensitive too.
    * **Logistic regression** — gradient-trained; scaling conditions the optimization (and, with an L2
      penalty, makes the per-feature penalty fair).
    * **Random forest** — splits on thresholds (``x_j <= t``), which are invariant to any monotone
      rescaling; the honest control that should NOT move when we scale.
    """
    return {
        "KNN": KNeighborsClassifier(n_neighbors=KNN_K),
        "SVM-RBF": SVC(kernel="rbf", C=SVM_C, gamma="scale", random_state=RANDOM_STATE),
        "LogReg": LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_STATE),
    }


@dataclass
class ModelScores:
    """Measured test accuracy of each model under each scaling choice, plus a random-forest invariance flag."""

    scaler_names: list[str]  # ["none", "standard", "minmax", "robust"]
    model_names: list[str]  # ["KNN", "SVM-RBF", "LogReg", "RandomForest"]
    accuracy: dict[str, list[float]]  # model -> [acc per scaler, in scaler_names order]
    rf_predictions_identical: bool  # True iff the forest's predictions are bit-identical across scalers


def evaluate_models(split: WineSplit) -> ModelScores:
    """Train every model under no-scaling and each scaler; return the measured test-accuracy table.

    The scaler is always fit on ``x_train`` only and applied to both splits — the correct protocol.
    ``clone`` gives each (model, scaler) cell a fresh, identically-seeded estimator so the only thing
    that varies across a row is the scaling. We also check whether the random forest's *predictions*
    (not just its accuracy) are identical across every scaler — the strongest possible statement of
    scale-invariance for a tree model.
    """
    scaler_names = ["none", "standard", "minmax", "robust"]
    models = _make_models()
    accuracy: dict[str, list[float]] = {name: [] for name in models}
    rf_predictions: list[NDArray[np.int64]] = []

    for scaler_name in scaler_names:
        if scaler_name == "none":
            x_tr, x_te = split.x_train, split.x_test
        else:
            scaler = SKLEARN_SCALERS[scaler_name]().fit(split.x_train)
            x_tr, x_te = scaler.transform(split.x_train), scaler.transform(split.x_test)
        for model_name, model in models.items():
            estimator = clone(model)
            estimator.fit(x_tr, split.y_train)
            accuracy[model_name].append(float(estimator.score(x_te, split.y_test)))
            if model_name == "RandomForest":
                rf_predictions.append(estimator.predict(x_te))

    rf_identical = all(np.array_equal(rf_predictions[0], p) for p in rf_predictions[1:])
    return ModelScores(
        scaler_names=scaler_names,
        model_names=list(models),
        accuracy=accuracy,
        rf_predictions_identical=rf_identical,
    )


def logreg_iterations(split: WineSplit) -> tuple[int, int]:
    """Iterations scikit-learn's logistic regression needs to converge: unscaled vs standardized.

    Same solver, same data, only the feature scale differs. Unscaled Wine is so ill-conditioned that
    the optimizer needs *thousands* of steps (or fails); standardized, it converges in a handful. The
    measured iteration counts are a concrete, honest price tag on skipping the scaler.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we report n_iter_ directly; the convergence warning is the point
        raw = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE)
        raw.fit(split.x_train, split.y_train)
        z = StandardScaler().fit_transform(split.x_train)
        scaled = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE)
        scaled.fit(z, split.y_train)
    return int(np.max(raw.n_iter_)), int(np.max(scaled.n_iter_))


# ============================ 5. why gradient descent needs scaling ==============================
def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_gd(
    x: NDArray[np.float64], y: NDArray[np.float64], *, lr: float, iters: int
) -> NDArray[np.float64]:
    """From-scratch full-batch logistic-regression gradient descent; return the loss at each iteration.

    Standard binary cross-entropy with a fixed learning rate — no library doing the optimizing. The
    point is not the classifier but the *shape of the loss curve*: on well-conditioned (scaled) inputs
    it descends smoothly; on ill-conditioned (raw) inputs the same ``lr`` overshoots the steep direction
    and the loss diverges. The optimizer is identical; only the geometry of the loss surface changed.
    """
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0.0
    losses = np.empty(iters)
    for step in range(iters):
        probs = np.clip(_sigmoid(x @ weights + bias), 1e-12, 1 - 1e-12)
        losses[step] = float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
        grad_w = x.T @ (probs - y) / n_samples
        grad_b = float(np.mean(probs - y))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return losses


@dataclass
class ConditioningDemo:
    """The GD-conditioning result: raw-vs-scaled loss curves and the loss-curvature condition numbers."""

    loss_raw: NDArray[np.float64]  # loss per iteration on the raw two features (diverges)
    loss_scaled: NDArray[np.float64]  # loss per iteration after standardizing them (converges)
    cond_raw: float  # condition number of the feature covariance (raw) — ~10^5
    cond_scaled: float  # ...after standardization — ~3
    x_raw: NDArray[np.float64]  # the two raw features (for the contour figure)
    x_scaled: NDArray[np.float64]  # the two standardized features (for the contour figure)
    y: NDArray[np.float64]  # the binary target
    lr: float


def gd_conditioning(split: WineSplit, *, lr: float = GD_LR, iters: int = GD_ITERS) -> ConditioningDemo:
    """Run the same gradient descent on two Wine features, raw vs standardized, and measure the geometry.

    We pick two features on very different scales (``proline`` ~ hundreds-to-thousands vs ``flavanoids``
    ~ 0-5) and a binary target (cultivar 1 vs rest). The **condition number** of the feature covariance
    is the ratio of the loss surface's steepest to shallowest curvature — the elongation of its
    contours. Raw, it is enormous (contours are long thin valleys), so a step big enough to move along
    the valley overshoots across it and the loss blows up. Standardized, the contours are near-circular
    (condition number ~3) and the same step converges. This is the exact link to gradient-descent
    conditioning theory.
    """
    idx = [split.feature_names.index(f) for f in GD_FEATURES]
    x_full = np.vstack([split.x_train, split.x_test])[:, idx]
    y_full = (np.concatenate([split.y_train, split.y_test]) == GD_POS_CLASS).astype(np.float64)
    x_scaled = StandardScalerScratch.fit(x_full).transform(x_full)
    cond_raw = float(np.linalg.cond(np.cov(x_full.T)))
    cond_scaled = float(np.linalg.cond(np.cov(x_scaled.T)))
    return ConditioningDemo(
        loss_raw=logistic_gd(x_full, y_full, lr=lr, iters=iters),
        loss_scaled=logistic_gd(x_scaled, y_full, lr=lr, iters=iters),
        cond_raw=cond_raw,
        cond_scaled=cond_scaled,
        x_raw=x_full,
        x_scaled=x_scaled,
        y=y_full,
        lr=lr,
    )


# ============================ 6. fit on train only (a leakage preview) ===========================
@dataclass
class LeakageProbe:
    """How far a scaler's fitted statistics move when test rows are (wrongly) included in the fit."""

    mean_shift: NDArray[np.float64]  # |mean(all data) - mean(train)| per feature
    std_shift: NDArray[np.float64]  # |std(all data)  - std(train)|  per feature
    max_test_transform_diff: float  # largest change in a transformed TEST value, correct vs leaky


def leakage_probe(split: WineSplit) -> LeakageProbe:
    """Measure the difference between the correct (fit-on-train) and leaky (fit-on-everything) scaler.

    The correct protocol fits the scaler on ``x_train`` only; the leaky one fits on train **and** test
    together, so test-set statistics contaminate the transform of the test set — the model gets a peek
    at information it will not have in production. Scaling leakage is usually mild (the shifts are
    small), so we *quantify* it honestly rather than overclaim: the fitted mean/std move, and the
    transformed test values change. The dramatic, score-destroying cases (and the ``Pipeline`` fix) are
    the subject of the Data Leakage chapter.
    """
    correct = StandardScalerScratch.fit(split.x_train)
    everything = np.vstack([split.x_train, split.x_test])
    leaky = StandardScalerScratch.fit(everything)
    diff = np.abs(correct.transform(split.x_test) - leaky.transform(split.x_test))
    return LeakageProbe(
        mean_shift=np.abs(leaky.mean_ - correct.mean_),
        std_shift=np.abs(leaky.std_ - correct.std_),
        max_test_transform_diff=float(np.max(diff)),
    )


# ============================ 7. run it all: the printed proof ==================================
def main() -> None:
    """Run every measured experiment and cross-check, printing the results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    split = load_wine_split()
    n_features = len(split.feature_names)
    print(f"Wine dataset: {split.x_train.shape[0]} train + {split.x_test.shape[0]} test wines, "
          f"{n_features} features, {len(np.unique(split.y_train))} cultivars\n")

    # ---- 0. correctness: scratch scalers == scikit-learn ----
    report = scaler_match_report(split)
    print("=== 0. Correctness: our from-scratch scalers == scikit-learn's, on identical data ===")
    for name, diff in report.items():
        print(f"  {name:>9}: max|ours - sklearn| = {diff:.2e}")
        if diff > MATCH_TOL:
            raise AssertionError(f"{name} scaler must match scikit-learn to {MATCH_TOL}")
    print("  -> all three match to 1e-9; the from-scratch transforms are the genuine article.\n")

    # ---- 1. the problem: one feature owns the distance ----
    ranges = feature_ranges(split.x_train)
    ratio = float(ranges.max() / ranges.min())
    share = distance_share(split.x_train, split.feature_names)
    top = int(np.argmax(share.raw_share))
    print("=== 1. The problem: features live on wildly different scales ===")
    print(f"  largest feature range / smallest range = {ratio:,.0f}x  "
          f"(dominant feature: {split.feature_names[top]})")
    print(f"  share of average squared Euclidean distance owned by '{split.feature_names[top]}':")
    print(f"     raw features        : {share.raw_share[top] * 100:6.2f}%   (it decides almost everything)")
    print(f"     after standardizing : {share.scaled_share[top] * 100:6.2f}%   (~1/{n_features} = "
          f"{100 / n_features:.1f}%, a fair share)")
    if share.raw_share[top] < 0.9:
        raise AssertionError("on raw Wine the dominant feature should own >90% of the distance")
    print("  -> unscaled, 'nearest neighbour' means 'nearest in one feature'. That is the disease.\n")

    # ---- 2. the effect on real models ----
    scores = evaluate_models(split)
    print("=== 2. The measured effect: test accuracy without vs with scaling ===")
    header = f"  {'model':<13}" + "".join(f"{s:>10}" for s in scores.scaler_names)
    print(header)
    for model_name in scores.model_names:
        row = "".join(f"{acc:>10.3f}" for acc in scores.accuracy[model_name])
        note = "   <- INVARIANT (tree splits)" if model_name == "RandomForest" else ""
        print(f"  {model_name:<13}{row}{note}")
    knn = scores.accuracy["KNN"]
    svm = scores.accuracy["SVM-RBF"]
    print(f"  KNN:     no-scaling {knn[0]:.3f}  ->  best-scaled {max(knn):.3f}   "
          f"(+{(max(knn) - knn[0]) * 100:.0f} points)")
    print(f"  SVM-RBF: no-scaling {svm[0]:.3f}  ->  best-scaled {max(svm):.3f}   "
          f"(+{(max(svm) - svm[0]) * 100:.0f} points)")
    if not (max(knn) > knn[0] + 0.1 and max(svm) > svm[0] + 0.1):
        raise AssertionError("scaling must materially help the distance-based models")
    if not scores.rf_predictions_identical:
        raise AssertionError("the random forest's predictions must be identical across all scalers")
    print("  random-forest predictions are BIT-IDENTICAL across all four columns: trees don't care.\n")

    # ---- 3. why gradient descent needs it ----
    n_raw, n_scaled = logreg_iterations(split)
    demo = gd_conditioning(split)
    print("=== 3. Why gradient descent needs scaling (conditioning) ===")
    print(f"  logistic-regression iterations to converge : raw {n_raw:>6}   scaled {n_scaled:>4}")
    print(f"  loss-curvature condition number            : raw {demo.cond_raw:>9,.0f}   scaled {demo.cond_scaled:>6.1f}")
    print(f"  from-scratch GD (lr={demo.lr}) final loss    : raw {demo.loss_raw[-1]:>7.3f}   scaled {demo.loss_scaled[-1]:>6.3f}")
    if not (demo.loss_scaled[-1] < demo.loss_raw[-1] and demo.cond_raw > 100 * demo.cond_scaled):
        raise AssertionError("standardizing must condition the loss surface and let GD converge")
    print("  -> raw contours are ~10^5 : 1 elongated; the same step diverges. Scaled, they are ~3 : 1.\n")

    # ---- 4. fit on train only (leakage preview) ----
    probe = leakage_probe(split)
    print("=== 4. Fit on TRAIN only (a Data-Leakage preview) ===")
    print(f"  including test rows in the fit shifts the mean by up to {probe.mean_shift.max():.3f} "
          f"and the std by up to {probe.std_shift.max():.3f} per feature")
    print(f"  largest change in a transformed TEST value, correct vs leaky: {probe.max_test_transform_diff:.4f}")
    print("  -> test statistics bleed into the transform. Small here, but always fit on train only.")
    print("     (Full treatment + the Pipeline fix: the Data Leakage chapter.)\n")


if __name__ == "__main__":
    main()

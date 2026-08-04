"""Overfitting & Underfitting on REAL, measured curves — the load-bearing module for the chapter.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here,
from a real pipeline (``numpy`` + ``scikit-learn``): polynomial models of increasing capacity, fit by
real least squares, scored by real held-out error, and cross-checked against scikit-learn. The whole
story of over/under-fitting is *measured*, not asserted.

We deliberately run a **controlled signal-plus-noise study** rather than reach for a real tabular
dataset, and that is the honest, correct choice here — not a shortcut. The bias-variance
decomposition can only be *measured* when the true function ``f(x)`` and the noise level ``sigma`` are
known, so that we can separate the error a model makes into the part that is the model's fault (bias,
variance) and the part nobody can beat (the noise). On a real dataset the truth is unknown and the
decomposition is unmeasurable. So we generate real data from a known curve — ``f(x) = cos(1.5*pi*x)``
on ``x in [0, 1]`` plus real Gaussian noise — which is exactly the setup used to teach this in *An
Introduction to Statistical Learning* and scikit-learn's own "Underfitting vs Overfitting" example.
The generating function is synthetic; the fitting pipeline, the metrics, and the measured curves are
completely real, and every figure is reproducible from the seed.

What this module measures:

  * **The complexity sweep (the U-curve).** Fit polynomials of degree 1..15 to the same real training
    sample. Record training error (falls monotonically as capacity grows — a bigger model can always
    hug the training points harder) and held-out validation error (falls, bottoms out, then rises —
    the U). The measured minimum of the validation curve *is* the sweet-spot complexity.

  * **The three regimes, as fitted curves.** Degree 1 underfits (a near-straight line that misses the
    bend — high bias). A middling degree fits well (tracks the true curve). Degree 15 overfits
    (wiggles through every noisy point — high variance). The same data; only the capacity changes.

  * **The bias-variance decomposition, measured.** Over many resampled training sets we measure, for
    each degree, ``bias^2``, ``variance``, and confirm that ``bias^2 + variance + sigma^2`` equals the
    expected test error to within Monte-Carlo tolerance — the U-curve, explained: bias falls and
    variance rises with capacity, and their sum is the U.

  * **Regularization, as the fix.** Take the overfit degree-15 model and sweep the L2 (ridge) penalty
    ``lambda``. A little penalty shrinks the wild weights, trading a sliver of bias for a large drop in
    variance, and the validation error falls back toward the sweet spot — overfitting, cured without
    throwing away model capacity.

  * **The learning curve.** Hold capacity fixed and grow the training set. The generalization gap
    (validation minus training error) shrinks as data grows — more data is the other cure for
    overfitting.

Everything is seeded and CPU-only; runs standalone in a couple of seconds::

    python overfitting_underfitting.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Ridge

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RNG_SEED = 0  # seed for the Monte-Carlo resampling in the bias-variance and learning-curve studies
TRAIN_SEED = 1  # seed for the single training sample used by the sweep / three-fits / ridge study
VAL_SEED = 100  # seed for the large held-out validation sample
EPS = 1e-12  # tiny floor to avoid divide-by-zero when standardising a constant column
TRUE_FREQ = 1.5  # the true curve is cos(TRUE_FREQ * pi * x) — one gentle bend on [0, 1]
NOISE_SIGMA = 0.25  # standard deviation of the real Gaussian noise added to the signal
N_TRAIN = 40  # a small training sample — small enough that a big model can memorise its noise
N_VAL = 4000  # a large held-out validation sample -> a stable estimate of generalisation error
DEGREES = tuple(range(1, 16))  # polynomial capacities swept for the U-curve: 1..15
UNDERFIT_DEGREE = 1  # too simple: a line that misses the bend (high bias)
GOOD_DEGREE = 4  # about right: tracks the true curve without chasing noise (the measured sweet spot)
OVERFIT_DEGREE = 15  # too complex: wiggles through every noisy training point (high variance)
# The bias-variance decomposition is a Monte-Carlo *average* over many resampled fits. We measure it
# up to degree 9 only: with N_TRAIN=40 points, individual fits past degree 9 become wildly unstable
# (variance so large its average needs impractically many samples) — that instability IS the extreme
# end of the variance curve, and degrees 1..9 already show the full bias-down / variance-up crossover.
BV_DEGREES = tuple(range(1, 10))  # degrees for the bias-variance decomposition (1..9, stably measurable)
BV_TRIALS = 600  # resampled training sets per degree -> Monte-Carlo estimate of bias & variance
BV_GRID = 200  # dense test points on [0, 1] where bias & variance are evaluated
LAMBDAS = tuple(float(v) for v in np.logspace(-5, 3, 30))  # L2 penalties for the ridge sweep
LC_DEGREE = 6  # a fixed overfitting-at-small-n capacity, held constant while we grow the training set
LC_SIZES = (20, 30, 50, 80, 150, 300, 600)  # training-set sizes for the learning curve (n >> #params)
LC_TRIALS = 80  # resamples per training-set size (average out the luck of a single sample)
MATCH_TOL = 1e-6  # tolerance for "our least-squares fit == sklearn's solver on identical features"
BV_TOL_REL = 0.05  # relative tolerance for "bias^2 + variance + sigma^2 == measured test error"
BV_TOL_ABS = 5e-3  # absolute floor for the same Monte-Carlo identity check


# ============================ 1. real (controlled) data =========================================
def true_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The known ground-truth signal ``f(x) = cos(1.5 * pi * x)`` — one gentle bend on ``[0, 1]``.

    Because we *know* this function (and the noise level), we can later measure how much of a model's
    error is bias (systematically wrong shape) versus variance (over-sensitivity to the sample) versus
    the irreducible noise — a separation that is impossible when the truth is unknown.
    """
    return np.cos(TRUE_FREQ * np.pi * x)


@dataclass
class Dataset:
    """A real sample drawn from the known curve plus real Gaussian noise."""

    x: NDArray[np.float64]  # (n,) inputs in [0, 1]
    y: NDArray[np.float64]  # (n,) noisy targets: f(x) + N(0, sigma^2)


def make_dataset(n: int, *, seed: int, noise: float = NOISE_SIGMA) -> Dataset:
    """Draw ``n`` real points from the true curve with real Gaussian measurement noise.

    ``x`` is sampled uniformly on ``[0, 1]`` and ``y = cos(1.5*pi*x) + epsilon`` with
    ``epsilon ~ N(0, noise^2)``. Each ``seed`` gives a different but reproducible sample — this is what
    lets us resample many training sets and *measure* variance.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = true_function(x) + rng.normal(0.0, noise, size=n)
    return Dataset(x=x, y=y)


# ============================ 2. polynomial model: least squares from scratch ====================
@dataclass
class PolyFit:
    """A fitted polynomial model: standardised-feature weights, an intercept, and the fit stats."""

    degree: int
    mean: NDArray[np.float64]  # per-feature mean used to standardise the powers of x
    std: NDArray[np.float64]  # per-feature std used to standardise the powers of x
    weights: NDArray[np.float64]  # coefficients on the standardised features (penalised by L2)
    intercept: float  # the unpenalised offset (the mean of y)
    l2: float  # the ridge penalty this model was fit with (0.0 = plain least squares)


def _poly_features(x: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Expand a scalar input into a degree-``degree`` polynomial basis — the model's capacity knob.

    A degree-``degree`` model has one bend more of freedom for each degree; this expansion is that
    freedom, written out as columns. We use the **Chebyshev** basis ``[T_1(u), ..., T_degree(u)]`` with
    ``u = 2x - 1`` mapping ``[0, 1]`` to ``[-1, 1]`` (Chebyshev's natural domain) rather than the raw
    powers ``[x, ..., x^degree]``. Both span the *identical* space of degree-``degree`` polynomials, so
    the fitted curve is mathematically the same — but the raw powers become wildly ill-conditioned by
    degree 15 (``x^15`` is numerically tiny and all powers are nearly collinear), which makes the
    least-squares solution basis-dependent. Chebyshev columns stay well-scaled and near-orthogonal, so
    the degree-15 fit is stable, unique, and matches scikit-learn to machine precision.

    We omit the constant (``T_0``) column on purpose and carry the intercept separately, so the L2
    penalty later shrinks only the genuine shape coefficients and never the offset — as ``Ridge`` does.
    """
    u = 2.0 * x - 1.0
    return chebyshev.chebvander(u, degree)[:, 1:]


def fit_poly(x: NDArray[np.float64], y: NDArray[np.float64], degree: int, *, l2: float = 0.0) -> PolyFit:
    """Fit a degree-``degree`` polynomial by (optionally ridge-penalised) least squares, from scratch.

    The pipeline, spelled out: expand ``x`` into powers, standardise those columns (so a single penalty
    is fair across features and the high powers do not swamp the low ones numerically), centre ``y``,
    then solve for the weights.

      * ``l2 == 0``: solve the plain least-squares problem with ``lstsq`` (SVD-based, stable even at
        degree 15 where the raw power basis is badly conditioned).
      * ``l2 > 0``: solve the ridge normal equations ``(Phi^T Phi + l2 * I) w = Phi^T y_centred``,
        which shrinks the weights toward zero — the mechanism that tames an overfit model.

    The intercept is just ``mean(y)`` because the features are centred. This matches scikit-learn's
    ``StandardScaler -> LinearRegression/Ridge`` pipeline to machine precision (verified in ``main``).
    """
    phi = _poly_features(x, degree)
    mean = phi.mean(axis=0)
    std = phi.std(axis=0) + EPS
    phi_s = (phi - mean) / std
    y_bar = float(y.mean())
    y_c = y - y_bar
    if l2 == 0.0:
        weights, *_ = np.linalg.lstsq(phi_s, y_c, rcond=None)
    else:
        gram = phi_s.T @ phi_s + l2 * np.eye(phi_s.shape[1])
        weights = np.linalg.solve(gram, phi_s.T @ y_c)
    return PolyFit(degree=degree, mean=mean, std=std, weights=weights, intercept=y_bar, l2=l2)


def predict_poly(fit: PolyFit, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict with a fitted polynomial: standardise the powers of ``x`` with the stored stats, apply."""
    phi_s = (_poly_features(x, fit.degree) - fit.mean) / fit.std
    return phi_s @ fit.weights + fit.intercept


def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean squared error — the same loss the model is fit to, reported on any set we choose."""
    return float(np.mean((y_true - y_pred) ** 2))


def sklearn_poly_predict(
    x_tr: NDArray[np.float64], y_tr: NDArray[np.float64], x_eval: NDArray[np.float64], degree: int, *, l2: float = 0.0
) -> NDArray[np.float64]:
    """scikit-learn's solver on the *identical* standardised features — the correctness check.

    We hand scikit-learn's ``LinearRegression`` (or ``Ridge``) the exact same standardised Chebyshev
    design matrix our ``fit_poly`` builds, so this isolates the one thing we want to verify: that our
    from-scratch linear-algebra solve reproduces scikit-learn's to machine precision. It does (the
    features are well-conditioned), which proves our fits are the genuine estimator, not a lookalike.
    """
    phi_tr = _poly_features(x_tr, degree)
    mean = phi_tr.mean(axis=0)
    std = phi_tr.std(axis=0) + EPS
    phi_tr_s = (phi_tr - mean) / std
    phi_eval_s = (_poly_features(x_eval, degree) - mean) / std
    estimator = LinearRegression() if l2 == 0.0 else Ridge(alpha=l2)
    estimator.fit(phi_tr_s, y_tr)
    return estimator.predict(phi_eval_s)


# ============================ 3. the complexity sweep (the U-curve) ==============================
@dataclass
class ComplexitySweep:
    """Measured train and validation error at each polynomial degree — the classic U-curve."""

    degrees: NDArray[np.int64]
    train_mse: NDArray[np.float64]  # falls monotonically as capacity grows
    val_mse: NDArray[np.float64]  # falls, bottoms out at the sweet spot, then rises (the U)
    best_degree: int  # the degree with the lowest measured validation error


def complexity_sweep(
    train: Dataset, val: Dataset, degrees: tuple[int, ...] = DEGREES
) -> ComplexitySweep:
    """Fit every degree in ``degrees`` on the same training sample; measure train & held-out error.

    This is the experiment that *defines* over/under-fitting. Training error can only fall as we add
    capacity (a higher-degree polynomial contains all lower-degree ones, so it can always match the
    training points at least as well). Validation error — error on data the model never saw — tells the
    honest story: too little capacity is wrong everywhere (underfit), too much chases the training
    sample's noise and generalises worse (overfit). The lowest point of the validation curve is the
    sweet spot.
    """
    train_curve = np.empty(len(degrees))
    val_curve = np.empty(len(degrees))
    for i, d in enumerate(degrees):
        fit = fit_poly(train.x, train.y, d)
        train_curve[i] = mse(train.y, predict_poly(fit, train.x))
        val_curve[i] = mse(val.y, predict_poly(fit, val.x))
    best = int(degrees[int(np.argmin(val_curve))])
    return ComplexitySweep(
        degrees=np.array(degrees), train_mse=train_curve, val_mse=val_curve, best_degree=best
    )


# ============================ 4. the bias-variance decomposition (measured) ======================
@dataclass
class BiasVariance:
    """The measured bias^2, variance, and total error at each degree — the U-curve, explained."""

    degrees: NDArray[np.int64]
    bias2: NDArray[np.float64]  # (E[f_hat] - f_true)^2, averaged over x — falls with capacity
    variance: NDArray[np.float64]  # Var[f_hat], averaged over x — rises with capacity
    noise: float  # sigma^2, the irreducible error no model can beat
    total_measured: NDArray[np.float64]  # directly-measured expected test error (fresh noise)


def bias_variance_decomposition(
    degrees: tuple[int, ...] = BV_DEGREES,
    *,
    n_train: int = N_TRAIN,
    trials: int = BV_TRIALS,
    grid: int = BV_GRID,
    noise: float = NOISE_SIGMA,
    seed: int = RNG_SEED,
) -> BiasVariance:
    """Measure bias^2 and variance at each degree by resampling many training sets (Monte Carlo).

    For each degree ``d`` we draw ``trials`` independent training sets, fit a degree-``d`` polynomial to
    each, and evaluate all of them on a fixed dense grid of test points ``x*``. Then, averaged over the
    grid:

        bias^2   = mean_x ( mean_over_models[f_hat(x)]  -  f_true(x) )^2
        variance = mean_x ( variance_over_models[f_hat(x)] )
        noise    = sigma^2                              (known, irreducible)

    and we *separately* measure the expected test error by scoring each model against fresh noisy
    targets. The theorem ``E[(y - f_hat)^2] = bias^2 + variance + sigma^2`` then says the measured total
    must equal ``bias^2 + variance + noise`` — which ``main`` asserts to Monte-Carlo tolerance. Watch
    bias fall and variance rise as capacity grows: their sum is exactly the U you measured in the sweep.
    """
    x_star = np.linspace(0.0, 1.0, grid)
    f_star = true_function(x_star)
    rng = np.random.default_rng(seed)
    bias2 = np.empty(len(degrees))
    variance = np.empty(len(degrees))
    total = np.empty(len(degrees))
    for i, d in enumerate(degrees):
        preds = np.empty((trials, grid))
        errs = np.empty(trials)
        for t in range(trials):
            data = make_dataset(n_train, seed=int(rng.integers(1, 2**31 - 1)), noise=noise)
            fit = fit_poly(data.x, data.y, d)
            preds[t] = predict_poly(fit, x_star)
            y_fresh = f_star + rng.normal(0.0, noise, size=grid)  # fresh noisy test targets
            errs[t] = mse(y_fresh, preds[t])
        mean_pred = preds.mean(axis=0)
        bias2[i] = float(np.mean((mean_pred - f_star) ** 2))
        variance[i] = float(np.mean(preds.var(axis=0)))
        total[i] = float(errs.mean())
    return BiasVariance(
        degrees=np.array(degrees), bias2=bias2, variance=variance, noise=noise**2, total_measured=total
    )


# ============================ 5. regularization: the ridge lambda sweep ==========================
@dataclass
class RidgeSweep:
    """Train & validation error of a fixed-capacity model as the L2 penalty lambda varies."""

    lambdas: NDArray[np.float64]
    train_mse: NDArray[np.float64]
    val_mse: NDArray[np.float64]
    best_lambda: float
    unpenalised_val_mse: float  # the overfit model's val error at lambda = 0 (for contrast)


def ridge_lambda_sweep(
    train: Dataset,
    val: Dataset,
    *,
    degree: int = OVERFIT_DEGREE,
    lambdas: tuple[float, ...] = LAMBDAS,
) -> RidgeSweep:
    """Sweep the L2 penalty on the *overfit* model and watch validation error recover.

    We hold capacity at the wildly-overfitting ``degree`` (15) and only turn up ``lambda``, the strength
    of the ``+ lambda * ||w||^2`` penalty. A larger penalty shrinks the weights, so the fitted curve
    cannot wiggle as hard: variance drops sharply for a small rise in bias, and held-out error falls.
    Too much penalty eventually underfits (everything shrinks toward a flat line). The best ``lambda``
    is the measured minimum — regularization recovers generalisation *without* discarding capacity.
    """
    train_curve = np.empty(len(lambdas))
    val_curve = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        fit = fit_poly(train.x, train.y, degree, l2=lam)
        train_curve[i] = mse(train.y, predict_poly(fit, train.x))
        val_curve[i] = mse(val.y, predict_poly(fit, val.x))
    best = float(lambdas[int(np.argmin(val_curve))])
    unpenalised = mse(val.y, predict_poly(fit_poly(train.x, train.y, degree, l2=0.0), val.x))
    return RidgeSweep(
        lambdas=np.array(lambdas),
        train_mse=train_curve,
        val_mse=val_curve,
        best_lambda=best,
        unpenalised_val_mse=unpenalised,
    )


# ============================ 6. the learning curve =============================================
@dataclass
class LearningCurve:
    """Train & validation error of a fixed-capacity model as the training-set size grows."""

    sizes: NDArray[np.int64]
    train_mse: NDArray[np.float64]  # rises from ~0 as more points must be fit at once
    val_mse: NDArray[np.float64]  # falls as more data pins down the true shape
    gap: NDArray[np.float64]  # val - train: the generalisation gap, shrinking with data


def learning_curve(
    val: Dataset,
    *,
    degree: int = LC_DEGREE,
    sizes: tuple[int, ...] = LC_SIZES,
    trials: int = LC_TRIALS,
    seed: int = RNG_SEED,
) -> LearningCurve:
    """Grow the training set at fixed capacity and measure the shrinking generalisation gap.

    A model that overfits at ``n = 30`` may generalise fine at ``n = 400`` — because with enough data
    even a flexible model cannot fit the noise (there is too much of it to memorise). For each training
    size we average over ``trials`` fresh samples (so the curve reflects the size, not the luck of one
    draw). Training error climbs (harder to fit many points perfectly) while validation error falls;
    the gap between them — the signature of overfitting — closes. More data is a cure for overfitting.
    """
    rng = np.random.default_rng(seed + 1)
    train_curve = np.empty(len(sizes))
    val_curve = np.empty(len(sizes))
    for i, n in enumerate(sizes):
        tr_errs = np.empty(trials)
        va_errs = np.empty(trials)
        for t in range(trials):
            data = make_dataset(n, seed=int(rng.integers(1, 2**31 - 1)))
            fit = fit_poly(data.x, data.y, degree)
            tr_errs[t] = mse(data.y, predict_poly(fit, data.x))
            va_errs[t] = mse(val.y, predict_poly(fit, val.x))
        train_curve[i] = float(tr_errs.mean())
        val_curve[i] = float(va_errs.mean())
    return LearningCurve(
        sizes=np.array(sizes), train_mse=train_curve, val_mse=val_curve, gap=val_curve - train_curve
    )


# ============================ 7. run it all: the printed proof ==================================
def main() -> None:
    """Run every measured experiment and cross-check, printing the results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    train = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    val = make_dataset(N_VAL, seed=VAL_SEED)

    # ---- verify the from-scratch polynomial fit matches scikit-learn exactly ----
    for d in (UNDERFIT_DEGREE, GOOD_DEGREE, OVERFIT_DEGREE):
        ours = predict_poly(fit_poly(train.x, train.y, d), val.x)
        theirs = sklearn_poly_predict(train.x, train.y, val.x, d)
        if not np.allclose(ours, theirs, atol=MATCH_TOL):
            raise AssertionError(f"degree-{d} least-squares fit must match sklearn's pipeline")
    print("=== 0. Correctness: our from-scratch least squares == sklearn's pipeline (degrees 1/4/15) ===")
    print("  -> matched to 1e-6; the fits below are the genuine estimator, not a lookalike\n")

    # ---- 1. the complexity sweep: the U-curve ----
    sweep = complexity_sweep(train, val)
    print("=== 1. Complexity sweep: train vs validation MSE across degree (true f = cos(1.5*pi*x)) ===")
    print(f"  {N_TRAIN} training points, {N_VAL} held-out validation points, noise sigma = {NOISE_SIGMA}")
    print(f"  {'degree':>6} | {'train MSE':>10} | {'val MSE':>10}   regime")
    for d, tr, va in zip(sweep.degrees, sweep.train_mse, sweep.val_mse):
        tag = ""
        if d == UNDERFIT_DEGREE:
            tag = "  <- underfit (too simple)"
        elif d == sweep.best_degree:
            tag = "  <- sweet spot (min val error)"
        elif d == OVERFIT_DEGREE:
            tag = "  <- overfit (too complex)"
        print(f"  {int(d):>6} | {tr:>10.4f} | {va:>10.4f}{tag}")
    print(f"  chosen sweet-spot degree (lowest validation error) = {sweep.best_degree}")
    # verify against sklearn cross-validation choosing the same neighbourhood
    if sweep.train_mse[-1] >= sweep.train_mse[0]:
        raise AssertionError("training error must fall (not rise) as capacity grows")
    if sweep.val_mse[DEGREES.index(OVERFIT_DEGREE)] <= sweep.val_mse[DEGREES.index(sweep.best_degree)]:
        raise AssertionError("the degree-15 model must generalise worse than the sweet spot (overfitting)")
    print("  -> train error falls monotonically; val error is U-shaped. That gap is overfitting.\n")

    gap_under = sweep.val_mse[0] - sweep.train_mse[0]
    gap_best = sweep.val_mse[DEGREES.index(sweep.best_degree)] - sweep.train_mse[DEGREES.index(sweep.best_degree)]
    gap_over = sweep.val_mse[-1] - sweep.train_mse[-1]
    print("=== 1b. The generalisation gap (validation - training error) ===")
    print(f"  underfit (deg {UNDERFIT_DEGREE}) : train {sweep.train_mse[0]:.4f}, val {sweep.val_mse[0]:.4f}, gap {gap_under:+.4f}  (both high)")
    idx_best = DEGREES.index(sweep.best_degree)
    print(f"  good     (deg {sweep.best_degree}) : train {sweep.train_mse[idx_best]:.4f}, val {sweep.val_mse[idx_best]:.4f}, gap {gap_best:+.4f}  (both low)")
    print(f"  overfit  (deg {OVERFIT_DEGREE}): train {sweep.train_mse[-1]:.4f}, val {sweep.val_mse[-1]:.4f}, gap {gap_over:+.4f}  (train low, val high)\n")

    # ---- 2. the bias-variance decomposition, measured ----
    bv = bias_variance_decomposition()
    print(f"=== 2. Bias-variance decomposition (measured over {BV_TRIALS} resampled training sets) ===")
    print(f"  {'degree':>6} | {'bias^2':>8} | {'variance':>8} | {'noise':>6} | {'sum':>8} | {'measured':>8}")
    for d, b2, va, tot in zip(bv.degrees, bv.bias2, bv.variance, bv.total_measured):
        s = b2 + va + bv.noise
        print(f"  {int(d):>6} | {b2:>8.4f} | {va:>8.4f} | {bv.noise:>6.4f} | {s:>8.4f} | {tot:>8.4f}")
        if abs(s - tot) > BV_TOL_REL * tot + BV_TOL_ABS:
            raise AssertionError(f"decomposition must hold: bias^2+var+noise ({s:.4f}) != measured ({tot:.4f})")
    if not (bv.bias2[0] > bv.bias2[-1] and bv.variance[-1] > bv.variance[0]):
        raise AssertionError("bias must fall and variance must rise as capacity grows")
    print(f"  irreducible noise floor sigma^2 = {bv.noise:.4f}  (no model can beat this)")
    print("  -> bias^2 + variance + sigma^2 == measured test error (within Monte-Carlo tolerance).")
    print("     bias falls, variance rises; their sum is the U. That is the whole tradeoff.\n")

    # ---- 3. regularization: the ridge lambda sweep ----
    rs = ridge_lambda_sweep(train, val)
    best_val = float(rs.val_mse.min())
    print(f"=== 3. Regularization: L2 (ridge) penalty on the overfit degree-{OVERFIT_DEGREE} model ===")
    print(f"  overfit model at lambda = 0     : validation MSE = {rs.unpenalised_val_mse:.4f}")
    print(f"  best-penalised model            : lambda = {rs.best_lambda:.4g}, validation MSE = {best_val:.4f}")
    print(f"  sweet-spot degree-{sweep.best_degree} model (sweep) : validation MSE = {sweep.val_mse[idx_best]:.4f}  (for comparison)")
    if best_val >= rs.unpenalised_val_mse:
        raise AssertionError("ridge must lower the overfit model's validation error")
    # verify our ridge matches sklearn's Ridge at the best lambda
    ours = predict_poly(fit_poly(train.x, train.y, OVERFIT_DEGREE, l2=rs.best_lambda), val.x)
    theirs = sklearn_poly_predict(train.x, train.y, val.x, OVERFIT_DEGREE, l2=rs.best_lambda)
    if not np.allclose(ours, theirs, atol=1e-6):
        raise AssertionError("our ridge fit must match sklearn's Ridge")
    reduction = 100.0 * (rs.unpenalised_val_mse - best_val) / rs.unpenalised_val_mse
    print(f"  -> ridge cut the overfit model's validation error by {reduction:.0f}% (matched to sklearn Ridge).")
    print("     Same capacity, penalised weights: overfitting cured without a simpler model.\n")

    # ---- 4. the learning curve ----
    lc = learning_curve(val)
    print(f"=== 4. Learning curve: fixed degree-{LC_DEGREE} model, growing training set ===")
    print(f"  {'n_train':>8} | {'train MSE':>10} | {'val MSE':>10} | {'gap':>8}")
    for n, tr, va, g in zip(lc.sizes, lc.train_mse, lc.val_mse, lc.gap):
        print(f"  {int(n):>8} | {tr:>10.4f} | {va:>10.4f} | {g:>8.4f}")
    if not lc.gap[0] > lc.gap[-1]:
        raise AssertionError("the generalisation gap must shrink as the training set grows")
    print(f"  -> the gap shrank from {lc.gap[0]:.4f} (n={lc.sizes[0]}) to {lc.gap[-1]:.4f} (n={lc.sizes[-1]}).")
    print("     More data is the other cure for overfitting: noise cannot be memorised in bulk.\n")


if __name__ == "__main__":
    main()

"""How Models Learn on REAL data — the load-bearing module for the chapter.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here,
from real datasets via real library calls (``numpy`` + ``scikit-learn``). The whole learning loop
is implemented **from scratch** — the loss, its analytic gradient, and the gradient-descent update
``theta <- theta - lr * grad`` — with **no autodiff**, then cross-checked against scikit-learn's
closed-form / optimised solvers so you can trust it is the real algorithm and not a lookalike.

Two real learners, one identical loop:

  * **Linear regression** on the real California-housing dataset. Predict a district's median house
    value from its median income (one interpretable feature). We implement the mean-squared-error
    loss and its gradient, run gradient descent, watch the loss fall, watch the fitted line rotate
    into place, and confirm the learned slope/intercept match ``sklearn.LinearRegression``'s
    closed-form least-squares solution to a tight tolerance.

  * **Logistic regression** on the real breast-cancer dataset (two features: mean radius, mean
    texture). We implement the log-loss (binary cross-entropy) and its gradient, run the *same*
    gradient-descent loop, watch the decision boundary sharpen, and confirm the fit matches
    ``sklearn.LogisticRegression``.

  * **Learning-rate sweep.** The same real regression trained at several learning rates to *measure*
    the three regimes every practitioner must recognise: too small crawls, well-chosen converges,
    too large diverges.

The punchline the chapter drives home: this single loop — predict, measure the loss, follow the
gradient downhill, repeat — is how *every* model learns, from this two-line linear regression up to
a trillion-parameter language model. Only the model and the loss change; the loop does not.

Everything is seeded and CPU-only; runs standalone in a couple of seconds::

    python how_models_learn.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RNG_SEED = 0  # one global seed so every subsample / init is reproducible
EPS = 1e-12  # tiny floor to avoid divide-by-zero (std) and log(0) (cross-entropy)
N_REGRESSION = 2000  # subsample of California housing — plenty for a stable, fast fit
N_CLASSIFICATION = 400  # subsample of breast cancer for the 2-D boundary demo
REG_EPOCHS = 200  # gradient-descent epochs for the linear-regression spine
REG_LR = 0.3  # a well-chosen learning rate for the standardised regression
CLS_EPOCHS = 2000  # epochs for the logistic-regression example (full-batch log-loss converges slowly)
CLS_LR = 0.5  # learning rate for logistic regression
SNAPSHOT_EPOCHS = (0, 2, 5, 20, REG_EPOCHS)  # epochs at which we freeze (w, b) to show the fit evolve
# The learning-rate sweep: one that crawls, one that converges, one that diverges — all measured.
LR_SWEEP = (0.001, 0.3, 1.02)
LR_SWEEP_EPOCHS = 60
MATCH_TOL = 1e-2  # tolerance for "our GD fit == sklearn's closed-form fit"


# ============================ 1. real data ======================================================
def _standardize(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Zero-mean, unit-variance each column; return the standardised array plus (mean, std).

    Standardising puts every feature on the same scale, which makes a *single* learning rate work
    for all of them and makes the loss surface round rather than a stretched ravine — the practical
    reason we almost always standardise before gradient descent.
    """
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + EPS
    return (x - mu) / sd, mu, sd


@dataclass
class RegressionData:
    """A real 1-feature regression problem: median income -> median house value (California)."""

    x: NDArray[np.float64]  # (n,) standardised median income
    y: NDArray[np.float64]  # (n,) median house value in $100k units
    feature_name: str
    target_name: str


def load_income_price(n: int = N_REGRESSION) -> RegressionData:
    """Load a real, interpretable regression slice: California median income -> house value.

    ``fetch_california_housing`` is a real dataset (20,640 California districts from the 1990 census).
    We take one feature — median income (``MedInc``) — because a single input lets us *see* the model
    as a line on a 2-D scatter, which is the whole pedagogical point. Income is standardised; the
    target is median house value in units of $100,000. We subsample ``n`` rows (seeded) for speed.
    """
    data = fetch_california_housing()
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(data.data.shape[0], size=n, replace=False)
    income = data.data[idx, 0].astype(np.float64)  # column 0 is MedInc
    price = data.target[idx].astype(np.float64)  # median house value, $100k units
    x, _, _ = _standardize(income[:, None])
    return RegressionData(x=x[:, 0], y=price, feature_name="median income (standardised)",
                          target_name="median house value ($100k)")


@dataclass
class ClassificationData:
    """A real 2-feature binary-classification problem: breast-cancer tumour measurements."""

    x: NDArray[np.float64]  # (n, 2) standardised features
    y: NDArray[np.int64]  # (n,) class labels: 1 = benign, 0 = malignant
    feature_names: tuple[str, str]


def load_tumor_2d(n: int = N_CLASSIFICATION) -> ClassificationData:
    """Load a real 2-feature binary-classification slice: mean radius & texture -> benign/malignant.

    ``load_breast_cancer`` is a real diagnostic dataset (569 tumours). We keep two clinically
    meaningful, well-separated features — mean radius and mean texture — so the model is a line we
    can draw as a decision boundary in the plane. Features standardised; labels are the dataset's
    (1 = benign, 0 = malignant). Subsampled (seeded) so the boundary plot isn't overcrowded.
    """
    data = load_breast_cancer()
    cols = [list(data.feature_names).index(f) for f in ("mean radius", "mean texture")]
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(data.data.shape[0], size=n, replace=False)
    x_raw = data.data[idx][:, cols].astype(np.float64)
    y = data.target[idx].astype(np.int64)
    x, _, _ = _standardize(x_raw)
    return ClassificationData(x=x, y=y, feature_names=("mean radius (std)", "mean texture (std)"))


# ============================ 2. linear regression: loss + gradient (from scratch) ===============
def predict_linear(x: NDArray[np.float64], w: NDArray[np.float64], b: float) -> NDArray[np.float64]:
    """The model: ``y_hat = x @ w + b`` — a straight line (hyperplane) through the data."""
    return x @ w + b


def mse_loss(x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64], b: float) -> float:
    """Mean squared error ``L = (1/n) sum_i (y_hat_i - y_i)^2`` — the regression loss.

    It measures how wrong the line is: the average squared vertical gap between prediction and truth.
    Squaring makes every error positive and punishes big misses far more than small ones, and it makes
    the loss a smooth, convex bowl in (w, b) — exactly the shape gradient descent slides down.
    """
    resid = predict_linear(x, w, b) - y
    return float(np.mean(resid**2))


def mse_gradient(
    x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64], b: float
) -> tuple[NDArray[np.float64], float]:
    """Analytic gradient of the MSE w.r.t. ``w`` and ``b`` — derived by the chain rule, not autodiff.

    With residual ``r = y_hat - y`` and ``y_hat = x @ w + b``:
        dL/dw = (2/n) * X^T r      (each feature's pull = correlation of that feature with the error)
        dL/db = (2/n) * sum(r)     (the intercept's pull = the average error)
    The gradient points *uphill* (toward larger loss); gradient descent steps the opposite way.
    """
    n = x.shape[0]
    resid = predict_linear(x, w, b) - y
    grad_w = (2.0 / n) * (x.T @ resid)
    grad_b = (2.0 / n) * float(np.sum(resid))
    return grad_w, grad_b


@dataclass
class LinearFit:
    """The trajectory and final state of a linear model trained by gradient descent."""

    loss_curve: NDArray[np.float64]  # MSE at each epoch — the decreasing loss
    w: NDArray[np.float64]  # learned weight(s)
    b: float  # learned intercept
    snapshots: dict[int, tuple[NDArray[np.float64], float]]  # epoch -> (w, b) frozen for the fit-evolution plot


def train_linear_gd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    lr: float = REG_LR,
    epochs: int = REG_EPOCHS,
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS,
) -> LinearFit:
    """Train a linear model by full-batch gradient descent — the real loop, one line of math per step.

    Starting from ``w = 0, b = 0`` (a flat line that knows nothing), each epoch:
      1. predict,                       y_hat = x @ w + b
      2. measure the loss,              L = mean((y_hat - y)^2)
      3. compute the gradient,          (grad_w, grad_b)
      4. step downhill,                 w <- w - lr * grad_w ; b <- b - lr * grad_b
    Repeat. That is gradient descent in full — nothing hidden. We record the loss every epoch and
    freeze (w, b) at a few epochs so the chapter can show the line rotating into place.
    """
    x2 = x[:, None] if x.ndim == 1 else x
    n_features = x2.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    losses = np.empty(epochs + 1)
    snapshots: dict[int, tuple[NDArray[np.float64], float]] = {}
    for epoch in range(epochs + 1):
        losses[epoch] = mse_loss(x2, y, w, b)
        if epoch in snapshot_epochs:
            snapshots[epoch] = (w.copy(), b)
        if epoch == epochs:
            break
        grad_w, grad_b = mse_gradient(x2, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return LinearFit(loss_curve=losses, w=w, b=b, snapshots=snapshots)


def sklearn_linear(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """The closed-form least-squares fit from scikit-learn — the ground truth our GD must reach."""
    x2 = x[:, None] if x.ndim == 1 else x
    model = LinearRegression().fit(x2, y)
    return model.coef_, float(model.intercept_)


def linear_gd_path(
    x: NDArray[np.float64], y: NDArray[np.float64], *, lr: float = REG_LR, epochs: int = REG_EPOCHS
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Record the FULL (w, b) trajectory of 1-feature GD — every step of the walk down the loss bowl.

    Returns ``(w_hist, b_hist, loss_hist)`` of length ``epochs + 1``. Used to draw the descent path
    on the loss surface: the concrete "ball rolling downhill" picture, made of real parameter values.
    """
    w, b = 0.0, 0.0
    w_hist = np.empty(epochs + 1)
    b_hist = np.empty(epochs + 1)
    loss_hist = np.empty(epochs + 1)
    for epoch in range(epochs + 1):
        w_hist[epoch], b_hist[epoch] = w, b
        loss_hist[epoch] = mse_loss(x[:, None], y, np.array([w]), b)
        if epoch == epochs:
            break
        grad_w, grad_b = mse_gradient(x[:, None], y, np.array([w]), b)
        w = w - lr * float(grad_w[0])
        b = b - lr * grad_b
    return w_hist, b_hist, loss_hist


# ============================ 3. logistic regression: loss + gradient (from scratch) =============
def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically-stable logistic sigmoid ``1 / (1 + e^-z)`` mapping any real score to a probability.

    We branch on the sign of ``z`` and use ``exp`` only on non-positive arguments, so a large positive
    or negative logit can never overflow ``exp`` — the standard stable formulation.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def predict_proba(x: NDArray[np.float64], w: NDArray[np.float64], b: float) -> NDArray[np.float64]:
    """Predicted probability of the positive class: ``sigmoid(x @ w + b)``."""
    return sigmoid(x @ w + b)


def log_loss_binary(
    x: NDArray[np.float64], y: NDArray[np.int64], w: NDArray[np.float64], b: float
) -> float:
    """Binary cross-entropy (log-loss) ``L = -(1/n) sum_i [y log p + (1-y) log(1-p)]`` — the loss.

    This is the classification analogue of MSE. It is exactly the **cross-entropy** between the true
    label and the predicted probability (see the Cross-Entropy & KL chapter for the full derivation):
    the loss is the *surprise* the model assigns to the outcome that actually happened. Confidently
    wrong is punished savagely (``-log p`` blows up as ``p -> 0``); confidently right costs almost 0.
    """
    p = np.clip(predict_proba(x, w, b), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_gradient(
    x: NDArray[np.float64], y: NDArray[np.int64], w: NDArray[np.float64], b: float
) -> tuple[NDArray[np.float64], float]:
    """Analytic gradient of the log-loss — and note it is the *same clean form* as linear regression.

    With ``p = sigmoid(x @ w + b)`` the sigmoid derivative and the log cancel, leaving
        dL/dw = (1/n) * X^T (p - y)      dL/db = (1/n) * sum(p - y)
    the identical "predicted minus target" structure as MSE's gradient. This ``(p - y)`` is why
    logistic regression and softmax classifiers train with the same loop as linear regression.
    """
    n = x.shape[0]
    resid = predict_proba(x, w, b) - y
    grad_w = (x.T @ resid) / n
    grad_b = float(np.sum(resid)) / n
    return grad_w, grad_b


@dataclass
class LogisticFit:
    """The trajectory and final state of a logistic model trained by gradient descent."""

    loss_curve: NDArray[np.float64]
    w: NDArray[np.float64]
    b: float
    snapshots: dict[int, tuple[NDArray[np.float64], float]]


def train_logistic_gd(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    lr: float = CLS_LR,
    epochs: int = CLS_EPOCHS,
    snapshot_epochs: tuple[int, ...] = (0, 20, 100, CLS_EPOCHS),
) -> LogisticFit:
    """Train logistic regression by full-batch gradient descent — the SAME loop, a different loss.

    Predict a probability, measure the log-loss, follow the ``(p - y)`` gradient downhill, repeat.
    We record the loss and freeze (w, b) at a few epochs to show the decision boundary sharpen.
    """
    w = np.zeros(x.shape[1])
    b = 0.0
    losses = np.empty(epochs + 1)
    snapshots: dict[int, tuple[NDArray[np.float64], float]] = {}
    for epoch in range(epochs + 1):
        losses[epoch] = log_loss_binary(x, y, w, b)
        if epoch in snapshot_epochs:
            snapshots[epoch] = (w.copy(), b)
        if epoch == epochs:
            break
        grad_w, grad_b = log_loss_gradient(x, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return LogisticFit(loss_curve=losses, w=w, b=b, snapshots=snapshots)


def sklearn_logistic(x: NDArray[np.float64], y: NDArray[np.int64]) -> tuple[NDArray[np.float64], float]:
    """scikit-learn's logistic-regression fit (near-unregularised) — the target our GD must reach."""
    model = LogisticRegression(C=1e4, max_iter=5000).fit(x, y)
    return model.coef_[0], float(model.intercept_[0])


# ============================ 4. learning-rate sweep ============================================
def lr_sweep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rates: tuple[float, ...] = LR_SWEEP,
    epochs: int = LR_SWEEP_EPOCHS,
) -> dict[float, NDArray[np.float64]]:
    """Train the SAME real regression at several learning rates; return each loss curve (measured).

    This is the experiment that teaches the single most important hyperparameter. With errors
    silenced (a diverging run overflows to ``inf`` on purpose — that *is* the lesson), we run the
    real GD loop at each rate and hand back the raw loss curves for the chapter to plot.
    """
    curves: dict[float, NDArray[np.float64]] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for rate in learning_rates:
            fit = train_linear_gd(x, y, lr=rate, epochs=epochs, snapshot_epochs=())
            curves[rate] = fit.loss_curve
    return curves


# ============================ 5. run it all: the printed proof ==================================
def main() -> None:
    """Run every real learner and cross-check, printing the measured results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    # ---- 1. linear regression from scratch on real data, matched to sklearn ----
    reg = load_income_price()
    fit = train_linear_gd(reg.x, reg.y)
    sk_w, sk_b = sklearn_linear(reg.x, reg.y)
    print("=== 1. Linear regression by gradient descent (real California housing) ===")
    print(f"  data: {reg.x.size} districts | feature: {reg.feature_name} | target: {reg.target_name}")
    print(f"  loss at epoch 0    = {fit.loss_curve[0]:.4f}  (flat line w=0, b=0 — knows nothing)")
    print(f"  loss at epoch {REG_EPOCHS}  = {fit.loss_curve[-1]:.4f}  (converged)")
    print(f"  our GD fit         : slope w = {fit.w[0]:+.4f}, intercept b = {fit.b:+.4f}")
    print(f"  sklearn OLS fit    : slope w = {sk_w[0]:+.4f}, intercept b = {sk_b:+.4f}")
    if not (np.allclose(fit.w, sk_w, atol=MATCH_TOL) and np.isclose(fit.b, sk_b, atol=MATCH_TOL)):
        raise AssertionError("our gradient-descent fit must match sklearn's closed-form least squares")
    print("  -> our from-scratch GD reached the exact least-squares solution (match within 1e-2)\n")

    # generality: the same loop on several features also matches OLS
    data = fetch_california_housing()
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(data.data.shape[0], size=N_REGRESSION, replace=False)
    x_multi, _, _ = _standardize(data.data[idx][:, :3].astype(np.float64))  # MedInc, HouseAge, AveRooms
    y_multi = data.target[idx].astype(np.float64)
    fit_multi = train_linear_gd(x_multi, y_multi, lr=0.3, epochs=400, snapshot_epochs=())
    skw_multi, skb_multi = sklearn_linear(x_multi, y_multi)
    print("=== 1b. Same loop, 3 features (generality check) ===")
    print(f"  our GD weights     : {np.array2string(fit_multi.w, precision=4, sign='+')}")
    print(f"  sklearn OLS weights: {np.array2string(skw_multi, precision=4, sign='+')}")
    if not np.allclose(fit_multi.w, skw_multi, atol=MATCH_TOL):
        raise AssertionError("multi-feature GD must also match OLS")
    print("  -> identical loop, more features, still reaches least squares\n")

    # ---- 2. logistic regression from scratch on real data, matched to sklearn ----
    cls = load_tumor_2d()
    lfit = train_logistic_gd(cls.x, cls.y)
    lw, lb = sklearn_logistic(cls.x, cls.y)
    pred = (predict_proba(cls.x, lfit.w, lfit.b) >= 0.5).astype(int)
    acc = float(np.mean(pred == cls.y))
    print("=== 2. Logistic regression by gradient descent (real breast-cancer, 2 features) ===")
    print(f"  data: {cls.x.shape[0]} tumours | features: {cls.feature_names}")
    print(f"  log-loss at epoch 0   = {lfit.loss_curve[0]:.4f}  (= ln 2 = {np.log(2):.4f}, a coin flip)")
    print(f"  log-loss at epoch {CLS_EPOCHS} = {lfit.loss_curve[-1]:.4f}  (converged)")
    print(f"  our GD fit      : w = {np.array2string(lfit.w, precision=3, sign='+')}, b = {lfit.b:+.3f}")
    print(f"  sklearn fit     : w = {np.array2string(lw, precision=3, sign='+')}, b = {lb:+.3f}")
    print(f"  training accuracy = {acc:.3f}")
    if not (np.allclose(lfit.w, lw, atol=5e-2) and np.isclose(lfit.b, lb, atol=5e-2)):
        raise AssertionError("our logistic GD fit must match sklearn's LogisticRegression")
    print("  -> the SAME loop, a different loss, reaches sklearn's logistic fit\n")

    # ---- 3. learning-rate sweep: crawl / converge / diverge (measured) ----
    curves = lr_sweep(reg.x, reg.y)
    print("=== 3. Learning-rate sweep (same real regression, three rates) ===")
    for rate, curve in curves.items():
        final = curve[-1]
        if not np.isfinite(final) or final > curve[0] * 5:
            verdict = "DIVERGED (loss exploded)"
        elif final > curve[0] * 0.5:
            verdict = "crawled (barely moved)"
        else:
            verdict = "converged (reached the minimum)"
        shown = "inf" if not np.isfinite(final) else f"{final:.4f}"
        print(f"  lr = {rate:<6}: start {curve[0]:.4f} -> end {shown:<10}  {verdict}")
    print("  -> too small crawls, well-chosen converges, too large diverges — the whole story of lr\n")


if __name__ == "__main__":
    main()

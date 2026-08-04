"""Backpropagation from scratch on REAL data, VERIFIED three ways.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from a
real pipeline (``numpy`` for the from-scratch gradients, ``torch.autograd`` as the reference engine,
``scikit-learn`` for a real dataset, ``matplotlib`` only in the figure generator). The single lesson is the
**backward pass** — reverse-mode automatic differentiation — and it is proven correct three independent ways:

  1. **Against calculus, by hand.** Four worked examples the chapter traces symbol by symbol — a scalar graph
     ``f = (a + b)·c``, a sigmoid neuron with squared error, the softmax + cross-entropy head (whose gradient
     collapses to ``p − y``), and a full 2→2→2 network forward *and* backward — are each reproduced here and
     checked against ``torch.autograd`` to machine precision (max abs diff ~1e-16).

  2. **Against a numerical gradient (the from-scratch correctness proof).** A generic multi-layer perceptron's
     analytic backward pass is checked against a **centred finite-difference** gradient for *every* parameter.
     On float64 the median relative error is ~1e-10 and the max ~1e-7 (a handful of tiny-gradient parameters
     inflate the ratio) — both far below the ~1e-3 that would flag a bug, the backprop analogue of the
     "matches scikit-learn" proof in the sibling chapters. We also sweep the step size ``eps`` to expose the
     truncation/round-off **U-curve** that sets the sweet spot near 1e-6.

  3. **Against a real autodiff engine, then put to work.** The same from-scratch MLP is rebuilt in PyTorch;
     ``loss.backward()`` returns gradients identical to ours. Then the from-scratch net is *trained* with those
     hand-computed gradients (plain mini-batch SGD) on **scikit-learn digits** (1797 8×8 images, 10 classes,
     no download) — the loss falls and test accuracy climbs past 96%, so backprop is not just correct, it
     learns.

What this module measures (all real, all reproducible from the seed)::

  * ``worked_scalar_graph`` / ``worked_sigmoid_neuron`` / ``worked_softmax_ce`` / ``worked_two_layer_net`` —
    the four hand-traced examples, each returning its forward values and backward gradients and each
    cross-checked against autograd.
  * ``MLP`` — a from-scratch multi-layer perceptron (row-vector convention ``Z = X @ W + b``) whose
    ``.backward()`` implements the four backprop equations directly; used for the gradient check, the torch
    cross-check, and the digits training.
  * ``gradient_check`` — analytic vs centred-difference gradients for every parameter; returns the max
    relative error (the correctness proof).
  * ``epsilon_sweep`` — the finite-difference step-size U-curve (truncation vs round-off).
  * ``train_digits`` — real mini-batch SGD training driven by the from-scratch gradients; returns the loss
    history and train/test accuracy.
  * ``depth_gradient_profile`` — the per-layer error (``delta``) magnitude and weight-gradient norm pulled
    back through a deep net for sigmoid vs ReLU, i.e. the vanishing-gradient product made visible.

Everything is seeded and CPU-only; runs standalone in a few seconds::

    python backpropagation.py

Verified on Python 3.12 / numpy 2.4 / torch 2.12 / scikit-learn 1.9 (CPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0

Activation = Literal["tanh", "relu", "sigmoid"]

# ------------------------------------------------------------------------------------------------
# Elementary activations and their exact local derivatives (the diagonal Jacobians of the chapter).
# Each returns f(z); the *_grad returns f'(z) expressed in whatever is cheapest (the cached output
# for tanh/sigmoid, the pre-activation for relu) — exactly what a real autograd op caches.
# ------------------------------------------------------------------------------------------------


def _act_forward(z: np.ndarray, kind: Activation) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(0.0, z)
    return 1.0 / (1.0 + np.exp(-z))  # sigmoid


def _act_local_grad(z: np.ndarray, a: np.ndarray, kind: Activation) -> np.ndarray:
    """f'(z) — the elementwise local gradient the backward pass multiplies into the upstream error."""
    if kind == "tanh":
        return 1.0 - a * a  # 1 - tanh^2
    if kind == "relu":
        return (z > 0.0).astype(z.dtype)  # 1[z>0] — the router gate
    return a * (1.0 - a)  # sigmoid: a(1-a)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the log-sum-exp shift (subtract the max) so large logits never overflow."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, y_onehot: np.ndarray) -> float:
    """Mean softmax + cross-entropy loss, computed stably via log-sum-exp (never log of a raw probability)."""
    shift = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shift - np.log(np.exp(shift).sum(axis=-1, keepdims=True))
    return float(-(y_onehot * log_probs).sum(axis=-1).mean())


# ================================================================================================
# Worked example 1 — a scalar computational graph f = (a + b)·c
# ================================================================================================


@dataclass(frozen=True)
class ScalarGraph:
    """f = (a + b)·c: the forward value and the backward gradients to every leaf."""

    f: float
    grads: dict[str, float]  # dL/da, dL/db, dL/dc
    max_abs_diff_vs_torch: float


def worked_scalar_graph(a: float = 2.0, b: float = 1.0, c: float = 3.0) -> ScalarGraph:
    """Trace f = (a+b)·c forward and backward by hand, then confirm against torch.autograd."""
    s = a + b  # forward: cache s
    f = s * c
    # backward: seed df/df = 1; multiply gate swaps inputs, add gate distributes.
    df_ds, df_dc = c, s  # d(s·c)/ds = c, /dc = s
    df_da, df_db = df_ds * 1.0, df_ds * 1.0  # add distributes df_ds unchanged
    grads = {"a": df_da, "b": df_db, "c": df_dc}

    at, bt, ct = (torch.tensor(v, requires_grad=True) for v in (a, b, c))
    (((at + bt) * ct)).backward()
    diff = max(abs(grads[k] - t.grad.item()) for k, t in zip("abc", (at, bt, ct)))
    return ScalarGraph(f=f, grads=grads, max_abs_diff_vs_torch=diff)


# ================================================================================================
# Worked example 2 — a sigmoid neuron with squared-error loss
# ================================================================================================


@dataclass(frozen=True)
class SigmoidNeuron:
    z: float
    a: float
    loss: float
    dL_dw: float
    dL_db: float
    dL_dx: float
    max_abs_diff_vs_torch: float


def worked_sigmoid_neuron(x: float = 2.0, w: float = -1.0, b: float = 3.0, y: float = 0.0) -> SigmoidNeuron:
    """z = wx + b, a = sigmoid(z), L = 1/2 (a - y)^2 — traced backward, checked against autograd."""
    z = w * x + b
    a = 1.0 / (1.0 + np.exp(-z))
    loss = 0.5 * (a - y) ** 2
    dL_da = a - y  # d(1/2 (a-y)^2)/da
    dL_dz = dL_da * a * (1.0 - a)  # sigmoid local grad a(1-a)
    dL_dw, dL_db, dL_dx = dL_dz * x, dL_dz * 1.0, dL_dz * w  # through z = wx + b

    xt, wt, bt = (torch.tensor(v, requires_grad=True) for v in (x, w, b))
    (0.5 * (torch.sigmoid(wt * xt + bt) - y) ** 2).backward()
    diff = max(abs(g - t.grad.item()) for g, t in ((dL_dw, wt), (dL_db, bt), (dL_dx, xt)))
    return SigmoidNeuron(float(z), float(a), float(loss), float(dL_dw), float(dL_db), float(dL_dx), diff)


# ================================================================================================
# Worked example 3 — softmax + cross-entropy, whose gradient collapses to p - y
# ================================================================================================


@dataclass(frozen=True)
class SoftmaxCE:
    p: np.ndarray
    grad: np.ndarray  # dL/dz = p - y
    max_abs_diff_vs_torch: float


def worked_softmax_ce(logits: tuple[float, ...] = (2.0, 1.0, 0.1), true_class: int = 0) -> SoftmaxCE:
    """The classifier-head gradient: softmax(z) then CE has dL/dz = p - y. Verified against autograd."""
    z = np.array(logits)
    p = softmax(z)
    y = np.zeros_like(p)
    y[true_class] = 1.0
    grad = p - y  # the whole derivation collapses to predicted minus one-hot target

    zt = torch.tensor(z, requires_grad=True)
    F.cross_entropy(zt.unsqueeze(0), torch.tensor([true_class])).backward()
    diff = float(np.abs(grad - zt.grad.numpy()).max())
    return SoftmaxCE(p=p, grad=grad, max_abs_diff_vs_torch=diff)


# ================================================================================================
# Worked example 4 — a full 2->2->2 net, forward AND backward by hand
# ================================================================================================


@dataclass(frozen=True)
class TwoLayerNet:
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    p: np.ndarray
    loss: float
    delta2: np.ndarray
    dW2: np.ndarray
    db2: np.ndarray
    da1: np.ndarray
    delta1: np.ndarray
    dW1: np.ndarray
    db1: np.ndarray
    max_abs_diff_vs_torch: float


def worked_two_layer_net() -> TwoLayerNet:
    """The centrepiece: forward + backward on a 2->2->2 net (tanh hidden, softmax+CE), checked vs torch.

    Row-vector convention z = xW + b, so the matmul VJP reads dL/da_prev = W·delta and dL/dW = x^T·delta.
    """
    x = np.array([1.0, 2.0])
    w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b1 = np.array([0.1, 0.2])
    w2 = np.array([[0.5, 0.1], [0.2, 0.3]])
    b2 = np.array([0.1, 0.2])
    y = np.array([1.0, 0.0])  # true class 0

    # forward (cache z1, a1)
    z1 = x @ w1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    p = softmax(z2)
    loss = -np.log(p[0])

    # backward — the four backprop equations, single sample so no batch averaging
    delta2 = p - y  # softmax + CE
    dW2 = np.outer(a1, delta2)  # dL/dW2 = a1^T delta2
    db2 = delta2
    da1 = w2 @ delta2  # pull back through W2 (row convention: W·delta)
    delta1 = da1 * (1.0 - a1**2)  # tanh local grad
    dW1 = np.outer(x, delta1)
    db1 = delta1

    xt = torch.tensor(x)
    w1t, b1t = torch.tensor(w1, requires_grad=True), torch.tensor(b1, requires_grad=True)
    w2t, b2t = torch.tensor(w2, requires_grad=True), torch.tensor(b2, requires_grad=True)
    logits = torch.tanh(xt @ w1t + b1t) @ w2t + b2t
    F.cross_entropy(logits.unsqueeze(0), torch.tensor([0])).backward()
    diff = max(
        float(np.abs(g - t.grad.numpy()).max())
        for g, t in ((dW1, w1t), (db1, b1t), (dW2, w2t), (db2, b2t))
    )
    return TwoLayerNet(z1, a1, z2, p, float(loss), delta2, dW2, db2, da1, delta1, dW1, db1, diff)


# ================================================================================================
# A from-scratch multi-layer perceptron with a manual forward + backward pass
# ================================================================================================


@dataclass
class MLP:
    """Row-vector MLP (Z = X @ W + b) with a manual backward pass implementing the four backprop equations.

    Hidden layers use ``activation``; the output layer is linear and the loss is softmax + cross-entropy,
    so the output error seeds as ``p - Y`` (averaged over the batch). Parameters live in ``self.params`` as
    a flat dict so the gradient check can perturb every entry generically.
    """

    layer_sizes: list[int]
    activation: Activation = "relu"
    seed: int = SEED
    params: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        for i, (d_in, d_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            scale = np.sqrt(2.0 / d_in)  # He-style init so the forward signal keeps a healthy scale
            self.params[f"W{i}"] = rng.standard_normal((d_in, d_out)) * scale
            self.params[f"b{i}"] = np.zeros(d_out)

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return logits and the cache of pre-activations/activations the backward pass needs."""
        cache: dict[str, np.ndarray] = {"a-1": x}  # a^{-1} = input
        a = x
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"z{i}"] = z
            a = _act_forward(z, self.activation) if i < self.n_layers - 1 else z  # last layer linear
            cache[f"a{i}"] = a
        return a, cache  # a of the last layer is the logits

    def backward(self, cache: dict[str, np.ndarray], y_onehot: np.ndarray) -> dict[str, np.ndarray]:
        """The four backprop equations, applied from the output layer down; returns dL/d(param) for all."""
        n = y_onehot.shape[0]
        logits = cache[f"a{self.n_layers - 1}"]
        delta = (softmax(logits) - y_onehot) / n  # dL/dz_last = (p - y)/N  (mean reduction)
        grads: dict[str, np.ndarray] = {}
        for i in reversed(range(self.n_layers)):
            a_prev = cache[f"a{i - 1}"] if i > 0 else cache["a-1"]
            grads[f"W{i}"] = a_prev.T @ delta  # dL/dW = a_prev^T delta (outer-product rule)
            grads[f"b{i}"] = delta.sum(axis=0)  # dL/db = sum over batch (bias fan-out)
            if i > 0:  # pull the error back through W_i, then through the activation of layer i-1
                da_prev = delta @ self.params[f"W{i}"].T  # W^T delta (matmul VJP)
                delta = da_prev * _act_local_grad(cache[f"z{i - 1}"], cache[f"a{i - 1}"], self.activation)
        return grads

    def loss(self, x: np.ndarray, y_onehot: np.ndarray) -> float:
        logits, _ = self.forward(x)
        return cross_entropy(logits, y_onehot)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)


# ================================================================================================
# Gradient checking — the from-scratch correctness proof
# ================================================================================================


@dataclass(frozen=True)
class GradCheck:
    max_rel_error: float
    median_rel_error: float
    n_params: int
    analytic: np.ndarray  # flattened analytic gradient (for the scatter figure)
    numerical: np.ndarray  # flattened numerical gradient
    eps: float


def _relative_error(g_a: np.ndarray, g_n: np.ndarray, tiny: float = 1e-12) -> np.ndarray:
    return np.abs(g_a - g_n) / np.maximum(np.abs(g_a) + np.abs(g_n), tiny)


def gradient_check(mlp: MLP, x: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-5) -> GradCheck:
    """Compare the analytic backward pass to a centred finite difference for EVERY parameter.

    Returns the maximum relative error over all parameters — on float64 this is ~1e-10 for a correct
    backward pass. This is the executed proof that the from-scratch gradients are right.
    """
    _, cache = mlp.forward(x)
    analytic = mlp.backward(cache, y_onehot)
    a_flat, n_flat = [], []
    for name, theta in mlp.params.items():
        g_a = analytic[name]
        g_n = np.zeros_like(theta)
        it = np.nditer(theta, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = theta[idx]
            theta[idx] = orig + eps
            l_plus = mlp.loss(x, y_onehot)
            theta[idx] = orig - eps
            l_minus = mlp.loss(x, y_onehot)
            theta[idx] = orig  # restore
            g_n[idx] = (l_plus - l_minus) / (2 * eps)  # centred difference
            it.iternext()
        a_flat.append(g_a.ravel())
        n_flat.append(g_n.ravel())
    a_all, n_all = np.concatenate(a_flat), np.concatenate(n_flat)
    rel = _relative_error(a_all, n_all)
    return GradCheck(float(rel.max()), float(np.median(rel)), a_all.size, a_all, n_all, eps)


@dataclass(frozen=True)
class EpsPoint:
    eps: float
    numerical: float
    rel_error: float


def epsilon_sweep(w: float = 0.7, exps: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 10)) -> tuple[float, list[EpsPoint]]:
    """The finite-difference U-curve on f(w) = 1/2 (tanh 3w - 0.5)^2: too-large eps => truncation,
    too-small eps => round-off. Returns the analytic derivative and the (eps, numerical, rel_err) points."""

    def f(v: float) -> float:
        return 0.5 * (np.tanh(3 * v) - 0.5) ** 2

    t = np.tanh(3 * w)
    analytic = (t - 0.5) * (1 - t**2) * 3  # chain rule by hand
    points = []
    for e in exps:
        eps = 10.0**-e
        num = (f(w + eps) - f(w - eps)) / (2 * eps)
        rel = abs(num - analytic) / max(abs(num) + abs(analytic), 1e-15)
        points.append(EpsPoint(eps, float(num), float(rel)))
    return float(analytic), points


# ================================================================================================
# Cross-check the from-scratch MLP against PyTorch autograd
# ================================================================================================


@dataclass(frozen=True)
class TorchMatch:
    max_abs_diff: float
    all_close: bool


def torch_cross_check(mlp: MLP, x: np.ndarray, y_idx: np.ndarray) -> TorchMatch:
    """Rebuild the exact from-scratch net in torch, run loss.backward(), and compare gradients."""
    y_onehot = np.eye(mlp.layer_sizes[-1])[y_idx]
    _, cache = mlp.forward(x)
    ours = mlp.backward(cache, y_onehot)

    tensors = {k: torch.tensor(v, requires_grad=True) for k, v in mlp.params.items()}
    a = torch.tensor(x)
    for i in range(mlp.n_layers):
        z = a @ tensors[f"W{i}"] + tensors[f"b{i}"]
        a = z if i == mlp.n_layers - 1 else _torch_act(z, mlp.activation)
    F.cross_entropy(a, torch.tensor(y_idx)).backward()
    diff = max(float(np.abs(ours[k] - tensors[k].grad.numpy()).max()) for k in mlp.params)
    close = all(np.allclose(ours[k], tensors[k].grad.numpy(), atol=1e-10) for k in mlp.params)
    return TorchMatch(diff, close)


def _torch_act(z: torch.Tensor, kind: Activation) -> torch.Tensor:
    return {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[kind](z)


# ================================================================================================
# Put backprop to work: train the from-scratch net on real scikit-learn digits
# ================================================================================================


@dataclass(frozen=True)
class TrainResult:
    loss_history: list[float]
    train_acc: float
    test_acc: float
    epochs: int
    n_train: int
    n_test: int
    n_params: int


def load_digits_split(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardized scikit-learn digits (1797 8x8 images, 10 classes), stratified 75/25 train/test split."""
    data = load_digits()
    x_tr, x_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target
    )
    scaler = StandardScaler().fit(x_tr)
    return scaler.transform(x_tr), scaler.transform(x_te), y_tr, y_te


def train_digits(
    hidden: int = 64, epochs: int = 60, batch_size: int = 64, lr: float = 0.2, seed: int = SEED
) -> TrainResult:
    """Train the from-scratch MLP with mini-batch SGD driven by our own backward pass (no autograd)."""
    x_tr, x_te, y_tr, y_te = load_digits_split(seed)
    n_classes = int(y_tr.max()) + 1
    mlp = MLP([x_tr.shape[1], hidden, n_classes], activation="relu", seed=seed)
    y_onehot = np.eye(n_classes)[y_tr]
    rng = np.random.default_rng(seed)
    n = x_tr.shape[0]
    history: list[float] = [mlp.loss(x_tr, y_onehot)]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            _, cache = mlp.forward(x_tr[idx])
            grads = mlp.backward(cache, y_onehot[idx])
            for name in mlp.params:  # plain SGD: the optimizer's job, using backprop's gradient
                mlp.params[name] -= lr * grads[name]
        history.append(mlp.loss(x_tr, y_onehot))
    train_acc = float((mlp.predict(x_tr) == y_tr).mean())
    test_acc = float((mlp.predict(x_te) == y_te).mean())
    n_params = sum(p.size for p in mlp.params.values())
    return TrainResult(history, train_acc, test_acc, epochs, n, x_te.shape[0], n_params)


# ================================================================================================
# The vanishing-gradient product, measured: delta magnitude and weight-grad norm vs depth
# ================================================================================================


@dataclass(frozen=True)
class DepthProfile:
    activation: Activation
    delta_norms: list[float]  # ||delta^l|| pulled back, layer 1 -> L
    grad_norms: list[float]  # ||dL/dW^l|| per layer


def depth_gradient_profile(depth: int = 12, width: int = 48, activation: Activation = "sigmoid", seed: int = SEED) -> DepthProfile:
    """One backward pass through a deep net; return the per-layer delta magnitude and weight-grad norm.

    With sigmoid/tanh the backward product shrinks toward the input layers (vanishing gradients); ReLU keeps
    it roughly flat. Uses a small unit-variance random init deliberately (not He) so the effect is visible.
    """
    rng = np.random.default_rng(seed)
    sizes = [width] * (depth + 1)
    mlp = MLP(sizes, activation=activation, seed=seed)
    for i in range(mlp.n_layers):  # override with a plain small-scale init to expose the product
        mlp.params[f"W{i}"] = rng.standard_normal((width, width)) * 0.1
    x = rng.standard_normal((32, width))
    y_onehot = np.eye(width)[rng.integers(0, width, size=32)]
    _, cache = mlp.forward(x)
    # replicate backward while recording per-layer delta norm
    n = y_onehot.shape[0]
    logits = cache[f"a{mlp.n_layers - 1}"]
    delta = (softmax(logits) - y_onehot) / n
    delta_norms, grad_norms = [], []
    for i in reversed(range(mlp.n_layers)):
        a_prev = cache[f"a{i - 1}"] if i > 0 else cache["a-1"]
        grad_norms.append(float(np.linalg.norm(a_prev.T @ delta)))
        delta_norms.append(float(np.linalg.norm(delta)))
        if i > 0:
            da_prev = delta @ mlp.params[f"W{i}"].T
            delta = da_prev * _act_local_grad(cache[f"z{i - 1}"], cache[f"a{i - 1}"], activation)
    return DepthProfile(activation, delta_norms[::-1], grad_norms[::-1])


# ================================================================================================
# Report
# ================================================================================================


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"numpy {np.__version__} | torch {torch.__version__} | scikit-learn ", end="")
    import sklearn

    print(f"{sklearn.__version__}  (CPU, seed={SEED})\n")

    g = worked_scalar_graph()
    print("=== Worked example 1 — scalar graph f = (a+b)·c, a=2 b=1 c=3 ===")
    print(f"  f = {g.f:.0f}   grads dL/da,db,dc = {g.grads}   max|hand-torch| = {g.max_abs_diff_vs_torch:.1e}\n")

    s = worked_sigmoid_neuron()
    print("=== Worked example 2 — sigmoid neuron, x=2 w=-1 b=3 y=0 ===")
    print(f"  z={s.z:.0f}  a={s.a:.4f}  L={s.loss:.4f}")
    print(f"  dL/dw={s.dL_dw:.4f}  dL/db={s.dL_db:.4f}  dL/dx={s.dL_dx:.4f}   max|hand-torch|={s.max_abs_diff_vs_torch:.1e}\n")

    sc = worked_softmax_ce()
    print("=== Worked example 3 — softmax + CE, logits=[2,1,0.1], true class 0 ===")
    print(f"  p = {np.round(sc.p, 4)}   dL/dz = p - y = {np.round(sc.grad, 4)}   max|hand-torch| = {sc.max_abs_diff_vs_torch:.1e}\n")

    net = worked_two_layer_net()
    print("=== Worked example 4 — full 2->2->2 net, forward AND backward by hand ===")
    print(f"  z1={np.round(net.z1, 4)}  a1={np.round(net.a1, 4)}  z2={np.round(net.z2, 4)}")
    print(f"  p={np.round(net.p, 4)}  L={net.loss:.4f}   delta2 = p - y = {np.round(net.delta2, 4)}")
    print(f"  dW2=\n{np.round(net.dW2, 4)}")
    print(f"  dL/da1 = W2·delta2 = {np.round(net.da1, 4)}   delta1 = {np.round(net.delta1, 4)}")
    print(f"  dW1=\n{np.round(net.dW1, 4)}")
    print(f"  max|hand-torch| over all grads = {net.max_abs_diff_vs_torch:.1e}  (machine precision)\n")

    print("=== Gradient check — analytic backward vs centred finite difference (float64) ===")
    mlp = MLP([64, 16, 10], activation="tanh", seed=SEED)
    x_tr, _, y_tr, _ = load_digits_split()
    xb, yb = x_tr[:16], np.eye(10)[y_tr[:16]]
    gc = gradient_check(mlp, xb, yb, eps=1e-5)
    print(f"  parameters checked : {gc.n_params}")
    print(f"  median relative error : {gc.median_rel_error:.2e}   max : {gc.max_rel_error:.2e}  (<< 1e-3 => correct)")
    if not (gc.max_rel_error < 1e-3):  # hard gate: a broken backward pass must FAIL, not print a bad number and exit 0
        raise AssertionError(f"gradient check FAILED: max relative error {gc.max_rel_error:.2e} >= 1e-3")
    print()

    print("=== Epsilon U-curve on f(w) = 1/2 (tanh 3w - 0.5)^2 at w=0.7 ===")
    analytic, pts = epsilon_sweep()
    print(f"  analytic derivative = {analytic:.6f}")
    print(f"  {'eps':>8}{'numerical':>14}{'rel error':>14}")
    for p in pts:
        print(f"  {p.eps:>8.0e}{p.numerical:>14.6f}{p.rel_error:>14.2e}")
    print()

    print("=== Torch cross-check — from-scratch MLP gradients vs loss.backward() ===")
    tm = torch_cross_check(mlp, xb, y_tr[:16])
    print(f"  max abs diff = {tm.max_abs_diff:.2e}   allclose(atol=1e-10) = {tm.all_close}")
    if not tm.all_close:  # hard gate: from-scratch gradients MUST match the reference autodiff engine
        raise AssertionError(f"torch cross-check FAILED: max abs diff {tm.max_abs_diff:.2e} exceeds atol=1e-10")
    print()

    print("=== Train the from-scratch net on scikit-learn digits (SGD driven by backprop) ===")
    tr = train_digits()
    print(f"  net = 64 -> 64 -> 10 ReLU, {tr.n_params} params; {tr.n_train} train / {tr.n_test} test")
    print(f"  loss: {tr.loss_history[0]:.3f} -> {tr.loss_history[-1]:.3f} over {tr.epochs} epochs")
    print(f"  train accuracy = {tr.train_acc:.4f}   test accuracy = {tr.test_acc:.4f}\n")

    print("=== Vanishing product, measured — weight-grad norm across a 12-layer net ===")
    for kind in ("sigmoid", "relu"):
        prof = depth_gradient_profile(activation=kind)
        ratio = prof.grad_norms[-1] / max(prof.grad_norms[0], 1e-30)
        print(f"  {kind:<8}: layer1 ||dW||={prof.grad_norms[0]:.2e}  layerL ||dW||={prof.grad_norms[-1]:.2e}  (Lx{ratio:.0e} vs layer 1)")


if __name__ == "__main__":
    main()

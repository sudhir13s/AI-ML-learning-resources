"""Generate the step-by-step teaching notebook (02-Backpropagation-and-Computational-Graphs.ipynb).

The notebook mirrors ``backpropagation.py`` one measurement at a time, so a learner can open it, run every
cell live, and *see* backprop being built and proven correct: the scalar computational graph, the four
local-gradient gates on numbers, a sigmoid neuron, the softmax + cross-entropy shortcut (p − y), a full
2→2→2 net forward AND backward, the from-scratch MLP's manual backward pass, the finite-difference gradient
check for every parameter, the ε U-curve, the PyTorch cross-check, real training on scikit-learn digits, and
the vanishing-gradient product measured across depth. Each numbered step has a short markdown lead-in (the
intuition) followed by ONE focused code cell with real output. This generator writes the .ipynb; a separate
nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_02.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../02-Backpropagation-and-Computational-Graphs/code/02-Backpropagation-and-Computational-Graphs.ipynb"

This generator lives in the domain-level ``05. Deep_Learning/tools/`` folder; the notebook it writes (and the
module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so
the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "02-Backpropagation-and-Computational-Graphs" / "code"
NB_PATH = _CHAPTER_CODE / "02-Backpropagation-and-Computational-Graphs.ipynb"

_CELL_ID = 0


def _next_id() -> str:
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
    "# Backpropagation & computational graphs — a step-by-step, runnable notebook\n"
    "\n"
    "To train a network you need the gradient of the loss with respect to **every** weight. The naive way — "
    "nudge each weight and watch the loss — costs one forward pass *per weight*, hopeless for millions of "
    "parameters. **Backpropagation** computes *all* of them in a single backward pass, at about the cost of "
    "one extra forward pass. It is **reverse-mode automatic differentiation** on a computational graph: a "
    "forward pass caches values, then a backward pass sends *upstream gradient × local gradient* to each "
    "node, in reverse order.\n"
    "\n"
    "This notebook builds that from scratch in NumPy and proves it correct **three independent ways** on real "
    "data: (1) against hand calculus (four worked examples, each checked against PyTorch autograd to ~1e-16); "
    "(2) against a numerical finite-difference gradient for every parameter (the from-scratch correctness "
    "proof); and (3) against PyTorch autograd, after which the from-scratch net is *trained* on scikit-learn "
    "digits to 97.8% test accuracy. Every cell uses the **exact same functions** as the chapter and its "
    "figures (imported from `backpropagation.py`), so the numbers here are the numbers there.\n"
    "\n"
    "> Companion page: **Backpropagation & Computational Graphs**. Run top-to-bottom (Kernel → Restart & Run "
    "All); it is CPU-only, seeded, and finishes in a few seconds."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the from-scratch functions from the chapter module so this notebook runs the *same code* the "
    "page and figures use, and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import torch\n"
    "import sklearn\n"
    "\n"
    "from backpropagation import (\n"
    "    worked_scalar_graph, worked_sigmoid_neuron, worked_softmax_ce, worked_two_layer_net,\n"
    "    MLP, softmax, gradient_check, epsilon_sweep, torch_cross_check, train_digits,\n"
    "    depth_gradient_profile, load_digits_split,\n"
    ")\n"
    "\n"
    "np.random.seed(0)\n"
    "torch.manual_seed(0)\n"
    "print(f'numpy {np.__version__} | torch {torch.__version__} | scikit-learn {sklearn.__version__}  (CPU)')"
)

# ---- Step 1: scalar graph ----
add_md(
    "## Step 1 — A scalar computational graph: $f = (a+b)\\cdot c$\n"
    "\n"
    "The whole algorithm in miniature. **Forward:** compute $s=a+b$, then $f=s\\cdot c$, caching $s$ and $c$. "
    "**Backward:** seed $\\partial f/\\partial f = 1$, then push it back through each op using its *local* "
    "gradient — the multiply gate swaps its inputs ($\\partial(sc)/\\partial s = c$), the add gate distributes "
    "unchanged. With $a=2,b=1,c=3$ every gradient is $3$ (check: $f=ac+bc$, so $\\partial f/\\partial a=c=3$). "
    "We confirm against autograd."
)
add_code(
    "g = worked_scalar_graph(a=2.0, b=1.0, c=3.0)\n"
    "print(f'f = {g.f:.0f}')\n"
    "print(f'gradients  dL/da, dL/db, dL/dc = {g.grads}')\n"
    "print(f'max |hand - autograd| = {g.max_abs_diff_vs_torch:.1e}   (exact)')"
)

# ---- Step 2: the four gates ----
add_md(
    "## Step 2 — The four local-gradient gates, on numbers\n"
    "\n"
    "Almost every backward pass is built from four routing patterns. With an upstream gradient of $2$:\n"
    "\n"
    "- **add = distributor:** copies $2$ to *both* inputs (bias adds, residuals — gradient passes through).\n"
    "- **multiply = swapper:** each input's gradient is $2\\times$ *the other* input.\n"
    "- **max / ReLU = router:** the whole $2$ goes to the winner, $0$ to the loser (the dead-ReLU effect).\n"
    "- **copy / fan-out = adder:** a value used in several places gets the **sum** of the returning gradients "
    "(this is why `.grad` accumulates and why you call `zero_grad()`).\n"
    "\n"
    "The fan-out sum is the rule beginners drop: $f = x\\cdot x = x^2$ at $x=3$ sends back $3$ along each of the "
    "two edges, and the total is $3+3=6 = 2x$ — the factor of 2 in $2x$ *is* the two edges summing."
)
add_code(
    "up = 2.0\n"
    "x_, y_ = 3.0, 4.0\n"
    "print(f'add       x<-{up:.0f}, y<-{up:.0f}         (distributes unchanged)')\n"
    "print(f'multiply  x<-{up*y_:.0f}, y<-{up*x_:.0f}         (swaps: up*other input)')\n"
    "print(f'max(5,1)  winner<-{up:.0f}, loser<-0     (routes to the larger input)')\n"
    "x = 3.0\n"
    "grad_fanout = up_edge = x + x   # both multiply edges send back the *other* x = 3\n"
    "print(f'copy      f=x*x at x=3: grad = 3 + 3 = {grad_fanout:.0f} = 2x   (fan-out sums)')"
)

# ---- Step 3: sigmoid neuron ----
add_md(
    "## Step 3 — A sigmoid neuron, end to end\n"
    "\n"
    "One real neuron: $z = wx+b$, $a=\\sigma(z)$, squared-error loss $L=\\tfrac12(a-y)^2$, with $x=2,w=-1,b=3,"
    "y=0$. Backward, one factor at a time: $\\partial L/\\partial a = a-y$; through the sigmoid, "
    "$\\partial a/\\partial z = a(1-a)$; then through $z=wx+b$. Note $a(1-a)\\le 0.25$: every sigmoid a gradient "
    "passes shrinks it by at least $4\\times$ — the seed of vanishing gradients (Step 13)."
)
add_code(
    "s = worked_sigmoid_neuron(x=2.0, w=-1.0, b=3.0, y=0.0)\n"
    "print(f'forward :  z = {s.z:.0f},  a = sigma(z) = {s.a:.4f},  L = {s.loss:.4f}')\n"
    "print(f'backward:  dL/dw = {s.dL_dw:.4f},  dL/db = {s.dL_db:.4f},  dL/dx = {s.dL_dx:.4f}')\n"
    "print(f'max |hand - autograd| = {s.max_abs_diff_vs_torch:.1e}')"
)

# ---- Step 4: softmax + CE ----
add_md(
    "## Step 4 — Softmax + cross-entropy → $\\hat y - y$\n"
    "\n"
    "The classifier head has the most elegant gradient in deep learning. With logits $z$, softmax "
    "probabilities $p$, one-hot target $y$, and $L=-\\log p_t$, the softmax Jacobian $\\partial p_i/\\partial "
    "z_j = p_i(\\delta_{ij}-p_j)$ and the $-1/p_t$ from cross-entropy **cancel**, leaving simply\n"
    "\n"
    "$$\\frac{\\partial L}{\\partial z} = p - y.$$\n"
    "\n"
    "Predicted probabilities minus the one-hot target — *exactly how wrong each class is*. The correct class "
    "gets a negative gradient (push its logit up); the wrong classes get positive gradients (push theirs "
    "down). Frameworks fuse the two ops for this clean gradient and for log-sum-exp stability."
)
add_code(
    "sc = worked_softmax_ce(logits=(2.0, 1.0, 0.1), true_class=0)\n"
    "print(f'softmax p       = {np.round(sc.p, 4)}')\n"
    "print(f'dL/dz = p - y   = {np.round(sc.grad, 4)}   (class 0 negative, others positive)')\n"
    "print(f'max |hand - autograd| = {sc.max_abs_diff_vs_torch:.1e}')"
)

# ---- Step 5: 2->2->2 net ----
add_md(
    "## Step 5 — A full 2→2→2 net, forward AND backward by hand\n"
    "\n"
    "The centrepiece: input $x\\in\\mathbb{R}^2$, a tanh hidden layer, a linear output layer, softmax + "
    "cross-entropy. Row-vector convention $z = xW+b$, so the matmul VJP reads $\\partial L/\\partial a_{\\text"
    "{prev}} = W\\delta$ and $\\partial L/\\partial W = x^\\top\\delta$. The four backprop equations, on real "
    "numbers: $\\delta^2 = p-y$; $\\partial L/\\partial W^2 = (a^1)^\\top\\delta^2$; pull back "
    "$\\partial L/\\partial a^1 = W^2\\delta^2$; through tanh $\\delta^1 = \\partial L/\\partial a^1 \\odot "
    "(1-(a^1)^2)$; then $\\partial L/\\partial W^1 = x^\\top\\delta^1$. Every number matches autograd to "
    "machine precision — this figure's annotations are these exact values."
)
add_code(
    "net = worked_two_layer_net()\n"
    "print('FORWARD')\n"
    "print(f'  z1 = {np.round(net.z1,4)}   a1 = tanh(z1) = {np.round(net.a1,4)}')\n"
    "print(f'  z2 = {np.round(net.z2,4)}   p = softmax(z2) = {np.round(net.p,4)}   L = {net.loss:.4f}')\n"
    "print('BACKWARD')\n"
    "print(f'  delta2 = p - y = {np.round(net.delta2,4)}')\n"
    "print(f'  dL/dW2 =\\n{np.round(net.dW2,4)}')\n"
    "print(f'  dL/da1 = W2.delta2 = {np.round(net.da1,4)}   delta1 = {np.round(net.delta1,4)}')\n"
    "print(f'  dL/dW1 =\\n{np.round(net.dW1,4)}')\n"
    "print(f'max |hand - autograd| over ALL gradients = {net.max_abs_diff_vs_torch:.1e}  (machine precision)')"
)

# ---- Step 6: the MLP forward ----
add_md(
    "## Step 6 — A from-scratch MLP: the forward pass, caching activations\n"
    "\n"
    "Now scale up to a real network. `MLP` stores `(W, b)` per layer (row convention $Z = X @ W + b$), applies "
    "the hidden activation, and leaves the output layer linear (the loss is softmax + cross-entropy). The "
    "forward pass **caches** every pre-activation $z$ and activation $a$ — the backward pass needs them. Here "
    "is a 64→32→10 net on a batch of real digit images."
)
add_code(
    "x_tr, x_te, y_tr, y_te = load_digits_split()\n"
    "mlp = MLP([64, 32, 10], activation='relu', seed=0)\n"
    "logits, cache = mlp.forward(x_tr[:8])\n"
    "print(f'input batch      : {x_tr[:8].shape}   (8 flattened 8x8 digit images)')\n"
    "print(f'cached tensors   : {sorted(cache.keys())}')\n"
    "print(f'logits shape     : {logits.shape}   softmax row 0 sums to {softmax(logits)[0].sum():.4f}')"
)

# ---- Step 7: the MLP backward ----
add_md(
    "## Step 7 — The backward pass: the four equations, in code\n"
    "\n"
    "This is the entire algorithm. Seed the output error $\\delta = (p-y)/N$ (mean reduction over the batch), "
    "then for each layer from the output down: read off $\\partial L/\\partial W = a_{\\text{prev}}^\\top\\delta$ "
    "and $\\partial L/\\partial b = \\sum_{\\text{batch}}\\delta$, then pull the error back one layer with "
    "$W^\\top\\delta$ and multiply by the activation's local gradient. The gradient shapes must match the "
    "parameter shapes — that is how a missing transpose is caught."
)
add_code(
    "y_onehot = np.eye(10)[y_tr[:8]]\n"
    "grads = mlp.backward(cache, y_onehot)\n"
    "for name in ['W0', 'b0', 'W1', 'b1']:\n"
    "    assert grads[name].shape == mlp.params[name].shape\n"
    "    print(f'  grad {name}: shape {grads[name].shape} == param shape  (norm {np.linalg.norm(grads[name]):.4f})')"
)

# ---- Step 8: gradient check ----
add_md(
    "## Step 8 — Gradient check: analytic backward vs numerical finite difference\n"
    "\n"
    "The from-scratch correctness proof. For **every** parameter, compare the analytic backprop gradient to a "
    "**centred finite difference** $\\big(L(\\theta+\\epsilon)-L(\\theta-\\epsilon)\\big)/2\\epsilon$ using the "
    "robust *relative* error. On float64 the median relative error is ~1e-10 and the max ~1e-7 (a few "
    "small-gradient parameters inflate the ratio) — both far below the ~1e-3 that would flag a bug. This is an "
    "*independent* algorithm agreeing with backprop, so it rules out a shared mistake."
)
add_code(
    "check_net = MLP([64, 16, 10], activation='tanh', seed=0)\n"
    "gc = gradient_check(check_net, x_tr[:16], np.eye(10)[y_tr[:16]], eps=1e-5)\n"
    "print(f'parameters checked    : {gc.n_params}')\n"
    "print(f'median relative error : {gc.median_rel_error:.2e}')\n"
    "print(f'max    relative error : {gc.max_rel_error:.2e}   (<< 1e-3 => backward pass is correct)')\n"
    "# hard gate: a broken backward pass must FAIL here, not just print a bad number\n"
    "assert gc.max_rel_error < 1e-3, f'gradient check failed: max rel error {gc.max_rel_error:.2e}'\n"
    "print('OK: gradient check confirms the from-scratch backward pass')"
)

# ---- Step 9: epsilon U-curve ----
add_md(
    "## Step 9 — The ε U-curve: why the step size matters\n"
    "\n"
    "The finite-difference error is a **U** in $\\epsilon$. Too *large* an $\\epsilon$ leaves $O(\\epsilon^2)$ "
    "**truncation error** (the difference isn't the true derivative); too *small* an $\\epsilon$ subtracts two "
    "nearly-equal floats and amplifies floating-point **round-off**. The sweet spot for float64 sits near "
    "$\\epsilon\\approx 10^{-6}$. On $f(w)=\\tfrac12(\\tanh 3w - 0.5)^2$ at $w=0.7$ (analytic derivative "
    "$0.082173$):"
)
add_code(
    "analytic, pts = epsilon_sweep(w=0.7)\n"
    "print(f'analytic derivative = {analytic:.6f}\\n')\n"
    "print(f\"{'eps':>8}{'numerical':>14}{'rel error':>14}\")\n"
    "for p in pts:\n"
    "    print(f'  {p.eps:>8.0e}{p.numerical:>14.6f}{p.rel_error:>12.2e}')"
)

# ---- Step 10: torch cross-check ----
add_md(
    "## Step 10 — Cross-check against a real autodiff engine (PyTorch)\n"
    "\n"
    "Rebuild the exact from-scratch net in PyTorch, call `loss.backward()`, and compare gradients. They agree "
    "to machine epsilon — because autograd runs the *same* chain-rule VJPs we coded by hand. This is the "
    "'matches the real engine' proof."
)
add_code(
    "tm = torch_cross_check(check_net, x_tr[:16], y_tr[:16])\n"
    "print(f'max |from-scratch - autograd| = {tm.max_abs_diff:.2e}')\n"
    "print(f'allclose(atol=1e-10)          = {tm.all_close}')\n"
    "# hard gate: the from-scratch gradients must match the reference autodiff engine\n"
    "assert tm.all_close, f'torch cross-check failed: max abs diff {tm.max_abs_diff:.2e}'\n"
    "print('OK: from-scratch gradients match PyTorch autograd')"
)

# ---- Step 11: train on digits ----
add_md(
    "## Step 11 — Put backprop to work: train on scikit-learn digits\n"
    "\n"
    "The payoff. Feed the from-scratch gradients to plain mini-batch **SGD** (no autograd, no framework) and "
    "train a 64→64→10 ReLU network on 1,347 handwritten digits. The cross-entropy loss falls from ~2.8 toward "
    "zero and the held-out **test accuracy** climbs past 96% — proof the gradients are not just numerically "
    "correct but *useful*: they point downhill. Backprop computes the gradient; SGD (the optimizer) uses it."
)
add_code(
    "tr = train_digits(hidden=64, epochs=60, batch_size=64, lr=0.2)\n"
    "print(f'net = 64 -> 64 -> 10 ReLU, {tr.n_params} params; {tr.n_train} train / {tr.n_test} test')\n"
    "print(f'loss: {tr.loss_history[0]:.3f} -> {tr.loss_history[-1]:.3f} over {tr.epochs} epochs')\n"
    "print(f'train accuracy = {tr.train_acc:.4f}   test accuracy = {tr.test_acc:.4f}')"
)
add_code(
    "import matplotlib.pyplot as plt\n"
    "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
    "ax.plot(tr.loss_history, color='#5D4A8A', lw=2)\n"
    "ax.set_xlabel('epoch')\n"
    "ax.set_ylabel('training cross-entropy loss')\n"
    "ax.set_title(f'Backprop trains: loss {tr.loss_history[0]:.2f} -> {tr.loss_history[-1]:.3f}, "
    "test acc {tr.test_acc*100:.1f}%')\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.show()"
)

# ---- Step 12: vanishing gradients ----
add_md(
    "## Step 12 — The vanishing product, measured across depth\n"
    "\n"
    "The backward error recurrence $\\delta^l = ((W^{l+1})^\\top\\delta^{l+1})\\odot\\sigma'(z^l)$ multiplies "
    "$\\sim D$ factors as it flows toward the input. If those factors are consistently below one — sigmoid's "
    "$\\sigma'\\le 0.25$ — the gradient **vanishes** geometrically toward the early layers. We pull one "
    "backward pass through a 12-layer net and read the per-layer weight-gradient norm: sigmoid's is orders of "
    "magnitude smaller at layer 1 than at the output, while ReLU (local gradient exactly 1 in its active "
    "region) stays flat. This one product motivates ReLU, residual connections, normalization, and careful "
    "initialization."
)
add_code(
    "for kind in ('sigmoid', 'relu'):\n"
    "    prof = depth_gradient_profile(depth=12, activation=kind)\n"
    "    ratio = prof.grad_norms[0] / prof.grad_norms[-1]\n"
    "    print(f'{kind:<8}: ||dW|| layer 1 = {prof.grad_norms[0]:.2e},  layer L = {prof.grad_norms[-1]:.2e}'\n"
    "          f'   (layer 1 is {ratio:.0e}x the output layer)')"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "Backprop is **reverse-mode autodiff on the computational graph**: a forward pass caches values, then a "
    "backward pass sends *upstream × local gradient* to each node in reverse order, **summing** at fan-outs. "
    "It is the chain rule plus **dynamic programming** — each $\\delta$ computed once and reused — so *all* "
    "gradients come out in one pass, because the loss is a single scalar (many inputs, one output → reverse "
    "mode wins). The softmax+CE gradient is $p-y$; the matmul gradients are $\\delta x^\\top$ and "
    "$W^\\top\\delta$. We proved the from-scratch backward pass correct against hand calculus (~1e-16), a "
    "numerical gradient (median rel error ~1e-10), and PyTorch autograd, then trained a network with it to "
    "97.8% test accuracy. Backprop computes the gradient; the **optimizer** does the update.\n"
    "\n"
    "See the companion page for the full derivations, pitfalls, and references."
)


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()

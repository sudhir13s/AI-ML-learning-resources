"""Generate the step-by-step teaching notebook (04-How-Models-Learn.ipynb).

The notebook mirrors ``how_models_learn.py`` one operation at a time, so a beginner can open it, run
every cell live, and *see* how a model learns — from a line that knows nothing to a fitted model,
one downhill step at a time. Each numbered step has a short markdown lead-in (the intuition) followed
by ONE focused code cell with real output. This generator writes the .ipynb; a separate nbconvert
pass executes it headless so the outputs are embedded.

    python build_notebook_04.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../04-How-Models-Learn/code/04-How-Models-Learn.ipynb"

This generator lives in the domain-level ``00. Basics/tools/`` folder; the notebook it writes (and
the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep: the same algorithm, typed once in the
module, demonstrated step-by-step here.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "04-How-Models-Learn"
    / "code"
    / "04-How-Models-Learn.ipynb"
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
    "# How Models Learn — a step-by-step, runnable notebook\n"
    "\n"
    "A model starts out knowing **nothing**. This notebook shows, one operation at a time, exactly how "
    "it gets better — on **real data**, with a learning loop we build from scratch (no autodiff). It is "
    "the executable companion to the chapter and to `how_models_learn.py`; every function used here lives "
    "in that module, imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **watched**, on real datasets:\n"
    "\n"
    "1. a model that **knows nothing** (a flat line) and how we score its wrongness with a **loss**;\n"
    "2. the **gradient** — which way is downhill — and a single **gradient-descent step** by hand;\n"
    "3. the **full training loop**: predict → loss → gradient → update → repeat, with the loss falling;\n"
    "4. the fitted line **rotating into place**, and a proof it reaches the *exact* least-squares answer;\n"
    "5. gradient descent as a **ball rolling downhill** on the real loss surface;\n"
    "6. the **learning rate**: too small crawls, well-chosen converges, too large diverges;\n"
    "7. the **same loop** training a **logistic-regression** classifier — a different loss, identical loop.\n"
    "\n"
    "Everything runs on CPU in a couple of seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real functions from the chapter module (so this notebook uses the *exact same code* "
    "the figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "\n"
    "from how_models_learn import (\n"
    "    load_income_price, predict_linear, mse_loss, mse_gradient,\n"
    "    train_linear_gd, sklearn_linear, linear_gd_path, lr_sweep, LR_SWEEP,\n"
    "    load_tumor_2d, sigmoid, log_loss_binary, log_loss_gradient,\n"
    "    predict_proba, train_logistic_gd, sklearn_logistic,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__} '\n"
    "      f'| matplotlib {matplotlib.__version__}')"
)

# ---- Step 1: a model that knows nothing ----
add_md(
    "## Step 1 — A model that knows nothing\n"
    "\n"
    "Our model is the simplest possible: a straight line, $\\hat y = w\\,x + b$. Here $x$ is a California "
    "district's **median income** and $\\hat y$ is our prediction of its **median house value**. Learning "
    "means finding good numbers for the **slope** $w$ and **intercept** $b$.\n"
    "\n"
    "Before any learning we set $w = 0,\\ b = 0$ — a flat line pinned at zero. It predicts a house value of "
    "0 for *every* district, no matter the income. That is a model that knows nothing; watch how badly it "
    "does on real data."
)
add_code(
    "reg = load_income_price()\n"
    "print(f'{reg.x.size} real districts | feature: {reg.feature_name} | target: {reg.target_name}')\n"
    "\n"
    "w, b = np.array([0.0]), 0.0            # the model that knows nothing\n"
    "y_hat = predict_linear(reg.x[:, None], w, b)\n"
    "print(f'first 5 predictions : {np.round(y_hat[:5], 2)}   (all zero — it ignores income)')\n"
    "print(f'first 5 true values : {np.round(reg.y[:5], 2)}   ($100k units)')"
)

# ---- Step 2: measuring how wrong ----
add_md(
    "## Step 2 — Measuring *how wrong*: the loss\n"
    "\n"
    "To improve, the model first needs a number that says how wrong it is. That number is the **loss**. For "
    "regression we use the **mean squared error** — the average squared gap between prediction and truth:\n"
    "\n"
    "$$L(w, b) = \\frac{1}{n}\\sum_{i=1}^{n}\\big(\\hat y_i - y_i\\big)^2, \\qquad \\hat y_i = w\\,x_i + b.$$\n"
    "\n"
    "Squaring makes every error positive and punishes big misses far more than small ones. A perfect model "
    "has loss 0; our know-nothing model should have a big loss."
)
add_code(
    "loss0 = mse_loss(reg.x[:, None], reg.y, w, b)\n"
    "print(f'loss of the know-nothing model = {loss0:.4f}')\n"
    "print('this is just the mean of y squared, since y_hat = 0:',\n"
    "      round(float(np.mean(reg.y ** 2)), 4))\n"
    "print('\\nour whole job: make this number as small as the data allows.')"
)

# ---- Step 3: which way is downhill ----
add_md(
    "## Step 3 — Which way is downhill? The gradient\n"
    "\n"
    "We want to change $w$ and $b$ to *lower* the loss. But which direction lowers it? Calculus answers "
    "this: the **gradient** $\\nabla L = (\\partial L/\\partial w,\\ \\partial L/\\partial b)$ is the "
    "direction of steepest **increase**. So the direction of steepest **decrease** — downhill — is the "
    "*negative* gradient. For our MSE, the chain rule gives a clean form (derived in the chapter):\n"
    "\n"
    "$$\\frac{\\partial L}{\\partial w} = \\frac{2}{n}\\sum_i (\\hat y_i - y_i)\\,x_i, \\qquad "
    "\\frac{\\partial L}{\\partial b} = \\frac{2}{n}\\sum_i (\\hat y_i - y_i).$$\n"
    "\n"
    "Both are built from the **residual** $\\hat y_i - y_i$ (prediction minus truth). At our starting point "
    "the predictions are all 0, so every residual is $-y_i$ — the gradient will point us toward a positive "
    "slope and intercept, exactly as it should."
)
add_code(
    "grad_w, grad_b = mse_gradient(reg.x[:, None], reg.y, w, b)\n"
    "print(f'gradient at (w=0, b=0): dL/dw = {grad_w[0]:+.4f}, dL/db = {grad_b:+.4f}')\n"
    "print('both negative -> downhill is +w, +b (raise the slope and intercept). Makes sense:')\n"
    "print('the line is far too low, so lifting and tilting it up reduces the error.')"
)

# ---- Step 4: one step by hand ----
add_md(
    "## Step 4 — One gradient-descent step, by hand\n"
    "\n"
    "Now the single most important line in machine learning. We nudge each parameter a small amount in the "
    "downhill (negative-gradient) direction:\n"
    "\n"
    "$$w \\leftarrow w - \\eta\\,\\frac{\\partial L}{\\partial w}, \\qquad b \\leftarrow b - \\eta\\,"
    "\\frac{\\partial L}{\\partial b}.$$\n"
    "\n"
    "The number $\\eta$ (**eta**) is the **learning rate** — how big a step we take. Let's take one step "
    "with $\\eta = 0.3$ and confirm the loss goes **down**."
)
add_code(
    "eta = 0.3\n"
    "w_new = w - eta * grad_w\n"
    "b_new = b - eta * grad_b\n"
    "loss1 = mse_loss(reg.x[:, None], reg.y, w_new, b_new)\n"
    "print(f'before: w={w[0]:+.3f}, b={b:+.3f}, loss={loss0:.4f}')\n"
    "print(f'after : w={w_new[0]:+.3f}, b={b_new:+.3f}, loss={loss1:.4f}')\n"
    "print(f'loss dropped by {loss0 - loss1:.4f} in a single step. Now just repeat.')"
)

# ---- Step 5: the full loop ----
add_md(
    "## Step 5 — The full training loop: predict → loss → gradient → update → repeat\n"
    "\n"
    "Learning is that one step, done over and over. Each full pass is an **epoch**. `train_linear_gd` runs "
    "the loop from $w=0, b=0$ for 200 epochs, recording the loss each time. Watch it fall steeply at first "
    "(the gradient is large when we're very wrong) and then flatten as it approaches the best possible fit."
)
add_code(
    "fit = train_linear_gd(reg.x, reg.y)\n"
    "curve = fit.loss_curve\n"
    "print(f'loss at epoch 0   = {curve[0]:.4f}  (knows nothing)')\n"
    "print(f'loss at epoch 5   = {curve[5]:.4f}  (already most of the way)')\n"
    "print(f'loss at epoch 200 = {curve[-1]:.4f}  (converged)')\n"
    "\n"
    "plt.figure(figsize=(7, 3.8))\n"
    "plt.plot(range(curve.size), curve, lw=2, color='#8B3B4A')\n"
    "plt.xlabel('epoch')\n"
    "plt.ylabel('loss (MSE)')\n"
    "plt.title('learning is loss going down')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 6: watch the line fit ----
add_md(
    "## Step 6 — Watch the line fit itself to the data\n"
    "\n"
    "The falling loss *is* the line rotating into place. We froze $(w, b)$ at a few epochs during training; "
    "plotting each one over the real scatter shows the model physically fitting itself — from the flat "
    "know-nothing line at epoch 0 to the best straight-line fit by epoch 200."
)
add_code(
    "xs = np.linspace(reg.x.min(), reg.x.max(), 100)\n"
    "plt.figure(figsize=(7.5, 4.6))\n"
    "plt.scatter(reg.x, reg.y, s=8, alpha=0.15, color='#3A6B96', label='real districts')\n"
    "shades = plt.cm.Greens(np.linspace(0.35, 0.95, len(fit.snapshots)))\n"
    "for shade, epoch in zip(shades, sorted(fit.snapshots)):\n"
    "    ws, bs = fit.snapshots[epoch]\n"
    "    plt.plot(xs, ws[0] * xs + bs, color=shade, lw=2, label=f'epoch {epoch}')\n"
    "plt.xlabel(reg.feature_name)\n"
    "plt.ylabel(reg.target_name)\n"
    "plt.title('the line rotates from flat (epoch 0) into the best fit')\n"
    "plt.legend(fontsize=8, loc='upper left')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 7: match sklearn ----
add_md(
    "## Step 7 — We built the real thing: it matches scikit-learn\n"
    "\n"
    "Is our from-scratch gradient descent *actually correct*, or just close? Linear regression has a known "
    "closed-form answer — the **least-squares** solution — which `sklearn.LinearRegression` computes "
    "directly (no gradient descent at all). If our GD found the same slope and intercept, we didn't build a "
    "lookalike; we built the real algorithm and it converged to the exact right answer."
)
add_code(
    "sk_w, sk_b = sklearn_linear(reg.x, reg.y)\n"
    "print(f'our gradient descent : slope w = {fit.w[0]:+.5f}, intercept b = {fit.b:+.5f}')\n"
    "print(f'sklearn least squares: slope w = {sk_w[0]:+.5f}, intercept b = {sk_b:+.5f}')\n"
    "print('match within 1e-2:', bool(np.allclose(fit.w, sk_w, atol=1e-2)\n"
    "                                 and np.isclose(fit.b, sk_b, atol=1e-2)))\n"
    "print('\\n=> our loop reached the provably-optimal fit. It is the real thing.')"
)

# ---- Step 8: the loss surface ----
add_md(
    "## Step 8 — Gradient descent is a ball rolling downhill\n"
    "\n"
    "Here is the picture behind the whole method. The loss $L(w, b)$ is a **surface** over the two "
    "parameters — a bowl. Gradient descent drops a ball at our start $(0, 0)$ and lets it roll downhill, "
    "one step per epoch, until it settles at the bottom — the least-squares minimum (the star). We plot the "
    "real surface and the real path our training took."
)
add_code(
    "w_hist, b_hist, _ = linear_gd_path(reg.x, reg.y)\n"
    "\n"
    "# evaluate the real loss surface on a grid around the descent path\n"
    "wg = np.linspace(-0.3, 1.3, 60)\n"
    "bg = np.linspace(-0.2, 2.6, 60)\n"
    "WW, BB = np.meshgrid(wg, bg)\n"
    "Z = np.array([[mse_loss(reg.x[:, None], reg.y, np.array([w_]), b_)\n"
    "               for w_ in wg] for b_ in bg])\n"
    "\n"
    "plt.figure(figsize=(6.6, 5))\n"
    "cs = plt.contourf(WW, BB, Z, levels=25, cmap='Blues_r')\n"
    "plt.colorbar(cs, label='loss (MSE)')\n"
    "plt.plot(w_hist, b_hist, color='#8B3B4A', lw=1.8, marker='o', ms=2.5, label='descent path')\n"
    "plt.scatter([sk_w[0]], [sk_b], color='#2E7A5A', marker='*', s=220,\n"
    "            edgecolor='white', zorder=5, label='minimum')\n"
    "plt.xlabel('slope w')\n"
    "plt.ylabel('intercept b')\n"
    "plt.title('the ball rolls from (0,0) to the bottom of the bowl')\n"
    "plt.legend(loc='lower right')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: learning-rate sweep ----
add_md(
    "## Step 9 — The learning rate decides everything\n"
    "\n"
    "The step size $\\eta$ is the hyperparameter you *must* get roughly right. Same model, same data — only "
    "$\\eta$ changes:\n"
    "\n"
    "* **too small** ($\\eta = 0.001$): steps are tiny, the loss barely moves — it **crawls**;\n"
    "* **well-chosen** ($\\eta = 0.3$): steps are just right, the loss plunges — it **converges**;\n"
    "* **too large** ($\\eta = 1.02$): steps overshoot the minimum and grow — it **diverges** to infinity.\n"
    "\n"
    "We measure all three on the real regression (note the log scale — the diverging run climbs off the "
    "top)."
)
add_code(
    "curves = lr_sweep(reg.x, reg.y)\n"
    "for rate in LR_SWEEP:\n"
    "    c = curves[rate]\n"
    "    end = 'inf' if not np.isfinite(c[-1]) else f'{c[-1]:.3f}'\n"
    "    print(f'lr = {rate:<6}: start {c[0]:.3f} -> end {end}')\n"
    "\n"
    "plt.figure(figsize=(7.5, 4.2))\n"
    "labels = {LR_SWEEP[0]: 'too small (crawls)', LR_SWEEP[1]: 'good (converges)',\n"
    "          LR_SWEEP[2]: 'too large (diverges)'}\n"
    "colors = {LR_SWEEP[0]: '#4A5B6E', LR_SWEEP[1]: '#2E7A5A', LR_SWEEP[2]: '#8B3B4A'}\n"
    "for rate in LR_SWEEP:\n"
    "    c = np.clip(np.nan_to_num(curves[rate], posinf=1e6), 1e-3, 1e6)\n"
    "    plt.plot(range(c.size), c, lw=2, color=colors[rate], label=f'lr={rate} ({labels[rate]})')\n"
    "plt.yscale('log')\n"
    "plt.xlabel('epoch')\n"
    "plt.ylabel('loss (log scale)')\n"
    "plt.title('same model & data, only the step size changes')\n"
    "plt.legend(fontsize=8)\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: same loop, different loss ----
add_md(
    "## Step 10 — The same loop, a different loss: classification\n"
    "\n"
    "Does this loop only work for drawing lines through scatter plots? No — it is *how every model learns*. "
    "To prove it, we switch problems entirely: **classifying** real tumours as benign or malignant from two "
    "measurements (mean radius, mean texture). Two things change, the loop does not:\n"
    "\n"
    "* the model now outputs a **probability** via the **sigmoid**, $p = \\sigma(w\\!\\cdot\\!x + b) = "
    "1/(1+e^{-(w\\cdot x + b)})$, squashing any score into $[0, 1]$;\n"
    "* the loss becomes **log-loss** (binary cross-entropy), the *surprise* of the true label: "
    "$L = -\\frac{1}{n}\\sum_i [\\,y_i \\log p_i + (1-y_i)\\log(1-p_i)\\,]$.\n"
    "\n"
    "(The chapter on Cross-Entropy & KL derives this loss in full.) A know-nothing classifier predicts $p=0.5$ "
    "for everyone, giving loss $\\ln 2 \\approx 0.693$ — a coin flip."
)
add_code(
    "cls = load_tumor_2d()\n"
    "print(f'{cls.x.shape[0]} real tumours | features: {cls.feature_names}')\n"
    "print(f'class balance: {int((cls.y == 1).sum())} benign, {int((cls.y == 0).sum())} malignant')\n"
    "\n"
    "w0, b0 = np.zeros(2), 0.0                       # know-nothing classifier\n"
    "print(f'sigmoid(0) = {sigmoid(np.array([0.0]))[0]:.3f}  (predicts 0.5 for everyone)')\n"
    "print(f'log-loss of the coin-flip model = {log_loss_binary(cls.x, cls.y, w0, b0):.4f} '\n"
    "      f'(= ln 2 = {np.log(2):.4f})')"
)

# ---- Step 11: train logistic ----
add_md(
    "## Step 11 — Train the classifier with the *identical* loop\n"
    "\n"
    "`train_logistic_gd` is the same four steps — predict, measure loss, compute gradient, step downhill — "
    "with the sigmoid model and the log-loss. Watch the loss fall from $\\ln 2$ just like the MSE fell, and "
    "read off the final training accuracy."
)
add_code(
    "lfit = train_logistic_gd(cls.x, cls.y)\n"
    "lcurve = lfit.loss_curve\n"
    "acc = float(np.mean((predict_proba(cls.x, lfit.w, lfit.b) >= 0.5).astype(int) == cls.y))\n"
    "print(f'log-loss at epoch 0    = {lcurve[0]:.4f}  (= ln 2, a coin flip)')\n"
    "print(f'log-loss at epoch 2000 = {lcurve[-1]:.4f}  (converged)')\n"
    "print(f'final training accuracy = {acc:.3f}')\n"
    "\n"
    "plt.figure(figsize=(7, 3.6))\n"
    "plt.plot(range(lcurve.size), lcurve, lw=2, color='#5D4A8A')\n"
    "plt.axhline(np.log(2), ls='--', color='gray', label='coin flip: ln 2')\n"
    "plt.xlabel('epoch')\n"
    "plt.ylabel('log-loss')\n"
    "plt.title('the same loop, a different loss')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 12: the (p - y) gradient ----
add_md(
    "## Step 12 — The gradient is the *same shape* as before\n"
    "\n"
    "Why does the identical loop work for a completely different loss? Because the gradient of log-loss, "
    "after the sigmoid derivative and the log cancel, collapses to the very same form as MSE's:\n"
    "\n"
    "$$\\frac{\\partial L}{\\partial w} = \\frac{1}{n}\\sum_i (p_i - y_i)\\,x_i, \\qquad "
    "\\frac{\\partial L}{\\partial b} = \\frac{1}{n}\\sum_i (p_i - y_i).$$\n"
    "\n"
    "It is **prediction minus target** again — exactly the residual structure from Step 3. That is the deep "
    "reason one gradient-descent loop trains linear regression, logistic regression, and (with a softmax) "
    "the output layer of a neural network alike."
)
add_code(
    "gw, gb = log_loss_gradient(cls.x, cls.y, w0, b0)\n"
    "p0 = predict_proba(cls.x, w0, b0)\n"
    "manual_gw = (cls.x.T @ (p0 - cls.y)) / cls.x.shape[0]   # (1/n) X^T (p - y), by hand\n"
    "print(f'log_loss_gradient dL/dw = {np.round(gw, 4)}')\n"
    "print(f'by-hand (1/n)Xᵀ(p - y) = {np.round(manual_gw, 4)}')\n"
    "print('same (p - y) structure as linear regression:', bool(np.allclose(gw, manual_gw)))"
)

# ---- Step 13: match sklearn logistic ----
add_md(
    "## Step 13 — And it matches scikit-learn's classifier\n"
    "\n"
    "The same proof as Step 7, for classification: our from-scratch logistic GD should reach the same "
    "weights as `sklearn.LogisticRegression` (fit almost unregularised, so it targets the same maximum-"
    "likelihood solution)."
)
add_code(
    "lw, lb = sklearn_logistic(cls.x, cls.y)\n"
    "print(f'our GD    : w = {np.round(lfit.w, 3)}, b = {lfit.b:+.3f}')\n"
    "print(f'sklearn   : w = {np.round(lw, 3)}, b = {lb:+.3f}')\n"
    "print('match within 5e-2:', bool(np.allclose(lfit.w, lw, atol=5e-2)\n"
    "                                 and np.isclose(lfit.b, lb, atol=5e-2)))"
)

# ---- Step 14: the boundary sharpening ----
add_md(
    "## Step 14 — The decision boundary sharpening\n"
    "\n"
    "For classification, the visual payoff isn't a fitted line through points — it's the **decision "
    "boundary**: the line where the model is 50/50 ($p = 0.5$, i.e. $w\\!\\cdot\\!x + b = 0$). As training "
    "lowers the loss, that boundary swings into the place that best separates benign from malignant. We plot "
    "the frozen boundaries from early to late training."
)
add_code(
    "benign = cls.y == 1\n"
    "xlim = (cls.x[:, 0].min() - 0.3, cls.x[:, 0].max() + 0.3)\n"
    "plt.figure(figsize=(6.6, 5))\n"
    "plt.scatter(cls.x[benign, 0], cls.x[benign, 1], s=16, color='#2E7A5A', alpha=0.6, label='benign')\n"
    "plt.scatter(cls.x[~benign, 0], cls.x[~benign, 1], s=16, color='#8B3B4A', alpha=0.6, label='malignant')\n"
    "shades = plt.cm.Purples(np.linspace(0.45, 0.95, len(lfit.snapshots)))\n"
    "gx = np.array(xlim)\n"
    "for shade, epoch in zip(shades, sorted(lfit.snapshots)):\n"
    "    ws, bs = lfit.snapshots[epoch]\n"
    "    if abs(ws[1]) < 1e-6:\n"
    "        continue                                   # epoch-0 boundary is undefined (w=0)\n"
    "    plt.plot(gx, -(ws[0] * gx + bs) / ws[1], color=shade, lw=2, label=f'epoch {epoch}')\n"
    "plt.xlim(*xlim)\n"
    "plt.ylim(cls.x[:, 1].min() - 0.3, cls.x[:, 1].max() + 0.3)\n"
    "plt.xlabel(cls.feature_names[0])\n"
    "plt.ylabel(cls.feature_names[1])\n"
    "plt.title('the decision boundary learns to separate the classes')\n"
    "plt.legend(fontsize=8)\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "In one runnable notebook, on real data, we saw the whole of how a model learns:\n"
    "\n"
    "| Step | What we did | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–2 | a flat know-nothing line; its MSE | a **loss** scores how wrong the model is |\n"
    "| 3–4 | the gradient; one step by hand | the **negative gradient** points downhill; one step lowers loss |\n"
    "| 5–6 | the full loop; the line fitting | learning = repeating the step; loss falls, the line rotates in |\n"
    "| 7 | match `sklearn` least squares | our from-scratch GD reaches the **provably-optimal** fit |\n"
    "| 8 | the loss surface + path | gradient descent = **a ball rolling downhill** |\n"
    "| 9 | crawl / converge / diverge | the **learning rate** is the step size you must get right |\n"
    "| 10–14 | logistic regression, same loop | a different model & loss, the **identical loop** — matched to `sklearn` |\n"
    "\n"
    "One loop — **predict → measure the loss → follow the gradient downhill → repeat** — trains everything "
    "from this two-parameter line to a trillion-parameter language model. Only the model and the loss "
    "change; the loop is always the same. That is how models learn."
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

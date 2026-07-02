"""Generate the step-by-step teaching notebook (23-Cross-Entropy-and-KL-Divergence.ipynb).

The notebook mirrors ``cross_entropy_kl.py`` one operation at a time, so a reader can open it, run
every cell live, and *teach* cross-entropy and KL from it. Each numbered step has a short markdown
lead-in (the intuition) followed by ONE focused code cell with real output. This generator writes
the .ipynb; a separate nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_23.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../23-Cross-Entropy-and-KL-Divergence/code/23-Cross-Entropy-and-KL-Divergence.ipynb"

This generator lives in the domain-level ``01. Foundations/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a
hand-edited .ipynb) so the notebook and the module stay in lockstep: the same algorithm, typed once
in the module, demonstrated step-by-step here.
"""

from __future__ import annotations

import json
from pathlib import Path

# Written into the chapter's own code/ folder, one directory up from tools/ then into the chapter.
NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "23-Cross-Entropy-and-KL-Divergence"
    / "code"
    / "23-Cross-Entropy-and-KL-Divergence.ipynb"
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
    "# Cross-Entropy & KL Divergence — a step-by-step, runnable teaching notebook\n"
    "\n"
    "This notebook builds cross-entropy and KL divergence from the ground up on **real data**, one "
    "operation at a time. It is the executable companion to the chapter and to `cross_entropy_kl.py` — "
    "every function used here lives in that module, imported so the notebook and the module can never "
    "drift apart.\n"
    "\n"
    "By the end you will have **measured**, on real corpora and real datasets:\n"
    "\n"
    "1. **Entropy as expected surprise** — the letter distribution of a real corpus is ~4.19 bits/letter;\n"
    "2. the **binary-entropy curve**, peaking at exactly 1 bit for a fair coin;\n"
    "3. **cross-entropy** $H(p,q)=-\\sum p\\log q$ and its decomposition $H(p,q)=H(p)+D_{KL}(p\\|q)$;\n"
    "4. **KL $\\ge 0$** (Gibbs) and its **asymmetry** $D_{KL}(p\\|q)\\ne D_{KL}(q\\|p)$, both measured;\n"
    "5. **cross-entropy IS the classification loss** — a real softmax classifier trained by GD, its loss "
    "falling from $\\ln 10$, matched to `sklearn.log_loss`, with the **$(p-y)$ gradient** verified;\n"
    "6. **forward vs reverse KL** — fitting one Gaussian to a real bimodal distribution (cover vs seek);\n"
    "7. **perplexity** $= 2^{H}$ of real n-gram language models on real held-out text;\n"
    "8. the **Gaussian KL** closed form, cross-checked against a numeric integral.\n"
    "\n"
    "Everything runs on CPU in a few seconds, seeded for reproducibility. The first run downloads the "
    "20-newsgroups corpus once into scikit-learn's cache; after that it is fully offline."
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
    "import scipy\n"
    "import sklearn\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from cross_entropy_kl import (\n"
    "    shannon_entropy_bits, binary_entropy_bits, letter_distribution,\n"
    "    cross_entropy_bits, kl_divergence_bits, compare_distributions,\n"
    "    softmax, train_softmax_gd, load_digits_split, standardize,\n"
    "    real_bimodal_scores, mode_centers, fit_gaussian_forward_kl, fit_gaussian_reverse_kl,\n"
    "    build_lm_corpus, unigram_perplexity, bigram_perplexity,\n"
    "    kl_two_gaussians, kl_two_gaussians_numeric, N_CLASSES, RNG_SEED,\n"
    ")\n"
    "from sklearn.datasets import fetch_20newsgroups\n"
    "from sklearn.metrics import log_loss\n"
    "\n"
    "print(f'numpy {np.__version__} | scipy {scipy.__version__} | scikit-learn {sklearn.__version__} '\n"
    "      f'| matplotlib {matplotlib.__version__}')\n"
    "rng = np.random.default_rng(RNG_SEED)\n"
    "REMOVE = ('headers', 'footers', 'quotes')"
)

# ---- Step 1: entropy of a real corpus ----
add_md(
    "## Step 1 — Entropy is *expected surprise*, measured in bits\n"
    "\n"
    "Before cross-entropy or KL, we need entropy. The **surprise** of seeing a symbol of probability "
    "$p$ is $-\\log_2 p$ bits: a certain event ($p=1$) surprises you 0 bits; a one-in-a-million event "
    "surprises you ~20 bits. **Entropy** is the *average* surprise, $H(p)=-\\sum_i p_i\\log_2 p_i$ — and, "
    "by Shannon's source-coding theorem, the minimum average bits/symbol any lossless code can achieve.\n"
    "\n"
    "Let's measure it on a **real corpus**: the letter frequencies of the `sci.space` newsgroup. The "
    "answer (~4.19 bits/letter) is the classic figure for English text — well below the 4.70 bits a "
    "uniform 26-letter alphabet would need, because real letters are far from uniform."
)
add_code(
    "space = fetch_20newsgroups(subset='train', categories=['sci.space'], remove=REMOVE)\n"
    "letters, p_letters = letter_distribution(' '.join(space.data))\n"
    "\n"
    "H = shannon_entropy_bits(p_letters)\n"
    "print(f'corpus: {len(space.data)} documents, {len(letters)} distinct letters')\n"
    "print(f'H(letters) = {H:.4f} bits/letter   (English is famously ~4.1-4.2)')\n"
    "print(f'uniform max = log2(26) = {np.log2(26):.4f} bits')\n"
    "print(f'so real text saves {np.log2(26) - H:.3f} bits/letter over a uniform code')\n"
    "\n"
    "# the three most and least surprising letters\n"
    "surprise = -np.log2(p_letters)\n"
    "order = np.argsort(p_letters)[::-1]\n"
    "print('\\nmost frequent (least surprising):', [letters[i] for i in order[:5]])\n"
    "print('least frequent (most surprising):', [letters[i] for i in order[-5:]])"
)

# ---- Step 2: binary entropy curve ----
add_md(
    "## Step 2 — The binary-entropy curve: uncertainty of a coin\n"
    "\n"
    "The simplest entropy is a single yes/no outcome with probability $p$: "
    "$H(p) = -p\\log_2 p - (1-p)\\log_2(1-p)$. It is **0** at $p=0$ or $p=1$ (a certain coin — no "
    "surprise) and **maximal, exactly 1 bit, at $p=0.5$** (a fair coin — you truly can't predict it). "
    "This curve is the shape of every binary cross-entropy loss you'll ever train."
)
add_code(
    "grid = np.linspace(1e-6, 1 - 1e-6, 500)\n"
    "Hb = binary_entropy_bits(grid)\n"
    "print('H(0.5) =', round(binary_entropy_bits(np.array([0.5]))[0], 4), 'bit (max)')\n"
    "print('H(0.1) =', round(binary_entropy_bits(np.array([0.1]))[0], 4), 'bit')\n"
    "print('H(0.01)=', round(binary_entropy_bits(np.array([0.01]))[0], 4), 'bit')\n"
    "\n"
    "plt.figure(figsize=(6, 3.6))\n"
    "plt.plot(grid, Hb, lw=2)\n"
    "plt.axvline(0.5, ls='--', color='gray')\n"
    "plt.xlabel('p')\n"
    "plt.ylabel('H(p) bits')\n"
    "plt.title('binary entropy peaks at 1 bit (p=0.5)')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 3: cross-entropy ----
add_md(
    "## Step 3 — Cross-entropy: the cost of coding $p$ with the *wrong* model $q$\n"
    "\n"
    "Now bring in a second distribution. If the truth is $p$ but you build your code assuming $q$, each "
    "symbol costs $-\\log_2 q_i$ bits and you pay $p_i$ of them on average:\n"
    "\n"
    "$$H(p,q) = -\\sum_i p_i \\log_2 q_i.$$\n"
    "\n"
    "Two real distributions to compare: the letter frequencies of `sci.space` ($p$) and of "
    "`rec.sport.baseball` ($q$). Cross-entropy is minimised, over all $q$, exactly when $q=p$ — at which "
    "point it equals the entropy $H(p)$. So $H(p,q)\\ge H(p)$ always, with the *gap* being the KL "
    "divergence of the next step."
)
add_code(
    "baseball = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball'], remove=REMOVE)\n"
    "_, q_letters = letter_distribution(' '.join(baseball.data))\n"
    "\n"
    "H_p  = shannon_entropy_bits(p_letters)\n"
    "H_pq = cross_entropy_bits(p_letters, q_letters)\n"
    "H_pp = cross_entropy_bits(p_letters, p_letters)   # coding p with p == entropy of p\n"
    "print(f'H(p)        = {H_p:.4f} bits   (entropy of sci.space letters)')\n"
    "print(f'H(p, p)     = {H_pp:.4f} bits   (== H(p): the best possible)')\n"
    "print(f'H(p, q)     = {H_pq:.4f} bits   (coding sci.space letters with the baseball model)')\n"
    "print(f'gap H(p,q) - H(p) = {H_pq - H_p:.4f} bits  <- this is exactly D_KL(p||q)')"
)

# ---- Step 4: KL divergence + the identity ----
add_md(
    "## Step 4 — KL divergence: the *extra* bits, and $H(p,q)=H(p)+D_{KL}(p\\|q)$\n"
    "\n"
    "The gap you just saw has a name: the **Kullback–Leibler divergence**\n"
    "\n"
    "$$D_{KL}(p\\|q) = \\sum_i p_i \\log_2 \\frac{p_i}{q_i} = H(p,q) - H(p).$$\n"
    "\n"
    "It is the number of **wasted** bits per symbol from using model $q$ when the truth is $p$. We verify "
    "the decomposition $H(p,q) = H(p) + D_{KL}(p\\|q)$ numerically — it holds to machine precision because "
    "it is an identity, not an approximation."
)
add_code(
    "cmp = compare_distributions(p_letters, q_letters)\n"
    "print(f'H(p)              = {cmp.h_p:.4f} bits')\n"
    "print(f'D_KL(p||q)        = {cmp.kl_pq:.4f} bits   (the extra/wasted bits)')\n"
    "print(f'H(p) + D_KL(p||q) = {cmp.h_p + cmp.kl_pq:.4f} bits')\n"
    "print(f'H(p, q)           = {cmp.h_pq:.4f} bits')\n"
    "print('identity H(p,q) == H(p)+D_KL(p||q):', cmp.identity_holds)"
)

# ---- Step 5: KL >= 0 and asymmetry ----
add_md(
    "## Step 5 — KL is $\\ge 0$ (Gibbs) and **asymmetric** (a divergence, not a distance)\n"
    "\n"
    "Two properties that trip people up, both *measured* here:\n"
    "\n"
    "* **Non-negativity** ($D_{KL}\\ge 0$, zero iff $p=q$) — *Gibbs' inequality*, a consequence of the "
    "concavity of $\\log$ (Jensen). You can never do better than coding $p$ with $p$ itself.\n"
    "* **Asymmetry** — $D_{KL}(p\\|q)\\ne D_{KL}(q\\|p)$ in general. This is *the* reason KL is a "
    "*divergence*, not a *distance*: it has no symmetry and no triangle inequality.\n"
    "\n"
    "The two real corpora are so similar that their asymmetry is tiny, so we also show a clean, large "
    "asymmetry: real letters vs a **uniform** model. There, $D_{KL}(\\text{real}\\|\\text{uniform}) = "
    "\\log_2 26 - H(p)$ exactly — a lovely closed form."
)
add_code(
    "print(f'D_KL(p||q) = {cmp.kl_pq:.5f} bits')\n"
    "print(f'D_KL(q||p) = {cmp.kl_qp:.5f} bits   (different -> asymmetric)')\n"
    "print(f'D_KL(p||p) = {kl_divergence_bits(p_letters, p_letters):.5f} bits   (zero: q==p)')\n"
    "print('both KL >= 0:', cmp.kl_pq >= 0 and cmp.kl_qp >= 0, '(Gibbs)')\n"
    "\n"
    "uniform = np.full_like(p_letters, 1.0 / p_letters.size)\n"
    "kl_pu = kl_divergence_bits(p_letters, uniform)\n"
    "kl_up = kl_divergence_bits(uniform, p_letters)\n"
    "print(f'\\nD_KL(real||uniform) = {kl_pu:.4f} bits')\n"
    "print(f'D_KL(uniform||real) = {kl_up:.4f} bits   ({kl_up/kl_pu:.2f}x larger -> asymmetric)')\n"
    "print(f'check: log2(26) - H(p) = {np.log2(26) - cmp.h_p:.4f}  == D_KL(real||uniform)')"
)

# ---- Step 6: load digits, softmax classifier setup ----
add_md(
    "## Step 6 — Cross-entropy IS the classification loss (real digits)\n"
    "\n"
    "Here is why this topic dominates ML. When you train a classifier, the loss you minimise **is** the "
    "cross-entropy between the one-hot true labels and the predicted probabilities. For a single example "
    "with true class $c$, that cross-entropy collapses to the **negative log-likelihood** of the truth:\n"
    "\n"
    "$$H(\\text{one-hot}, \\hat p) = -\\sum_k \\mathbb{1}[k=c]\\log \\hat p_k = -\\log \\hat p_c.$$\n"
    "\n"
    "NLL == cross-entropy == $-\\log p(\\text{true class})$. We load the real 8×8 `load_digits` dataset "
    "and standardise it, ready to train a softmax classifier and watch its cross-entropy fall."
)
add_code(
    "x_tr, x_te, y_tr, y_te = load_digits_split()\n"
    "x_tr_s, x_te_s = standardize(x_tr, x_te)\n"
    "print('train:', x_tr_s.shape, '| test:', x_te_s.shape, '| classes:', N_CLASSES)\n"
    "print('a uniform guess has loss ln(10) =', round(np.log(10), 4), 'nats (the starting floor)')"
)

# ---- Step 7: train, watch loss fall, match sklearn ----
add_md(
    "## Step 7 — Train it by gradient descent and watch cross-entropy fall\n"
    "\n"
    "We train a softmax (multinomial logistic) classifier by full-batch gradient descent, recording the "
    "cross-entropy at every step. It starts at $\\ln 10 \\approx 2.30$ (a uniform guess over 10 classes) "
    "and falls as the model learns. At the end we confirm our hand-computed cross-entropy equals "
    "`sklearn.metrics.log_loss` to 6 decimals — it is literally the same quantity."
)
add_code(
    "trained = train_softmax_gd(x_tr_s, y_tr)\n"
    "curve = trained.loss_curve\n"
    "print(f'CE at step 0   = {curve[0]:.4f}  (= ln 10 = {np.log(10):.4f})')\n"
    "print(f'CE at step {curve.size} = {curve[-1]:.4f}  (converged)')\n"
    "\n"
    "n = x_tr_s.shape[0]\n"
    "our_ce = -np.mean(np.log(trained.probs_train[np.arange(n), y_tr]))\n"
    "sk_ce  = log_loss(y_tr, trained.probs_train)\n"
    "print(f'\\nour cross-entropy        = {our_ce:.6f}')\n"
    "print(f'sklearn.metrics.log_loss = {sk_ce:.6f}')\n"
    "print('they match:', bool(np.isclose(our_ce, sk_ce, atol=1e-6)))\n"
    "\n"
    "plt.figure(figsize=(7, 3.6))\n"
    "plt.plot(range(1, curve.size + 1), curve, lw=2)\n"
    "plt.axhline(np.log(10), ls='--', color='gray', label='ln(10): uniform guess')\n"
    "plt.xlabel('GD step')\n"
    "plt.ylabel('cross-entropy (nats)')\n"
    "plt.title('training a classifier = minimising cross-entropy')\n"
    "plt.legend()\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: NLL of one example, and the (p - y) gradient ----
add_md(
    "## Step 8 — NLL of a single example, and the famous $(p-y)$ gradient\n"
    "\n"
    "Two things worth seeing directly. First, the loss for one example really is just $-\\log$ of the "
    "probability it assigned the true class. Second — the reason softmax + cross-entropy is *the* pairing "
    "— the gradient of the loss with respect to the **logits** simplifies to the beautifully clean\n"
    "\n"
    "$$\\frac{\\partial \\mathcal{L}}{\\partial z} = \\hat p - y \\quad(\\text{predicted} - \\text{one-hot true}).$$\n"
    "\n"
    "No division, no softmax derivative left over — the messy Jacobian cancels the $1/\\hat p$ from the "
    "log. We verify it against a finite-difference gradient on a real example."
)
add_code(
    "i = 0\n"
    "true_c = y_tr[i]\n"
    "p_i = trained.probs_train[i]\n"
    "print(f'example {i}: true class = {true_c}, p(true) = {p_i[true_c]:.4f}')\n"
    "print(f'NLL = -log p(true) = {-np.log(p_i[true_c]):.4f}   (== its cross-entropy)')\n"
    "\n"
    "# verify d L_i / d logits == (p - y) by finite differences, on one example's logits\n"
    "W, b = trained.weights, trained.bias\n"
    "z = x_tr_s[i] @ W + b\n"
    "y_onehot = np.eye(N_CLASSES)[true_c]\n"
    "analytic = softmax(z[None])[0] - y_onehot          # the (p - y) claim\n"
    "eps = 1e-6\n"
    "numeric = np.zeros(N_CLASSES)\n"
    "for k in range(N_CLASSES):\n"
    "    zp = z.copy()\n"
    "    zp[k] += eps\n"
    "    zm = z.copy()\n"
    "    zm[k] -= eps\n"
    "    Lp = -np.log(softmax(zp[None])[0][true_c])\n"
    "    Lm = -np.log(softmax(zm[None])[0][true_c])\n"
    "    numeric[k] = (Lp - Lm) / (2 * eps)\n"
    "print('\\nanalytic (p - y):', np.round(analytic, 4))\n"
    "print('finite-difference:', np.round(numeric, 4))\n"
    "print('match:', bool(np.allclose(analytic, numeric, atol=1e-5)))"
)

# ---- Step 9: sklearn LogisticRegression as the cross-check ----
add_md(
    "## Step 9 — The same loss under `sklearn.LogisticRegression`\n"
    "\n"
    "`LogisticRegression` with the multinomial option minimises exactly this cross-entropy (it calls it "
    "'log loss'). We fit it on the same real data and report its held-out log-loss and accuracy — the "
    "cross-entropy on data the model never saw, the number you actually care about."
)
add_code(
    "from sklearn.linear_model import LogisticRegression\n"
    "clf = LogisticRegression(max_iter=2000, C=10.0).fit(x_tr_s, y_tr)\n"
    "test_ce = log_loss(y_te, clf.predict_proba(x_te_s))\n"
    "print(f'held-out cross-entropy (log-loss): {test_ce:.4f} nats')\n"
    "print(f'held-out accuracy                : {clf.score(x_te_s, y_te):.4f}')\n"
    "print('lower cross-entropy => better-calibrated probabilities, not just correct labels')"
)

# ---- Step 10: forward vs reverse KL, real bimodal ----
add_md(
    "## Step 10 — Forward vs reverse KL: fitting one Gaussian to a *real bimodal* distribution\n"
    "\n"
    "KL's asymmetry has teeth. Suppose your data $p$ has **two modes** but your model $q$ is a single "
    "Gaussian — it *cannot* match $p$. Which way you write the KL decides how it fails:\n"
    "\n"
    "* **Forward** $\\arg\\min_q D_{KL}(p\\|q)$ (this is MLE) is **mode-covering**: it is penalised wherever "
    "$p>0$ but $q\\approx 0$, so it stretches to cover *both* modes (a wide Gaussian = moment matching).\n"
    "* **Reverse** $\\arg\\min_q D_{KL}(q\\|p)$ (this is variational inference / the ELBO) is "
    "**mode-seeking**: it is penalised wherever $q>0$ but $p\\approx 0$, so it collapses onto *one* mode "
    "(a narrow spike).\n"
    "\n"
    "The real bimodal data: PC1 scores of two visually distinct digit classes (0 and 6), which separate "
    "into two bumps."
)
add_code(
    "scores = real_bimodal_scores()\n"
    "grid = np.linspace(scores.min() - 1.5, scores.max() + 1.5, 500)\n"
    "centers = 0.5 * (grid[:-1] + grid[1:])\n"
    "m0, m1 = mode_centers()\n"
    "print(f'real modes at PC1 = {m0:.3f} and {m1:.3f}  (two bumps)')\n"
    "\n"
    "fwd = fit_gaussian_forward_kl(scores, grid, centers)\n"
    "rev = fit_gaussian_reverse_kl(scores, grid, centers, mode_inits=(m0, m1))\n"
    "print(f'forward KL fit: mu={fwd.mu:+.3f}, sigma={fwd.sigma:.3f}  -> mode-COVERING (wide, spans both)')\n"
    "print(f'reverse KL fit: mu={rev.mu:+.3f}, sigma={rev.sigma:.3f}  -> mode-SEEKING (narrow, one mode)')\n"
    "print('reverse fit is narrower:', rev.sigma < fwd.sigma)\n"
    "\n"
    "xs = np.linspace(scores.min() - 1.5, scores.max() + 1.5, 400)\n"
    "\n"
    "def gauss(mu, s):\n"
    "    return np.exp(-0.5 * ((xs - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))\n"
    "\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)\n"
    "for ax, fit, ttl in ((a1, fwd, 'forward KL (cover)'), (a2, rev, 'reverse KL (seek)')):\n"
    "    ax.hist(scores, bins=40, density=True, alpha=0.4)\n"
    "    ax.plot(xs, gauss(fit.mu, fit.sigma), lw=2)\n"
    "    ax.set_title(ttl)\n"
    "    ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 11: connection to MLE ----
add_md(
    "## Step 11 — Why forward KL = maximum likelihood\n"
    "\n"
    "The forward-KL fit above was just the data's mean and variance — *moment matching*. That is no "
    "accident. Minimising $D_{KL}(\\hat p_{\\text{data}}\\|q_\\theta)$ over the model $q_\\theta$ is "
    "identical to **maximising the log-likelihood** of the data, because\n"
    "\n"
    "$$D_{KL}(\\hat p\\|q_\\theta) = \\underbrace{\\sum_x \\hat p(x)\\log \\hat p(x)}_{-H(\\hat p),\\ \\text{const in }\\theta} - \\underbrace{\\sum_x \\hat p(x)\\log q_\\theta(x)}_{\\frac{1}{N}\\sum_i \\log q_\\theta(x_i)}.$$\n"
    "\n"
    "The first term doesn't depend on $\\theta$, so minimising KL = maximising the average log-likelihood "
    "= minimising cross-entropy. We show it numerically: the forward-KL Gaussian's parameters equal the "
    "closed-form MLE (sample mean and std)."
)
add_code(
    "mle_mu, mle_sigma = scores.mean(), scores.std()\n"
    "print(f'MLE (sample mean, std)   : mu={mle_mu:+.4f}, sigma={mle_sigma:.4f}')\n"
    "print(f'forward-KL fit           : mu={fwd.mu:+.4f}, sigma={fwd.sigma:.4f}')\n"
    "print('identical:', bool(np.isclose(mle_mu, fwd.mu) and np.isclose(mle_sigma, fwd.sigma)))\n"
    "print('\\n=> minimising forward KL to the data == maximising likelihood == minimising cross-entropy')"
)

# ---- Step 12: perplexity ----
add_md(
    "## Step 12 — Perplexity: cross-entropy for language models\n"
    "\n"
    "Language models report **perplexity**, which is just cross-entropy in disguise:\n"
    "\n"
    "$$\\text{perplexity} = 2^{H} = 2^{-\\frac{1}{N}\\sum_i \\log_2 q(w_i\\mid\\text{context})}.$$\n"
    "\n"
    "It is the model's *effective branching factor* — how many equally-likely words it is choosing among "
    "at each step. Lower is better. We build two real n-gram models on real held-out newsgroup text — a "
    "context-free **unigram** and an interpolated **bigram** — and watch context lower the perplexity."
)
add_code(
    "train, test, vocab_size = build_lm_corpus()\n"
    "uni = unigram_perplexity(train, test, vocab_size)\n"
    "bi  = bigram_perplexity(train, test, vocab_size)\n"
    "print(f'corpus: {len(train):,} train tokens, {len(test):,} held-out, vocab {vocab_size:,}')\n"
    "print(f'{uni.name:<22}: {uni.bits_per_word:.4f} bits/word -> perplexity {uni.perplexity:.1f}')\n"
    "print(f'{bi.name:<22}: {bi.bits_per_word:.4f} bits/word -> perplexity {bi.perplexity:.1f}')\n"
    "print(f'the bigram is {uni.perplexity/bi.perplexity:.2f}x less perplexed (context helps)')"
)

# ---- Step 13: Gaussian KL closed form ----
add_md(
    "## Step 13 — The Gaussian KL closed form (the VAE regulariser)\n"
    "\n"
    "For two Gaussians the KL has a clean closed form — the only reason variational autoencoders can put "
    "a KL term in their loss cheaply:\n"
    "\n"
    "$$D_{KL}\\big(\\mathcal N(\\mu_0,\\sigma_0)\\,\\|\\,\\mathcal N(\\mu_1,\\sigma_1)\\big) = "
    "\\log\\frac{\\sigma_1}{\\sigma_0} + \\frac{\\sigma_0^2 + (\\mu_0-\\mu_1)^2}{2\\sigma_1^2} - \\frac12.$$\n"
    "\n"
    "We check the formula against a brute-force numeric integral — they agree to 3+ decimals, confirming "
    "the algebra."
)
add_code(
    "closed  = kl_two_gaussians(0.0, 1.0, 1.5, 2.0)\n"
    "numeric = kl_two_gaussians_numeric(0.0, 1.0, 1.5, 2.0)\n"
    "print(f'D_KL(N(0,1) || N(1.5,2))  closed form = {closed:.5f} nats')\n"
    "print(f'                          numeric int = {numeric:.5f} nats')\n"
    "print('match:', bool(np.isclose(closed, numeric, atol=1e-3)))\n"
    "\n"
    "# KL is 0 iff the Gaussians coincide, and grows as they drift apart\n"
    "print('\\nD_KL(N(0,1) || N(0,1))    =', round(kl_two_gaussians(0, 1, 0, 1), 6), '(identical -> 0)')\n"
    "print('D_KL(N(0,1) || N(3,1))    =', round(kl_two_gaussians(0, 1, 3, 1), 4), '(far apart -> large)')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "In one runnable notebook, on real data, we established:\n"
    "\n"
    "| Step | What we saw | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–2 | real letter entropy ~4.19 bits; binary-entropy curve | entropy = expected surprise, in bits |\n"
    "| 3–4 | $H(p,q)$ and $H(p,q)=H(p)+D_{KL}(p\\|q)$ | cross-entropy = floor + KL 'waste' |\n"
    "| 5 | $D_{KL}\\ge 0$ and asymmetric | KL is a divergence, not a distance |\n"
    "| 6–9 | softmax on digits, loss $\\downarrow$, $(p-y)$, sklearn match | **cross-entropy IS the loss** |\n"
    "| 10–11 | one Gaussian, two KLs; forward = MLE | cover vs seek; forward KL = max-likelihood |\n"
    "| 12 | unigram vs bigram perplexity | perplexity $=2^{\\text{cross-entropy}}$ |\n"
    "| 13 | Gaussian KL closed form == numeric | the VAE regulariser, verified |\n"
    "\n"
    "Every number here came from a real corpus or a real dataset and a real library call. That is the "
    "point: cross-entropy and KL are not abstractions to memorise — they are concrete, measurable "
    "quantities you can compute, see, and trust."
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

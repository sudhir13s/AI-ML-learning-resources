---
id: "03-supervised-learning/logistic-regression"
topic: "Logistic Regression"
parent: "03-supervised-learning"
level: beginner
prereqs: ["linear-regression", "sigmoid", "maximum-likelihood", "gradient-descent"]
interview_frequency: very-high
template: concept-deep
updated: 2026-06-22
---

# Logistic regression: from a linear score to a calibrated probability

Logistic regression is the most-used classification algorithm in the world, the first thing every ML course teaches after linear regression, and — not coincidentally — exactly what sits at the output of every classification neural network. The idea is elegant: take a plain linear score $w\cdot x + b$ (the same thing linear regression computes), and instead of using it directly, **squash it through the sigmoid** into a number between 0 and 1 that you can read as a **probability**. Then fit the weights by **maximum likelihood** — making the observed labels as probable as possible, which turns out to be identical to minimizing **cross-entropy** (log-loss). Despite the confusing name, logistic regression is a *classifier*, its decision boundary is a straight line, and it's a single neuron with a sigmoid activation. Master it and you've understood the bridge from linear models to deep learning.

By the end of this page you'll be able to:

- explain why you **can't** just use linear regression for classification, and what the sigmoid fixes;
- interpret the model's coefficients as **log-odds** (the classic interview follow-up);
- **derive** the cross-entropy loss from maximum likelihood and show its gradient is $(\hat y - y)\,x$;
- explain why log-loss (not MSE) — convexity *and* the right gradient;
- reason about the **linear decision boundary**, **regularization**, and the **softmax/multiclass** extension;
- implement it from scratch and verify the gradient against numerics and scikit-learn.

Intuition and pictures first, then the math (with sources), then runnable code.

> **Note:** "regression" in the name is historical — it predicts a continuous *probability*, then you threshold that probability to get a *class*. It's a classifier. (And softmax regression, its multiclass cousin, is literally the final layer of an image or text classifier.)

---

## The problem: why not linear regression?

You could try fitting $y \in \{0, 1\}$ with ordinary linear regression, but it breaks in three ways:

- **Outputs aren't probabilities** — $w\cdot x + b$ ranges over all reals; you'd get predictions like $-0.4$ or $1.7$, which can't be probabilities.
- **MSE is the wrong loss** — squared error on a 0/1 target is sensitive to outliers and, once you add a sigmoid, becomes **non-convex** with vanishing gradients on confident-wrong predictions (see [Loss Functions](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md)).
- **No notion of confidence** — you want $P(y=1 \mid x)$, a calibrated probability you can threshold and reason about, not an unbounded score.

Logistic regression fixes all three by passing the linear score through the sigmoid and fitting with the right (probabilistic) loss.

---

## The model: a linear score through the sigmoid

Compute a linear score $z = w\cdot x + b$, then map it to a probability with the **sigmoid**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad p = P(y=1 \mid x)$$

![The sigmoid curve mapping the linear score z = w·x + b on the x-axis to a probability p = σ(z) on the y-axis. It is S-shaped, passing through 0.5 at z = 0. Where z < 0 the probability is below 0.5 (predict class 0); where z > 0 it is above 0.5 (predict class 1); the decision threshold sits at p = 0.5, z = 0.](images/logreg_sigmoid.png)

The sigmoid squashes any real score into $(0, 1)$: a large positive score → near 1, a large negative score → near 0, and $z = 0$ → exactly $0.5$. You then classify by thresholding (usually at $0.5$, but the threshold is yours to tune for the precision/recall tradeoff — see [Classification Metrics](14-Classification-Metrics.md)).

---

## Odds and log-odds: what the coefficients mean

This is the interview question that separates people who memorized from people who understand. Invert the sigmoid: the linear score equals the **log-odds** (logit) of the probability:

$$z = w\cdot x + b = \log\frac{p}{1-p} = \text{logit}(p)$$

So **logistic regression is linear in the log-odds.** That gives each coefficient a precise meaning: increasing feature $x_j$ by one unit adds $w_j$ to the log-odds, i.e. **multiplies the odds by $e^{w_j}$**. (In the code, a weight of $4.0$ means a one-unit increase multiplies the odds of class 1 by $e^{4.0} \approx 55\times$.) That interpretability — coefficients as odds ratios — is a big reason logistic regression remains a default in medicine, finance, and the social sciences.

> *Where this comes from: logistic regression and the log-odds link are **The Regression Analysis of Binary Sequences** (Cox 1958); the clean modern treatment is **Speech and Language Processing** (Jurafsky & Martin) Ch. 5 — in the references.*

---

## The loss: maximum likelihood = cross-entropy

How do we fit $w, b$? **Maximum likelihood**: choose parameters that make the observed labels most probable. For one example, the model says the label has probability $p$ if $y=1$ and $1-p$ if $y=0$ — compactly $p^{y}(1-p)^{1-y}$. Taking the negative log of the likelihood over the dataset gives the **cross-entropy / log-loss**:

$$\mathcal{L} = -\frac{1}{n}\sum_i \big[\,y_i \log p_i + (1-y_i)\log(1-p_i)\,\big]$$

Maximizing likelihood = minimizing this. And it has a property MSE lacks here: composed with the sigmoid, **log-loss is convex**, so gradient descent (or Newton's method / IRLS) reliably finds the *global* optimum — no bad local minima.

> *Where this comes from: the MLE → cross-entropy derivation is **Speech and Language Processing** Ch. 5 and **CS229** notes §1.2; the convexity and IRLS view is **The Elements of Statistical Learning** Ch. 4 — references.*

---

## The gradient: strikingly like linear regression

Differentiate the log-loss with respect to the weights and — after the sigmoid's derivative cancels the log's, exactly as in [softmax + cross-entropy](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) — you get a beautifully simple result:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n}\sum_i (p_i - y_i)\,x_i = \frac{1}{n}X^\top(\hat y - y)$$

The gradient is the **prediction error $(\hat y - y)$ times the input** — *identical in form* to linear regression's gradient, just with $\hat y = \sigma(z)$ instead of $\hat y = z$. (The code confirms this against a numerical gradient to $10^{-12}$.) That clean error-times-input form is exactly why this generalizes straight to a neuron and to backprop.

> **Gotcha:** there's no closed-form solution for $w$ (unlike linear regression's normal equations) — the sigmoid makes it nonlinear. You *must* solve it iteratively (gradient descent / Newton-IRLS). The good news: convexity guarantees you'll converge to the unique optimum.

---

## The decision boundary is linear

Classify positive when $p \ge 0.5$, i.e. when $z = w\cdot x + b \ge 0$. The boundary $w\cdot x + b = 0$ is a **hyperplane** (a line in 2D, a plane in 3D) — so logistic regression is a **linear classifier**:

![Two clusters of 2D points (class 0 and class 1) with a logistic-regression model fit from scratch. The decision boundary where p = 0.5 is a straight line separating them, and the background is shaded by the sigmoid probability — deep blue (P→0) on the class-0 side, deep red (P→1) on the class-1 side, with a smooth gradient across the boundary showing the model's confidence.](images/logreg_boundary.png)

The line is the **boundary**; the sigmoid turns distance-from-the-line into a **confidence** (points far on the red side are near-certain class 1; points near the line are ~50/50). For non-linear boundaries you add polynomial/interaction features, use a kernel, or move to a non-linear model — logistic regression itself only draws straight lines.

---

## Regularization and the multiclass extension

- **Regularization.** Add an L2 penalty (ridge) to shrink weights and prevent overfitting, or L1 (lasso) for sparse feature selection — same idea as [Regularization](../../05.%20Deep_Learning/concepts/09-Regularization.md). scikit-learn's `LogisticRegression` regularizes by default (the `C` parameter is inverse strength).
- **Multiclass = softmax regression.** For $K$ classes, replace the sigmoid with the **softmax** over $K$ linear scores; this is **multinomial logistic regression**, and it's exactly the [softmax + cross-entropy](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) output layer of a classification neural network. Binary logistic regression is the $K=2$ special case.

> **Tip:** the connection to deep learning is the punchline. A logistic regression unit *is* a single neuron with a sigmoid; softmax regression *is* a one-layer net's output. Everything you know about its loss and gradient is exactly what backprop computes for that final layer.

---

## Worked example

A model has learned $w = [2.0]$ and $b = -1.0$ for a single feature $x$. For $x = 2$:

- **Linear score:** $z = 2.0 \cdot 2 - 1.0 = 3.0$.
- **Probability:** $p = \sigma(3.0) = 1/(1 + e^{-3}) = 0.953$ → predict class 1 (since $> 0.5$), with 95.3% confidence.
- **Log-odds:** $\log\frac{0.953}{0.047} = 3.0 = z$ ✓ — the score *is* the log-odds.
- **If the true label is $y = 1$:** loss $= -\log(0.953) = 0.048$ (small — confident and correct); gradient contribution $(p - y)x = (0.953 - 1)\cdot 2 = -0.094$ — a small nudge to raise $w$ slightly.

---

## Code: logistic regression from scratch (gradient verified)

```python
"""Logistic regression from scratch: the (p - y)x gradient and log-odds coefficients.
Verified on ml-py312, CPU (numpy)."""
import numpy as np
rng = np.random.default_rng(1)
sig = lambda z: 1 / (1 + np.exp(-z))

X = np.vstack([rng.normal([-1.3, -0.6], 0.85, (120, 2)),
               rng.normal([1.3, 0.8], 0.85, (120, 2))])
y = np.concatenate([np.zeros(120), np.ones(120)])
Xb = np.c_[np.ones(len(X)), X]                          # bias column

def log_loss(w):
    p = np.clip(sig(Xb @ w), 1e-12, 1 - 1e-12)
    return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

# the claimed gradient (1/n) X^T (p - y) vs a numerical gradient
w = rng.normal(size=3)
analytic = Xb.T @ (sig(Xb @ w) - y) / len(y)
numeric = np.array([(log_loss(w + e) - log_loss(w - e)) / 2e-5 for e in 1e-5*np.eye(3)])
print(f"gradient (1/n)X^T(p-y) vs numerical: max diff = {np.abs(analytic-numeric).max():.2e}")

w = np.zeros(3)                                          # fit (convex -> reliable)
for _ in range(4000):
    w -= 0.1 * Xb.T @ (sig(Xb @ w) - y) / len(y)
print(f"train accuracy = {((sig(Xb@w)>0.5)==y).mean():.3f}")
print(f"feature-1 weight w1={w[1]:.2f} -> +1 unit multiplies odds by e^w1 = {np.exp(w[1]):.1f}")
```

Output:

```
gradient (1/n)X^T(p-y) vs numerical: max diff = 5.36e-12
train accuracy = 0.983
feature-1 weight w1=4.02 -> +1 unit multiplies odds by e^w1 = 55.5
```

> **Note:** the gradient $(1/n)X^\top(p - y)$ matches the numerical gradient to $10^{-12}$ — confirming the clean error-times-input form. The fitted model reaches 98.3% accuracy, and the log-odds reading is concrete: a one-unit increase in feature 1 multiplies the odds of class 1 by $e^{4.02} \approx 55$. (scikit-learn's `LogisticRegression` finds weights in the same direction.)

---

## Where logistic regression is used

- **The default binary classifier** — spam, churn, fraud, medical diagnosis, credit scoring: fast, calibrated, interpretable.
- **Interpretable / regulated settings** — medicine, finance, social science prize the odds-ratio coefficients.
- **A strong baseline** — always try logistic regression before reaching for a complex model; it's hard to beat on linearly-separable problems.
- **The output layer of neural nets** — sigmoid (binary) or softmax (multiclass) on top of learned features is logistic/softmax regression.

> **Tip:** logistic regression is the canonical "explain a model end to end" interview. Be ready to walk sigmoid → probability → log-odds coefficients → cross-entropy from MLE → the $(\hat y - y)x$ gradient → linear boundary → regularization → softmax/NN connection. That single thread covers a remarkable amount of ML.

---

## Recap and rapid-fire

**If you remember nothing else:** logistic regression squashes a linear score $w\cdot x + b$ through the **sigmoid** into a probability, and fits by **maximum likelihood** = minimizing **cross-entropy** (convex, so reliable). Coefficients are **log-odds** (a unit change multiplies the odds by $e^{w_j}$), the gradient is the clean $(\hat y - y)x$, and the decision boundary is **linear**. Softmax regression is the multiclass version — and the output layer of every classification net.

**Quick-fire — say these out loud:**

- *Why the sigmoid?* Maps an unbounded linear score to a $(0,1)$ probability.
- *Why log-loss, not MSE?* With a sigmoid, log-loss is convex with a clean gradient; MSE is non-convex and its gradient vanishes on confident-wrong cases.
- *What do the coefficients mean?* Log-odds — a one-unit feature increase multiplies the odds by $e^{w_j}$.
- *What's the gradient?* $(1/n)X^\top(\hat y - y)$ — prediction error times input (same form as linear regression).
- *Closed-form solution?* No — solve iteratively (gradient descent / Newton-IRLS); convexity guarantees the global optimum.
- *What shape is the decision boundary?* Linear (a hyperplane $w\cdot x + b = 0$).
- *How do you get non-linear boundaries?* Add polynomial/interaction features, use a kernel, or switch models.
- *Multiclass?* Softmax (multinomial logistic) regression.
- *Connection to deep learning?* It's one neuron with a sigmoid; softmax regression is a net's output layer.

---

## References and further reading

The curated link library for this topic — videos, courses, interactive/visual resources, articles, papers, books, and internal cross-links — lives in a companion file so it can be reused as a standalone reference list:

**→ [Logistic Regression — references and further reading](02-Logistic-Regression.references.md)**

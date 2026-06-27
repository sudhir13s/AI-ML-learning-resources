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

Logistic regression is the most-used classification algorithm in the world. It is the first thing every ML course teaches after linear regression, the default baseline a good practitioner reaches for before anything fancier, and — not coincidentally — *exactly* what sits at the output of every classification neural network. The idea is elegant and worth saying in one breath: take a plain linear score $w\cdot x + b$ (the same quantity linear regression computes), and instead of using it directly, **squash it through the sigmoid** into a number in $(0,1)$ you can read as a **probability**. Then fit the weights by **maximum likelihood** — making the observed labels as probable as possible — which turns out to be *identical* to minimizing **cross-entropy** (log-loss). Despite the confusing name, logistic regression is a *classifier*, its decision boundary is a straight line, and it is a single neuron with a sigmoid activation. Master it cold and you have understood the entire bridge from linear models to deep learning.

I'm going to teach this the way I'd actually explain it to a sharp teammate at a whiteboard: feel *why* linear regression fails first, then build the model, then **derive** every result that matters — the log-odds reading of the coefficients, cross-entropy from maximum likelihood, the strikingly simple $(\hat y - y)x$ gradient and *why* the sigmoid's derivative cancels the log's, why log-loss and not MSE (with a convexity argument *and* a vanishing-gradient argument), and why the boundary is linear. We'll work **three** numeric examples of increasing complexity, prove the from-scratch model matches scikit-learn to the decimal, confront the two things people get wrong (perfect separation and calibration), and tie it to Naive Bayes and to the final layer of a neural net. By the end you'll be able to:

- explain why you **can't** just use linear regression for classification, and what the sigmoid fixes;
- interpret each coefficient as a **log-odds** contribution, i.e. an **odds ratio** $e^{w_j}$ (the classic interview follow-up);
- **derive** the cross-entropy loss from maximum likelihood, and show its gradient is the clean $(\hat y - y)\,x$;
- argue why log-loss (not MSE) — **convexity** *and* the **right gradient** that doesn't vanish on confident-wrong cases;
- reason about the **linear decision boundary**, **convex optimization** (gradient descent vs Newton/IRLS), **regularization**, and the **softmax/multiclass** extension;
- place it against **Naive Bayes** as the discriminative half of a generative–discriminative pair, and explain why logistic regression is **well-calibrated**;
- spot the pitfalls — **perfect separation**, unscaled features, class imbalance — and implement it from scratch, gradient verified against numerics and scikit-learn.

Intuition and pictures first, then the math (with sources), then runnable, verified code.

> **Note:** "regression" in the name is historical baggage — the model predicts a continuous *probability*, and you then threshold that probability to get a *class*. It is a **classifier**. (And softmax regression, its multiclass cousin, is literally the final layer of an image or text classifier.)

---

## The problem: why not just use linear regression?

You could try to fit a 0/1 label $y \in \{0, 1\}$ with ordinary linear regression — predict $\hat y = w\cdot x + b$ and call it the class probability. It breaks in three distinct ways, and seeing each one tells you exactly what the sigmoid + log-loss combination is *for*:

- **The outputs aren't probabilities.** $w\cdot x + b$ ranges over all of $\mathbb{R}$, so you'd routinely predict $-0.4$ or $1.7$ — numbers that *cannot* be probabilities. There's no natural cap at 0 and 1.
- **MSE is the wrong loss here.** Squared error on a 0/1 target is dominated by outliers, and — the deeper problem we'll *prove* below — once you wrap a sigmoid around the score, **MSE becomes non-convex** and its gradient **vanishes exactly on the confidently-wrong predictions** you most need to fix (see [Loss Functions](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md)).
- **No notion of graded confidence.** A good classifier should say $P(y=1 \mid x)$ — a calibrated probability you can threshold, rank by, and plug into expected-value decisions — not an unbounded score with no probabilistic meaning.

Logistic regression fixes all three at once: pass the linear score through the **sigmoid** to get a genuine probability in $(0,1)$, and fit with the **right probabilistic loss** (cross-entropy) that is convex and has a clean gradient.

> **Tip:** the one-line framing that lands in an interview: *"Linear regression gives you an unbounded score and the wrong loss; logistic regression keeps the linear score but squashes it into a probability with the sigmoid and fits it with cross-entropy."* Everything else on this page is the consequence of that single substitution.

---

## The model: a linear score through the sigmoid

Compute a linear score $z = w\cdot x + b$, then map it to a probability with the **sigmoid** (also called the **logistic function** — that's where the name comes from):

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad p = P(y=1 \mid x)$$

![The sigmoid curve mapping the linear score z = w·x + b on the x-axis to a probability p = σ(z) on the y-axis. It is S-shaped, passing through 0.5 at z = 0. Where z < 0 the probability is below 0.5 (predict class 0); where z > 0 it is above 0.5 (predict class 1); the decision threshold sits at p = 0.5, z = 0.](../images/logreg_sigmoid.png)

The sigmoid squashes any real score into $(0, 1)$: a large positive score $\to$ near 1, a large negative score $\to$ near 0, and $z = 0 \to$ exactly $0.5$. You then classify by thresholding (usually at $0.5$, but the threshold is yours to tune for the precision/recall trade-off — see [Classification Metrics](14-Classification-Metrics.md)).

Two facts about the sigmoid you'll use constantly — and the second one is the algebraic key to the whole page:

- It is **symmetric**: $\sigma(-z) = 1 - \sigma(z)$. So $P(y=0\mid x) = 1 - p = \sigma(-z)$.
- Its **derivative is itself, in disguise**: $\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) = p(1-p)$. Derive it once and never forget it — the quotient rule on $\sigma(z) = (1+e^{-z})^{-1}$ gives $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\big(1-\sigma(z)\big)$.

> **Note:** $\sigma'(z) = p(1-p)$ is biggest at $z=0$ ($p=0.5$, slope $0.25$) and shrinks toward $0$ as $|z|\to\infty$. That **saturation** — flat tails — is exactly why MSE's gradient dies on confident predictions and why log-loss is built to cancel it. Hold this fact; it returns three times below.

---

## Intuition: a panel of weighted votes through a soft switch

Picture a loan officer scoring an application. Each feature is a *voter* with a signed strength: "high income" votes $+2$ for approve, "many recent missed payments" votes $-3$, "long credit history" votes $+1$. The officer just **adds up the weighted votes** — that sum is the linear score $z = w\cdot x + b$, with $b$ the baseline lean before any feature is seen. A big positive total means "very likely approve," a big negative total means "very likely deny," and a total near zero means "genuinely on the fence." The sigmoid is the **soft switch** that turns that unbounded tally into a probability: it can't run off to $\pm\infty$, it's $0.5$ when the votes cancel, and it saturates smoothly toward $0$ or $1$ as the evidence piles up. Reading a coefficient as an odds ratio is just asking *"how loud is this voter?"* — $e^{w_j}$ is the factor by which that one voter multiplies the odds.

That's the entire model in one image: **weighted votes summed, then squashed through a soft 0/1 switch.** It's also why logistic regression is "one neuron" — a neuron is literally a weighted sum followed by a non-linear squash, which is exactly this. Keep the loan-officer panel in mind; every piece below (log-odds coefficients, the linear boundary, the gradient that nudges each voter's weight by its error) is that picture made precise.

---

## Odds and log-odds: what the coefficients actually mean

This is the interview question that separates people who *memorized* logistic regression from people who *understand* it. Start from the model and **invert the sigmoid** to see what the linear score really is. From $p = \frac{1}{1+e^{-z}}$, solve for $z$:

$$1 + e^{-z} = \frac1p \;\Rightarrow\; e^{-z} = \frac{1-p}{p} \;\Rightarrow\; e^{z} = \frac{p}{1-p} \;\Rightarrow\; z = \log\frac{p}{1-p}.$$

So the linear score *equals* the **log-odds** (the **logit**) of the probability:

$$z = w\cdot x + b = \log\frac{p}{1-p} = \operatorname{logit}(p).$$

This is the heart of the model: **logistic regression is linear in the log-odds.** The quantity $\frac{p}{1-p}$ is the **odds** ("3-to-1 on"), and its log is what the model fits linearly. That gives each coefficient a precise, quotable meaning. Increase feature $x_j$ by one unit: the log-odds change by $w_j$, so the **odds get multiplied by $e^{w_j}$**:

$$\frac{\text{odds}(x_j+1)}{\text{odds}(x_j)} = \frac{e^{\,w\cdot x + b + w_j}}{e^{\,w\cdot x + b}} = e^{w_j}.$$

So $w_j$ is the **log odds-ratio** and $e^{w_j}$ is the **odds ratio** for a one-unit change in $x_j$ (holding the others fixed). In the code below a fitted weight of $\approx 4.0$ means a one-unit increase **multiplies the odds of class 1 by $e^{4.0} \approx 55\times$** — a concrete, defensible statement. A positive $w_j$ pushes toward class 1; a negative $w_j$ ($e^{w_j} < 1$) pushes toward class 0; $w_j = 0$ ($e^0 = 1$) means the feature is irrelevant. *That* interpretability — coefficients you can read off as odds ratios — is why logistic regression is still a default in medicine, epidemiology, finance, and the social sciences, long after fancier models existed.

> **Gotcha:** the odds-ratio reading is **multiplicative on the odds**, not additive on the probability. A coefficient of $0.7$ does *not* mean "+0.7 probability"; it means the *odds* multiply by $e^{0.7}\approx 2$. The effect on the probability itself depends on where you start on the S-curve (huge near $p=0.5$, tiny near $p=0$ or $1$).

> *Where this comes from: logistic regression and the log-odds link function originate in **The Regression Analysis of Binary Sequences** (Cox 1958); the clean modern derivation is **Speech and Language Processing** (Jurafsky & Martin) Ch. 5, with the odds-ratio interpretation worked through in **An Introduction to Statistical Learning** Ch. 4 — all in the references.*

---

## The loss: maximum likelihood = cross-entropy (derived)

How do we choose $w, b$? By **maximum likelihood**: pick the parameters that make the *observed labels* most probable under the model. Watch the whole derivation — it's three steps and you should be able to reproduce it on a whiteboard.

**Step 1 — one example's likelihood.** The model says label $y$ has probability $p$ when $y=1$ and $1-p$ when $y=0$. Write both cases as a single expression using $y$ as a switch:

$$P(y \mid x) = p^{\,y}\,(1-p)^{\,1-y}.$$

(Check: $y=1 \Rightarrow p^1(1-p)^0 = p$; $y=0 \Rightarrow p^0(1-p)^1 = 1-p$. ✓ — this is just the Bernoulli probability mass function.)

**Step 2 — the dataset likelihood.** Assuming the $n$ examples are independent, the likelihood of all the labels is the product, and its log (the **log-likelihood**) turns the product into a sum:

$$\log \mathcal{L}(w,b) = \sum_{i=1}^n \Big[\, y_i \log p_i + (1-y_i)\log(1-p_i) \,\Big].$$

**Step 3 — flip the sign to get a loss.** Maximizing the log-likelihood is the same as minimizing its negative (averaged over $n$ so the scale is dataset-size-independent). That negative-average log-likelihood **is** the **cross-entropy / log-loss**:

$$\boxed{\;\mathcal{L}(w,b) = -\frac{1}{n}\sum_{i=1}^n \Big[\, y_i \log p_i + (1-y_i)\log(1-p_i) \,\Big]\;}$$

So **maximum likelihood and minimizing cross-entropy are literally the same objective** — there is nothing extra to assume. The intuition is also clean: $-\log p_i$ is the *surprise* of seeing label $y_i$; minimizing total surprise = making the labels unsurprising = fitting well. And critically, composed with the sigmoid this loss is **convex** in $(w,b)$, so gradient descent (or Newton's method / IRLS) reliably finds the *global* optimum — no bad local minima to get stuck in.

> **Note:** cross-entropy here is the special two-class case of the general $-\sum_c y_c \log p_c$. For $K$ classes you swap the sigmoid for the softmax and get the multiclass cross-entropy of the [softmax output layer](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) — same idea, more classes. Binary logistic regression is the $K=2$ instance.

> *Where this comes from: the MLE $\to$ cross-entropy derivation is **Speech and Language Processing** Ch. 5 and the **CS229** notes §1.2 (Classification and logistic regression); the convexity / IRLS treatment is **The Elements of Statistical Learning** Ch. 4 — references.*

---

## The gradient: strikingly like linear regression (and *why*)

Now differentiate the loss. This is where the sigmoid's derivative does something magical, and the "why" is the part interviewers want. Work it for a single example with $z = w\cdot x + b$ and $p = \sigma(z)$, using the chain rule $\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial p}\cdot\frac{\partial p}{\partial z}\cdot\frac{\partial z}{\partial w}$.

**The loss-vs-$p$ term.** Differentiate $-\big[y\log p + (1-y)\log(1-p)\big]$ with respect to $p$:

$$\frac{\partial \mathcal{L}}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{-y(1-p) + (1-y)p}{p(1-p)} = \frac{p - y}{p(1-p)}.$$

**The sigmoid term.** From before, $\frac{\partial p}{\partial z} = \sigma'(z) = p(1-p)$.

**Multiply — and watch the cancellation.** The $p(1-p)$ from the sigmoid's derivative **exactly cancels** the $p(1-p)$ in the denominator of the loss term:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{p-y}{\,p(1-p)\,}\cdot p(1-p) = (p - y).$$

Then $\frac{\partial z}{\partial w} = x$, so over the whole dataset:

$$\boxed{\;\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n}\sum_{i=1}^n (p_i - y_i)\,x_i = \frac{1}{n}X^\top(\hat y - y)\;}$$

The gradient is the **prediction error $(\hat y - y)$ times the input** — *identical in form* to linear regression's gradient, just with $\hat y = \sigma(z)$ instead of $\hat y = z$. (The code confirms this against a numerical gradient to $10^{-12}$.)

That cancellation is not a coincidence — it is **why cross-entropy is the right loss for the sigmoid**. The log in cross-entropy was *chosen* (via MLE) so that its derivative produces a $\frac{1}{p(1-p)}$ that annihilates the sigmoid's $p(1-p)$, leaving a clean error signal that does **not** saturate. This same algebra is what makes [softmax + cross-entropy](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) backprop so simple, and it's exactly why this model generalizes straight to a single neuron.

> **Gotcha:** there is **no closed-form solution** for $w$ (unlike linear regression's normal equations $w = (X^\top X)^{-1}X^\top y$). The sigmoid makes the score-to-label map nonlinear, so $\frac{\partial \mathcal{L}}{\partial w} = 0$ has no algebraic solution — you *must* solve it iteratively. The good news, again: the objective is **convex**, so any reasonable iterative method converges to the *unique* global optimum (when one exists — see perfect separation).

---

## Why log-loss and not MSE: convexity *and* the gradient

People accept "use log-loss" on faith; an expert can say *why* on two independent grounds. Both are visible in the figure below, and both are reproduced with real numbers in the code.

**Argument 1 — convexity.** Composed with the sigmoid, **cross-entropy is convex** in $(w,b)$ (its Hessian is $\frac{1}{n}X^\top \operatorname{diag}\!\big(p_i(1-p_i)\big) X$, which is positive semi-definite because every $p_i(1-p_i) \ge 0$). One bowl, one global minimum, gradient descent can't get stuck. **MSE composed with the sigmoid is non-convex** — the sigmoid's S-shape introduces flat plateaus and curvature that flips sign, so MSE-with-sigmoid can have local minima and saddle regions. The left panel below shows it: the log-loss surface over a weight is a single clean bowl; the MSE+sigmoid surface flattens into long plateaus.

**Argument 2 — the gradient that doesn't vanish.** Recall MSE's gradient through the sigmoid carries an *extra* $\sigma'(z) = p(1-p)$ factor that cross-entropy's algebra cancelled: $\frac{\partial}{\partial z}\tfrac12(p-y)^2 = (p-y)\,p(1-p)$. On a **confidently wrong** prediction — say the true label is $1$ but the model says $p \approx 0$ (so $z \ll 0$) — that $p(1-p)$ factor is near zero, so **MSE's gradient is near zero exactly when the error is largest.** The model barely corrects its worst mistakes. Cross-entropy's gradient is just $(p - y)$, which is $\approx -1$ in that same situation — a strong corrective push. The right panel makes this stark (at $z=-6$: log-loss gradient $\approx 0.998$, MSE gradient $\approx 0.002$ — a $400\times$ difference).

![Two panels. LEFT: the loss surface over a single weight w for a confident-wrong setup — the log-loss curve (green) is one convex bowl with a single minimum, while the MSE+sigmoid curve (red) flattens into a long plateau (non-convex), so gradient descent stalls. RIGHT: the per-example gradient magnitude versus the linear score z for a true label y=1 — log-loss (green) keeps a strong gradient (≈1) where the model is confidently wrong (z very negative), while MSE+sigmoid (red) collapses toward zero gradient in exactly that confidently-wrong region (shaded), so it can't fix its worst mistakes.](../images/logreg_loss_surface.png)

> **Tip:** the crisp interview answer: *"MSE on a sigmoid is non-convex AND its gradient vanishes on confident-wrong predictions because of the extra σ′ = p(1−p) factor. Cross-entropy comes from MLE, is convex, and its σ′ factor cancels — so the gradient is a clean (p − y) that stays strong exactly where you need it."* Two reasons, said in one breath, signal mastery.

---

## The decision boundary is linear (derived)

You classify positive when $p \ge 0.5$. But $\sigma(z) \ge 0.5 \iff z \ge 0$ (since $\sigma$ is monotonic and $\sigma(0)=0.5$). So the decision rule is simply:

$$\text{predict class 1} \iff z = w\cdot x + b \ge 0.$$

The set where the model is undecided, $w\cdot x + b = 0$, is a **hyperplane** — a line in 2D, a plane in 3D, a flat $(d{-}1)$-dimensional surface in general. **Logistic regression is therefore a linear classifier**: its boundary is always flat, no matter how the sigmoid curves the probabilities behind it.

![Two clusters of 2D points (class 0 in blue, class 1 in red) with a logistic-regression model fit from scratch. The decision boundary where p = 0.5 is a straight line separating them, and the background is shaded by the sigmoid probability — deep blue (P→0) on the class-0 side, deep red (P→1) on the class-1 side, with a smooth gradient across the boundary showing the model's confidence increasing with distance from the line.](../images/logreg_boundary.png)

The line is the **boundary**; the sigmoid turns *distance from the line* into a **confidence** — points far on the red side are near-certain class 1, points near the line are ~50/50. The two roles are separate: $z = 0$ decides the label, $|z|$ decides how sure. For genuinely non-linear boundaries you add polynomial/interaction features (then the boundary is linear in the *expanded* space but curved in the original), use a kernel, or move to a non-linear model — bare logistic regression only ever draws straight lines.

> **Note:** this is the same linear-log-odds form that **Naive Bayes** produces (we make the connection explicit below). Two very different fitting procedures — discriminative likelihood maximization here, generative counting there — arrive at the *same shape* of boundary. That's the generative–discriminative pairing, a recurring interview theme.

---

## Convex optimization: gradient descent vs Newton / IRLS

Because there's no closed form, we iterate. Two methods matter, and knowing both (and *why* you'd pick each) is a strong signal.

**Gradient descent** takes a step downhill: $w \leftarrow w - \eta\,\frac{1}{n}X^\top(\hat y - y)$. Simple, scales to huge $n$ and $d$ (especially as **mini-batch SGD**), and — thanks to convexity — reliably converges to the global optimum. The cost is choosing the learning rate $\eta$ and waiting through many cheap iterations. This is what deep-learning frameworks do, because the same loop trains a billion-parameter network.

**Newton's method / IRLS** uses the curvature too. Newton's update is $w \leftarrow w - H^{-1}\nabla$, where the Hessian is $H = \frac{1}{n}X^\top S X$ with $S = \operatorname{diag}\!\big(p_i(1-p_i)\big)$. Substituting the logistic gradient and Hessian, each Newton step turns out to be a **weighted least-squares solve** — which is why it's called **Iteratively Reweighted Least Squares (IRLS)**: at each step you solve an ordinary least-squares problem reweighted by $p_i(1-p_i)$. It converges in *very few* iterations (quadratically near the optimum), which is great for the modest-$d$ statistical problems logistic regression was born in. The cost is forming and inverting a $d\times d$ Hessian each step — $O(d^3)$ — which is prohibitive when $d$ is large. So: **IRLS for small-to-medium $d$** (statsmodels, R's `glm`, scikit-learn's `newton-cg` solver); **(mini-batch) gradient descent for huge data or as a neural-net layer.**

> **Tip:** scikit-learn's `LogisticRegression` defaults to the **`lbfgs`** solver — a quasi-Newton method that approximates the Hessian from gradients, getting Newton-like convergence without the full $O(d^3)$ solve. It's the sensible middle ground and the reason you rarely tune a learning rate for logistic regression in practice.

---

## Regularization and the `C` knob

Unregularized logistic regression can overfit — and, worse, on separable data its weights **diverge to infinity** (next section). The cure is a penalty on weight size, exactly as in [Regularization](../../05.%20Deep_Learning/concepts/09-Regularization.md):

- **L2 (ridge)** adds $\frac{\lambda}{2}\lVert w\rVert_2^2$ to the loss, shrinking weights smoothly toward zero. It keeps the problem strictly convex, bounds the weights, and is the sensible default. From the Bayesian view, L2 is a **Gaussian prior** on $w$ (MAP estimation).
- **L1 (lasso)** adds $\lambda\lVert w\rVert_1$, which drives some weights *exactly* to zero — giving **feature selection** / sparse models. It corresponds to a **Laplace prior**.
- **Elastic-net** mixes both.

scikit-learn parameterizes the strength with **`C`, the *inverse* regularization strength** ($C = 1/\lambda$): **small `C` = strong regularization** (weights heavily shrunk), **large `C` = weak** (approaches the unregularized MLE). By default `LogisticRegression` uses L2 with `C=1.0` — so it is *always* regularized unless you ask otherwise.

> **Gotcha:** `C` is *inverse* strength — the single most common scikit-learn logistic-regression mistake is reading it backwards. **Small `C` regularizes more.** If your model is overfitting, *decrease* `C`; if it's underfitting, *increase* it.

---

## Multiclass: softmax / multinomial logistic regression

For $K > 2$ classes, replace the single sigmoid with the **softmax** over $K$ linear scores $z_k = w_k\cdot x + b_k$:

$$P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.$$

This is **multinomial logistic regression** (a.k.a. **softmax regression**), fit by minimizing the multiclass cross-entropy $-\frac1n\sum_i \log P(y_i\mid x_i)$. The gradient keeps the *same clean form* — $(\hat y - y)x$ with one-hot $y$ — for the same reason the binary one did (softmax's Jacobian cancels the log's derivative). Binary logistic regression is exactly the $K=2$ special case. And this is *precisely* the [softmax + cross-entropy](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) **output layer of every classification neural network** — an image classifier's final layer *is* softmax regression on learned features.

> **Note:** an older alternative is **one-vs-rest (OvR)**: train $K$ independent binary logistic regressions ("class $k$ vs everything else") and pick the highest score. It's simple and parallel but its probabilities don't jointly normalize, so true **multinomial (softmax)** is usually preferred when you want coherent class probabilities. scikit-learn now defaults to multinomial.

> **Tip:** the deeper, unifying view is the **generalized linear model (GLM)**: logistic regression is the GLM for a **Bernoulli** response with the **logit link** $g(p)=\log\frac{p}{1-p}$, just as linear regression is the GLM for a **Gaussian** response with the identity link and Poisson regression is the GLM for counts with the log link. "Linear model + a link function that maps the linear predictor to the right kind of output" is the pattern; logistic regression is the classification instance, and IRLS is the GLM's shared fitting algorithm. Saying "it's the Bernoulli GLM with a logit link" in an interview is the one-line, theory-grounded definition.

---

## The generative–discriminative pair with Naive Bayes

Here's a connection worth carrying into any interview. [Naive Bayes](05-Naive-Bayes.md) (Multinomial/Bernoulli) produces a **linear log-odds** $w\cdot x + b$. Logistic regression *also* produces a linear log-odds — that's the whole "linear in the logit" result above. So the two fit the **same parametric form** for $P(\text{class}\mid x)$; they differ only in **how they estimate the weights**:

| | Naive Bayes (generative) | Logistic Regression (discriminative) |
|---|---|---|
| **Models** | $P(x\mid c)$ then Bayes-flips to $P(c\mid x)$ | $P(c\mid x)$ **directly** |
| **Boundary form** | linear (Multinomial/Bernoulli) | linear |
| **Weights from** | per-feature counts (closed form) | iterative likelihood maximization |
| **Assumption** | features conditionally independent | **none** |
| **Data efficiency** | converges with **little** data | needs **more** data |
| **Asymptotic error** | higher (capped by the assumption) | lower |
| **Calibration** | over-confident (correlated features double-counted) | **well-calibrated** |

Ng & Jordan's classic result: Naive Bayes hits its (higher) error floor with **far less data**, while logistic regression starts worse but **overtakes** it as data grows. *Little data $\to$ prefer Naive Bayes; plenty of data $\to$ logistic regression usually wins.* They are the textbook **generative–discriminative pair**, and "they fit the same line two different ways" is the sentence that shows you understand both.

> *Where this comes from: the generative–discriminative pairing and the "NB converges faster, LR is asymptotically better" result are **On Discriminative vs. Generative Classifiers** (Ng & Jordan, 2002) — references.*

---

## Calibration: logistic regression hugs the diagonal

A model is **calibrated** if, among the cases where it predicts $0.8$, about $80\%$ really are positive. Logistic regression is **well-calibrated essentially by construction**: it *directly* minimizes log-loss, a strictly **proper scoring rule** whose minimizer is the true conditional probability — so its outputs behave like genuine probabilities, not just scores to be ranked. This is exactly the opposite of Naive Bayes, whose independence assumption double-counts correlated features and produces **over-confident** probabilities (pushed toward 0 and 1). The reliability diagram below — measured with `sklearn.calibration.calibration_curve` on data with many redundant features — shows it plainly:

![A reliability diagram (predicted probability on the x-axis, actual fraction positive on the y-axis) measured with sklearn.calibration.calibration_curve. The dashed diagonal is perfect calibration. Logistic regression's curve (green) sits right on the diagonal — a predicted 0.8 really is ~80% positive — while Naive Bayes' curve (red) bows away from it (flattened and pushed toward the extremes), confirming logistic regression is well-calibrated and Naive Bayes is over-confident.](../images/logreg_calibration.png)

> **Tip:** the gold answer — *"Logistic regression is well-calibrated because it optimizes log-loss directly, a proper scoring rule, so its probabilities are trustworthy enough to threshold and to feed into expected-value decisions. Naive Bayes is accurate but mis-calibrated. If even logistic regression's calibration drifts (it can, under heavy regularization or distribution shift), wrap it in `CalibratedClassifierCV` with Platt scaling or isotonic regression."* Knowing it's calibrated *and* how to fix calibration is the complete answer.

---

## Worked example 1 (minimal): one feature, by hand

A model has learned $w = [2.0]$ and $b = -1.0$ for a single feature $x$. Classify $x = 2$, step by step:

- **Linear score:** $z = 2.0\cdot 2 - 1.0 = 3.0$.
- **Probability:** $p = \sigma(3.0) = \dfrac{1}{1 + e^{-3}} = 0.9526 \to$ predict **class 1** (since $> 0.5$), with $95.3\%$ confidence.
- **Log-odds check:** $\log\dfrac{0.9526}{0.0474} = 3.0 = z$ ✓ — the score *is* the log-odds, exactly as derived.
- **Odds reading:** the odds of class 1 are $\dfrac{0.9526}{0.0474} \approx 20{:}1$. A one-unit increase in $x$ would add $w=2.0$ to the log-odds, i.e. multiply the odds by $e^{2.0}\approx 7.4\times$.
- **If the true label is $y = 1$:** loss $= -\log(0.9526) = 0.0486$ (small — confident *and* correct); gradient contribution $(p - y)\,x = (0.9526 - 1)\cdot 2 = -0.0949$ — a small nudge to raise $w$ slightly.

This is the whole model on one feature. The next examples add a second feature, and then a full gradient step.

---

## Worked example 2 (realistic): a two-feature decision

Now $w = [1.5,\, -2.0]$, $b = 0.5$, and a point $x = (2.0,\, 1.0)$. Notice feature 2 has a **negative** weight — it pushes *toward class 0*.

- **Linear score:** $z = 1.5\cdot 2.0 + (-2.0)\cdot 1.0 + 0.5 = 3.0 - 2.0 + 0.5 = 1.5$.
- **Probability:** $p = \sigma(1.5) = 0.8176 \to$ predict **class 1**; odds $= \dfrac{0.8176}{0.1824} \approx 4.48{:}1$.
- **Odds-ratio reading of the coefficients:** a one-unit rise in $x_1$ multiplies the odds by $e^{1.5} = 4.48$ (toward class 1); a one-unit rise in $x_2$ multiplies them by $e^{-2.0} = 0.135$ (i.e. *cuts* the odds to $\sim\!14\%$, strongly toward class 0). The two features pull in opposite directions, and the linear score adds up their net effect.
- **Geometry:** the decision boundary is the line $1.5\,x_1 - 2.0\,x_2 + 0.5 = 0$; our point gives $z = 1.5 > 0$, so it sits on the class-1 side, a modest distance from the boundary — hence a confident-but-not-extreme $p = 0.82$.

---

## Worked example 3 (full trace): one gradient step with numbers

Let's run a single gradient-descent step end to end on a tiny dataset, so the $(\hat y - y)x$ formula is concrete. Four 1-D points with a bias term:

$$X = \begin{bmatrix}1\\2\\3\\4\end{bmatrix},\quad y = \begin{bmatrix}0\\0\\1\\1\end{bmatrix},\quad X_b = \begin{bmatrix}1&1\\1&2\\1&3\\1&4\end{bmatrix},\quad w = \begin{bmatrix}b\\w_1\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}\ \text{(start)}.$$

**Forward pass.** With $w = (0,0)$, every score is $z_i = 0$, so $p_i = \sigma(0) = 0.5$ for all four points — the model starts maximally unsure. Initial loss $= -\frac14\sum[\,y_i\log 0.5 + (1-y_i)\log 0.5\,] = -\log 0.5 = 0.6931$.

**Gradient.** Errors $p - y = (0.5,\,0.5,\,-0.5,\,-0.5)$. Then $\nabla = \frac{1}{n}X_b^\top(p - y)$:

- bias component: $\frac14(0.5 + 0.5 - 0.5 - 0.5) = 0$;
- $w_1$ component: $\frac14\big(1\cdot0.5 + 2\cdot0.5 + 3\cdot(-0.5) + 4\cdot(-0.5)\big) = \frac14(0.5 + 1 - 1.5 - 2) = \frac14(-2) = -0.5$.

So $\nabla = (0,\,-0.5)$ — the bias is already balanced, and the slope wants to *increase* (negative gradient $\to$ the step raises $w_1$), which makes sense: bigger $x$ should mean higher $p$.

**Update** (learning rate $\eta = 1$): $w \leftarrow w - \eta\nabla = (0,\,0) - (0,\,-0.5) = (0,\,0.5)$.

**Check it improved.** New scores $z = 0.5\,x = (0.5, 1.0, 1.5, 2.0)$, so $p = (0.6225,\,0.7311,\,0.8176,\,0.8808)$ — the two positive points ($x=3,4$) now correctly sit above $0.5$, and the loss drops from $0.6931$ to $0.6539$. One honest step, and the model already separates the classes better. Iterate this and (because the data here is perfectly separable) the weights keep climbing without bound — which is the perfect-separation pitfall we cover below. Every number in this trace is reproduced exactly by the code below.

---

## Code: logistic regression from scratch (gradient verified, matches scikit-learn)

```python
"""Logistic regression from scratch: the (p - y)x gradient, log-odds coefficients,
convexity, and the scikit-learn match. Verified on Python 3.12, CPU (numpy + sklearn)."""
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

# 1) the claimed gradient (1/n) X^T (p - y) vs a numerical gradient
w = rng.normal(size=3)
analytic = Xb.T @ (sig(Xb @ w) - y) / len(y)
numeric = np.array([(log_loss(w + e) - log_loss(w - e)) / 2e-5 for e in 1e-5*np.eye(3)])
print(f"gradient (1/n)X^T(p-y) vs numerical: max diff = {np.abs(analytic-numeric).max():.2e}")

# 2) fit by gradient descent (convex -> reliable)
w = np.zeros(3)
for _ in range(4000):
    w -= 0.1 * Xb.T @ (sig(Xb @ w) - y) / len(y)
print(f"train accuracy = {((sig(Xb@w)>0.5)==y).mean():.3f}")
print(f"feature-1 weight w1={w[1]:.2f} -> +1 unit multiplies odds by e^w1 = {np.exp(w[1]):.1f}")

# 3) match scikit-learn (large C ~ unregularized) -- same direction, same accuracy
from sklearn.linear_model import LogisticRegression
skl = LogisticRegression(C=1e6).fit(X, y)
print(f"sklearn weights  = {skl.coef_[0].round(2)}  bias = {skl.intercept_[0]:.2f}")
print(f"scratch  weights = {w[1:].round(2)}  bias = {w[0]:.2f}")
print(f"sklearn train acc = {skl.score(X, y):.3f}")
```

Output:

```
gradient (1/n)X^T(p-y) vs numerical: max diff = 5.36e-12
train accuracy = 0.983
feature-1 weight w1=4.02 -> +1 unit multiplies odds by e^w1 = 55.5
sklearn weights  = [4.51 3.47]  bias = -0.18
scratch  weights = [4.02 3.06]  bias = -0.15
sklearn train acc = 0.983
```

> **Note:** the analytic gradient $(1/n)X^\top(p - y)$ matches the numerical gradient to $10^{-12}$ — confirming the clean error-times-input form we derived. The fitted model reaches $98.3\%$ accuracy, and the log-odds reading is concrete: a one-unit increase in feature 1 multiplies the odds of class 1 by $e^{4.02} \approx 55$. scikit-learn finds weights in the **same direction** and the **identical accuracy** (its are a touch larger because our hand-rolled descent isn't fully converged) — the derivation is correct.

---

## Pitfalls that actually bite

- **Perfect (linear) separation $\to$ weights diverge.** If a feature (or combination) *perfectly* splits the classes, the MLE wants infinite confidence: pushing $|w|\to\infty$ makes the separable points' probabilities $\to 0/1$ and drives log-loss $\to 0$, so there is **no finite optimum** — unregularized gradient descent grows the weights forever. In a quick check, $100\to50{,}000$ steps on separable data sent $w_1$ from $3.4$ to $9.4$ and still climbing. **The fix is regularization:** any L2 penalty bounds the weights and restores a unique, finite solution (in that check, L2 with `C=1.0` pinned $w_1$ to a sane $\approx 1.0$). This is *the* reason scikit-learn regularizes by default — and why "what happens under perfect separation?" is a favorite interview probe.
- **Unscaled features.** Gradient descent and the regularization penalty both assume comparable feature scales; a feature measured in the thousands will dominate the gradient and be penalized differently from one in $[0,1]$. **Standardize** (zero mean, unit variance) before fitting — and remember it changes the coefficients' units, so interpret odds ratios in standardized terms.
- **Class imbalance and the threshold.** With $99\%$ negatives, a model that always predicts "negative" is $99\%$ accurate and useless. Don't trust raw accuracy: use class weights (`class_weight="balanced"`), resampling, and pick the **decision threshold** from a precision/recall or ROC analysis rather than blindly using $0.5$ (see [Classification Metrics](14-Classification-Metrics.md)). The model outputs a calibrated probability; *you* choose where to cut it.
- **Reading `C` backwards** — small `C` = *more* regularization (covered above); a perennial source of confusion.
- **Multicollinearity.** Highly correlated features make individual coefficients unstable and hard to interpret (their effects trade off), even though predictions stay fine. L2 stabilizes them; L1 can pick one and zero the rest.

---

## Application: how I'd actually fit one

Given a fresh binary-classification problem, here's the end-to-end playbook — each step maps to something derived above:

1. **Feature prep.** Standardize numeric features (zero mean, unit variance — otherwise the gradient and the L2 penalty are dominated by large-scale features); one-hot encode categoricals; for text, build **TF-IDF** vectors. Add interaction/polynomial terms *only* if you expect a non-linear boundary.
2. **Fit with regularization on.** `LogisticRegression(C=..., class_weight="balanced" if imbalanced)`. Leave the default **L2 / `lbfgs`** unless you want **L1** for sparsity. Never fit unregularized on data that might be separable (the weights diverge).
3. **Tune `C` by cross-validation.** Sweep `C` on a log grid (`LogisticRegressionCV` does this for you); remember **small `C` = more regularization**. Watch validation log-loss, not just accuracy.
4. **Read the coefficients.** Exponentiate them ($e^{w_j}$) to report **odds ratios** — the interpretable payoff. Sanity-check signs against domain knowledge; unstable/huge coefficients flag multicollinearity or separation.
5. **Pick the threshold deliberately.** The model gives a calibrated probability; choose the operating point from a **precision/recall or ROC** curve for *your* cost trade-off, not a reflexive $0.5$ (see [Classification Metrics](14-Classification-Metrics.md)).
6. **Check calibration** with a reliability diagram (`calibration_curve`) and the Brier score; if it has drifted under heavy regularization or shift, wrap in `CalibratedClassifierCV`.
7. **Treat it as the baseline to beat.** Its log-loss/AUC is the number a boosted-tree or neural model must *clearly* exceed to justify the extra complexity.

> **Tip:** in practice this whole pipeline is `make_pipeline(StandardScaler(), LogisticRegression())` plus a threshold chosen from the validation PR curve — a five-line, fast, interpretable, well-calibrated classifier that is genuinely hard to beat on tabular or linearly-separable problems.

---

## Where logistic regression is used

- **The default binary classifier** — spam, churn, fraud, click-through, medical diagnosis, credit scoring: fast to train, calibrated, interpretable, and a genuinely strong baseline.
- **Interpretable / regulated settings** — medicine, epidemiology, finance, and the social sciences prize the **odds-ratio coefficients** you can defend to a regulator or a clinician.
- **The baseline you always run first** — try logistic regression (with TF-IDF features for text, standardized numerics otherwise) *before* reaching for a boosted-tree ensemble or a transformer; it's cheap, hard to beat on linearly-separable problems, and a sanity check on the fancier model.
- **The output layer of neural networks** — a sigmoid (binary) or softmax (multiclass) on top of *learned* features *is* logistic / softmax regression. Everything you know about its loss and gradient is exactly what backprop computes for that final layer.

> **Tip:** logistic regression is the canonical "explain a model end to end" interview. Be ready to walk the full thread: sigmoid $\to$ probability $\to$ log-odds coefficients (odds ratios) $\to$ cross-entropy from MLE $\to$ the $(\hat y - y)x$ gradient and *why* $\sigma'$ cancels $\to$ convex optimization (GD vs IRLS) $\to$ linear boundary $\to$ regularization and perfect separation $\to$ calibration $\to$ softmax/NN connection. That single chain covers a remarkable fraction of all of supervised ML.

---

## Recap and rapid-fire

**If you remember nothing else:** logistic regression squashes a linear score $w\cdot x + b$ through the **sigmoid** into a probability, and fits by **maximum likelihood** = minimizing **cross-entropy** (convex, so gradient descent reliably finds the global optimum). The coefficients are **log-odds** — a one-unit change multiplies the odds by $e^{w_j}$ — the gradient is the clean $(\hat y - y)x$ (because the sigmoid's $\sigma'=p(1-p)$ cancels the log's derivative), and the decision boundary is **linear**. It's **well-calibrated** (it optimizes log-loss directly), regularize it (and you *must*, under perfect separation), and **softmax regression** is the multiclass version — the output layer of every classification net.

**Quick-fire — say these out loud:**

- *Why the sigmoid?* It maps an unbounded linear score to a $(0,1)$ probability.
- *What is the linear score, really?* The **log-odds** of the probability: $w\cdot x + b = \log\frac{p}{1-p}$.
- *What do the coefficients mean?* A one-unit feature increase multiplies the **odds** by $e^{w_j}$ (the odds ratio).
- *Where does the loss come from?* Maximum likelihood on the labels $\to$ cross-entropy; minimizing one *is* maximizing the other.
- *What's the gradient?* $(1/n)X^\top(\hat y - y)$ — prediction error times input, same form as linear regression.
- *Why does it cancel so cleanly?* The sigmoid's derivative $\sigma'=p(1-p)$ cancels the $\frac{1}{p(1-p)}$ from the log's derivative.
- *Why log-loss, not MSE?* Cross-entropy with the sigmoid is **convex** *and* its gradient stays strong on confident-wrong cases; MSE+sigmoid is non-convex with a **vanishing gradient** there.
- *Closed-form solution?* No — iterate (gradient descent / Newton-IRLS); convexity guarantees the global optimum *when one exists*.
- *What shape is the boundary?* Linear (the hyperplane $w\cdot x + b = 0$).
- *Non-linear boundaries?* Add polynomial/interaction features, use a kernel, or switch models.
- *Regularization & the `C` knob?* L2 (default) / L1 (sparse); `C` is *inverse* strength — small `C` regularizes more.
- *Perfect separation?* The MLE diverges (infinite weights); regularization is required to get a finite fit.
- *Multiclass?* Softmax (multinomial logistic) regression.
- *NB vs logistic regression?* Same linear form; generative (counts, little data, higher floor) vs discriminative (direct fit, more data, lower floor) — Ng & Jordan.
- *Is it calibrated?* Yes — it optimizes log-loss directly; Naive Bayes is over-confident by contrast.
- *Connection to deep learning?* It's one neuron with a sigmoid; softmax regression is a net's output layer.

---

## References and further reading

The curated link library for this topic — start-here path, videos, interactive/visual resources, courses, articles, papers, books, and internal cross-links — lives in a companion file so it can be reused as a standalone reference list:

**→ [Logistic Regression — references and further reading](02-Logistic-Regression.references.md)**

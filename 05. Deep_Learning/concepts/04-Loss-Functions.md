---
id: "05-deep-learning/loss-functions"
topic: "Loss Functions (MSE · cross-entropy · and friends)"
parent: "05-deep-learning"
level: beginner
prereqs: ["feedforward-networks", "probability"]
interview_frequency: high
template: concept-deep
updated: 2026-06-21
---

# Loss functions: the number the network is trying to make small

Every neural network learns by playing the same game: make a prediction, measure how wrong it is with a single number, and nudge the weights to make that number smaller. That single number is the **loss**, and choosing it is one of the most consequential decisions in all of ML — because the loss is the *only* thing the optimizer actually cares about. Pick the wrong loss and you can train a perfectly good architecture to do exactly the wrong thing. The loss has two jobs: **score** how bad a prediction is, and — because we learn by gradient descent — **supply a useful gradient** that points the weights toward "less wrong." A good loss does both; a bad one scores fine but gives gradients that vanish or mislead.

By the end of this page you'll be able to:

- explain what a loss is and the two jobs it must do (score **and** gradient);
- derive **MSE** from Gaussian maximum likelihood, and reason about **MAE/Huber** and outlier sensitivity;
- derive **cross-entropy** from maximum likelihood / negative log-likelihood, and connect it to entropy and **KL divergence**;
- show why **softmax + cross-entropy** has the beautifully clean gradient $(\hat y - y)$ — and why MSE on a classifier does *not*;
- pick the right loss per task, and recognize **hinge, focal, and label-smoothing** when they come up;
- implement MSE and cross-entropy from scratch and confirm the gradients against PyTorch.

Intuition and pictures first, then the derivations (with sources), then runnable code.

> **Note:** almost every loss in supervised learning is secretly **maximum likelihood** — "choose parameters that make the observed data most probable." MSE falls out of assuming Gaussian noise; cross-entropy falls out of assuming a categorical/Bernoulli label. Seeing that one principle behind both is the thing that makes losses click instead of feeling like a grab-bag of formulas.

---

## The problem: turning "how wrong" into one differentiable number

We have a model that outputs predictions $\hat y$ and ground-truth targets $y$. To train it we need a function $L(\hat y, y)$ that is:

- **a single scalar** — so we can say one run is better than another and minimize it;
- **minimized by the right answer** — smallest when $\hat y = y$ (or matches the true distribution);
- **differentiable** (almost everywhere) — so backprop can compute $\partial L / \partial \hat y$ and push the weights downhill.

The catch is the gradient. Two losses can agree on *which* prediction is best yet give wildly different gradients along the way — and the gradient is what training actually follows. That's the thread running through this whole page: we don't just want a loss that *scores* correctly, we want one whose *slope* keeps pointing usefully even when the model is badly wrong.

---

## Regression: MSE, MAE, and Huber

When the target is a real number (a price, a temperature), the natural losses measure the **error** $e = \hat y - y$:

- **Mean Squared Error**: $L_{\text{MSE}} = \frac{1}{N}\sum (\hat y_i - y_i)^2$ — squares the error.
- **Mean Absolute Error**: $L_{\text{MAE}} = \frac{1}{N}\sum |\hat y_i - y_i|$ — takes the absolute value.
- **Huber**: quadratic for small errors, linear for large ones — a tunable hybrid.

![Loss versus prediction error for the three regression losses. MSE is a parabola that blows up quadratically for large errors (very outlier-sensitive); MAE is a V-shape that grows linearly (robust, but with a non-smooth kink at zero); Huber with delta=1 matches MSE near zero and MAE far out, combining smoothness with robustness.](images/loss_regression.png)

The picture *is* the intuition. MSE's quadratic arms mean a single big outlier dominates the loss — the model bends to chase it. MAE grows only linearly, so it's **robust to outliers**, but its kink at zero gives a constant-magnitude gradient (it doesn't ease off as you get close, and it's non-differentiable exactly at 0). **Huber** gets the best of both: smooth quadratic near zero (good gradients as you converge), linear far out (robust to outliers). The knob $\delta$ sets the crossover.

> *Where MSE comes from: assume the target is the model's output plus **Gaussian noise**, $y = \hat y + \mathcal{N}(0,\sigma^2)$. Maximizing the likelihood of the data is then **equivalent to minimizing $\sum(\hat y - y)^2$** — MSE is Gaussian maximum likelihood. The derivation is in **Pattern Recognition and Machine Learning** (Bishop) §1.2.5 and **Deep Learning** (Goodfellow et al.) §5.5; Huber is Huber (1964). All in the references.*

> **Tip:** "MSE vs MAE?" → MSE if large errors really are worse and your data is clean; MAE/Huber if you have outliers you don't want to chase. MSE penalizes a 10-unit miss 100×, a 1-unit miss; MAE penalizes it only 10×.

---

## Classification: cross-entropy

For classification the model outputs a **probability distribution** over classes (via softmax), and we want it to put as much probability as possible on the true class. The loss is **cross-entropy** — for one example with true class distribution $y$ (one-hot) and predicted probabilities $\hat y$:

$$L_{\text{CE}} = -\sum_{c} y_c \log \hat y_c \;=\; -\log \hat y_{\text{true}}$$

Because $y$ is one-hot, the sum collapses to a single term: **the negative log-probability the model assigned to the correct class.** That's the whole loss.

![The cross-entropy penalty as a function of the probability the model gave the true class. It is the curve minus-log-p: near zero loss when the model is confident and correct (p near 1), and exploding toward infinity as the model confidently assigns the true class a low probability. Marked points: p=0.9 gives 0.11, p=0.5 gives 0.69, p=0.1 gives 2.30.](images/loss_crossentropy.png)

The shape is the entire intuition: assigning the true class $p = 0.9$ costs almost nothing (0.11), but $p = 0.1$ — confident *and wrong* — costs 2.30, and as $p \to 0$ the penalty goes to **infinity**. Cross-entropy punishes confident mistakes mercilessly, which is exactly what you want from a probabilistic classifier.

**Binary vs categorical.** With two classes (one sigmoid output $p$), this becomes **binary cross-entropy** $-[y\log p + (1-y)\log(1-p)]$; with $K$ classes (softmax) it's the **categorical** form above. Same idea, same MLE origin.

> *Where it comes from: cross-entropy is the **negative log-likelihood** of a categorical/Bernoulli label — maximizing likelihood = minimizing cross-entropy. Derived in **Deep Learning** (Goodfellow et al.) §6.2.1.1 and **d2l.ai** Ch. 4 (softmax regression); the information-theory grounding is below.*

---

## The magic: softmax + cross-entropy → gradient $(\hat y - y)$

Here's the result that makes softmax and cross-entropy inseparable. Feed logits $z$ through softmax to get $\hat y$, then through cross-entropy against one-hot $y$. The gradient of the loss with respect to the **logits** is just:

$$\frac{\partial L_{\text{CE}}}{\partial z} = \hat y - y$$

Predicted-probability minus the truth. No softmax derivative left over, no division, no mess — just the error. Let me show why. For the true-class logit $z_k$ (where $y_k = 1$), cross-entropy is $L = -\log \hat y_k$ with $\hat y_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$. The softmax Jacobian is $\frac{\partial \hat y_i}{\partial z_j} = \hat y_i(\delta_{ij} - \hat y_j)$. Applying the chain rule and summing over classes, every term telescopes and you are left with $\partial L / \partial z_i = \hat y_i - y_i$. The softmax's $\hat y(1-\hat y)$ factor that would otherwise cause vanishing gradients is **exactly cancelled** by the $1/\hat y$ from differentiating $\log$. That cancellation is not luck — softmax and cross-entropy are designed as a matched pair.

> *Where this derivation lives: the full step-by-step is in **d2l.ai** §4.1.2 and Bishop §4.3.2; Brandon Rohrer's "Softmax and its derivative" (references) walks the cancellation. The code below confirms it against autograd.*

> **Gotcha — don't use MSE on a classifier.** Put MSE on top of a sigmoid/softmax and the gradient picks up that $\hat y(1-\hat y)$ factor, which is **near zero exactly when the model is confidently wrong** (a saturated sigmoid). So the most badly-wrong examples produce almost no gradient — training stalls. Cross-entropy has no such factor; its gradient stays $(\hat y - y)$, large precisely when the prediction is far off. (MSE+sigmoid is also non-convex.) The code makes this concrete.

---

## Why cross-entropy: the information-theory view

Cross-entropy isn't an arbitrary formula — it's a quantity from information theory. The **cross-entropy** between a true distribution $p$ and a model distribution $q$ is $H(p,q) = -\sum p_c \log q_c$, and it decomposes as:

$$H(p, q) = \underbrace{H(p)}_{\text{entropy of the data}} + \underbrace{D_{\text{KL}}(p \,\|\, q)}_{\text{extra cost of using } q \text{ instead of } p}$$

Since $H(p)$ (the data's own entropy) is fixed, **minimizing cross-entropy is identical to minimizing the KL divergence** from your model to the true label distribution — i.e. making your predicted distribution as close as possible to reality. That's the principled reason it's *the* classification loss: it directly minimizes the statistical distance between what the model believes and what's true.

---

## Other losses worth knowing

- **Hinge loss** ($\max(0, 1 - y\cdot\hat y)$) — the SVM loss; cares only about a margin, not calibrated probabilities. Still appears in some embedding/ranking setups.
- **Focal loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)) — cross-entropy reweighted by $(1-\hat y)^\gamma$ so easy, well-classified examples contribute little; built for extreme class imbalance (dense object detection).
- **Label smoothing** ([Szegedy et al. 2016](https://arxiv.org/abs/1512.00567)) — replace one-hot targets with, say, $0.9$ / $0.1$-spread so the model doesn't become pathologically over-confident; improves calibration and is standard in modern training.
- **Contrastive / triplet / InfoNCE** — losses over *pairs* that pull similar items together and push dissimilar apart; the backbone of self-supervised and embedding learning.

> **Tip:** the loss encodes *what you actually want*. Class imbalance → focal or class weights. Over-confidence → label smoothing. Want embeddings, not labels → contrastive. Reaching for a fancier loss is often the cleanest fix for a task-specific problem.

---

## Numerical stability: never softmax-then-log

Computing softmax and then taking its log separately overflows: a large logit makes $e^{z}$ explode, and a tiny probability makes $\log(\hat y) \to -\infty$. The fix is the **log-sum-exp** trick — compute $\log \text{softmax}$ directly and in a numerically stable way (subtract the max logit first). This is why every framework gives you a *fused* `cross_entropy` / `log_softmax` that takes **logits**, not probabilities.

> **Gotcha:** pass **logits** to `nn.CrossEntropyLoss` / `F.cross_entropy`, not softmax outputs. Applying softmax yourself and then the loss double-applies it and is numerically worse. The code uses `log_softmax` for exactly this reason.

---

## Worked examples

**Cross-entropy, one example.** True class is #1; the model outputs probabilities $\hat y = [0.7, 0.2, 0.1]$. Loss $= -\log(0.7) = 0.357$. If it had been confident-correct, $\hat y = [0.99, \dots]$, loss $= -\log(0.99) = 0.010$. If confident-wrong, $\hat y_{\text{true}} = 0.01$, loss $= -\log(0.01) = 4.61$. Same example, a 460× swing in penalty — that's the $-\log$ curve doing its job.

**MSE, a tiny batch.** Predictions $\hat y = [2.5, 0.0, 2.1]$, targets $y = [3.0, -0.5, 2.0]$. Errors $e = [-0.5, 0.5, 0.1]$. $L_{\text{MSE}} = \frac{1}{3}(0.25 + 0.25 + 0.01) = 0.17$. The per-element gradient is $\frac{2}{N}e = [-0.33, 0.33, 0.067]$ — proportional to the error, so the worst-fit elements get pushed hardest.

---

## Where each loss is used

- **MSE / Huber** — regression: forecasting, value functions in RL, autoencoder reconstruction.
- **Cross-entropy** — virtually all classification, **including every token of LLM next-token prediction** (softmax over the vocabulary + cross-entropy is the pretraining loss).
- **Focal / weighted CE** — imbalanced detection and classification.
- **Contrastive / InfoNCE** — self-supervised pretraining, retrieval/embedding models, CLIP.

> **Tip:** when an LLM is "trained to predict the next token," the loss is plain categorical cross-entropy over the vocabulary — the same formula on this page, just with a 50,000-way softmax. Everything here scales straight up to frontier models.

---

## Code: MSE and cross-entropy from scratch, gradients checked

The headline check: the softmax+cross-entropy gradient really is $(\hat y - y)$, matched against autograd. Runs on CPU in a second.

```python
"""Cross-entropy + MSE from scratch, and the softmax+CE gradient = (p - y).
Verified against torch autograd on Python 3.12 (torch 2.12), CPU."""
import torch, torch.nn.functional as F
torch.manual_seed(0)
N, C = 4, 3                                  # 4 examples, 3 classes
z = torch.randn(N, C, requires_grad=True)    # logits
y_idx = torch.tensor([0, 2, 1, 0])           # true class per example
Y = F.one_hot(y_idx, C).float()              # one-hot targets

logp = F.log_softmax(z, dim=1)               # stable log(softmax) — never softmax-then-log
ce = -(Y * logp).sum(1).mean()               # mean cross-entropy
print(f"cross-entropy = {ce.item():.4f}   (torch = {F.cross_entropy(z, y_idx).item():.4f})")

ce.backward()                                # the claim: dCE/dz = (softmax(z) - Y)/N
p = F.softmax(z, dim=1)
print(f"max|autograd - (p - y)/N| = {(z.grad - (p - Y)/N).abs().max():.2e}   <- gradient = (p - y)")

# why NOT MSE on a saturated sigmoid: sigmoid'(z) kills the gradient when confident
s = torch.sigmoid(torch.tensor(8.0))         # confident output ~0.9997
print(f"sigmoid(8)={s:.4f}, its derivative={ (s*(1-s)).item():.2e}  -> MSE+sigmoid gradient ~vanishes")
```

Output:

```
cross-entropy = 1.0820   (torch = 1.0820)
max|autograd - (p - y)/N| = 1.49e-08   <- gradient = (p - y)
sigmoid(8)=0.9997, its derivative=3.35e-04  -> MSE+sigmoid gradient ~vanishes
```

> **Note:** the middle line is the payoff — the analytic $(\hat y - y)/N$ matches PyTorch's autograd to $10^{-8}$, confirming the clean gradient is real. The last line quantifies the MSE-on-classifier trap: at a confident output the sigmoid derivative is $\sim 3\times10^{-4}$, so an MSE gradient there is essentially zero even when the answer is wrong.

---

## Recap and rapid-fire

**If you remember nothing else:** the loss is the single number training minimizes, and it must give a useful *gradient*, not just a correct *score*. **MSE** (Gaussian MLE) fits regression but is outlier-sensitive; **MAE/Huber** are robust. **Cross-entropy** (categorical MLE = minimizing KL to the data) fits classification, punishes confident-wrong toward infinity, and pairs with softmax to give the clean gradient $(\hat y - y)$ — which is exactly why you never use MSE on a classifier.

**Quick-fire — say these out loud:**

- *What two jobs does a loss do?* Score the prediction **and** supply a useful gradient.
- *Where does MSE come from?* Maximum likelihood under Gaussian noise.
- *Where does cross-entropy come from?* Negative log-likelihood of a categorical/Bernoulli label.
- *MSE vs MAE?* MSE squares (outlier-sensitive, smooth); MAE is linear (robust, kink at 0); Huber blends them.
- *Why cross-entropy over MSE for classification?* Clean $(\hat y - y)$ gradient that stays strong when wrong; MSE+sigmoid's gradient vanishes on confident mistakes (and is non-convex).
- *Cross-entropy vs KL divergence?* $H(p,q) = H(p) + D_{KL}(p\|q)$; since $H(p)$ is fixed, minimizing CE = minimizing KL to the truth.
- *Softmax+CE gradient?* $\partial L/\partial z = \hat y - y$ (the softmax factor cancels the log's).
- *Logits or probabilities into the loss?* Logits — frameworks fuse `log_softmax` for numerical stability.
- *A loss for class imbalance? for over-confidence?* Focal loss / weighted CE; label smoothing.
- *What loss trains an LLM?* Categorical cross-entropy over the vocabulary, per token.

---

## References and further reading

The curated link library for this topic — videos, courses, interactive/visual resources, articles, papers, books, and internal cross-links — lives in a companion file so it can be reused as a standalone reference list:

**→ [Loss Functions — references and further reading](04-Loss-Functions.references.md)**

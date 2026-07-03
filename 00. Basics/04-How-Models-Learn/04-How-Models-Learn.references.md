---
id: "00-basics/how-models-learn/references"
topic: "How Models Learn — References"
parent: "00-basics/how-models-learn"
type: references
updated: 2026-07-03
---

# How Models Learn — references and further reading

> Companion link library for **[How Models Learn](04-How-Models-Learn.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Everything here is free / open, and every "Source / derivation" citation on the concept page appears below.

**Start here — suggested path**:
1. **Feel it move** — watch [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (**3Blue1Brown**). *The ball-rolling-downhill picture on a real loss surface — the best 20 minutes for making learning click.*
2. **See the algorithm with numbers** — watch [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) (**StatQuest**). *The exact loop — predict, loss, gradient, step — worked on concrete numbers, no hand-waving.*
3. **Get loss precisely** — read [Google ML Crash Course — Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss) then [Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent) (**Google**). *What the loss measures and how the update step uses its gradient, with interactive widgets.*
4. **Build the whole loop in code** — watch [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU) (**Samson Zhang**). *Forward pass → loss → gradient → update in plain NumPy — the same thing this chapter's module does.*
5. **Learn to measure honestly** — read [Google — Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) + watch [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw). *Why train / validation / test (and k-fold) keep you from fooling yourself.*

**Videos**:
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the canonical visual intuition for minimising a loss by walking downhill.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — the algorithm with real numbers through every step; demystifies the math.
- [Stochastic Gradient Descent, Clearly Explained](https://www.youtube.com/watch?v=vMh0zPT0tLI) — **StatQuest** — why we use mini-batches and what "stochastic" buys us at scale.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest** — train/val/test and k-fold, the honest way to estimate performance.
- [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU) — **Samson Zhang** — the full learn loop (forward → loss → backprop → update) in NumPy, no frameworks.
- [Gradient Descent — Andrew Ng (ML Specialization, C1)](https://www.youtube.com/watch?v=4b4MUYve_U8) — **Andrew Ng / DeepLearning.AI** — the cost function and the gradient-descent update derived from scratch, the classic treatment.

**Courses (free)**:
- [Google ML Crash Course — Linear Regression (Loss & Gradient Descent)](https://developers.google.com/machine-learning/crash-course/linear-regression/loss) — **Google** — the cleanest loss → gradient-descent walkthrough, with interactive widgets and a learning-rate playground.
- [Machine Learning Specialization — Course 1 (free to audit)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — derives cost functions and gradient descent from first principles.
- [Kaggle Learn — Intro to ML (Model Validation)](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on train/validation split and measuring error in a few notebooks.
- [fast.ai — Practical Deep Learning, Lesson on SGD](https://course.fast.ai/) — **Jeremy Howard** — SGD built from scratch, the practitioner's view of the loop.

**Interactive & visual**:
- [Google — Gradient Descent & Learning Rate playground](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent) — **Google** — drag the learning rate and watch the loss converge, crawl, or diverge — the sweep from this chapter, interactive.
- [TensorFlow Playground](https://playground.tensorflow.org/) — **Google** — train a small network in the browser and watch the loss fall as the boundary forms; the loop made tangible.

**Articles / blogs (free, no paywall)**:
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — the reference survey of SGD, momentum, and Adam; read once the intuition lands.
- [Gradient Descent — explained](https://www.ibm.com/think/topics/gradient-descent) — **IBM** — a clear text walkthrough of the optimization loop and the learning rate.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn docs** — train/val/test and k-fold with runnable code.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — how the loop behaves in practice, and the failure modes (great once you've trained a few models).

**Key sources / derivations** (cited on the concept page):
- [Méthode générale pour la résolution des systèmes d'équations simultanées](https://www.academie-sciences.fr/pdf/dossiers/Cauchy/Cauchy_pdf/CR1847_t25_p536_538.pdf) — **Cauchy (1847)** — the original note introducing the gradient-descent idea (step in the negative-gradient direction).
- [Deep Learning — Ch. 4 (numerical computation / gradient-based optimization) & Ch. 8 (optimization for training)](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — the modern reference for gradient descent on ML objectives; Ch. 5.1.4 gives MSE as a maximum-likelihood loss.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder (2016)** — the survey behind the "batch vs SGD vs Adam" note.
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — why generalisation (the test set) is the real goal, not training error.

**Books (free, with chapters)**:
- [Dive into Deep Learning — Ch. 3 "Linear Neural Networks" & Ch. 4.4 "Model Selection"](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — loss, gradient descent, and train/val/test built from scratch with runnable code; the direct companion to this chapter.
- [Neural Networks and Deep Learning — Ch. 1–2](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — gradient descent and how learning works, from zero, beautifully explained.
- [An Introduction to Statistical Learning — Ch. 5 "Resampling Methods"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free PDF; cross-validation and honest error estimation.

**In this platform**:
- Concept page (full explanation): [How Models Learn](04-How-Models-Learn.md)
- The calculus underneath the gradient: [08 Derivatives and Gradients](../../01.%20Foundations/08-Derivatives-and-Gradients/08-Derivatives-and-Gradients.md)
- The theory of the loop (convergence, learning-rate bounds, SGD, momentum): [13 Gradient Descent Theory](../../01.%20Foundations/13-Gradient-Descent-Theory/13-Gradient-Descent-Theory.md)
- The classification loss, derived in full: [23 Cross-Entropy and KL Divergence](../../01.%20Foundations/23-Cross-Entropy-and-KL-Divergence/23-Cross-Entropy-and-KL-Divergence.md)
- Next concept — why low training loss isn't enough: [05 Overfitting and Underfitting](../05-Overfitting-and-Underfitting/05-Overfitting-and-Underfitting.md)
- Put it all together on a real dataset: [12 Your First ML Project](../12-Your-First-ML-Project/12-Your-First-ML-Project.md)
- Go deeper — backprop & optimizers: [05. Deep Learning](../../05.%20Deep_Learning/README.md)

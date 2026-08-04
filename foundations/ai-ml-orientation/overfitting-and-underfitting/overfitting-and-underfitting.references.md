---
id: "00-basics/overfitting-and-underfitting/references"
topic: "Overfitting & Underfitting — References"
parent: "00-basics/overfitting-and-underfitting"
type: references
updated: 2026-07-03
---

# Overfitting & Underfitting — references and further reading

> Companion link library for **[Overfitting & Underfitting](overfitting-and-underfitting.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Everything here is free / open, and every "Source / derivation" citation on the concept page appears below.

**Start here — suggested path**:
1. **Feel the tradeoff** — play with [MLU-Explain: The Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/) (**Amazon**). *Drag model complexity and watch bias fall while variance rises — the crossover this chapter measures, animated under your cursor.*
2. **Get the framework** — watch [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) (**StatQuest**). *The tradeoff that explains why over- and under-fitting happen, in six clear minutes.*
3. **Read it precisely** — [Google ML Crash Course — Overfitting & Generalization](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) (**Google**). *Generalization, the train/validation gap, and why simpler can be better, with widgets.*
4. **Learn the cures** — watch [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) + [Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) (**StatQuest**). *Regularization and CV — your two main weapons against overfitting.*
5. **See it in code** — run [scikit-learn: Underfitting vs Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) (**scikit-learn**). *Fit too-simple / too-complex / just-right models and watch the curves — the same experiment as this chapter's module.*

**Videos**:
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — the canonical explanation of the tradeoff that underlies over/under-fitting.
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest** — how the L2 penalty shrinks weights and trades a little bias for less variance; the cure, visually.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest** — how to detect overfitting honestly with train/validation/test and k-fold before it bites you.
- [Regularization in a Neural Network | Andrew Ng](https://www.youtube.com/watch?v=6g0t3Phly2M) — **Andrew Ng / DeepLearning.AI** — bias/variance diagnosis and regularization from the deep-learning course, from first principles.
- [The Bias Variance Trade-off (Learning From Data, Lecture 8)](https://www.youtube.com/watch?v=zrEyxfl2-a8) — **Yaser Abu-Mostafa (Caltech CS156)** — the rigorous, geometry-driven derivation of the bias-variance decomposition; the deepest of the videos here.
- [Overfitting, Underfitting, and Bad Data](https://www.youtube.com/watch?v=0RT2Q0qwXSA) — **IBM Technology** — what goes wrong and how to spot it, concisely.

**Courses (free)**:
- [Google ML Crash Course — Overfitting & Generalization](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) — **Google** — the train/validation gap, generalization, and regularization, with interactive widgets.
- [Machine Learning Specialization — Course 1 (free to audit)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — bias/variance diagnosis and regularization, taught from scratch.
- [Learning From Data (CS156)](https://work.caltech.edu/telecourse.html) — **Yaser Abu-Mostafa (Caltech)** — a free full course; the bias-variance and generalization lectures are the theoretical gold standard.
- [Kaggle Learn — Intro to ML (Underfitting & Overfitting)](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — a hands-on lesson tuning model complexity to the sweet spot.

**Interactive & visual**:
- [MLU-Explain: The Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/) — **Amazon** — a beautiful interactive: drag complexity, watch bias fall and variance rise and the U form.
- [Underfitting vs. Overfitting (scikit-learn)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) — **scikit-learn docs** — runnable code that fits and plots all three regimes on the same cos-curve setup this chapter uses.
- [Validation curves & learning curves](https://scikit-learn.org/stable/modules/learning_curve.html) — **scikit-learn docs** — how to plot the U-curve and the learning curve for any model, with runnable code.

**Articles / blogs (free, no paywall)**:
- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — the classic, clearest essay on the decomposition, with the bullseye diagram everyone borrows.
- [What Is Overfitting vs. Underfitting?](https://www.ibm.com/think/topics/overfitting-vs-underfitting) — **IBM** — both failure modes side by side with detection and fixes.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn docs** — train/validation/test and k-fold with runnable code.

**Key sources / derivations** (cited on the concept page):
- [Neural Networks and the Bias/Variance Dilemma](https://direct.mit.edu/neco/article/4/1/1/5620/Neural-Networks-and-the-Bias-Variance-Dilemma) — **Geman, Bienenstock & Doursat (1992), *Neural Computation***  — the paper that crystallised the bias-variance decomposition for machine learning.
- [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) — **Hoerl & Kennard (1970), *Technometrics*** — the origin of L2 (ridge) regularization; the title names the bias-for-variance trade.
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — "overfitting has many faces"; the clearest short framing of generalization vs. memorization.
- [Reconciling modern machine-learning practice and the bias–variance trade-off](https://arxiv.org/abs/1812.11118) — **Belkin et al. (2019)** — "double descent": where the classic U-curve bends back down in heavily overparameterised models.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — **Srivastava et al. (2014)** — the canonical variance-reducing regularizer for deep nets.

**Books (free, with chapters)**:
- [An Introduction to Statistical Learning — Ch. 2.2 "Assessing Model Accuracy" & Ch. 5 "Resampling", Ch. 6.2 "Shrinkage"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free PDF; the definitive beginner treatment of the bias-variance trade-off, cross-validation, and ridge/lasso. The source for this chapter's derivations.
- [Deep Learning — Ch. 5.4 "Estimators, Bias and Variance" & Ch. 7 "Regularization"](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — the modern reference for the decomposition and the full menu of regularizers.
- [Dive into Deep Learning — Ch. 3.6 "Generalization" & 3.7 "Weight Decay"](https://d2l.ai/chapter_linear-regression/generalization.html) — **Zhang et al.** — underfitting/overfitting and L2 regularization built from scratch with runnable experiments.
- [Neural Networks and Deep Learning — Ch. 3 "Overfitting and regularization"](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — free; intuition plus the standard fixes, beautifully explained.

**In this platform**:
- Concept page (full explanation): [Overfitting & Underfitting](overfitting-and-underfitting.md)
- Previous concept — how the training loss falls in the first place: [04 How Models Learn](../how-models-learn/how-models-learn.md)
- The classification loss used across the repo, derived: [23 Cross-Entropy and KL Divergence](../../mathematical-foundations/cross-entropy-and-kl-divergence/cross-entropy-and-kl-divergence.md)
- The theory of the training loop (convergence, SGD, why the loss falls): [13 Gradient Descent Theory](../../mathematical-foundations/gradient-descent-theory/gradient-descent-theory.md)
- Put it all together on a real dataset with a train/validation split: [12 Your First ML Project](../../programming-and-data-foundations/your-first-ml-project/your-first-ml-project.md)
- Go deeper — ridge, lasso, and regularization in linear models: [03. Supervised Learning](../../../core-machine-learning/supervised-learning/README.md)
- Go deeper — dropout, weight decay, early stopping, augmentation: [05. Deep Learning](../../../deep-learning/README.md)

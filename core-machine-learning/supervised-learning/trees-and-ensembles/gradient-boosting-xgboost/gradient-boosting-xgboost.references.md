---
id: "03-supervised-learning/gradient-boosting/references"
topic: "Gradient Boosting — References"
parent: "03-supervised-learning/gradient-boosting"
type: references
updated: 2026-07-03
---

# Gradient Boosting — references and further reading

> Companion link library for **[Gradient Boosting (XGBoost · LightGBM · CatBoost)](gradient-boosting-xgboost.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Gradient Boost Part 1: Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc) (**StatQuest**). *Trees fitting residuals one after another — the whole idea.*
2. **See why it works** — read [How to explain gradient boosting](https://explained.ai/gradient-boosting/) (**Parr & Howard**). *The "gradient descent in function space" view, with pictures.*
3. **Get the math** — watch [Gradient Boost Part 2: Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs) (**StatQuest**) + read [ESL Ch. 10](https://hastie.su.domains/ElemStatLearn/). *Pseudo-residuals, the additive model, shrinkage.*
4. **Read the sources** — [Friedman: Greedy Function Approximation](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) → [XGBoost paper](https://arxiv.org/abs/1603.02754). *The original algorithm, then XGBoost's 2nd-order objective + regularization.*
5. **Make it concrete** — read the [XGBoost "Introduction to Boosted Trees"](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) tutorial and tune a model on [Kaggle's Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning).

**Videos**:
- [Gradient Boost Part 1: Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc) — **StatQuest (Josh Starmer)** — the clearest "fit trees to residuals" intuition.
- [Gradient Boost Part 2: Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs) — **StatQuest (Josh Starmer)** — pseudo-residuals, learning rate, leaf outputs.
- [Gradient Boost Part 3: Classification](https://www.youtube.com/watch?v=jxuNLH5dXCs) — **StatQuest (Josh Starmer)** — boosting for classification via log-odds and log-loss.
- [Gradient Boost Part 4: Classification Details](https://www.youtube.com/watch?v=StWY5QWMXCw) — **StatQuest (Josh Starmer)** — the log-odds bookkeeping and Newton leaf values for classification boosting, worked out.
- [XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E) — **StatQuest (Josh Starmer)** — how XGBoost builds regression trees from similarity scores and gain, from scratch.
- [XGBoost Part 3: Mathematical Details](https://www.youtube.com/watch?v=ZVFeW798-2I) — **StatQuest (Josh Starmer)** — the 2nd-order Taylor objective behind XGBoost, derived.

**Interactive & visual**:
- [How to explain gradient boosting](https://explained.ai/gradient-boosting/) — **Terence Parr & Jeremy Howard** — the best free visual deep-dive: residuals → gradient → function-space descent, with full derivations and figures.

**Courses (free)**:
- [Kaggle Learn — Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) — **Kaggle** — hands-on XGBoost, early stopping, and tuning; the fastest path to using it well.
- [Google ML Crash Course — Decision Forests (Gradient Boosting)](https://developers.google.com/machine-learning/decision-forests) — **Google** — free mini-course covering trees → forests → gradient-boosted trees.

**Articles / blogs (free, no paywall)**:
- [Introduction to Boosted Trees (XGBoost docs)](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) — **XGBoost** — the official, math-grounded tutorial on the regularized objective.
- [LightGBM — Features](https://lightgbm.readthedocs.io/en/stable/Features.html) — **Microsoft / LightGBM** — what makes it fast: histograms, leaf-wise growth, GOSS, EFB.
- [CatBoost — Documentation](https://catboost.ai/docs/en/) — **Yandex / CatBoost** — ordered boosting and categorical-feature handling, by the authors.

**Key papers**:
- [A Decision-Theoretic Generalization of On-Line Learning (AdaBoost)](https://cseweb.ucsd.edu/~yfreund/papers/adaboost.pdf) — **Freund & Schapire (1997)** — the original boosting algorithm; the ancestor gradient boosting generalizes.
- [Additive Logistic Regression: A Statistical View of Boosting](https://projecteuclid.org/journals/annals-of-statistics/volume-28/issue-2/Additive-logistic-regression-a-statistical-view-of-boosting/10.1214/aos/1016218223.full) — **Friedman, Hastie & Tibshirani (2000)** — shows AdaBoost = forward stagewise additive modeling under exponential loss (the bridge to gradient boosting).
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) — **Friedman (2001)** — the original algorithm; "gradient descent in function space." The primary source for the pseudo-residual = negative-gradient derivation and the additive update.
- [Stochastic Gradient Boosting](https://www.sciencedirect.com/science/article/abs/pii/S0167947301000652) — **Friedman (2002)** — adds row subsampling (`subsample` < 1) to boosting: a bagging-like randomization that reduces variance and often improves generalization.
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — **Chen & Guestrin (2016)** — 2nd-order objective, regularization, sparsity-aware splits, system design.
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — **Ke et al. (2017)** — histogram binning + leaf-wise growth for speed at scale.
- [CatBoost: Unbiased Boosting with Categorical Features](https://arxiv.org/abs/1706.09516) — **Prokhorenkova et al. (2018)** — ordered boosting + native categorical handling.
- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874) — **Lundberg & Lee (2017)** — Shapley-value attributions; TreeSHAP is the standard way to explain boosted trees.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) — **Grinsztajn, Oyallon & Varoquaux (2022)** — the careful benchmark explaining why boosted trees still win on tabular data.

**Books (free chapters)**:
- [The Elements of Statistical Learning — Ch. 10 "Boosting and Additive Trees"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the definitive treatment (AdaBoost → gradient boosting → regularization).
- [An Introduction to Statistical Learning (ISLR) — Ch. 8.2 "Bagging, Random Forests, Boosting"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied, intuitive version with labs.

**In this platform**:
- Concept page (full explanation): [Gradient Boosting](gradient-boosting-xgboost.md)
- Runnable code: [gradient_boosting.py](code/gradient_boosting.py) — the from-scratch boosting loop, the pseudo-residual = negative-gradient check, the sklearn verification, the staged train/validation curve, the learning-rate sweep, the residual-shrinking movie, the XGBoost leaf-weight/gain worked example, the log-loss classifier, and the model comparison (real California Housing / Breast Cancer, real XGBoost) · [10-Gradient-Boosting-XGBoost.ipynb](code/gradient-boosting-xgboost.ipynb) — the 13-step, run-live notebook
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../../../AI-ML-intuition/architectural-mechanisms/composition/bagging-and-boosting-intuition.md) · [2.05 Gradient Descent & SGD](../../../../../AI-ML-intuition/learning-and-optimization/first-order-optimization/gradient-descent-and-stochastic-gradient-descent-intuition.md)
- Base learner and siblings: [Decision Trees](../decision-trees/decision-trees.md) (the weak learner boosting adds) · [Bagging](../bagging/bagging.md) / [Random Forests](../random-forests/random-forests.md) (↓ variance by averaging — the mirror image of boosting's ↓ bias) · [Bias–Variance Tradeoff](../../../model-selection-and-evaluation/bias-variance-tradeoff/bias-variance-tradeoff.md) (why each ensemble is configured as it is)
- Why overfitting / early stopping matters: [Overfitting & Underfitting](../../../../foundations/ai-ml-orientation/overfitting-and-underfitting/overfitting-and-underfitting.md) — the U-curve the staged train/validation curve traces (early-stop at its minimum)
- Downstream: [Deployment & MLOps](../../../../deployment-and-mlops/README.md) — serving, monitoring, and versioning boosted-tree models in production
- Math prerequisites: [01. Foundations](../../../../foundations/mathematical-foundations/README.md) — gradient descent, Taylor expansion, loss functions

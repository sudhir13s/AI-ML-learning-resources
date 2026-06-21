---
id: "03-supervised-learning/stacking-blending/references"
topic: "Stacking & Blending — References"
parent: "03-supervised-learning/stacking-blending"
type: references
updated: 2026-06-22
---

# Stacking & Blending — references and further reading

> Companion link library for **[Stacking & Blending](11-Stacking-and-Blending.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Ensemble Learning: Bagging, Boosting & Stacking in 4 minutes](https://www.youtube.com/watch?v=eLt4a8-316E) (**AssemblyAI**). *See where stacking sits among ensembles: a meta-model on top of base models.*
2. **See it built** — watch [Stacking (Ensemble Methods, L07.7)](https://www.youtube.com/watch?v=8T2emza6g80) (**Sebastian Raschka**). *A clear academic walkthrough of the two-level architecture and out-of-fold predictions.*
3. **Get the math** — read [ESL Ch. 8.8 "Model Averaging and Stacking"](https://hastie.su.domains/ElemStatLearn/). *Why out-of-fold predictions and a simple meta-learner avoid overfitting.*
4. **Read the source** — skim [Wolpert: Stacked Generalization](https://www.semanticscholar.org/paper/Stacked-generalization-Wolpert/82e34a7d1c7e58e6a9b32cf45fae67aca308e02a) and [Breiman: Stacked Regressions](https://link.springer.com/article/10.1007/BF00117832). *The original idea and the regression form with non-negativity constraints.*
5. **Make it concrete** — implement with [scikit-learn `StackingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization). *Stack diverse base models under a logistic-regression meta-learner; verify it beats each alone.*

**Videos**:
- [Stacking (Ensemble Methods, L07.7)](https://www.youtube.com/watch?v=8T2emza6g80) — **Sebastian Raschka** — the clearest academic walkthrough of the meta-learner and out-of-fold predictions.
- [Ensemble Learning: Bagging, Boosting & Stacking in 4 minutes](https://www.youtube.com/watch?v=eLt4a8-316E) — **AssemblyAI** — the fastest map of where stacking fits among ensembles.
- [Stacking Classifier — Ensemble Classifiers](https://www.youtube.com/watch?v=sBrQnqwMpvA) — **Bhavesh Bhatt** — a hands-on code build of a stacking ensemble end to end.
- [StatQuest: AdaBoost, and the ensemble big picture](https://www.youtube.com/watch?v=LsK-xG1cLYA) — **StatQuest (Josh Starmer)** — the boosting contrast that frames why stacking adds a third, learned-combination strategy.

**Courses (free)**:
- [CS229: Machine Learning — Lecture notes (Ensembles)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — ensembling and model combination in the bias–variance frame.
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — the ensemble context (bagging/boosting) stacking builds on, free.
- [Kaggle Learn — Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) — **Kaggle** — the competition workflow where stacking and blending shine.

**Articles / blogs (free, no paywall)**:
- [Ensemble methods — Stacked generalization (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization) — **scikit-learn** — the practical reference: `StackingClassifier`/`StackingRegressor`, `cv`, `final_estimator`, `passthrough`.
- [Stacking Ensemble Machine Learning with Python](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) — **Jason Brownlee** — a free, end-to-end worked example of stacking and blending.
- [Blending Ensemble Machine Learning with Python](https://machinelearningmastery.com/blending-ensemble-machine-learning-with-python/) — **Jason Brownlee** — the holdout-based blending variant, contrasted with k-fold stacking.
- [The Bias–Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html) — **Scott Fortmann-Roe** — the error-decomposition lens explaining why combining diverse models helps.
- [Combine predictors using stacking (worked example)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html) — **scikit-learn** — the official end-to-end stacking example, with code and measured gains.
- [Super Learner ensemble in Python](https://machinelearningmastery.com/super-learner-ensemble-in-python/) — **Jason Brownlee** — the formal "super learner" (k-fold-OOF stacking with provable guarantees), implemented from scratch.

**Key papers**:
- [Stacked Generalization](https://www.semanticscholar.org/paper/Stacked-generalization-Wolpert/82e34a7d1c7e58e6a9b32cf45fae67aca308e02a) — **Wolpert (1992)** — the original paper that introduced stacking and the level-0/level-1 framing (abstract + free PDF links).
- [Stacked Regressions](https://link.springer.com/article/10.1007/BF00117832) — **Breiman (1996)** — the regression form of stacking and the value of non-negativity constraints on the meta-weights.
- [Issues in Stacked Generalization](https://www.jair.org/index.php/jair/article/view/10228) — **Ting & Witten (1999)** — the empirical study of *why* and *when* stacking helps, and the role of probabilities as meta-features.
- [Super Learner](https://biostats.bepress.com/ucbbiostat/paper222/) — **van der Laan, Polley & Hubbard (2007)** — the formal CV-selected stack and its **oracle inequality** (cross-validated risk is asymptotically within a $(1+\varepsilon)$ factor of the best candidate, penalty $O(\log M/n)$); open U.C. Berkeley working paper.

**Books (free chapters)**:
- [The Elements of Statistical Learning — Ch. 8.8 "Model Averaging and Stacking"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment of stacking and model combination.
- [An Introduction to Statistical Learning (ISLR) — Ch. 8 "Tree-Based Methods"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — the ensemble foundation (bagging/forests/boosting) stacking sits atop, with labs.
- [Dive into Deep Learning — model combination context](https://d2l.ai/) — **Zhang et al.** — ensembling among modern methods, with runnable code.

**In this platform**:
- Concept page (full explanation): [Stacking & Blending](11-Stacking-and-Blending.md)
- The three ensemble families: [Bagging](08-Bagging.md) (average copies → ↓variance) · [Random Forests](09-Random-Forests.md) (bagging + feature randomness) · [Gradient Boosting (XGBoost)](10-Gradient-Boosting-XGBoost.md) (chain copies → ↓bias) — stacking combines *different* algorithms via a learned meta-learner.
- The heart of leak-free stacking: [Cross-Validation](13-Cross-Validation.md) — out-of-fold predictions are k-fold CV repurposed as meta-features.
- The error-decomposition lens: [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) — why decorrelated (diverse) base learners make combination pay off.
- Concept depth (the *why*): [AI-ML-intuition 3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)

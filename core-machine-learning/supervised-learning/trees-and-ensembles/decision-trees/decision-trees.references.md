---
id: "03-supervised-learning/decision-trees/references"
topic: "Decision Trees — References"
parent: "03-supervised-learning/decision-trees"
type: references
updated: 2026-07-03
---

# Decision Trees — references and further reading

> Companion link library for **[Decision Trees](decision-trees.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Decision and Classification Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) (**StatQuest**), then play with [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/). *See splits carve up the space before any impurity formula.*
2. **See why splits work** — read [Interpretable ML — "Decision Tree"](https://christophm.github.io/interpretable-ml-book/tree.html). *Gini/entropy and information gain made concrete, plus why trees are interpretable.*
3. **Get the math** — read [ISLR Ch. 8.1](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 9.2](https://hastie.su.domains/ElemStatLearn/). *Greedy recursive binary splitting, pruning, regression vs classification trees.*
4. **Read the source** — [Induction of Decision Trees (ID3)](https://hunch.net/~coms-4771/quinlan.pdf). *Quinlan's original information-gain algorithm; CART (Gini) is the scikit-learn variant.*
5. **Make it concrete** — fit and visualize one with [scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html); watch `max_depth` trade bias for variance.

**Videos**:
- [Decision and Classification Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=_L39rN6gz7Y) — **StatQuest (Josh Starmer)** — the gentle, from-scratch build of a classification tree with Gini.
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) — **StatQuest (Josh Starmer)** — the original deep-dive on splitting, impurity, and overfitting.
- [Regression Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=g9c66TUylZ4) — **StatQuest (Josh Starmer)** — how trees handle continuous targets via variance reduction.
- [Decision Tree Classification, Clearly Explained!](https://www.youtube.com/watch?v=ZVR2Way4nwQ) — **Normalized Nerd** — a clean visual walkthrough of CART splitting and Gini, with a worked example.

**Interactive & visual**:
- [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/) — **Amazon (Jared Wilber)** — fully interactive: grow the tree, watch splits and impurity update live.
- [How to visualize decision trees](https://explained.ai/decision-tree-viz/) — **Terence Parr & Prince Grover** — a deep, visual essay (the dtreeviz library) on what a tree actually computes.

**Courses (free)**:
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — builds your first decision tree, underfitting/overfitting, in code.
- [CS229: Machine Learning — Decision Trees notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — concise, rigorous coverage of splitting criteria and tree learning.
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — a full free mini-course on trees → forests → boosting.

**Articles / blogs (free, no paywall)**:
- [Interpretable ML — Decision Tree](https://christophm.github.io/interpretable-ml-book/tree.html) — **Christoph Molnar** — the clearest free explanation of splits, impurity, and interpretability.
- [Decision Trees (scikit-learn user guide)](https://scikit-learn.org/stable/modules/tree.html) — **scikit-learn** — the practical reference: criteria, pruning, and the math of the splits.

**Key papers**:
- [Induction of Decision Trees (ID3)](https://hunch.net/~coms-4771/quinlan.pdf) — **Quinlan (1986)** — the original information-gain (entropy) tree-induction algorithm.
- [Induction of Decision Trees (Springer)](https://link.springer.com/article/10.1007/BF00116251) — **Quinlan (1986)** — the same landmark paper, journal version; CART (Breiman et al., 1984) is the Gini-based sibling used by scikit-learn.
- [C4.5: Programs for Machine Learning (review)](https://link.springer.com/article/10.1007/BF00993309) — **Quinlan (1993)** — ID3's successor: gain ratio (fixes the high-cardinality bias), continuous features, missing values, and post-pruning.
- [Constructing Optimal Binary Decision Trees is NP-complete](https://people.csail.mit.edu/rivest/pubs/HR76.pdf) — **Hyafil & Rivest (1976)** — the hardness result (author-hosted PDF) that forces trees to be grown greedily rather than globally optimized.
- [Permutation importance vs impurity-based (MDI) bias](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) — **scikit-learn** — shows impurity importance inflating a high-cardinality feature, and how permutation importance corrects it.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 8 "Tree-Based Methods"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — trees → bagging → forests → boosting, applied with labs.
- [The Elements of Statistical Learning — Ch. 9.2 "Tree-Based Methods"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous CART treatment (splitting, pruning, instability).
- [Interpretable Machine Learning — "Decision Tree"](https://christophm.github.io/interpretable-ml-book/tree.html) — **Christoph Molnar** — free online book; trees as the canonical interpretable model.

**In this platform**:
- Concept page (full explanation): [Decision Trees](decision-trees.md)
- Runnable code: [decision_trees.py](code/decision_trees.py) — the from-scratch tree, Gini/entropy, best-split search, sklearn verification, depth sweep, importance, and regression staircase (real Iris / Breast Cancer / Diabetes) · [07-Decision-Trees.ipynb](code/decision-trees.ipynb) — the 13-step, run-live notebook
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../../../AI-ML-intuition/objectives-and-evaluation/generalization/bias-variance-tradeoff-intuition.md) · [3.08 Ensembles (Bagging/Boosting)](../../../../../AI-ML-intuition/architectural-mechanisms/composition/bagging-and-boosting-intuition.md)
- Why no scaling (scale-invariance): [Feature Scaling & Normalization](../../../../foundations/data-preparation/feature-scaling-and-normalization/feature-scaling-and-normalization.md) — trees split on thresholds, so a monotone rescale leaves every split unchanged (the random forest there is the scale-invariant control)
- Why overfitting matters: [Overfitting & Underfitting](../../../../foundations/ai-ml-orientation/overfitting-and-underfitting/overfitting-and-underfitting.md) — the U-curve the depth sweep traces
- Trees as the base learner of ensembles: [Bias–Variance Tradeoff](../../../model-selection-and-evaluation/bias-variance-tradeoff/bias-variance-tradeoff.md) (why a single tree is high-variance) · [Bagging](../bagging/bagging.md) / [Random Forests](../random-forests/random-forests.md) (↓ variance) · [Gradient Boosting (XGBoost)](../gradient-boosting-xgboost/gradient-boosting-xgboost.md) (↓ bias)
- Math prerequisites: [01. Foundations](../../../../foundations/mathematical-foundations/README.md) — entropy, information gain, Gini impurity

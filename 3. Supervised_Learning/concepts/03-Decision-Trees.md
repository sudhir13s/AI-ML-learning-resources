---
id: "03-supervised-learning/decision-trees"
topic: "Decision Trees"
parent: "03-supervised-learning"
level: beginner
prereqs: ["entropy", "gini-impurity", "supervised-learning-basics"]
interview_frequency: very-high
updated: 2026-06-19
---

# Decision Trees
> Recursively split the feature space with axis-aligned questions ("is feature ≤ t?"), choosing each
> split to most reduce impurity (Gini / entropy for classification, variance for regression), until
> leaves are pure enough. A single, fully interpretable model — and the building block of Random
> Forests and Gradient Boosting.

**Why it matters:** the gateway to all of tree-based ML, which still dominates tabular data. Expect to
explain how a split is chosen (information gain / Gini), why trees overfit and how pruning / `max_depth`
/ `min_samples_leaf` control it, why they need no feature scaling, and the bias–variance reason a
*single* deep tree is high-variance (motivating ensembles).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Decision and Classification Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y), then play with [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/). *See splits carve up the space before any impurity formula.*
2. **See why splits work** — read [Interpretable ML — "Decision Tree"](https://christophm.github.io/interpretable-ml-book/tree.html). *Gini/entropy and information gain made concrete, plus why trees are so interpretable.*
3. **Get the math** — read [ISLR Ch. 8.1 "The Basics of Decision Trees"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 9.2 "Tree-Based Methods"](https://hastie.su.domains/ElemStatLearn/). *Greedy recursive binary splitting, cost-complexity pruning, and regression vs classification trees.*
4. **Read the sources** — [Induction of Decision Trees (ID3)](https://hunch.net/~coms-4771/quinlan.pdf). *Quinlan's original information-gain algorithm; CART (Gini) is the scikit-learn variant.*
5. **Make it concrete** — implement with [scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html), then visualize one. *Fit on a tiny dataset, plot the tree, and watch `max_depth` trade bias for variance.*

## 🎓 Courses (free)
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — builds your first decision tree, underfitting/overfitting, in code, free.
- [CS229: Machine Learning — Decision Trees notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — concise, rigorous coverage of splitting criteria and tree learning.
- [Google ML Crash Course — Decision Forests](https://developers.google.com/machine-learning/decision-forests) — **Google** — a full free mini-course on trees → forests → boosting.

## 🎥 Videos
- [Decision and Classification Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=_L39rN6gz7Y) — **StatQuest (Josh Starmer)** — the gentle, from-scratch build of a classification tree with Gini.
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) — **StatQuest (Josh Starmer)** — the original deep-dive on splitting, impurity, and overfitting.
- [Regression Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=g9c66TUylZ4) — **StatQuest (Josh Starmer)** — how trees handle continuous targets via variance reduction.
- [Decision Tree Classification, Clearly Explained!](https://www.youtube.com/watch?v=ZVR2Way4nwQ) — **Normalized Nerd** — a clean visual walkthrough of CART splitting and Gini, with a worked example.

## 📄 Key Papers
- [Induction of Decision Trees (ID3)](https://hunch.net/~coms-4771/quinlan.pdf) — **Quinlan (1986)** — the original information-gain tree-induction algorithm; foundational, free PDF.
- [Induction of Decision Trees (Springer)](https://link.springer.com/article/10.1007/BF00116251) — **Quinlan (1986)** — the same landmark paper, publisher page; CART (Breiman et al., 1984) is the Gini-based sibling used by scikit-learn.

## 📰 Articles / Blogs (free, no paywall)
- [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/) — **Amazon (Jared Wilber)** — fully interactive: grow the tree, watch splits and impurity update live.
- [Interpretable ML — Decision Tree](https://christophm.github.io/interpretable-ml-book/tree.html) — **Christoph Molnar** — the clearest free explanation of splits, impurity, and interpretability.
- [How to visualize decision trees](https://explained.ai/decision-tree-viz/) — **Terence Parr & Prince Grover** — deep, visual, free essay on what a tree actually computes.
- [Decision Trees (scikit-learn user guide)](https://scikit-learn.org/stable/modules/tree.html) — **scikit-learn** — the practical reference: criteria, pruning, and the math of the splits.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 8 "Tree-Based Methods"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — trees → bagging → forests → boosting, applied with labs.
- [The Elements of Statistical Learning — **Ch. 9.2 "Tree-Based Methods"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous CART treatment (splitting, pruning, instability).
- [Interpretable Machine Learning — **"Decision Tree" chapter**](https://christophm.github.io/interpretable-ml-book/tree.html) — **Christoph Molnar** — free online book; trees as the canonical interpretable model.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md) · [3.08 Ensembles (Bagging/Boosting)](../../../AI-ML-intuition/Module_3_Evaluation/3.08_Ensembles_Bagging_Boosting.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — entropy, information gain, Gini impurity.
- Next concepts: [04 Gradient Boosting (XGBoost)](04-Gradient-Boosting-XGBoost.md) · Random Forests *(coming soon)* · Bagging *(coming soon)* · Bias–Variance Tradeoff *(coming soon)*
- Related domain: [2. Data Preprocessing](../../2.%20Data_Preprocessing/README.md) — trees need no scaling but care about encoding and leakage.

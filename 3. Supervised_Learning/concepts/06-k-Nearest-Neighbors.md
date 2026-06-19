---
id: "03-supervised-learning/k-nearest-neighbors"
topic: "k-Nearest Neighbors (k-NN)"
parent: "03-supervised-learning"
level: beginner
prereqs: ["distance-metrics", "supervised-learning-basics"]
interview_frequency: high
updated: 2026-06-19
---

# k-Nearest Neighbors (k-NN)
> Classify (or regress) a point by looking at its `k` closest training examples and taking a majority
> vote (or average). No training step — the model *is* the data. The canonical lazy, instance-based,
> non-parametric algorithm, and the cleanest introduction to distance metrics and the curse of dimensionality.

**Why it matters:** a favorite interview warm-up that probes whether you understand bias–variance,
distance metrics, and scaling. Expect: how `k` trades bias for variance (small `k` = high variance),
why features *must* be standardized, why Euclidean distance breaks down in high dimensions (the curse
of dimensionality), and the prediction-time cost (no training, expensive inference) that motivates
KD-trees / ball-trees and approximate nearest-neighbor search.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: k-Nearest Neighbors](https://www.youtube.com/watch?v=HVXime0nQeI). *See the vote among neighbors decide the label — the whole algorithm in one video.*
2. **See `k` in action** — play with [scikit-learn k-NN docs + plots](https://scikit-learn.org/stable/modules/neighbors.html). *Watch the decision boundary go from jagged (small `k`, high variance) to smooth (large `k`, high bias).*
3. **Get the math** — read [ISLR Ch. 2.2.3 + Ch. 4.7.6 "KNN"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [ESL Ch. 13.3](https://hastie.su.domains/ElemStatLearn/). *The Bayes-classifier connection and why k-NN approximates it as data grows.*
4. **Understand the failure mode** — read about the [curse of dimensionality (scikit-learn / ESL Ch. 2.5)](https://hastie.su.domains/ElemStatLearn/). *Why all points become equidistant in high dimensions, and why scaling and dimensionality reduction help.*
5. **Make it concrete** — implement with [scikit-learn `KNeighborsClassifier`](https://scikit-learn.org/stable/modules/neighbors.html). *Standardize features, sweep `k` with cross-validation, and compare Euclidean vs Manhattan distance.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 1](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — covers k-NN among the foundational classifiers; free to audit.
- [CS229: Machine Learning — Lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — non-parametric methods and the bias–variance view, rigorously.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on first classifiers and over/underfitting, in code, free.

## 🎥 Videos
- [StatQuest: K-nearest neighbors, Clearly Explained](https://www.youtube.com/watch?v=HVXime0nQeI) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for the neighbor vote.
- [K Nearest Neighbors (KNN) — visual walkthrough](https://www.youtube.com/watch?v=0Lt9w-BxKFQ) — **Normalized Nerd** — a clean visual build of the algorithm and decision boundary.
- [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — how you actually pick `k` without fooling yourself.
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — why small `k` is high-variance and large `k` is high-bias.

## 📄 Key Papers
- [Nearest Neighbor Pattern Classification](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf) — **Cover & Hart (1967)** — the foundational paper proving the 1-NN error is at most twice the Bayes error; author-hosted PDF.
- [An Algorithm for Finding Best Matches in Logarithmic Expected Time (KD-trees)](http://i.stanford.edu/pub/cstr/reports/cs/tr/75/482/CS-TR-75-482.pdf) — **Friedman, Bentley & Finkel (1977)** — how to make nearest-neighbor search fast; Stanford tech-report PDF, free.

## 📰 Articles / Blogs (free, no paywall)
- [Nearest Neighbors (scikit-learn user guide)](https://scikit-learn.org/stable/modules/neighbors.html) — **scikit-learn** — the practical reference: distance metrics, KD-tree/ball-tree, weighting.
- [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) — **Amazon** — how to actually score a k-NN classifier once you've built it.
- [The Curse of Dimensionality (ESL excerpt, Ch. 2.5)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — why distance-based methods degrade as dimensions grow.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 2.2.3 & 4.7.6 (KNN)**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — k-NN as the running example for bias–variance and the Bayes boundary.
- [The Elements of Statistical Learning — **Ch. 13.3 "k-Nearest-Neighbor Classifiers"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment, including the curse of dimensionality.
- [Dive into Deep Learning — **Ch. 19.5 (k-NN background)**](https://d2l.ai/) — **Zhang et al.** — instance-based learning framed alongside modern methods, with code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine Distance](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) · [1.09 Manhattan (L1) Distance](../../../AI-ML-intuition/Module_1_Representation/1.09_Manhattan_L1_Distance.md) · [1.10 Mahalanobis Distance](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — distance metrics, norms, the curse of dimensionality.
- Related concepts: Bias–Variance Tradeoff *(coming soon)* · Cross-Validation *(coming soon)* — how you pick `k`.
- Related domain: [2. Data Preprocessing](../../2.%20Data_Preprocessing/README.md) — k-NN is acutely sensitive to feature scaling and encoding.

---
id: "04-unsupervised-learning/anomaly-outlier-detection"
topic: "Anomaly / Outlier Detection (Isolation Forest · LOF · One-Class SVM)"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["k-means", "dbscan", "decision-trees", "kernel-methods"]
interview_frequency: high
updated: 2026-06-19
---

# Anomaly / Outlier Detection
> Flag the rare points that don't fit the bulk of the data — fraud, defects, intrusions, sensor faults.
> Three workhorse unsupervised approaches: **Isolation Forest** (anomalies are easy to isolate with
> random splits), **Local Outlier Factor** (compare a point's density to its neighbors'), and
> **One-Class SVM** (learn a boundary around "normal").

**Why it matters:** the practical "how would you catch fraud with no labels?" question. Interviews probe
the framing (unsupervised vs semi-supervised "novelty" detection), why **Isolation Forest** is the
go-to (linear time, handles high dimensions, no distance metric needed — anomalies need fewer splits to
isolate), how **LOF** catches *local* outliers that global methods miss (a point loose relative to its
neighborhood, even in a sparse region), where **One-Class SVM** fits (a kernelized boundary, sensitive
to scaling and `ν`), and the evaluation trap: with imbalanced data, accuracy lies — use precision/recall
and PR-AUC.

**⭐ Start here — suggested path:**

1. **Frame the problem** — watch [Unsupervised Anomaly Detection Explained](https://www.youtube.com/watch?v=QZNEJHbophM). *Where clustering, density, and isolation methods each apply.*
2. **Learn the go-to method** — [Isolation Forest: a tree-based approach for outlier detection](https://www.youtube.com/watch?v=kqAxfOPlr1U). *Why random partitioning isolates anomalies in fewer splits — the key intuition.*
3. **See local density** — [Local Outlier Factor, explained with an example](https://www.youtube.com/watch?v=8W3mTEKTORg). *Why a point can be normal globally but an outlier relative to its neighbors.*
4. **Read the sources** — [Isolation Forest (Liu, Ting & Zhou, 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) → [LOF (Breunig et al., 2000)](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf). *The path-length and local-density definitions in their original form.*
5. **Make it concrete** — code all three with the [scikit-learn outlier detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html) and compare them on the [anomaly-comparison demo](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html). *Seeing where each fails cements the choice.*

## 🎓 Courses (free)
- [scikit-learn — Novelty and Outlier Detection user guide](https://scikit-learn.org/stable/modules/outlier_detection.html) — **scikit-learn** — Isolation Forest, LOF, One-Class SVM, and Elliptic Envelope side by side, with code.
- [scikit-learn — LOF outlier detection example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html) — **scikit-learn** — a runnable LOF walkthrough doubling as a focused mini-lesson.

## 🎥 Videos
- [Unsupervised Anomaly Detection Explained: Clustering, Density & Isolation Forest](https://www.youtube.com/watch?v=QZNEJHbophM) — **Data Science Garage** — the landscape: which family to reach for and why.
- [Isolation Forest: A Tree-Based Approach for Outlier Detection](https://www.youtube.com/watch?v=kqAxfOPlr1U) — **Pratik Nabriya** — the isolation intuition and the path-length score, clearly explained.
- [Local Outlier Factor (LOF) — Explained with Example](https://www.youtube.com/watch?v=8W3mTEKTORg) — **Endless Engineering** — local reachability density worked through step by step.
- [Unsupervised Anomaly Detection with Isolation Forest](https://www.youtube.com/watch?v=5p8B2Ikcw-k) — **Elena Sharova (PyData London)** — a practitioner's end-to-end talk with real data and pitfalls.

## 📄 Key Papers
- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) — **Liu, Ting & Zhou (2008)** — the linear-time, distance-free method that's the modern default.
- [LOF: Identifying Density-Based Local Outliers](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) — **Breunig, Kriegel, Ng & Sander (2000)** — the local-density formulation; the origin of "local" outlier detection.
- [Support Vector Method for Novelty Detection (One-Class SVM)](https://proceedings.neurips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf) — **Schölkopf et al. (2000)** — learning a boundary around the normal region via a kernel.

## 📰 Articles / Blogs (free, no paywall)
- [Comparing anomaly detection algorithms (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html) — **scikit-learn** — the side-by-side picture of where each method succeeds and fails.
- [Outlier detection with Local Outlier Factor (scikit-learn)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html) — **scikit-learn** — copy-paste LOF recipe with score interpretation.
- [Novelty and outlier detection (scikit-learn user guide)](https://scikit-learn.org/stable/modules/outlier_detection.html) — **scikit-learn** — the practical reference distinguishing outlier vs novelty detection.

## 📚 Books (free, with chapters)
- [The Elements of Statistical Learning — **§14.3 "Cluster Analysis"** (density & outliers)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the density framing that underpins LOF-style detection.
- [Mining of Massive Datasets — **Ch. 7 "Clustering"**](http://infolab.stanford.edu/~ullman/mmds/ch7.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; clustering and the "noise"/outlier view at scale.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Distances (Euclidean vs Cosine)](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) — density and "local" outliers are defined by the distance metric
- Related: [03 DBSCAN](03-DBSCAN.md) — its "noise" label *is* a form of outlier detection · [10 Kernel Density Estimation](10-Kernel-Density-Estimation.md) — density-based scoring
- Method ties: [01 K-Means](01-K-Means-Clustering.md) (distance-to-centroid scoring) · One-Class SVM uses [the kernel trick → Foundations](../../1.%20Foundations/Maths%20for%20AI-ML/README.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../1.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

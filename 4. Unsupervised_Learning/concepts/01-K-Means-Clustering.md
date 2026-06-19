---
id: "04-unsupervised-learning/k-means"
topic: "K-Means Clustering"
parent: "04-unsupervised-learning"
level: beginner
prereqs: ["linear-algebra", "euclidean-distance", "variance"]
interview_frequency: very-high
updated: 2026-06-19
---

# K-Means Clustering
> Partition `n` points into `k` clusters by alternately assigning each point to its nearest
> centroid and recomputing each centroid as the mean of its members — minimizing within-cluster
> squared distance (inertia). The first clustering algorithm everyone learns, and still a baseline.

**Why it matters:** the canonical "explain k-means" question — the Lloyd iteration and why it always
converges (to a *local* optimum), why k-means++ initialization matters, how to pick `k` (elbow vs
silhouette), the spherical/equal-variance assumptions it bakes in, and why it fails on non-convex or
unequal-density clusters (the setup for DBSCAN and GMMs).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA), then play with [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/). *See the assign-then-update loop animate before any math.*
2. **See why it works** — [Victor Lavrenko: K-means, how it works](https://www.youtube.com/watch?v=_aWzGGNrcic). *The objective it minimizes and why each step never increases it (so it converges).*
3. **Get the math** — [CS229 notes 7a (k-means)](https://cs229.stanford.edu/notes2020spring/cs229-notes7a.pdf) + [Andrew Ng: Clustering lecture](https://www.youtube.com/watch?v=0D4LnsJr85Y). *Inertia, the optimization objective, random restarts, and choosing `k`.*
4. **Read the sources** — [Lloyd (1982)](https://ieeexplore.ieee.org/document/1056489) → [k-means++ (Arthur & Vassilvitskii, 2007)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf). *The original quantization algorithm, then the smart seeding that fixed bad initialization.*
5. **Make it concrete** — code it with the [scikit-learn KMeans guide](https://scikit-learn.org/stable/modules/clustering.html#k-means) and try [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). *Implementing the loop + choosing `k` cements it.*

## 🎓 Courses (free)
- [Machine Learning Specialization (Course 3: Unsupervised Learning)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — free to audit; the clearest structured walkthrough of k-means and choosing `k`.
- [scikit-learn — Clustering user guide (K-means)](https://scikit-learn.org/stable/modules/clustering.html#k-means) — **scikit-learn** — the practical reference: inertia, k-means++, mini-batch, and pitfalls, with runnable code.

## 🎥 Videos
- [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for the assign/update loop and picking `k`.
- [K-means clustering: how it works](https://www.youtube.com/watch?v=_aWzGGNrcic) — **Victor Lavrenko (Edinburgh)** — the objective function and *why* the iteration converges.
- [Clustering: K-means and Hierarchical](https://www.youtube.com/watch?v=QXOkPvFM6NU) — **Luis Serrano** — illustrations-over-formulas; great mental picture and the link to hierarchical methods.
- [Clustering — Lecture 13](https://www.youtube.com/watch?v=0D4LnsJr85Y) — **Andrew Ng (Stanford)** — the optimization objective, random initialization, and the elbow method in one lecture.

## 📄 Key Papers
- [Least Squares Quantization in PCM (Lloyd's algorithm)](https://ieeexplore.ieee.org/document/1056489) — **Lloyd (1982, written 1957)** — the original iterative quantization that *is* k-means.
- [Some Methods for Classification and Analysis of Multivariate Observations](https://projecteuclid.org/euclid.bsmsp/1200512992) — **MacQueen (1967)** — coins "k-means" and gives the online formulation.
- [k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) — **Arthur & Vassilvitskii (2007)** — the probabilistic seeding that makes k-means robust (now the default).

## 📰 Articles / Blogs (free, no paywall)
- [In Depth: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html) — **Jake VanderPlas (Python Data Science Handbook)** — derivation + code + failure modes, fully free.
- [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) — **Naftali Harris** — interactive playground; watch initialization and convergence live.
- [Determine the optimal value of K](https://www.geeksforgeeks.org/machine-learning/ml-determine-the-optimal-value-of-k-in-k-means-clustering/) — **GeeksforGeeks** — practical elbow + silhouette walkthrough.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLP) — **Ch. 12.4 "Clustering Methods"**](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; k-means applied with Python labs.
- [The Elements of Statistical Learning — **§14.3 "Cluster Analysis"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the rigorous treatment of k-means and its objective.
- [Mathematics for Machine Learning — **Ch. 10 (PCA) / clustering context**](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the linear-algebra footing behind distances and centroids.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.18 K-Means Clustering](../../../AI-ML-intuition/Module_1_Representation/1.18_KMeans_Clustering.md)
- Next concepts: [02 Hierarchical Clustering](02-Hierarchical-Clustering.md) · [03 DBSCAN](03-DBSCAN.md) · [04 Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../1.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

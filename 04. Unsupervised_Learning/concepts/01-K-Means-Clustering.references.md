---
id: "04-unsupervised-learning/k-means/references"
topic: "K-Means Clustering — References"
parent: "04-unsupervised-learning/k-means"
type: references
updated: 2026-06-22
---

# K-Means Clustering — references and further reading

> Companion link library for **[K-Means Clustering](01-K-Means-Clustering.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. Every link verified (HTTP 200).

**Start here — suggested path**:
1. **Build intuition** — watch [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) (**Josh Starmer**), then play with [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) (**Naftali Harris**). *See the assign-then-update loop animate before any math.*
2. **See why it works** — watch [K-means, how it works](https://www.youtube.com/watch?v=_aWzGGNrcic) (**Victor Lavrenko**). *The objective it minimizes and why each step never increases it (so it converges).*
3. **Get the math** — read [CS229 notes 7a (k-means)](https://cs229.stanford.edu/notes2020spring/cs229-notes7a.pdf) (**Stanford / Ng**) + watch [Clustering — Lecture 13](https://www.youtube.com/watch?v=0D4LnsJr85Y) (**Andrew Ng**). *Inertia, alternating minimization, restarts, and choosing k.*
4. **Read the sources** — [Lloyd (1982)](https://ieeexplore.ieee.org/document/1056489) → [k-means++ (Arthur & Vassilvitskii, 2007)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf). *The original quantization algorithm, then the smart seeding that fixed bad initialization.*
5. **Make it concrete** — code it with the [scikit-learn KMeans guide](https://scikit-learn.org/stable/modules/clustering.html#k-means) and run [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). *Implementing the loop + choosing k cements it.*

**Videos**:
- [StatQuest: K-means clustering, Clearly Explained](https://www.youtube.com/watch?v=4b5d3muPQmA) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for the assign/update loop and picking k.
- [K-means clustering: how it works](https://www.youtube.com/watch?v=_aWzGGNrcic) — **Victor Lavrenko (Edinburgh)** — the objective function and *why* the iteration converges.
- [Clustering: K-means and Hierarchical](https://www.youtube.com/watch?v=QXOkPvFM6NU) — **Luis Serrano** — illustrations-over-formulas; the clearest mental picture and the link to hierarchical methods.
- [Clustering — Lecture 13](https://www.youtube.com/watch?v=0D4LnsJr85Y) — **Andrew Ng (Stanford)** — the optimization objective, random initialization, and the elbow method in one lecture.

**Interactive & visual**:
- [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) — **Naftali Harris** — interactive playground; step through initialization and convergence live, including pathological inits.
- [K-means assumptions (failure modes)](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html) — **scikit-learn** — runnable plots of exactly when k-means breaks (anisotropic, unequal-variance, unevenly-sized blobs).
- [Mini-batch K-means vs K-means](https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html) — **scikit-learn** — side-by-side of the large-scale variant and the quality/speed trade-off.

**Courses (free)**:
- [Machine Learning Specialization — Course 3: Unsupervised Learning](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — free to audit; the clearest structured walkthrough of k-means and choosing k.
- [CS229: Machine Learning — k-means notes (7a)](https://cs229.stanford.edu/notes2020spring/cs229-notes7a.pdf) — **Stanford (Ng)** — the objective, the alternating-minimization derivation, and convergence, rigorously.

**Articles / blogs (free, no paywall)**:
- [In Depth: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html) — **Jake VanderPlas (Python Data Science Handbook)** — derivation + code + failure modes, fully free and runnable.
- [scikit-learn — Clustering user guide (K-means)](https://scikit-learn.org/stable/modules/clustering.html#k-means) — **scikit-learn** — the practical reference: inertia, k-means++, mini-batch, complexity, and pitfalls.
- [Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) — **scikit-learn** — the silhouette method worked end to end, with per-cluster plots.
- [Determine the optimal value of K](https://www.geeksforgeeks.org/machine-learning/ml-determine-the-optimal-value-of-k-in-k-means-clustering/) — **GeeksforGeeks** — a practical elbow + silhouette walkthrough with code.
- [Silhouette (clustering)](https://en.wikipedia.org/wiki/Silhouette_(clustering)) — **Wikipedia** — the $s(i) = (b-a)/\max(a,b)$ definition and properties, with references.

**Key papers**:
- [Least Squares Quantization in PCM (Lloyd's algorithm)](https://ieeexplore.ieee.org/document/1056489) — **Lloyd (1982, written 1957)** — the original iterative quantization that *is* k-means.
- [Some Methods for Classification and Analysis of Multivariate Observations](https://projecteuclid.org/euclid.bsmsp/1200512992) — **MacQueen (1967)** — coins "k-means" and gives the online (incremental) formulation.
- [k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) — **Arthur & Vassilvitskii (2007)** — the $D^2$-weighted seeding and the $O(\log k)$-competitive guarantee (now the default).
- [Estimating the number of clusters via the Gap Statistic](https://web.stanford.edu/~hastie/Papers/gap.pdf) — **Tibshirani, Walther & Hastie (2001)** — a principled, null-referenced way to choose k.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLP) — **Ch. 12.4 "Clustering Methods"**](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; k-means applied with Python labs.
- [The Elements of Statistical Learning — **§14.3 "Cluster Analysis"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the rigorous treatment of k-means, its objective, and choosing k.
- [Mathematics for Machine Learning](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the linear-algebra footing (norms, distances, means) behind centroids and the squared-distance objective.

**In this platform**:
- Concept page (full explanation): [K-Means Clustering](01-K-Means-Clustering.md)
- Concept depth (the *why*): [AI-ML-intuition 1.18 K-Means Clustering](../../../AI-ML-intuition/Module_1_Representation/1.18_KMeans_Clustering.md)
- Next concepts: [02 Hierarchical Clustering](02-Hierarchical-Clustering.md) · [03 DBSCAN](03-DBSCAN.md) · [04 Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md) · [05 Spectral Clustering](05-Spectral-Clustering.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

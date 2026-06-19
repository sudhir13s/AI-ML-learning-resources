---
id: "04-unsupervised-learning/hierarchical-clustering"
topic: "Hierarchical Clustering"
parent: "04-unsupervised-learning"
level: beginner
prereqs: ["euclidean-distance", "k-means"]
interview_frequency: high
updated: 2026-06-19
---

# Hierarchical Clustering
> Build a tree of clusters (a *dendrogram*) — either bottom-up (agglomerative: start with every point
> its own cluster and repeatedly merge the closest pair) or top-down (divisive: split one big cluster).
> No need to pre-commit to `k`; you cut the tree at the height you want.

**Why it matters:** the classic "k-means vs hierarchical" question — when you *don't* know `k`, when you
want a nested structure (taxonomies, gene expression), and the crux: **linkage** (single → chaining,
complete → compact, average, Ward → minimizes variance) and the distance metric, which together decide
the cluster shapes. Plus reading a dendrogram and the `O(n²)`–`O(n³)` cost that limits it to small `n`.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo). *See points merge into a dendrogram and how you "cut" it to get clusters.*
2. **See why it works** — [Victor Lavrenko: Agglomerative clustering, how it works](https://www.youtube.com/watch?v=XJ3194AmH40). *Why merging the closest pair each step builds a valid hierarchy.*
3. **Get the linkages** — [Hierarchical clustering — explained](https://www.youtube.com/watch?v=uWf__KIKzPQ) (TileStats) + the [IR Book Ch. 17](https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html). *Single vs complete vs average vs Ward — the choice that changes everything.*
4. **Read the source** — [Ward's minimum-variance method (Ward, 1963)](https://www.cs.cmu.edu/~roni/11761/Presentations/Ward1963.pdf). *The most-used linkage; understand the variance criterion it optimizes.*
5. **Make it concrete** — code it with [scikit-learn AgglomerativeClustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) and plot a [SciPy dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html). *Building + cutting a dendrogram cements it.*

## 🎓 Courses (free)
- [scikit-learn — Hierarchical clustering user guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) — **scikit-learn** — linkage options, connectivity constraints, and runnable examples.
- [Introduction to Information Retrieval — **Ch. 17 (Hierarchical clustering)**](https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html) — **Manning, Raghavan & Schütze (Stanford)** — free online chapter; the cleanest text treatment of linkages.

## 🎥 Videos
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) — **StatQuest (Josh Starmer)** — gentle, from-scratch build of a dendrogram and how to read it.
- [Agglomerative Clustering: how it works](https://www.youtube.com/watch?v=XJ3194AmH40) — **Victor Lavrenko (Edinburgh)** — the merge rule and why the hierarchy is well-defined.
- [Hierarchical clustering — explained](https://www.youtube.com/watch?v=uWf__KIKzPQ) — **TileStats** — clear comparison of single/complete/average/Ward linkages on real data.
- [Clustering: K-means and Hierarchical](https://www.youtube.com/watch?v=QXOkPvFM6NU) — **Luis Serrano** — illustrated contrast with k-means; great for the "which to use" question.

## 📄 Key Papers
- [Hierarchical Grouping to Optimize an Objective Function (Ward's method)](https://www.cs.cmu.edu/~roni/11761/Presentations/Ward1963.pdf) — **Ward (1963)** — the minimum-variance linkage that's the modern default.
- [Methods of Hierarchical Clustering](https://arxiv.org/abs/1105.0121) — **Murtagh & Contreras (2011)** — free survey of all the linkage algorithms and their complexity; the reference overview.

## 📰 Articles / Blogs (free, no paywall)
- [Hierarchical clustering (SciPy reference)](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) — **SciPy** — the `linkage`/`dendrogram`/`fcluster` API with worked examples.
- [Plotting a dendrogram (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html) — **scikit-learn** — copy-paste recipe to visualize the tree.
- [Comparing clustering algorithms](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods) — **scikit-learn** — the side-by-side table of when hierarchical wins vs k-means/DBSCAN.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLP) — **Ch. 12.4.2 "Hierarchical Clustering"**](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; dendrograms and linkages with Python labs.
- [The Elements of Statistical Learning — **§14.3.12 "Hierarchical Clustering"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the rigorous linkage comparison.

## 🔗 In this platform
- Compare with: [01 K-Means Clustering](01-K-Means-Clustering.md) · [03 DBSCAN](03-DBSCAN.md) · [04 Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../1.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

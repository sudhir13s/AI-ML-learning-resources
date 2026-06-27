---
id: "04-unsupervised-learning/hierarchical-clustering/references"
topic: "Hierarchical Clustering — References"
parent: "04-unsupervised-learning/hierarchical-clustering"
type: references
updated: 2026-06-22
---

# Hierarchical Clustering — references and further reading

> Companion link library for **[Hierarchical Clustering](02-Hierarchical-Clustering.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) (**Josh Starmer**). *See points merge into a dendrogram and how you "cut" it to get clusters.*
2. **See why it works** — watch [Agglomerative clustering: how it works](https://www.youtube.com/watch?v=XJ3194AmH40) (**Victor Lavrenko**). *Why merging the closest pair each step builds a valid hierarchy.*
3. **Get the linkages** — read [Introduction to Information Retrieval, Ch. 17](https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html) (**Manning, Raghavan & Schütze**). *Single vs complete vs average vs Ward — the choice that changes everything.*
4. **Read the source** — read [Ward's minimum-variance method (Ward, 1963)](https://www.cs.cmu.edu/~roni/11761/Presentations/Ward1963.pdf). *The most-used linkage; understand the variance criterion it optimizes.*
5. **Make it concrete** — code it with [scikit-learn AgglomerativeClustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) + the [SciPy hierarchy API](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html). *Build, plot, and cut a dendrogram — that cements it.*

**Videos**:
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) — **StatQuest (Josh Starmer)** — gentle, from-scratch build of a dendrogram and how to read it; the best first watch.
- [Agglomerative Clustering: how it works](https://www.youtube.com/watch?v=XJ3194AmH40) — **Victor Lavrenko (Edinburgh)** — the merge rule and why the hierarchy is well-defined.
- [Hierarchical clustering — explained](https://www.youtube.com/watch?v=uWf__KIKzPQ) — **TileStats** — clear comparison of single/complete/average/Ward linkages on real data.
- [Clustering: K-means and Hierarchical](https://www.youtube.com/watch?v=QXOkPvFM6NU) — **Luis Serrano** — illustrated contrast with k-means; great for the "which to use" question.

**Courses (free)**:
- [scikit-learn — Hierarchical clustering user guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) — **scikit-learn** — linkage options, connectivity constraints, and runnable examples; the practical reference.
- [Introduction to Information Retrieval — Ch. 17 (Hierarchical clustering)](https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html) — **Manning, Raghavan & Schütze (Stanford)** — free online chapter; the cleanest text treatment of linkages.
- [CS229: Machine Learning — main notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — unsupervised learning and clustering in the broader ML context.

**Articles / blogs (free, no paywall)**:
- [SciPy `cluster.hierarchy` reference](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) — **SciPy** — the `linkage` / `dendrogram` / `fcluster` / `cophenet` API with worked examples; the canonical implementation.
- [Plotting a dendrogram (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html) — **scikit-learn** — copy-paste recipe to visualize the tree.
- [Comparing clustering algorithms](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods) — **scikit-learn** — the side-by-side table of when hierarchical wins vs k-means/DBSCAN/GMM.
- [`scipy.cluster.hierarchy.linkage` (linkage matrix + methods)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) — **SciPy** — exact definitions of single/complete/average/ward/centroid/median and the Lance–Williams update they use.

**Key papers**:
- [Hierarchical Grouping to Optimize an Objective Function (Ward's method)](https://www.cs.cmu.edu/~roni/11761/Presentations/Ward1963.pdf) — **Ward (1963)** — the minimum-variance linkage that's the modern default; author-hosted PDF, the original source of the variance criterion derived on the concept page.
- [A General Theory of Classificatory Sorting Strategies (Lance–Williams)](https://www.semanticscholar.org/paper/A-general-theory-of-classificatory-sorting-1.-Lance-Williams/49263e8c7a40bef34e2dc7e338f013acf6c8e84e) — **Lance & Williams (1967)** — the recurrence that updates the proximity matrix in O(1) and unifies all linkages.
- [A Statistical Method for Evaluating Systematic Relationships (UPGMA)](https://www.semanticscholar.org/paper/A-statistical-method-for-evaluating-systematic-Sokal-Michener/01402a8f6fc7b7a64bff85b9efb8ea82a26a3f4e) — **Sokal & Michener (1958)** — average linkage (UPGMA), the workhorse of phylogenetics.
- [Modern Hierarchical, Agglomerative Clustering Algorithms](https://arxiv.org/abs/1109.2378) — **Müllner (2011)** — the fast O(n² ) nearest-neighbor-chain algorithms behind `fastcluster`/scipy; the performance reference.
- [Methods of Hierarchical Clustering](https://arxiv.org/abs/1105.0121) — **Murtagh & Contreras (2011)** — free survey of all the linkage algorithms and their complexity; the reference overview.
- [Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward's Criterion?](https://link.springer.com/article/10.1007/s00357-014-9161-z) — **Murtagh & Legendre (2014)** — clears up the two incompatible "Ward" implementations (Ward1 vs Ward2) and which one scipy uses.

**Books (free chapters)**:
- [The Elements of Statistical Learning — §14.3.12 "Hierarchical Clustering"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the rigorous linkage comparison and the variance/cophenetic treatment.
- [An Introduction to Statistical Learning (ISLP) — Ch. 12.4.2 "Hierarchical Clustering"](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; dendrograms and linkages with Python labs.
- [Introduction to Data Mining — Ch. 8 "Cluster Analysis"](https://www-users.cse.umn.edu/~kumar001/dmbook/index.php) — **Tan, Steinbach, Karpatne & Kumar** — free chapter PDFs; the clearest textbook walkthrough of agglomerative merges, MIN/MAX/group-average, and Ward.

**In this platform**:
- Concept page (full explanation): [Hierarchical Clustering](02-Hierarchical-Clustering.md)
- Compare with the siblings: [K-Means Clustering](01-K-Means-Clustering.md) · [DBSCAN](03-DBSCAN.md) · [Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md) · [Spectral Clustering](05-Spectral-Clustering.md)
- Foundations (the *why* behind the distance): [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine Distance](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

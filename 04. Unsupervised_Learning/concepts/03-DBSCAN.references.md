---
id: "04-unsupervised-learning/dbscan/references"
topic: "DBSCAN — References"
parent: "04-unsupervised-learning/dbscan"
type: references
updated: 2026-06-22
---

# DBSCAN — references and further reading

> Companion link library for **[DBSCAN](03-DBSCAN.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (Ester, Sander, Kriegel, Campello, McInnes) or a recognized deep explainer — chosen for depth on *this* topic, not popularity. Every link verified (HTTP 200).

**Start here — suggested path:**

1. **Build intuition** — watch [Clustering with DBSCAN, Clearly Explained](https://www.youtube.com/watch?v=RDZUdRSDOok) (**StatQuest**), then play with [Visualizing DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/). *Watch density reachability grow clusters with the slider and leave noise behind.*
2. **Get the point taxonomy** — watch [DBSCAN - Explained](https://www.youtube.com/watch?v=WoR_crzMAhQ) (**DataMListic**). *Core / border / noise and density-reachability — the formal idea behind the picture.*
3. **Choose the parameters** — watch [DBSCAN Clustering Explained](https://www.youtube.com/watch?v=ry7oCBSzFlc) (**M Iqbal**). *How `ε` and `minPts` interact, and reading `ε` off the k-distance elbow.*
4. **Read the source** — the [DBSCAN paper (Ester et al., 1996)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf). *The original density definitions — directly density-reachable, density-reachable, density-connected — straight from the Test-of-Time paper.*
5. **See the successor for varying density** — read [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). *Mutual-reachability + the condensed tree — DBSCAN generalized to all densities at once.*
6. **Make it concrete** — code it with [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) and run the [DBSCAN demo](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html). *Tuning `ε`/`minPts` on real data cements it.*

**Videos:**
- [Clustering with DBSCAN, Clearly Explained!!!](https://www.youtube.com/watch?v=RDZUdRSDOok) — **StatQuest (Josh Starmer)** — the gentle, visual intro to density clustering and why it beats k-means on weird shapes.
- [DBSCAN - Explained](https://www.youtube.com/watch?v=WoR_crzMAhQ) — **DataMListic** — core/border/noise points and density-reachability, concisely and correctly.
- [DBSCAN Clustering Explained](https://www.youtube.com/watch?v=ry7oCBSzFlc) — **M Iqbal** — walks through `ε`/`minPts` choices and the k-distance plot on real examples.
- [DBSCAN Clustering Coding Tutorial in Python & scikit-learn](https://www.youtube.com/watch?v=VO_uzCU_nKw) — **Greg Hogg** — implement and tune it end-to-end in code.
- [HDBSCAN, Fast Density-Based Clustering — the How and the Why](https://www.youtube.com/watch?v=dGsxd67IFiU) — **John Healy (PyData)** — an HDBSCAN *co-author* explains mutual reachability, the condensed tree, and why it beats DBSCAN on varying density.

**Interactive & visual:**
- [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) — **Naftali Harris** — the interactive playground; set `ε`/`minPts` and watch the "stain" spread through dense regions live.
- [Comparing Python Clustering Algorithms](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html) — **McInnes, Healy & Astels** — side-by-side of k-means, DBSCAN, HDBSCAN and others on the same datasets — the clearest "which one when" visual.

**Courses (free):**
- [scikit-learn — DBSCAN user guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan) — **scikit-learn** — definitions, parameter guidance, and a runnable demo; the practical reference.
- [scikit-learn — HDBSCAN user guide](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) — **scikit-learn** — the modern, density-robust successor you should mention when DBSCAN's single `ε` struggles.
- [scikit-learn — OPTICS user guide](https://scikit-learn.org/stable/modules/clustering.html#optics) — **scikit-learn** — the reachability-plot method that orders points across all `ε` at once.

**Articles / blogs (free, no paywall):**
- [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) — **McInnes, Healy & Astels** — the clearest explanation anywhere of mutual reachability, the minimum spanning tree, and the condensed cluster tree.
- [Understanding HDBSCAN and Density-Based Clustering](https://pberba.github.io/stats/2020/01/17/hdbscan/) — **Pepe Berba** — a careful, illustrated walk from DBSCAN's limits to HDBSCAN's mutual-reachability fix; excellent and free.
- [DBSCAN demo (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html) — **scikit-learn** — copy-paste recipe with core-sample masking and metrics.
- [Comparing clustering algorithms on toy datasets](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) — **scikit-learn** — the canonical grid showing DBSCAN/HDBSCAN succeed on moons/circles where k-means fails.
- [Wikipedia — DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) — **Wikipedia** — a solid, well-referenced summary of the algorithm, complexity, and the DBSCAN\* variant.

**Papers:**
- [A Density-Based Algorithm for Discovering Clusters (DBSCAN)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) — **Ester, Kriegel, Sander & Xu (1996)** — the original; defines core/density-reachable/density-connected. SIGKDD Test-of-Time Award winner.
- [OPTICS: Ordering Points To Identify the Clustering Structure](https://www.dbs.ifi.lmu.de/Publikationen/Papers/OPTICS.pdf) — **Ankerst, Breunig, Kriegel & Sander (1999)** — the reachability-plot method that captures all `ε` at once; author-hosted PDF.
- [Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN)](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14) — **Campello, Moulavi & Sander (2013)** — the hierarchical, varying-density successor and its stability-based extraction.
- [Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection](https://link.springer.com/article/10.1007/s10618-015-0444-8) — **Campello, Moulavi, Zimek & Sander (2015)** — the extended HDBSCAN journal treatment (mutual reachability, GLOSH outlier scores).
- [Accelerated Hierarchical Density Based Clustering](https://arxiv.org/abs/1705.07321) — **McInnes & Healy (2017)** — the fast `hdbscan` implementation everyone uses, with the algorithm explained.
- [DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN](https://www.dbs.ifi.lmu.de/~zimek/publications/TODS2017/TODS-DBSCAN.pdf) — **Schubert, Sander, Ester, Kriegel & Xu (2017)** — the authors' modern guidance on parameters, complexity, and common misconceptions.

**Books (free chapters):**
- [Mining of Massive Datasets — Ch. 7 "Clustering"](http://infolab.stanford.edu/~ullman/mmds/ch7.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF chapter; clustering at scale, including density-based ideas.
- [The Elements of Statistical Learning — §14.3 "Cluster Analysis"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; places density clustering against k-means and hierarchical methods.
- [An Introduction to Statistical Learning — Ch. 12 "Unsupervised Learning"](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF + Python labs; the gentle companion to ESL for clustering.

**In this platform:**
- Concept page (full explanation): [DBSCAN](03-DBSCAN.md)
- Contrast with: [01 K-Means Clustering](01-K-Means-Clustering.md) (centroid vs density) · [02 Hierarchical Clustering](02-Hierarchical-Clustering.md) · [04 Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md) (soft, probabilistic clustering)
- Builds on this: [09 Anomaly / Outlier Detection](09-Anomaly-Outlier-Detection.md) — DBSCAN's "noise" label is a form of outlier detection.
- Prereqs (the *why* behind the metric): [k-Nearest Neighbors](../../03.%20Supervised_Learning/concepts/04-k-Nearest-Neighbors.md) (the same neighborhood/KD-tree machinery) · [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine Distance](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) — density/neighborhoods are defined by the distance metric.
- Field overview: [4. Unsupervised Learning](../README.md)

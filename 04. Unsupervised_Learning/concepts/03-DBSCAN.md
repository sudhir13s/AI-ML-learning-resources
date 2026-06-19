---
id: "04-unsupervised-learning/dbscan"
topic: "DBSCAN (Density-Based Clustering)"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["euclidean-distance", "k-means"]
interview_frequency: high
updated: 2026-06-19
---

# DBSCAN — Density-Based Spatial Clustering
> Group points that are *densely packed* (≥ `minPts` neighbors within radius `ε`) and mark sparse
> points as noise. Finds arbitrarily-shaped clusters, doesn't need `k` in advance, and is the go-to
> when k-means' spherical assumption breaks.

**Why it matters:** the canonical "why not k-means here?" answer — DBSCAN handles non-convex clusters
and outliers natively. Interviews probe the two hyperparameters (`ε`, `minPts`), the point taxonomy
(core / border / noise), how you pick `ε` from a k-distance plot, the `O(n log n)` complexity with a
spatial index, and its weakness on **varying-density** clusters (the motivation for HDBSCAN).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Clustering with DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok), then play with [Visualizing DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/). *See density reachability grow clusters and leave noise behind.*
2. **See why it works** — [DBSCAN, explained](https://www.youtube.com/watch?v=WoR_crzMAhQ) (DataMListic). *Core/border/noise points and density-reachability, the formal idea behind the picture.*
3. **Get the parameters** — [DBSCAN Clustering Explained](https://www.youtube.com/watch?v=ry7oCBSzFlc). *How `ε` and `minPts` interact, and choosing `ε` from the k-distance elbow.*
4. **Read the source** — [DBSCAN paper (Ester et al., 1996)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) → [HDBSCAN — how it works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). *The original density definitions, then the hierarchical extension that handles varying density.*
5. **Make it concrete** — code it with [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) and run the [DBSCAN demo](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html). *Tuning `ε`/`minPts` on real data cements it.*

## 🎓 Courses (free)
- [scikit-learn — DBSCAN user guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan) — **scikit-learn** — definitions, parameter guidance, and a runnable demo; the practical reference.
- [scikit-learn — HDBSCAN user guide](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) — **scikit-learn** — the modern, density-robust successor you should mention when DBSCAN struggles.

## 🎥 Videos
- [Clustering with DBSCAN, Clearly Explained!!!](https://www.youtube.com/watch?v=RDZUdRSDOok) — **StatQuest (Josh Starmer)** — gentle, visual intro to density clustering and why it beats k-means on weird shapes.
- [DBSCAN - Explained](https://www.youtube.com/watch?v=WoR_crzMAhQ) — **DataMListic** — core/border/noise points and density-reachability, concisely.
- [DBSCAN Clustering Explained](https://www.youtube.com/watch?v=ry7oCBSzFlc) — **M Iqbal** — walks through `ε`/`minPts` choices on real examples.
- [DBSCAN Clustering Coding Tutorial in Python & Scikit-Learn](https://www.youtube.com/watch?v=VO_uzCU_nKw) — **Greg Hogg** — implement and tune it end-to-end in code.

## 📄 Key Papers
- [A Density-Based Algorithm for Discovering Clusters (DBSCAN)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) — **Ester, Kriegel, Sander & Xu (1996)** — the original; defines core/density-reachable/density-connected. (Test-of-time award winner.)
- [DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN](https://www.dbs.ifi.lmu.de/~zimek/publications/TODS2017/TODS-DBSCAN.pdf) — **Schubert, Sander, Ester, Kriegel & Xu (2017)** — the authors' modern guidance on parameters and complexity.

## 📰 Articles / Blogs (free, no paywall)
- [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) — **Naftali Harris** — interactive; watch `ε`/`minPts` reshape clusters live.
- [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) — **McInnes, Healy & Astels** — the clearest explanation of the density-hierarchy extension, fully free.
- [DBSCAN demo (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html) — **scikit-learn** — copy-paste recipe with core-sample masking and metrics.

## 📚 Books (free, with chapters)
- [Mining of Massive Datasets — **Ch. 7 "Clustering"**](http://infolab.stanford.edu/~ullman/mmds/ch7.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF chapter; clustering at scale, including density-based ideas.
- [The Elements of Statistical Learning — **§14.3 "Cluster Analysis"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; places DBSCAN-style density clustering against k-means and hierarchical.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Distances (Euclidean vs Cosine)](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) — density/neighborhoods are defined by the distance metric
- Compare with: [01 K-Means Clustering](01-K-Means-Clustering.md) · [02 Hierarchical Clustering](02-Hierarchical-Clustering.md) · [04 Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md)
- Related: Anomaly / Outlier Detection (coming soon) — DBSCAN's "noise" label is a form of outlier detection.
- Field overview: [4. Unsupervised Learning](../README.md)

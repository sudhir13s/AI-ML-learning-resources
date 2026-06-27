---
id: "04-unsupervised-learning/anomaly-outlier-detection/references"
topic: "Anomaly / Outlier Detection — References"
parent: "04-unsupervised-learning/anomaly-outlier-detection"
type: references
updated: 2026-06-22
---

# Anomaly / Outlier Detection — references and further reading

> Companion link library for **[Anomaly / Outlier Detection](09-Anomaly-Outlier-Detection.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame the problem** — watch [Unsupervised Anomaly Detection Explained](https://www.youtube.com/watch?v=QZNEJHbophM) (**Data Science Garage**). *Where clustering, density, and isolation methods each apply.*
2. **Learn the go-to method** — watch [Isolation Forest: a tree-based approach](https://www.youtube.com/watch?v=kqAxfOPlr1U) (**Pratik Nabriya**). *Why random partitioning isolates anomalies in fewer splits — the key intuition.*
3. **See local density** — watch [Local Outlier Factor, explained with an example](https://www.youtube.com/watch?v=8W3mTEKTORg) (**Endless Engineering**). *Why a point can be normal globally but an outlier relative to its neighbours.*
4. **Read the sources** — [Isolation Forest (Liu, Ting & Zhou, 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) → [LOF (Breunig et al., 2000)](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf). *The path-length and local-density definitions in their original form.*
5. **Make it concrete** — code all three with the [scikit-learn outlier detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html) and compare them on the [anomaly-comparison demo](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html). *Seeing where each fails cements the choice.*

**Videos**:
- [Unsupervised Anomaly Detection Explained: Clustering, Density & Isolation Forest](https://www.youtube.com/watch?v=QZNEJHbophM) — **Data Science Garage** — the landscape: which family to reach for and why.
- [Isolation Forest: A Tree-Based Approach for Outlier Detection](https://www.youtube.com/watch?v=kqAxfOPlr1U) — **Pratik Nabriya** — the isolation intuition and the path-length score, clearly explained.
- [Local Outlier Factor (LOF) — Explained with Example](https://www.youtube.com/watch?v=8W3mTEKTORg) — **Endless Engineering** — local reachability density worked through step by step.
- [Anomaly Detection: Algorithm (Gaussian/statistical)](https://www.youtube.com/watch?v=g2YBWQnqOpw) — **Andrew Ng (Stanford / DeepLearning.AI)** — the density-estimation framing and why it differs from supervised learning.
- [Unsupervised Anomaly Detection with Isolation Forest (PyData)](https://www.youtube.com/watch?v=5p8B2Ikcw-k) — **Elena Sharova (PyData London)** — a practitioner's end-to-end talk with real data and pitfalls.

**Interactive & visual**:
- [Comparing anomaly detection algorithms](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html) — **scikit-learn** — the side-by-side decision surfaces of all the major detectors on toy datasets; the clearest "where each fails" picture.
- [Outlier detection with Local Outlier Factor](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html) — **scikit-learn** — a runnable LOF visualization with score interpretation.

**Courses (free)**:
- [scikit-learn — Novelty and Outlier Detection user guide](https://scikit-learn.org/stable/modules/outlier_detection.html) — **scikit-learn** — Isolation Forest, LOF, One-Class SVM, and Elliptic Envelope side by side, with the outlier-vs-novelty distinction made precise.
- [Machine Learning Specialization — Anomaly Detection](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) — **Andrew Ng (DeepLearning.AI)** — the Gaussian density model and choosing features/thresholds; free to audit.

**Articles / blogs (free, no paywall)**:
- [Anomaly Detection — A Survey (PDF)](https://www.vs.inf.ethz.ch/edu/HS2011/CPS/papers/chandola09_anomaly-detection-survey.pdf) — **Chandola, Banerjee & Kumar (2009)** — the canonical taxonomy (point/contextual/collective, supervised/semi/unsupervised) every later treatment cites.
- [PyOD: A Python Toolbox for Scalable Outlier Detection](https://pyod.readthedocs.io/en/latest/) — **Yue Zhao et al.** — the reference library; 40+ detectors with a unified API and model-combination operators for ensembling.
- [Detecting and Treating Outliers — the modified z-score (MAD)](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) — **NIST/SEMATECH e-Handbook of Statistical Methods** — the authoritative reference for the robust z-score, MAD, and Grubbs' test, with the constants derived.
- [Outlier detection with Scikit-learn (Mahalanobis / EllipticEnvelope)](https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html) — **scikit-learn** — robust (MCD) vs empirical covariance for Mahalanobis-based detection, with the χ² intuition.

**Key papers**:
- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) — **Liu, Ting & Zhou (2008)** — the linear-time, distance-free method that's the modern default; the source of the path-length score and `c(n)`.
- [LOF: Identifying Density-Based Local Outliers](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) — **Breunig, Kriegel, Ng & Sander (2000)** — the local-density formulation; the origin of reachability distance, lrd, and the LOF ratio.
- [Support Vector Method for Novelty Detection (One-Class SVM)](https://proceedings.neurips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf) — **Schölkopf, Platt, Shawe-Taylor, Smola & Williamson (1999/2001)** — learning a boundary around the normal region via a kernel; the source of the ν-property.
- [Support Vector Data Description (SVDD)](https://link.springer.com/article/10.1023/B:MACH.0000008084.60811.49) — **Tax & Duin (2004)** — the smallest-enclosing-hypersphere cousin of One-Class SVM.
- [Extended Isolation Forest](https://arxiv.org/abs/1811.02141) — **Hariri, Carrasco Kind & Brunner (2019)** — random-hyperplane splits that remove the axis-parallel bias of the original.
- [Detecting outliers: do not use standard deviation around the mean, use absolute deviation around the median (PDF)](https://dipot.ulb.ac.be/dspace/bitstream/2013/139499/1/Leys_MAD_final-sans%20marque.pdf) — **Leys, Ley, Klein, Bernard & Licata (2013)** — the modern, widely-cited case for the MAD-based modified z-score over the mean/σ rule; open institutional PDF.

**Books (free chapters)**:
- [Outlier Analysis (2nd ed.)](http://charuaggarwal.net/outlierbook.pdf) — **Charu Aggarwal (IBM)** — the definitive textbook on the whole field; free author PDF, covers every family in this page in depth.
- [The Elements of Statistical Learning — §14.3 (density estimation & outliers)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the density framing that underpins LOF-style detection.
- [Mining of Massive Datasets — Ch. 7 "Clustering"](http://infolab.stanford.edu/~ullman/mmds/ch7.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; clustering and the "noise"/outlier view at scale.

**In this platform**:
- Concept page (full explanation): [Anomaly / Outlier Detection](09-Anomaly-Outlier-Detection.md)
- Related unsupervised methods: [DBSCAN](../03-DBSCAN/03-DBSCAN.md) — its "noise" label *is* a form of outlier detection · [Kernel Density Estimation](../10-Kernel-Density-Estimation/10-Kernel-Density-Estimation.md) — density-based scoring · [K-Means Clustering](../01-K-Means-Clustering/01-K-Means-Clustering.md) (distance-to-centroid scoring)
- Evaluation under imbalance: [Classification Metrics](../../03.%20Supervised_Learning/14-Classification-Metrics/14-Classification-Metrics.md) — the ROC-AUC vs PR-AUC contrast, derived
- Boundary method foundations: [Support Vector Machines](../../03.%20Supervised_Learning/06-Support-Vector-Machines/06-Support-Vector-Machines.md) — the kernel + margin machinery One-Class SVM reuses
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Distances (Euclidean vs Cosine)](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) — density and "local" outliers are defined by the distance metric
- Field overview: [4. Unsupervised Learning](../README.md)

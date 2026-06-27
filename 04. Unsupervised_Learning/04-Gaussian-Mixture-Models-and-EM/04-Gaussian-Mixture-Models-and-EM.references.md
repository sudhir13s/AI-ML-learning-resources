---
id: "04-unsupervised-learning/gmm-em/references"
topic: "Gaussian Mixture Models & EM — References"
parent: "04-unsupervised-learning/gmm-em"
type: references
updated: 2026-06-22
---

# Gaussian Mixture Models & EM — references and further reading

> Companion link library for **[Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md)** (the concept page). Curated links — external sources *and* internal cross-links — kept separate so this can double as a standalone reference list. Grouped by type, best-first. Every entry is a primary author or a recognized deep explainer, chosen for depth on *this* topic, and every link verified.

**Start here — suggested path**:
1. **Build intuition** — watch [Gaussian Mixture Models](https://www.youtube.com/watch?v=q71Niz856KE) (**Luis Serrano**). *Soft clustering with bell curves; see why "soft" beats "hard."*
2. **See why EM works** — watch [EM algorithm: how it works](https://www.youtube.com/watch?v=REypj2sy_5U) (**Victor Lavrenko**). *The E/M loop and why each iteration improves the likelihood.*
3. **Get the math** — read [CS229 Notes — The EM Algorithm](https://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf) (**Andrew Ng, Stanford**). *Responsibilities, the ELBO/Jensen lower bound, and the closed-form M-step.*
4. **Read the source** — skim [Maximum Likelihood from Incomplete Data via the EM Algorithm](https://web.mit.edu/6.435/www/Dempster77.pdf) (**Dempster, Laird & Rubin, 1977**). *The paper that unified EM; GMM fitting is its special case.*
5. **Make it concrete** — fit and select with [scikit-learn — Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html) + the [BIC selection example](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html). *Covariance types, EM fitting, and choosing k.*

**Videos**:
- [Clustering (4): Gaussian Mixture Models and EM](https://www.youtube.com/watch?v=qMTuMa86NzU) — **Alexander Ihler (UC Irvine)** — the rigorous derivation of responsibilities and the M-step updates.
- [EM algorithm: how it works](https://www.youtube.com/watch?v=REypj2sy_5U) — **Victor Lavrenko (Edinburgh)** — the E/M loop and *why* the likelihood keeps improving, with a clean worked example.
- [(ML 16.3) Expectation-Maximization (EM) algorithm](https://www.youtube.com/watch?v=AnbiNaVp3eQ) — **mathematicalmonk** — the general EM algorithm and its lower-bound/Jensen justification, beyond just GMMs.
- [Gaussian Mixture Models](https://www.youtube.com/watch?v=q71Niz856KE) — **Luis Serrano** — illustrations-over-formulas intro to soft clustering; the best first watch.
- [Gaussian Mixture Models for Clustering](https://www.youtube.com/watch?v=DODphRRL79c) — **Serrano.Academy** — the companion walkthrough connecting GMMs to k-means and EM.
- [Model-based clustering: an introduction to GMMs](https://www.youtube.com/watch?v=h7RVeO-P3zc) — **Mario Castro** — places GMMs in the broader model-based-clustering view.

**Courses (free)**:
- [scikit-learn — Gaussian mixture models user guide](https://scikit-learn.org/stable/modules/mixture.html) — **scikit-learn** — covariance types, EM fitting, BIC selection, and Bayesian GMMs, with code.
- [CS229 Notes — Mixtures of Gaussians and the EM Algorithm](https://cs229.stanford.edu/notes2020spring/cs229-notes7b.pdf) — **Andrew Ng (Stanford)** — the GMM-specific EM derivation (a companion to the general EM notes below).
- [Machine Learning Specialization (Course 3: Unsupervised Learning)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — free to audit; clustering and the soft-assignment intuition behind EM.

**Articles / blogs (free, no paywall)**:
- [In Depth: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html) — **Jake VanderPlas (Python Data Science Handbook)** — GMM as soft k-means, covariance shapes, and density estimation, with runnable code.
- [Soft Clustering: Gaussian Mixture Models](https://www.serrano.academy/unsupervised-machine-learning/gaussian-mixture-models) — **Luis Serrano** — the written companion to the video; clean intuition, no paywall.
- [Gaussian Mixture Model (wiki)](https://brilliant.org/wiki/gaussian-mixture-model/) — **Brilliant** — a concise, well-illustrated reference for the density and the EM updates.
- [Introduction to EM](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) — **Matthew Stephens (fiveMinuteStats)** — a tight, worked introduction to the EM lower bound.
- [GMM selection by BIC (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html) — **scikit-learn** — choosing the number of components and covariance type in practice.
- [GMM covariance types (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html) — **scikit-learn** — full vs tied vs diag vs spherical, drawn side by side.

**Key papers**:
- [Maximum Likelihood from Incomplete Data via the EM Algorithm](https://web.mit.edu/6.435/www/Dempster77.pdf) — **Dempster, Laird & Rubin (1977)** — the foundational paper that unified EM; the theory under GMM fitting.
- [A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants](https://www.cs.toronto.edu/~radford/ftp/emk.pdf) — **Neal & Hinton (1998)** — the ELBO / free-energy view that explains *why* EM works (and seeds variational inference).

**Books (free chapters / companions)**:
- [Pattern Recognition and Machine Learning — **Ch. 9 "Mixture Models and EM"**](https://www.bishopbook.com/) — **Christopher Bishop** — the canonical, exhaustive derivation of GMMs, responsibilities, and EM (free PDF on the book site).
- [Probabilistic Machine Learning: An Introduction — **mixture models & EM**](https://probml.github.io/pml-book/book1.html) — **Kevin Murphy** — modern, rigorous treatment with code; free online.
- [The Elements of Statistical Learning — **§8.5 "The EM Algorithm"** (and §6.8 mixtures)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; EM derived as maximizing a likelihood lower bound.
- [Mathematics for Machine Learning — **Ch. 11 "Density Estimation with GMMs"**](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the full GMM + EM derivation from first principles.

**In this platform**:
- Concept page (full explanation): [Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM.md)
- The hard-assignment special case: [01 K-Means Clustering](01-K-Means-Clustering.md) (k-means = GMM with equal spherical covariance in the zero-variance limit)
- Compare with: [02 Hierarchical Clustering](02-Hierarchical-Clustering.md) · [03 DBSCAN](03-DBSCAN.md) · [05 Spectral Clustering](05-Spectral-Clustering.md) (for non-Gaussian, non-convex clusters)
- Puts the density to work: [09 Anomaly & Outlier Detection](09-Anomaly-Outlier-Detection.md) · [10 Kernel Density Estimation](10-Kernel-Density-Estimation.md)
- Same Gaussian family, supervised: [Gaussian Naive Bayes](../../03.%20Supervised_Learning/concepts/05-Naive-Bayes.md) (the diagonal-covariance, labeled case)
- Cluster learned embeddings: [07 t-SNE](07-t-SNE.md) · [08 UMAP](08-UMAP.md)
- Concept depth (the *why*): [AI-ML-intuition 5.06 GMMs & EM](../../../AI-ML-intuition/Module_5_Generation/5.06_GMMs_and_EM.md) · [0.02 Distributions & the Gaussian](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

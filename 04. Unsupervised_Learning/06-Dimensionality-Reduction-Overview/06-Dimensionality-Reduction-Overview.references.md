---
id: "04-unsupervised-learning/dimensionality-reduction-overview/references"
topic: "Dimensionality Reduction — References"
parent: "04-unsupervised-learning/dimensionality-reduction-overview"
type: references
updated: 2026-06-22
---

# Dimensionality Reduction — references and further reading

> Companion link library for **[Dimensionality Reduction — Overview](06-Dimensionality-Reduction-Overview.md)** (the concept page, PCA-centric). Curated links — external sources *and* internal cross-links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a primary author or a recognized deep explainer, chosen for depth on *this* topic, and every link verified.

**Start here — suggested path:**
1. **Build intuition** — watch [Luis Serrano: Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=g-Hb26agBFg). *The "best camera angle on a cloud" picture — variance and projection without heavy math.*
2. **See it move** — play with [Setosa: PCA, interactive](https://setosa.io/ev/principal-component-analysis/). *Drag the cloud and watch the principal axes rotate to follow it; the best visual intuition, fully free.*
3. **Get the math (two views)** — read [Shlens, "A Tutorial on PCA"](https://arxiv.org/abs/1404.1100). *Derives PCA from variance maximization AND from the SVD — exactly the two derivations interviews want.*
4. **Map the landscape** — skim the [scikit-learn comparison of manifold methods](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html). *See PCA (linear) next to t-SNE / Isomap / LLE / UMAP on the same data.*
5. **Make it concrete** — apply PCA with the [scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html#pca), inspect the explained-variance ratio, then jump to the [t-SNE](07-t-SNE.md) / [UMAP](08-UMAP.md) pages for the non-linear methods.

**Videos:**
- [PCA, main ideas in only 5 minutes](https://www.youtube.com/watch?v=HMOI_lkzW08) — **StatQuest (Josh Starmer)** — the fastest correct mental model of components and scree.
- [PCA, Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — **StatQuest (Josh Starmer)** — components, loadings, and scree plots built from scratch.
- [Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=g-Hb26agBFg) — **Luis Serrano** — illustrations-over-formulas intro; the best first watch for variance/projection.
- [Singular Value Decomposition (SVD) and PCA](https://www.youtube.com/watch?v=gXbThCXjZFM) — **Steve Brunton (UW)** — the SVD view of PCA, the numerically preferred route, derived cleanly.
- [PyData: PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes (UMAP author)** — the unifying tour across linear and non-linear methods.

**Interactive & visual:**
- [Principal Component Analysis — interactive explainer](https://setosa.io/ev/principal-component-analysis/) — **Victor Powell & Lewis Lehe (Setosa)** — drag the data, watch the components rotate.
- [Comparison of Manifold Learning methods](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) — **scikit-learn** — PCA vs t-SNE vs Isomap vs LLE on one dataset; the clearest "which method" picture.

**Courses (free):**
- [scikit-learn — Decomposition (PCA, kernel PCA, truncated SVD, ICA, NMF)](https://scikit-learn.org/stable/modules/decomposition.html#pca) — **scikit-learn** — the linear family with the explained-variance API and runnable code.
- [scikit-learn — Manifold learning user guide](https://scikit-learn.org/stable/modules/manifold.html) — **scikit-learn** — the non-linear side (Isomap, LLE, t-SNE) placed next to PCA.
- [CS229: Machine Learning — PCA & ICA lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Andrew Ng)** — PCA and ICA derived rigorously, the variance and SVD views together.

**Articles / blogs (free, no paywall):**
- [In Depth: Principal Component Analysis](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) — **Jake VanderPlas (Python Data Science Handbook)** — PCA for visualization, noise filtering, and compression, with code.
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) — **Google PAIR** — how UMAP and t-SNE distort distances, and why their geometry isn't literal — the caveats every practitioner needs.
- [A Tutorial on Principal Components Analysis](https://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf) — **Lindsay I. Smith (Otago)** — the classic from-scratch walkthrough: covariance, eigenvectors, and a fully worked numeric example, free PDF.
- [Principal Components Analysis (lecture notes)](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf) — **Cosma Shalizi (CMU)** — a rigorous yet readable derivation of PCA as variance maximization and reconstruction (the Hotelling/Pearson views), free.

**Key papers:**
- [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100) — **Jonathon Shlens (2014)** — the most-cited free PCA tutorial; derives PCA from variance and from the SVD.
- [On Lines and Planes of Closest Fit to Systems of Points in Space](https://pca.narod.ru/pearson1901.pdf) — **Karl Pearson (1901)** — the original PCA paper, the reconstruction-error ("closest fit") view; free scanned PDF.
- [The Approximation of One Matrix by Another of Lower Rank](https://link.springer.com/article/10.1007/BF02288367) — **Eckart & Young (1936)** — the theorem that the truncated SVD is the optimal low-rank approximation (= min reconstruction error).
- [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.mlpack.org/papers/kpca.pdf) — **Schölkopf, Smola & Müller (1998)** — the original kernel-PCA paper.
- [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — **van der Maaten, Postma & van den Herik (2009)** — free survey benchmarking PCA against the non-linear methods; the landscape in one paper.
- [A Global Geometric Framework for Nonlinear Dimensionality Reduction (Isomap)](https://wearables.cc.gatech.edu/paper_of_week/isomap.pdf) — **Tenenbaum, de Silva & Langford (2000)** — geodesic-distance manifold learning; unrolls the Swiss roll; free PDF.
- [Nonlinear Dimensionality Reduction by Locally Linear Embedding (LLE)](https://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf) — **Roweis & Saul (2000)** — local-neighbour reconstruction embedding; free PDF.

**Books (free, with chapters):**
- [Mathematics for Machine Learning — Ch. 10 "Dimensionality Reduction with PCA"](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — PCA derived from first principles (projection, eigenvectors, SVD); the cleanest modern treatment.
- [An Introduction to Statistical Learning (ISLP) — Ch. 12.2 "Principal Components Analysis"](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; PCA intuition + the labs.
- [The Elements of Statistical Learning — Ch. 14.5 "Principal Components, Curves and Surfaces"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; PCA and its non-linear generalizations, the reference treatment.

**In this platform:**
- Concept page (full explanation): [Dimensionality Reduction — Overview](06-Dimensionality-Reduction-Overview.md)
- Non-linear siblings (the manifold methods): [t-SNE](07-t-SNE.md) · [UMAP](08-UMAP.md)
- The PCA/SVD math foundations (the *why*): [Foundations — Maths for AI-ML](../../01.%20Foundations/Maths%20for%20AI-ML/README.md) · [AI-ML-intuition 1.05 Spectral Methods (PCA / SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- The failure mode that motivates reduction: [k-Nearest Neighbors (curse of dimensionality)](../../03.%20Supervised_Learning/concepts/04-k-Nearest-Neighbors.md)
- Learned non-linear compression: [Autoencoders → Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
- Acutely relevant preprocessing: [2. Data Preprocessing](../../02.%20Data_Preprocessing/concepts/README.md) — scaling/standardization, which PCA depends on
- Field overview: [4. Unsupervised Learning](../README.md)

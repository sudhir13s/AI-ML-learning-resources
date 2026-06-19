---
id: "04-unsupervised-learning/dimensionality-reduction-overview"
topic: "Dimensionality Reduction — overview (PCA · SVD framing)"
parent: "04-unsupervised-learning"
level: beginner
prereqs: ["linear-algebra", "variance", "eigendecomposition"]
interview_frequency: very-high
updated: 2026-06-19
---

# Dimensionality Reduction — Overview
> Re-express high-dimensional data in far fewer coordinates while keeping what matters — for
> visualization, denoising, compression, and beating the curse of dimensionality. **PCA** (a linear
> projection onto the directions of greatest variance, computed via the SVD) is the workhorse; non-linear
> manifold methods (t-SNE, UMAP) and learned methods (autoencoders) extend it.

**Why it matters:** the framing question that organizes a whole interview area. You should be able to
say *why* you reduce dimensions (curse of dimensionality, correlated features, plotting, speed), draw
the **linear vs non-linear** split, explain PCA as variance maximization = reconstruction-error
minimization = the truncated SVD, and know when PCA's linearity fails so you reach for t-SNE/UMAP (for
visualization) or autoencoders (for learned, non-linear compression). This card is the map; deep PCA/SVD
math lives in Foundations, and each non-linear method has its own card.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Luis Serrano: Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=g-Hb26agBFg). *The "best camera angle on a cloud of points" picture — variance and projection without heavy math.*
2. **See why it works** — [StatQuest: PCA, Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ). *Principal components as variance-ranked directions, and how the scree plot tells you how many to keep.*
3. **Get the math** — read [Foundations — PCA/SVD (this platform)](../../01.%20Foundations/Maths%20for%20AI-ML/README.md) + [scikit-learn PCA user guide](https://scikit-learn.org/stable/modules/decomposition.html#pca). *Eigendecomposition of the covariance = the SVD; explained-variance ratio.*
4. **Map the landscape** — skim [scikit-learn: manifold learning](https://scikit-learn.org/stable/modules/manifold.html) and the [comparison of methods](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html). *See PCA (linear) next to t-SNE/Isomap/UMAP (non-linear) on the same data.*
5. **Make it concrete** — apply PCA with the [scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html#pca), inspect the explained-variance ratio, then jump to the [07 t-SNE](07-t-SNE.md) / [08 UMAP](08-UMAP.md) cards. *Choosing the number of components cements the trade-off.*

## 🎓 Courses (free)
- [scikit-learn — Decomposition (PCA & friends)](https://scikit-learn.org/stable/modules/decomposition.html#pca) — **scikit-learn** — PCA, truncated SVD, kernel PCA, and the explained-variance API with code.
- [scikit-learn — Manifold learning user guide](https://scikit-learn.org/stable/modules/manifold.html) — **scikit-learn** — the non-linear side (Isomap, LLE, t-SNE) placed next to PCA, with runnable examples.

## 🎥 Videos
- [Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=g-Hb26agBFg) — **Luis Serrano** — illustrations-over-formulas intro; the best first watch for the variance/projection idea.
- [StatQuest: PCA, Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — **StatQuest (Josh Starmer)** — components, loadings, and scree plots from scratch.
- [PCA clearly explained](https://www.youtube.com/watch?v=_UVHneBUBW0) — **StatQuest (Josh Starmer)** — the geometry of projection and why the top components keep the most information.
- [PyData: PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes** — the unifying tour across linear and non-linear methods; the big-picture talk.

## 📄 Key Papers
- [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100) — **Jonathon Shlens (2014)** — the most-cited free PCA tutorial; derives PCA from variance and from the SVD, the two views interviews want.
- [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — **van der Maaten, Postma & van den Herik (2009)** — free survey benchmarking PCA against the non-linear methods; the landscape in one paper.

## 📰 Articles / Blogs (free, no paywall)
- [In Depth: Principal Component Analysis](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) — **Jake VanderPlas (Python Data Science Handbook)** — PCA for visualization, noise filtering, and compression, with code.
- [Comparison of Manifold Learning methods (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) — **scikit-learn** — PCA vs t-SNE vs Isomap vs LLE on one dataset; the clearest "which method" picture.
- [Principal Component Analysis — interactive explainer](https://setosa.io/ev/principal-component-analysis/) — **Victor Powell & Lewis Lehe (Setosa)** — drag the data and watch the components rotate; the best visual intuition, fully free.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLP) — **Ch. 12.2 "Principal Components Analysis"**](https://www.statlearning.com/) — **James, Witten, Hastie, Tibshirani & Taylor** — free PDF; PCA intuition + the dimensionality-reduction labs.
- [Mathematics for Machine Learning — **Ch. 10 "Dimensionality Reduction with PCA"**](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; PCA derived from first principles (projection, eigenvectors, SVD).
- [The Elements of Statistical Learning — **§14.5 "Principal Components, Curves and Surfaces"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; PCA and its non-linear generalizations.

## 🔗 In this platform
- The PCA/SVD math (eigendecomposition, variance maximization, the SVD view): [Foundations — Maths for AI-ML](../../01.%20Foundations/Maths%20for%20AI-ML/README.md) · concept depth [AI-ML-intuition 1.05 Spectral Methods (PCA / SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Non-linear methods: [07 t-SNE](07-t-SNE.md) · [08 UMAP](08-UMAP.md)
- Learned (non-linear) compression: [Autoencoders → Deep Learning](../../05.%20Deep_Learning/README.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

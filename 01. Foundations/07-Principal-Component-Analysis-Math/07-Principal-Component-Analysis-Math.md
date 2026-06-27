---
id: "01-foundations/pca-math"
topic: "Principal Component Analysis (PCA) — the math"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/eigenvalues-and-eigenvectors", "01-foundations/singular-value-decomposition"]
interview_frequency: very-high
updated: 2026-06-20
---

# Principal Component Analysis (PCA) — the math
> PCA finds the orthogonal directions of maximum variance in your data — the eigenvectors of the
> covariance matrix (equivalently the right singular vectors of the centered data). Projecting onto
> the top components gives the best linear, lowest-distortion compression. It's the canonical
> bridge from linear algebra to machine learning.

**Why it matters:** "derive PCA" is one of the most common ML-math interview questions. You should
be able to set it up two ways — maximize projected variance (Rayleigh quotient → top eigenvectors
of the covariance) and minimize reconstruction error — and explain the SVD route, why you center
(and sometimes standardize), and how explained-variance ratios pick `k`.

**⭐ Start here — suggested path:**

1. **Intuition first** — watch [StatQuest: PCA, Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) (or the [5-minute version](https://www.youtube.com/watch?v=HMOI_lkzW08)). *Builds the "rotate to variance axes" picture without heavy math.*
2. **The two derivations** — read [MML Ch. 10 (Dimensionality Reduction with PCA)](https://mml-book.github.io/book/mml-book.pdf). *Max-variance and min-reconstruction-error views, both worked.*
3. **The SVD route** — watch [Steve Brunton: PCA](https://www.youtube.com/watch?v=fkf4IBRSeEc). *Why PCA is just SVD of the centered data matrix.*
4. **Pin the prerequisites** — review [04 Eigenvalues](../04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md) + [06 SVD](../06-Singular-Value-Decomposition/06-Singular-Value-Decomposition.md). *PCA is their direct application; make sure the covariance-eigenvector link is solid.*
5. **Connect to ML** — read [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md). *The ML payoff and how PCA sits next to t-SNE/UMAP.*

## 🎓 Courses (free)
- [Stanford CS229 — Lecture notes on PCA](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf) — **Stanford (Ng et al.)** — the rigorous variance-maximization derivation.
- [Mathematics for ML: PCA (course resources)](https://mml-book.github.io/) — **Deisenroth et al.** — the companion course's PCA chapter and exercises.

## 🎥 Videos
- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — **StatQuest (Josh Starmer)** — the gold-standard intuition build-up.
- [StatQuest: PCA main ideas in only 5 minutes](https://www.youtube.com/watch?v=HMOI_lkzW08) — **StatQuest (Josh Starmer)** — the fast refresher.
- [Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=fkf4IBRSeEc) — **Steve Brunton** — PCA as SVD of centered data, with the math.
- [Eigenvectors and eigenvalues | Ch. 14](https://www.youtube.com/watch?v=PFDu9oVAE-g) — **3Blue1Brown** — the eigen-intuition PCA rests on.

## 📄 Key Papers
- [MML book — Ch. 10 "Dimensionality Reduction with PCA"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — both PCA derivations end-to-end.
- [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100) — **Jonathon Shlens (2014)** — the widely-cited, beginner-friendly tutorial linking PCA and SVD.

## 📰 Articles / Blogs (free, no paywall)
- [Making sense of PCA, eigenvectors & eigenvalues](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues) — **amoeba (CrossValidated)** — the famous, deeply intuitive top answer; fully free.
- [CS229 PCA notes](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf) — **Stanford** — the clean derivation as a PDF.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 10 (PCA)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — the definitive free chapter.
- [An Introduction to Statistical Learning — **Ch. 12 (Unsupervised Learning / PCA)**](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — applied PCA with the variance-explained view.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) · [1.11–1.12 t-SNE/UMAP](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra, row 1.6)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [04 Eigenvalues](../04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md) · [06 SVD](../06-Singular-Value-Decomposition/06-Singular-Value-Decomposition.md)
- Applied: nonlinear dimensionality reduction (t-SNE/UMAP) → [Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
</content>

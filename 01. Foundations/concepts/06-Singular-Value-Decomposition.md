---
id: "01-foundations/singular-value-decomposition"
topic: "Singular Value Decomposition (SVD)"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/eigenvalues-and-eigenvectors"]
interview_frequency: high
updated: 2026-06-20
---

# Singular Value Decomposition (SVD)
> Every matrix — square or not — factors as `A = UΣVᵀ`: rotate (`Vᵀ`), scale by singular values
> (`Σ`), rotate again (`U`). It's the most general "find the natural axes" tool in linear algebra,
> giving the best low-rank approximation of any matrix and powering PCA, recommender systems,
> pseudo-inverses, and matrix compression.

**Why it matters:** SVD is arguably the single most useful decomposition in ML. Interviewers ask
how SVD relates to eigen-decomposition of `AᵀA`, why the top-k singular triplets give the optimal
rank-k approximation (Eckart–Young), how SVD yields PCA, and where it appears in LSA, LoRA, and
collaborative filtering.

**⭐ Start here — suggested path:**

1. **Big-picture intuition** — watch [Steve Brunton: SVD Overview](https://www.youtube.com/watch?v=gXbThCXjZFM). *The rotate–scale–rotate picture and why it's everywhere in data science.*
2. **The math** — watch [Brunton: SVD Mathematical Overview](https://www.youtube.com/watch?v=nbBvuuNVfco), then read [MML Ch. 4.5 (SVD)](https://mml-book.github.io/book/mml-book.pdf). *Connects `A = UΣVᵀ` to the eigenvectors of `AᵀA` and `AAᵀ`.*
3. **The authoritative lecture** — watch [Gilbert Strang: SVD (MIT 18.06 Lec 29)](https://www.youtube.com/watch?v=TX_vooSnhm8). *Strang's geometric derivation and the four-subspaces tie-in.*
4. **Low-rank & applications** — read [Brunton's SVD chapter notes](https://databookuw.com/) (Data-Driven Science, free PDF). *Eckart–Young, truncation, and real applications.*
5. **Connect to ML** — read [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md). *How SVD becomes PCA and dimensionality reduction.*

## 🎓 Courses (free)
- [MIT 18.06 — Singular Value Decomposition (Lec 29)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — the definitive SVD lecture.
- [MIT 18.065 — Matrix Methods (SVD-centric)](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) — **Gilbert Strang (MIT OCW)** — an entire data-focused course built around SVD.

## 🎥 Videos
- [Singular Value Decomposition (SVD): Overview](https://www.youtube.com/watch?v=gXbThCXjZFM) — **Steve Brunton** — the clearest high-level intuition.
- [SVD: Mathematical Overview](https://www.youtube.com/watch?v=nbBvuuNVfco) — **Steve Brunton** — derivation via `AᵀA` eigenvectors.
- [Singular Value Decomposition (18.06 Lec 29)](https://www.youtube.com/watch?v=TX_vooSnhm8) — **Gilbert Strang (MIT OCW)** — full lecture with the geometry.
- [Singular Value Decomposition (the SVD)](https://www.youtube.com/watch?v=mBcLRGuAFUk) — **Gilbert Strang (MIT OCW, 18.065)** — the data-methods framing.

## 📄 Key Papers
- [MML book — Ch. 4.5 "Singular Value Decomposition"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — `A = UΣVᵀ`, relation to eigendecomposition, low-rank approximation.
- [Finding Structure with Randomness (randomized SVD)](https://arxiv.org/abs/0909.4061) — **Halko, Martinsson & Tropp (2011)** — the standard method for SVD at scale.

## 📰 Articles / Blogs (free, no paywall)
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) — **David Austin (AMS Feature Column)** — a beautifully illustrated, free SVD walkthrough.
- [CS229 Linear Algebra Review — SVD & PSD matrices](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — the ML-oriented summary.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 4.5 (SVD)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — SVD, low-rank approximation, and the link to PCA.
- [Data-Driven Science & Engineering — **Ch. 1 (SVD)**](https://databookuw.com/) — **Brunton & Kutz** — SVD-first treatment with code; chapter PDFs free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra, row 1.5)](../Maths%20for%20AI-ML/README.md)
- Prereq: [04 Eigenvalues & Eigenvectors](04-Eigenvalues-and-Eigenvectors.md) · Next: [07 PCA — the math](07-Principal-Component-Analysis-Math.md)
- Applied: dimensionality reduction for visualization (t-SNE/UMAP) → [Unsupervised Learning](../../04.%20Unsupervised_Learning/concepts/README.md)
</content>

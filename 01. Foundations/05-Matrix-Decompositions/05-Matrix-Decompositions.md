---
id: "01-foundations/matrix-decompositions"
topic: "Matrix Decompositions (LU · QR · Cholesky)"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/matrices-and-matrix-operations", "01-foundations/norms-inner-products-orthogonality"]
interview_frequency: medium
updated: 2026-06-20
---

# Matrix Decompositions — LU · QR · Cholesky
> Factor a matrix into simpler pieces so the hard problems become easy: **LU** turns solving
> `Ax = b` into two triangular solves, **QR** orthogonalizes columns for stable least squares, and
> **Cholesky** factors a positive-definite (covariance) matrix as `LLᵀ`. These are the numerical
> engines under regression, Gaussian models, and optimization.

**Why it matters:** real ML rarely inverts a matrix directly — it decomposes. Interviewers ask why
QR is preferred over the normal equations for least squares (conditioning), what positive-definite
means and why Cholesky needs it, and how sampling from a multivariate Gaussian uses the Cholesky
factor of the covariance.

**⭐ Start here — suggested path:**

1. **Why factor at all** — read [MML Ch. 4 (Matrix Decompositions)](https://mml-book.github.io/book/mml-book.pdf), intro + Cholesky section. *Frames decompositions as "change to a basis where the problem is trivial."*
2. **QR & Gram–Schmidt** — watch [Gilbert Strang: Gram–Schmidt & A = QR (MIT 18.06 Lec 17)](https://www.youtube.com/watch?v=0MtwqhIwdrI). *The orthogonalization that makes least squares stable.*
3. **Cholesky for PD matrices** — watch [QR / Cholesky walkthrough](https://www.youtube.com/watch?v=FAnNBw7d0vg) and read the Cholesky section of MML. *The `LLᵀ` factorization used for covariance & sampling.*
4. **Tie to least squares** — read [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) (PD matrices, least squares). *Connects factorization to fitting models.*

## 🎓 Courses (free)
- [MIT 18.06 — Orthogonality, Gram–Schmidt & A = QR (Lec 17)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — QR via Gram–Schmidt; LU appears in the elimination lectures.
- [Khan Academy — Alternate bases & Gram–Schmidt](https://www.khanacademy.org/math/linear-algebra/alternate-bases) — **Khan Academy** — orthonormal bases and the Gram–Schmidt process behind QR.

## 🎥 Videos
- [Gram–Schmidt and A = QR (18.06 Lec 17)](https://www.youtube.com/watch?v=0MtwqhIwdrI) — **Gilbert Strang (MIT OCW)** — the QR decomposition from orthogonalization.
- [QR decomposition (for square matrices)](https://www.youtube.com/watch?v=FAnNBw7d0vg) — **The Bright Side of Mathematics** — clean step-by-step QR.
- [QR decomposition](https://www.youtube.com/watch?v=J41Ypt6Mftc) — **Dr Peyam** — an alternative derivation to reinforce the idea.
- [Linear transformations and matrices | Ch. 3](https://www.youtube.com/watch?v=kYB8IZa5AuE) — **3Blue1Brown** — the change-of-basis lens that makes "factor into simpler maps" intuitive.

## 📄 Key Papers
- [MML book — Ch. 4 "Matrix Decompositions"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — Cholesky, eigendecomposition, and the role of decompositions in ML.
- [The Matrix Cookbook — decompositions & solves](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — **Petersen & Pedersen** — reference identities for LU/QR/Cholesky and Gaussian densities.

## 📰 Articles / Blogs (free, no paywall)
- [Mathematics for ML (course notes) — positive-definite matrices & Cholesky](https://gwthomas.github.io/docs/math4ml.pdf) — **Garrett Thomas (Stanford)** — PD matrices and decompositions for ML.
- [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — positive (semi)definite matrices and least squares, the prerequisites for these factorizations.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 4 (Matrix Decompositions)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — Cholesky and eigendecomposition with ML applications.
- [Introduction to Applied Linear Algebra (VMLS) — **Ch. 5 (Linear Independence/QR), Ch. 11 (Least Squares)**](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — **Boyd & Vandenberghe** — QR factorization and least-squares solves.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) · [1.10 Mahalanobis Distance (covariance/Cholesky)](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra, row 1.5)](../Maths%20for%20AI-ML/README.md)
- Prereq: [04 Eigenvalues & Eigenvectors](../04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md) · Next: [06 Singular Value Decomposition](../06-Singular-Value-Decomposition/06-Singular-Value-Decomposition.md)
</content>

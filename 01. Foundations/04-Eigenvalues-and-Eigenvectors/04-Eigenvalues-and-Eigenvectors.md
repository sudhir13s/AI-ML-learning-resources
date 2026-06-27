---
id: "01-foundations/eigenvalues-and-eigenvectors"
topic: "Eigenvalues & Eigenvectors"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/matrices-and-matrix-operations"]
interview_frequency: high
updated: 2026-06-20
---

# Eigenvalues & Eigenvectors
> An eigenvector is a direction a matrix only *stretches* (never rotates); its eigenvalue is the
> stretch factor. Diagonalizing a matrix by its eigenvectors reveals the "natural axes" of the
> transformation — the idea behind PCA, spectral clustering, PageRank, and the stability of
> iterative methods.

**Why it matters:** eigen-decomposition is the entry point to PCA and SVD, and it explains why
covariance matrices have orthogonal principal axes, what the Graph Laplacian's spectrum encodes,
and why repeated multiplication by a matrix converges to its dominant eigenvector. Expect "derive
PCA from the covariance eigenvectors" and "what does a negative/zero eigenvalue mean?"

**⭐ Start here — suggested path:**

1. **See it move** — watch [3B1B: Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g). *The "axes that don't get knocked off their span" picture makes the definition obvious.*
2. **Compute a few** — watch [StatQuest / Professor Dave: finding eigenvalues & eigenvectors](https://www.youtube.com/watch?v=TQvxWaQnrqI), then do [Khan: eigen-everything](https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors). *The characteristic polynomial `det(A − λI) = 0` by hand.*
3. **Diagonalization & spectral theorem** — read [MML Ch. 4.2–4.4](https://mml-book.github.io/book/mml-book.pdf). *Eigendecomposition, symmetric ⇒ orthogonal eigenvectors (the fact PCA relies on).*
4. **The full lecture** — watch [MIT 18.06: Eigenvalues & Eigenvectors (Lec 21)](https://www.youtube.com/watch?v=lXNXrLcoerU). *Strang ties eigenvalues to stability, powers of a matrix, and diagonalization.*
5. **Connect to ML** — read [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md). *Where eigen-thinking becomes dimensionality reduction.*

## 🎓 Courses (free)
- [MIT 18.06 — Eigenvalues & Eigenvectors (Lec 21–22)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — diagonalization, powers, and the spectral theorem.
- [Khan Academy — Eigen-everything](https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors) — **Khan Academy** — definitions and computation with exercises.

## 🎥 Videos
- [Eigenvectors and eigenvalues | Ch. 14](https://www.youtube.com/watch?v=PFDu9oVAE-g) — **3Blue1Brown** — the definitive visual intuition.
- [Eigenvalues & Eigenvectors (18.06 Lec 21)](https://www.youtube.com/watch?v=lXNXrLcoerU) — **Gilbert Strang (MIT OCW)** — full lecture: diagonalization and stability.
- [Finding Eigenvalues and Eigenvectors](https://www.youtube.com/watch?v=TQvxWaQnrqI) — **Professor Dave Explains** — clean worked example of the characteristic equation.
- [Abstract vector spaces | Ch. 16](https://www.youtube.com/watch?v=TgKwz5Ikpc8) — **3Blue1Brown** — why eigenvectors are basis-independent (a recurring interview subtlety).

## 📄 Key Papers
- [MML book — Ch. 4 "Matrix Decompositions"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — eigendecomposition, the spectral theorem, and diagonalization.
- [CS229 Linear Algebra Review — eigenvalues/eigenvectors](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — the ML-oriented summary, including symmetric/PSD matrices.

## 📰 Articles / Blogs (free, no paywall)
- [Immersive Linear Algebra — Ch. 10 "Eigenvalues and Eigenvectors"](https://immersivemath.com/ila/ch10_eigen/ch10.html) — **Ström, Åström & Akenine-Möller** — interactive eigenvectors with worked examples.
- [Mathematics for ML (course notes) — eigenstuff & spectral theorem](https://gwthomas.github.io/docs/math4ml.pdf) — **Garrett Thomas (Stanford)** — concise ML-focused treatment.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 4.1–4.4 (Matrix Decompositions)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — determinant, eigenvalues, eigendecomposition.
- [Introduction to Applied Linear Algebra (VMLS) — **eigenvalues & dynamics**](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — **Boyd & Vandenberghe** — applied eigen-analysis.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) · [1.04 Graph Representations (Laplacian spectrum)](../../../AI-ML-intuition/Module_1_Representation/1.04_Graph_Representations.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra)](../Maths%20for%20AI-ML/README.md)
- Prereq: [02 Matrices & Matrix Operations](../02-Matrices-and-Matrix-Operations/02-Matrices-and-Matrix-Operations.md) · Next: [05 Matrix Decompositions](../05-Matrix-Decompositions/05-Matrix-Decompositions.md) · [06 SVD](../06-Singular-Value-Decomposition/06-Singular-Value-Decomposition.md) · [07 PCA (math)](../07-Principal-Component-Analysis-Math/07-Principal-Component-Analysis-Math.md)
</content>

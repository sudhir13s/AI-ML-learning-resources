---
id: "14-advanced-math/functional-analysis"
topic: "Functional Analysis"
parent: "14-advanced-research-mathematics"
level: advanced
prereqs: ["linear-algebra", "real-analysis", "measure-theory"]
interview_frequency: low
updated: 2026-06-20
---

# Functional Analysis — Banach & Operator Theory
> Linear algebra in infinite dimensions: complete normed spaces (Banach), inner-product spaces
> (Hilbert), bounded linear operators, dual spaces, and spectral theory. The framework that makes
> "a function as a point in a space" and "an operator as a matrix" rigorous — the language of RKHS,
> kernel methods, Fourier analysis, PDE-views of diffusion, and operator-theoretic deep learning.

**Why it matters:** every time you treat a *function* as a vector (kernels, GPs, Fourier, neural
operators) you're standing in a Banach or Hilbert space. The Riesz representation theorem is *why*
RKHS evaluation is an inner product; spectral theory is *why* the graph/Fourier transforms
diagonalize operators. This is the connective tissue under cards 3, 11, and 12.

**⭐ Start here — suggested path:**

1. **Anchor on Banach spaces** — watch [Lecture 1: Basic Banach Space Theory](https://www.youtube.com/watch?v=uoL4lQxfgwg) (MIT 18.102). *Completeness + norms is the whole game; start where the rigor starts.*
2. **Get to Hilbert spaces** — watch [Lecture 14: Basic Hilbert Space Theory](https://www.youtube.com/watch?v=EBdgFFf54U0). *Inner products, orthogonality, projections — the setting for least-squares, Fourier, and RKHS.*
3. **Read it carefully** — work the MIT [18.102 course materials](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/) alongside the lectures. *Notes + problem sets are where the definitions become reflexes.*
4. **See the key theorems** — focus on Riesz representation, Hahn–Banach, and the open-mapping theorem in [Salamon's free text](https://people.math.ethz.ch/~salamon/PREPRINTS/funcana.pdf). *These four theorems are the "big results" everything else cites.*
5. **Aim at the payoff** — read the spectral-theorem chapter, then jump to [RKHS (card 3)](03-Hilbert-Spaces-and-RKHS.md). *Spectral theory + Riesz is exactly what kernels need.*

## 🎓 Courses (free)
- [MIT 18.102 — Introduction to Functional Analysis (Spring 2021)](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/) — **Casey Rodriguez (MIT OCW)** — full video course + notes + problem sets, Banach → Hilbert → spectral theory.
- [Functional Analysis — free lecture text](https://people.math.ethz.ch/~salamon/PREPRINTS/funcana.pdf) — **Bühler & Salamon (ETH Zürich)** — a complete, rigorous course text, openly posted.
- [Functional Analysis — lecture notes](https://web.math.princeton.edu/~js129/PDFs/teaching/MAT520_fall_2023/MAT520_Lecture_Notes.pdf) — **Princeton MAT520** — concise graduate notes covering the core theorems.

## 🎥 Videos
- [MIT 18.102 — Lecture 1: Basic Banach Space Theory](https://www.youtube.com/watch?v=uoL4lQxfgwg) — **Casey Rodriguez (MIT OCW)** — completeness, norms, and why infinite dimensions are different.
- [MIT 18.102 — Lecture 14: Basic Hilbert Space Theory](https://www.youtube.com/watch?v=EBdgFFf54U0) — **Casey Rodriguez (MIT OCW)** — inner products, orthogonal projection, and Riesz.
- [The Geometric Anatomy of Theoretical Physics — Lec 09: tangent vector spaces (linear-algebra primer)](https://www.youtube.com/watch?v=UPGoXBfm6Js) — **Frederic Schuller** — rigorous vector-space thinking that transfers directly to function spaces.
- [Lecture 2 on kernel methods: RKHS](https://www.youtube.com/watch?v=2uvpOKoiYoI) — **Julien Mairal (Inria)** — sees functional analysis put to work building a Hilbert space of functions.

## 📄 Key Papers
- [Kernel methods in machine learning](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-3/Kernel-methods-in-machine-learning/10.1214/009053607000000677.full) — **Hofmann, Schölkopf & Smola (2008)** — the canonical bridge from operator/Hilbert-space theory to ML.
- [A Primer on Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/1408.0952) — **Manton & Amblard (2015)** — functional analysis assembled from scratch toward RKHS, free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Functional Analysis — lecture text (Ch. 1: normed & Banach spaces)](https://people.math.ethz.ch/~salamon/PREPRINTS/funcana.pdf) — **Bühler & Salamon** — clean, self-contained chapters you can read linearly.
- [From Zero to Reproducing Kernel Hilbert Spaces in Twelve Pages or Less](http://users.umiacs.umd.edu/~hal3//docs/daume04rkhs.pdf) — **Hal Daumé III** — the fastest path from inner-product spaces to operators that matter in ML.

## 📚 Books (free, with chapters)
- [Functional Analysis — **Ch. 1 (Banach spaces) & Ch. 5 (Hilbert spaces)**](https://people.math.ethz.ch/~salamon/PREPRINTS/funcana.pdf) — **Bühler & Salamon (ETH)** — graduate-standard, fully free.
- [Mathematics for Machine Learning — **Ch. 3 (Analytic Geometry: inner products, norms, projections)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the finite-dimensional on-ramp before infinite dimensions.

## 🔗 In this platform
- Foundations (the basics this builds on): [Norms, Inner Products & Orthogonality](../../01.%20Foundations/concepts/03-Norms-Inner-Products-and-Orthogonality.md) · [Eigenvalues & Eigenvectors](../../01.%20Foundations/concepts/04-Eigenvalues-and-Eigenvectors.md) · [SVD](../../01.%20Foundations/concepts/06-Singular-Value-Decomposition.md)
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Next concepts: [03 Hilbert Spaces & RKHS](03-Hilbert-Spaces-and-RKHS.md) · [12 Fourier Analysis & Signal Processing](12-Fourier-Analysis-and-Signal-Processing.md)
</content>

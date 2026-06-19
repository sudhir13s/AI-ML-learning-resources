---
id: "01-foundations/norms-inner-products-orthogonality"
topic: "Norms, Inner Products & Orthogonality"
parent: "01-foundations"
level: beginner
prereqs: ["01-foundations/vectors-and-vector-spaces"]
interview_frequency: high
updated: 2026-06-20
---

# Norms, Inner Products & Orthogonality
> The inner (dot) product measures *alignment*; a norm measures *length*; orthogonality means
> "no shared direction." Together they give geometry to vector spaces — angles, distances,
> projections — which is exactly what cosine similarity, L1/L2 regularization, least squares, and
> attention's scaled dot-product are built from.

**Why it matters:** the dot product is the most-used operation in ML. Interviewers ask why cosine
similarity divides out magnitude, the difference between L1 and L2 norms (and why one induces
sparsity), what an orthogonal matrix preserves, and why attention scales `QKᵀ` by `1/√d`. All of it
is inner-products and norms.

**⭐ Start here — suggested path:**

1. **Dot product, geometrically** — watch [3B1B: Dot products and duality](https://www.youtube.com/watch?v=LyGKycYT2v0). *Why `a·b` relates to projection and angle, not just a sum of products.*
2. **Norms & distances in ML** — read [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) and [1.09 Manhattan/L1](../../../AI-ML-intuition/Module_1_Representation/1.09_Manhattan_L1_Distance.md). *The exact place these definitions get used.*
3. **Formalize it** — read [MML Ch. 3 (Analytic Geometry)](https://mml-book.github.io/book/mml-book.pdf): inner products, norms, orthogonality, projections. *Definitions and the Cauchy–Schwarz inequality you'll quote.*
4. **Orthogonality & projection** — watch [MIT 18.06: Orthogonal Vectors & Subspaces](https://www.youtube.com/watch?v=YzZUIYRCE38). *Sets up least squares = projection onto a subspace.*
5. **Practice** — do [Khan: Projections](https://www.youtube.com/watch?v=27vT-NWuw0M) and norm exercises. *Cement projection formulas and L1/L2 mechanics.*

## 🎓 Courses (free)
- [MIT 18.06 — Orthogonality & Projections (Lec 14–16)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — orthogonal subspaces, projections, least squares.
- [Khan Academy — Alternate coordinate systems (bases, orthogonality)](https://www.khanacademy.org/math/linear-algebra/alternate-bases) — **Khan Academy** — orthonormal bases, projections, Gram–Schmidt with exercises.

## 🎥 Videos
- [Dot products and duality | Ch. 9](https://www.youtube.com/watch?v=LyGKycYT2v0) — **3Blue1Brown** — the geometry behind the dot product.
- [Orthogonal Vectors and Subspaces (18.06 Lec 14)](https://www.youtube.com/watch?v=YzZUIYRCE38) — **Gilbert Strang (MIT OCW)** — orthogonality and the four-subspaces picture.
- [Projection Matrices and Least Squares (18.06 Lec 16)](https://www.youtube.com/watch?v=osh80YCg_GM) — **Gilbert Strang (MIT OCW)** — projection = best approximation in a subspace.
- [Introduction to projections](https://www.youtube.com/watch?v=27vT-NWuw0M) — **Khan Academy** — the projection formula, worked step by step.

## 📄 Key Papers
- [MML book — Ch. 3 "Analytic Geometry"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — inner products, norms, orthogonality, projections, Cauchy–Schwarz.
- [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — norms, inner products, and orthogonal matrices in ML notation.

## 📰 Articles / Blogs (free, no paywall)
- [Immersive Linear Algebra — Ch. 3 "The Dot Product"](https://immersivemath.com/ila/ch03_dotproduct/ch03.html) — **Ström, Åström & Akenine-Möller** — interactive dot product, length, and angle.
- [A visual explanation of L1 vs L2 regularization](https://explained.ai/regularization/) — **Terence Parr & Jeremy Howard** — why the L1 norm produces sparse solutions, geometrically.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 3 (Analytic Geometry)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — norms, inner products, orthonormal bases, projections.
- [Introduction to Applied Linear Algebra (VMLS) — **Ch. 3 (Norm & Distance), Ch. 5 (Orthogonality)**](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — **Boyd & Vandenberghe** — norms, distance, angle, Gram–Schmidt.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.06 Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) · [1.07–1.08 Euclidean vs Cosine](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) · [1.09 Manhattan/L1](../../../AI-ML-intuition/Module_1_Representation/1.09_Manhattan_L1_Distance.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra)](../Maths%20for%20AI-ML/README.md)
- Prereq: [01 Vectors & Vector Spaces](01-Vectors-and-Vector-Spaces.md) · Related: [12 Convexity](12-Convexity.md) (norm balls)
</content>

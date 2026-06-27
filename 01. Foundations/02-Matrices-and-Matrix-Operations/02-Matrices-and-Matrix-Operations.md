---
id: "01-foundations/matrices-and-matrix-operations"
topic: "Matrices & Matrix Operations"
parent: "01-foundations"
level: beginner
prereqs: ["01-foundations/vectors-and-vector-spaces"]
interview_frequency: high
updated: 2026-06-20
---

# Matrices & Matrix Operations
> A matrix is a linear transformation written down as numbers: it stretches, rotates, projects, and
> mixes space. Multiplication is *composition* of those transformations; rank, inverse, and
> determinant tell you what the transformation keeps, destroys, or can undo. This is the literal
> arithmetic of every dense layer and attention block.

**Why it matters:** "what does this matrix do to space?" is the single most useful linear-algebra
question, and it powers interview staples — why `(AB)ᵀ = BᵀAᵀ`, when an inverse exists, what rank
deficiency means for least squares, and why a layer `Wx + b` is just an affine map.

**⭐ Start here — suggested path:**

1. **See transformations** — watch [3B1B: Linear transformations & matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE), then [Matrix multiplication as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU). *The single best reframing: a matrix is a function on space, and multiplication is doing one after another.*
2. **Get the mechanics** — read [MML Ch. 2.2–2.3](https://mml-book.github.io/book/mml-book.pdf) (matrix multiplication, inverse, transpose). *Locks in the algebra rules and when they apply.*
3. **Practice & extend** — do [Khan: Matrix transformations](https://www.khanacademy.org/math/linear-algebra/matrix-transformations) (multiplication, inverse, determinant). *Build fluency with the operations you'll do by hand in interviews.*
4. **Determinant intuition** — watch [3B1B: The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk). *Why determinant = signed volume scaling, and why det = 0 means non-invertible.*
5. **Connect to ML** — see matrices-as-maps in [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md). *A weight matrix is exactly this transformation.*

## 🎓 Courses (free)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — elimination, multiplication, inverses, the four subspaces.
- [Khan Academy — Matrix transformations](https://www.khanacademy.org/math/linear-algebra/matrix-transformations) — **Khan Academy** — multiplication, inverse, and determinant with exercises.

## 🎥 Videos
- [Linear transformations and matrices | Ch. 3](https://www.youtube.com/watch?v=kYB8IZa5AuE) — **3Blue1Brown** — a matrix *is* what it does to the basis vectors.
- [Matrix multiplication as composition | Ch. 4](https://www.youtube.com/watch?v=XkY2DOUCWMU) — **3Blue1Brown** — why `AB` means "apply B, then A" (and why order matters).
- [The determinant | Ch. 6](https://www.youtube.com/watch?v=Ip3X9LOh2dk) — **3Blue1Brown** — determinant as signed area/volume scaling.
- [Three-dimensional linear transformations | Ch. 5](https://www.youtube.com/watch?v=rHLEWRxRGiM) — **3Blue1Brown** — the same picture in 3D, where multiplication intuition matters most.

## 📄 Key Papers
- [MML book — Ch. 2.2–2.4](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — matrix multiplication, inverse/transpose, rank, and solving linear systems.
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — **Petersen & Pedersen** — the canonical free reference of matrix identities and derivative rules.

## 📰 Articles / Blogs (free, no paywall)
- [Immersive Linear Algebra — Ch. 6 "The Matrix"](https://immersivemath.com/ila/ch06_matrices/ch06.html) — **Ström, Åström & Akenine-Möller** — interactive matrices and transformations.
- [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — matrix operations, trace, rank, and the conventions used throughout ML.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 2 (Linear Algebra)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — matrices, multiplication, inverses, systems of equations.
- [Introduction to Applied Linear Algebra (VMLS) — **Ch. 6–11 (Matrices)**](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — **Boyd & Vandenberghe** — matrices, matrix-vector products, and least squares, applied and free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.04 Graph Representations](../../../AI-ML-intuition/Module_1_Representation/1.04_Graph_Representations.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra)](../Maths%20for%20AI-ML/README.md)
- Prereq: [01 Vectors & Vector Spaces](../01-Vectors-and-Vector-Spaces/01-Vectors-and-Vector-Spaces.md) · Next: [03 Norms, Inner Products & Orthogonality](../03-Norms-Inner-Products-and-Orthogonality/03-Norms-Inner-Products-and-Orthogonality.md) · [04 Eigenvalues & Eigenvectors](../04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md)
</content>

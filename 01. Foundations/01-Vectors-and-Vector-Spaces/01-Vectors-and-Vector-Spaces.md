---
id: "01-foundations/vectors-and-vector-spaces"
topic: "Vectors & Vector Spaces"
parent: "01-foundations"
level: beginner
prereqs: []
interview_frequency: high
updated: 2026-06-20
---

# Vectors & Vector Spaces
> A vector is both a list of numbers and an arrow in space; a vector space is the set of all such
> arrows you can scale and add. Span, linear independence, and basis are the vocabulary for "how
> many independent directions does my data really have?" — the foundation under embeddings,
> features, and every linear model.

**Why it matters:** every datapoint, embedding, and weight row is a vector. Interviewers probe
whether you understand span/basis/independence (e.g. "what does it mean for features to be linearly
dependent?", "what's the dimension of this column space?") because rank, collinearity, and
degrees-of-freedom arguments fall straight out of it.

**⭐ Start here — suggested path:**

1. **Build geometric intuition** — watch [3Blue1Brown: Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs) then [Linear combinations, span & basis](https://www.youtube.com/watch?v=k7RM-ot2NWY). *See vectors as arrows and "span" as the region you can reach before any algebra.*
2. **Pin down the definitions** — read [MML Ch. 2.1–2.4](https://mml-book.github.io/book/mml-book.pdf) (vector spaces, linear independence, basis & rank). *Turns the pictures into precise, exam-ready definitions.*
3. **Work problems** — do [Khan Academy: Vectors & spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces). *Repetition on independence/span is what makes it stick.*
4. **Connect to ML** — read why features-as-vectors and span matter in [AI-ML-intuition 1.01 One-Hot](../../../AI-ML-intuition/Module_1_Representation/1.01_One-Hot_Encoding.md). *Grounds the abstraction in real representation choices.*

## 🎓 Courses (free)
- [MIT 18.06 — The Geometry of Linear Equations (Lec 1)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — starts from vectors and linear combinations; the canonical course.
- [Khan Academy — Vectors and spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces) — **Khan Academy** — guided exercises on span, independence, basis, subspaces.

## 🎥 Videos
- [Vectors | Essence of Linear Algebra Ch. 1](https://www.youtube.com/watch?v=fNk_zzaMoSs) — **3Blue1Brown** — the "what is a vector, really" opener.
- [Linear combinations, span, and basis vectors | Ch. 2](https://www.youtube.com/watch?v=k7RM-ot2NWY) — **3Blue1Brown** — span and basis made visual.
- [The Geometry of Linear Equations (18.06 Lec 1)](https://www.youtube.com/watch?v=J7DzL2_Na80) — **Gilbert Strang (MIT OCW)** — row vs column picture of a linear system.
- [Nonsquare matrices as transformations between dimensions | Ch. 8](https://www.youtube.com/watch?v=v8VSDg_WQlA) — **3Blue1Brown** — why vectors can live in different-dimensional spaces.

## 📄 Key Papers
- [MML book — Ch. 2 "Linear Algebra"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — definitions of vector spaces, independence, basis & rank with worked examples.
- [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford (Ng et al.)** — the applied reference sheet for vectors, subspaces, and notation used across ML.

## 📰 Articles / Blogs (free, no paywall)
- [Immersive Linear Algebra — Ch. 2 "Vectors"](https://immersivemath.com/ila/ch02_vectors/ch02.html) — **Ström, Åström & Akenine-Möller** — the first fully interactive LA textbook; drag the figures.
- [Mathematics for ML (course notes) — vectors & spaces](https://gwthomas.github.io/docs/math4ml.pdf) — **Garrett Thomas (Stanford)** — concise ML-focused notes on vector spaces and bases.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 2.1–2.4 (Linear Algebra)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — vector spaces, linear independence, basis, rank.
- [Introduction to Applied Linear Algebra (VMLS) — **Ch. 1–5**](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — **Boyd & Vandenberghe** — vectors, linear functions, norms; applied and free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.01 One-Hot Encoding](../../../AI-ML-intuition/Module_1_Representation/1.01_One-Hot_Encoding.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
- Curriculum context: [Maths for AI-ML — Phase 1 (Linear Algebra)](../Maths%20for%20AI-ML/README.md)
- Next concepts: [02 Matrices & Matrix Operations](../02-Matrices-and-Matrix-Operations/02-Matrices-and-Matrix-Operations.md) · [03 Norms, Inner Products & Orthogonality](../03-Norms-Inner-Products-and-Orthogonality/03-Norms-Inner-Products-and-Orthogonality.md)
</content>

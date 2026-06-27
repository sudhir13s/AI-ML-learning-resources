---
id: "15-advanced-math/differential-geometry-manifolds"
topic: "Differential Geometry & Manifolds"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["linear-algebra", "multivariable-calculus", "topology"]
interview_frequency: low
updated: 2026-06-20
---

# Differential Geometry & Manifolds
> The calculus of curved spaces: smooth manifolds (spaces that look locally like ℝⁿ), tangent spaces,
> Riemannian metrics, geodesics, and optimization *on* manifolds. The language behind the manifold
> hypothesis, t-SNE/UMAP, natural gradients (info geometry), and constrained optimization where the
> feasible set is itself a curved surface (orthogonal matrices, the simplex, low-rank manifolds).

**Why it matters:** "real data lives on a low-dimensional manifold" is a load-bearing assumption in
representation learning — and Riemannian optimization (gradient on the tangent space, then retract
back) is how you train with hard geometric constraints (Stiefel/Grassmann manifolds, hyperbolic
embeddings). This card supplies the tangent-space/metric vocabulary that information geometry (card 8)
and geometric deep learning (card 11) reuse.

**⭐ Start here — suggested path:**

1. **Build manifold intuition** — watch [Manifolds, Tangent Spaces, and Coordinate Basis](https://www.youtube.com/watch?v=Ys_8Ty_I5XI). *"Locally like ℝⁿ" plus the tangent space, visually.*
2. **Nail the tangent space** — watch [Differential structures: tangent vector spaces (Schuller, Lec 09)](https://www.youtube.com/watch?v=UPGoXBfm6Js). *The rigorous definition — derivations vs curves — that everything else builds on.*
3. **Read a gentle text** — work [MML Ch. 3 (geometry) + Pennsylvania's Riemannian notes, early chapters](https://www.cis.upenn.edu/~cis6100/Riemann.pdf). *Metrics, geodesics, and the exponential map.*
4. **See it in ML** — connect to the manifold hypothesis via the [t-SNE/UMAP intuition](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md). *Where "data on a manifold" stops being a metaphor.*
5. **Optimize on manifolds** — read the Riemannian-optimization chapter (gradient → retraction), then bridge to [information geometry](../08-Information-Geometry/08-Information-Geometry.md). *Natural gradient is Riemannian gradient descent on a statistical manifold.*

## 🎓 Courses (free)
- [An Introduction to Riemannian Geometry — lecture notes](https://www.cis.upenn.edu/~cis6100/Riemann.pdf) — **Jean Gallier (UPenn)** — manifolds, metrics, geodesics, and connections aimed at CS/ML readers, free PDF.
- [Mathematics for Machine Learning — analytic geometry & optimization](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the inner-product/geometry foundation, free book.
- [Geometric Deep Learning — proto-book & lectures](https://geometricdeeplearning.com/) — **Bronstein, Bruna, Cohen & Veličković** — manifolds, symmetry, and geometry for ML, fully free.

## 🎥 Videos
- [Manifolds, Tangent Spaces, and Coordinate Basis](https://www.youtube.com/watch?v=Ys_8Ty_I5XI) — **The Cynical Philosopher** — the cleanest visual intro to manifolds and tangent spaces.
- [Differential structures: tangent vector spaces (Lec 09)](https://www.youtube.com/watch?v=UPGoXBfm6Js) — **Frederic Schuller** — the rigorous tangent-space construction, a graduate standard.
- [Differential Geometry: TpM the tangent space (Lec 12.5)](https://www.youtube.com/watch?v=275IwEl78aw) — **(differential geometry lecture series)** — tangent space as tangents to curves, complementary view.
- [AMMI Geometric Deep Learning — Lecture 1 (Introduction)](https://www.youtube.com/watch?v=PtA0lg_e5nA) — **Michael Bronstein** — manifolds & symmetry as the organizing principle for modern ML.

## 📄 Key Papers
- [Geometric deep learning: going beyond Euclidean data](https://arxiv.org/abs/1611.08097) — **Bronstein et al. (2017)** — the manifesto connecting manifolds/geometry to deep learning, free on arXiv.
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) — **Bronstein, Bruna, Cohen & Veličković (2021)** — the unifying geometric framework, free.

## 📰 Articles / Blogs (free, no paywall)
- [Manifolds: A Gentle Introduction](https://bjlkeng.github.io/posts/manifolds/) — **Brian Keng (Bounded Rationality)** — manifolds and tangent spaces explained for ML readers, free.
- [Geometric Deep Learning — blog & lecture hub](https://geometricdeeplearning.com/) — **Bronstein et al.** — the geometry-first view of deep learning, openly maintained.

## 📚 Books (free, with chapters)
- [An Introduction to Riemannian Geometry — **Ch. 1–3 (manifolds, tangent spaces, metrics)**](https://www.cis.upenn.edu/~cis6100/Riemann.pdf) — **Jean Gallier (UPenn)** — rigorous and CS-friendly, free PDF.
- [Mathematics for Machine Learning — **Ch. 3 (Analytic Geometry) & Ch. 5 (Vector Calculus)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the calculus-on-curved-spaces prerequisites, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.11–1.12 Dimensionality Reduction (t-SNE/UMAP)](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md) · [1.10 Mahalanobis Distance](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Foundations (the basics this builds on): [Derivatives & Gradients](../../01.%20Foundations/08-Derivatives-and-Gradients/08-Derivatives-and-Gradients.md) · [Jacobian & Hessian](../../01.%20Foundations/10-Jacobian-and-Hessian/10-Jacobian-and-Hessian.md)
- Prerequisite & next: [08 Information Geometry](../08-Information-Geometry/08-Information-Geometry.md) · [11 Spectral Graph Theory](../11-Spectral-Graph-Theory/11-Spectral-Graph-Theory.md)
- Related domain (dimensionality reduction): [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
</content>

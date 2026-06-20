---
id: "15-advanced-math/information-geometry"
topic: "Information Geometry"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["differential-geometry", "probability", "kl-divergence"]
interview_frequency: low
updated: 2026-06-20
---

# Information Geometry — Fisher–Rao & the Natural Gradient
> Treat a parametric family of distributions as a **Riemannian manifold** whose metric is the Fisher
> information matrix. Then KL divergence is (locally) squared distance, the steepest-descent direction
> becomes the *natural gradient* `F⁻¹∇L`, and statistical inference becomes geometry. Amari's program
> unifies the exponential family, the Fisher–Rao metric, and dual (e-/m-) connections.

**Why it matters:** the natural gradient is parameterization-invariant and converges faster than
vanilla SGD in ill-conditioned problems; it's the math behind K-FAC, TRPO/PPO's KL trust regions, and
mirror-descent variants. Information geometry is also the rigorous home of "KL is not a distance but
behaves like squared distance locally" — a subtle point interviewers love.

**⭐ Start here — suggested path:**

1. **Recall KL & entropy** — skim the platform's [Entropy/KL intuition](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md). *The Fisher metric is the second-order Taylor term of KL — know KL first.*
2. **Get the big picture** — watch [Introduction to Information Geometry](https://www.youtube.com/watch?v=w6r_jsEBlgU) (Nielsen). *Manifolds of distributions, the Fisher metric, dual connections, in one talk.*
3. **Read the elementary intro** — work [An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271) (Nielsen). *The single best free entry point: Fisher–Rao, divergences, exponential families.*
4. **Connect to optimization** — read [New insights on the natural gradient](https://arxiv.org/abs/1412.1193) (Martens). *Why `F⁻¹∇L` is the right step and how it relates to Gauss–Newton.*
5. **See it scale** — study [K-FAC](https://arxiv.org/abs/1503.05671) and watch [Computational Information Geometry](https://www.youtube.com/watch?v=X3cBhBA1nNw). *How natural gradient is made tractable for deep nets.*

## 🎓 Courses (free)
- [Computational Information Geometry — course hub](https://franknielsen.github.io/IG/index.html) — **Frank Nielsen (Sony CS Labs / École Polytechnique)** — slides, surveys, and lectures on the Fisher–Rao geometry of ML, free.
- [Mathematics for Machine Learning — geometry & probability prerequisites](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the differential-geometry and probability on-ramp, free book.

## 🎥 Videos
- ["Introduction to Information Geometry"](https://www.youtube.com/watch?v=w6r_jsEBlgU) — **Frank Nielsen** — the manifold-of-distributions picture, Fisher metric, and divergences.
- [Computational Information Geometry (MLSS 2015)](https://www.youtube.com/watch?v=X3cBhBA1nNw) — **Frank Nielsen** — information geometry aimed squarely at ML practitioners.
- [Understanding Kernels and Gaussian Processes](https://www.youtube.com/watch?v=1U5sIaTD6xA) — **Philipp Hennig (Tübingen)** — the geometry-of-probabilistic-models mindset that info-geometry formalizes.
- [Manifolds, Tangent Spaces, and Coordinate Basis](https://www.youtube.com/watch?v=Ys_8Ty_I5XI) — **The Cynical Philosopher** — the differential-geometry vocabulary (tangent space, metric) info geometry uses.

## 📄 Key Papers
- [An elementary introduction to information geometry](https://arxiv.org/abs/1808.08271) — **Frank Nielsen (2018)** — the accessible survey: Fisher–Rao metric, dual connections, exponential families, free on arXiv.
- [New insights and perspectives on the natural gradient method](https://arxiv.org/abs/1412.1193) — **James Martens (2014/2020)** — the definitive modern account of natural gradient & its Gauss–Newton links.
- [Revisiting Natural Gradient for Deep Networks](https://arxiv.org/abs/1301.3584) — **Pascanu & Bengio (2013)** — natural gradient brought into deep learning, free on arXiv.
- [Optimizing Neural Networks with Kronecker-factored Approximate Curvature (K-FAC)](https://arxiv.org/abs/1503.05671) — **Martens & Grosse (2015)** — makes the Fisher-metric step tractable at scale.

## 📰 Articles / Blogs (free, no paywall)
- [An Elementary Introduction to Information Geometry (Entropy journal, open access)](https://franknielsen.github.io/entropy-22-01100-v2.pdf) — **Frank Nielsen** — the polished, peer-reviewed open-access version, free PDF.
- [Information Geometry course materials](https://franknielsen.github.io/IG/index.html) — **Frank Nielsen** — curated slides, notes, and references kept current.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 6 (Probability & Distributions) & Ch. 7 (Continuous Optimization)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the probability + geometry foundation, free.
- [An Elementary Introduction to Information Geometry — **full monograph (open access)**](https://franknielsen.github.io/entropy-22-01100-v2.pdf) — **Frank Nielsen** — a complete, self-contained free reference on the Fisher–Rao geometry.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) · [2.07 Adam Optimizer](../../../AI-ML-intuition/Module_2_Optimization/2.07_Adam_Optimizer.md)
- Foundations (the basics this builds on): [Cross-Entropy & KL Divergence](../../01.%20Foundations/concepts/23-Cross-Entropy-and-KL-Divergence.md) · [Maximum Likelihood Estimation](../../01.%20Foundations/concepts/19-Maximum-Likelihood-Estimation.md)
- Prerequisite & next: [10 Differential Geometry & Manifolds](10-Differential-Geometry-and-Manifolds.md) · [03 Hilbert Spaces & RKHS](03-Hilbert-Spaces-and-RKHS.md)
- Related domain (applied optimizers): [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
</content>

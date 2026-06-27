---
id: "15-advanced-math/optimal-transport"
topic: "Optimal Transport (Wasserstein)"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["convex-analysis-duality", "probability", "linear-programming"]
interview_frequency: medium
updated: 2026-06-20
---

# Optimal Transport — Wasserstein Distances
> The mathematics of moving one probability distribution onto another at minimum cost: the
> Monge and Kantorovich formulations, the Wasserstein distance (a true metric on distributions that
> respects geometry), Kantorovich–Rubinstein duality, Brenier's theorem, and the entropic-regularized
> Sinkhorn algorithm that made OT computationally practical for ML.

**Why it matters:** the Wasserstein distance is the math under WGAN (and its weight-clipping /
gradient-penalty fixes), flow matching, domain adaptation, and distribution-shift metrics. Unlike KL,
it's finite and meaningful even for non-overlapping supports — which is *exactly* why it stabilizes
generative training. Kantorovich duality is a flagship use of the convex duality from card 4.

**⭐ Start here — suggested path:**

1. **Get the intuition** — watch [A Primer on Optimal Transport, Part 1](https://www.youtube.com/watch?v=6iR1E6t1MMQ) (Cuturi). *Monge → Kantorovich → Wasserstein, the conceptual ladder.*
2. **See the duality** — watch [Linking the Theory and Practice of Optimal Transport](https://www.youtube.com/watch?v=yiLS8JXJXig) (Solomon). *Kantorovich–Rubinstein duality, the form WGAN uses.*
3. **Read the reference** — work [Computational Optimal Transport, Ch. 2–4](https://optimaltransport.github.io/book/) (Peyré & Cuturi). *The free, definitive text: formulations, duality, Sinkhorn.*
4. **Understand Sinkhorn** — study the entropic-regularization chapter and the [arXiv version](https://arxiv.org/abs/1803.00567). *Entropic OT is what makes Wasserstein differentiable and GPU-friendly.*
5. **Connect to generative models** — read [WGAN](https://arxiv.org/abs/1701.07875), then watch [a deeper OT-for-ML talk](https://www.youtube.com/watch?v=k1CeOJdQQrc). *Where OT actually lives in modern ML research.*

## 🎓 Courses (free)
- [Computational Optimal Transport — free book & course](https://optimaltransport.github.io/book/) — **Gabriel Peyré & Marco Cuturi** — the definitive resource: Monge/Kantorovich, duality, Sinkhorn, applications, fully free.
- [Optimal Transport and Wasserstein Distance — lecture notes](https://www.stat.cmu.edu/~larry/=sml/Opt.pdf) — **Larry Wasserman (CMU)** — a crisp, statistics-flavored intro, free PDF.
- [Geometric / OT lectures hub](https://optimaltransport.github.io/) — **Peyré (CNRS/ENS)** — slides, course notes, and numerical tours, free.

## 🎥 Videos
- [A Primer on Optimal Transport, Part 1](https://www.youtube.com/watch?v=6iR1E6t1MMQ) — **Marco Cuturi (MLSS)** — Monge, Kantorovich, and the Wasserstein distance from scratch.
- [Linking the Theory and Practice of Optimal Transport](https://www.youtube.com/watch?v=yiLS8JXJXig) — **Justin Solomon (MIT, Harvard CMSA)** — duality and computation, clearly connected.
- [A Primer on Optimal Transport Theory and Algorithms (MLSS Kraków 2023)](https://www.youtube.com/watch?v=k1CeOJdQQrc) — **Marco Cuturi** — an updated, algorithm-focused tour incl. Sinkhorn.
- [Shape Analysis, Lecture 17: Optimal Transport](https://www.youtube.com/watch?v=ILRxpUKWGWA) — **Justin Solomon (MIT 6.838)** — OT applied to geometry, a vivid worked setting.

## 📄 Key Papers
- [Computational Optimal Transport (monograph)](https://arxiv.org/abs/1803.00567) — **Peyré & Cuturi (2019)** — the free arXiv monograph: formulations, duality, entropic OT, Sinkhorn.
- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895) — **Marco Cuturi (2013)** — entropic regularization, the paper that made OT scalable.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — **Arjovsky, Chintala & Bottou (2017)** — Wasserstein distance as a generative-model objective.

## 📰 Articles / Blogs (free, no paywall)
- [Computational Optimal Transport — online book chapters](https://optimaltransport.github.io/book/) — **Peyré & Cuturi** — every chapter readable online, with code notebooks.
- [Optimal Transport and Wasserstein Distance (CMU notes)](https://www.stat.cmu.edu/~larry/=sml/Opt.pdf) — **Larry Wasserman** — the fastest rigorous summary for ML readers, free.

## 📚 Books (free, with chapters)
- [Computational Optimal Transport — **Ch. 2 (Kantorovich), Ch. 3 (duality), Ch. 4 (Sinkhorn)**](https://optimaltransport.github.io/book/) — **Peyré & Cuturi** — the field's standard free text.
- [Mathematics for Machine Learning — **Ch. 6 (Probability) & Ch. 7 (Optimization)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the probability + LP/duality background, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.04 GANs & WGAN](../../../AI-ML-intuition/Module_5_Generation/5.04_GANs_and_WGAN.md) · [1.10 Mahalanobis Distance](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Foundations (the basics this builds on): [Cross-Entropy & KL Divergence](../../01.%20Foundations/concepts/23-Cross-Entropy-and-KL-Divergence.md)
- Prerequisite & next: [04 Convex Analysis & Duality](04-Convex-Analysis-and-Duality.md) · [08 Information Geometry](08-Information-Geometry.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
</content>

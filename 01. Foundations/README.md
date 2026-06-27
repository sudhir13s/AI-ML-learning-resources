---
id: "01-foundations"
topic: "Math & Programming Foundations"
level: beginner
prereqs: []
updated: 2026-06-27
---

# Math & Programming Foundations
> The linear algebra, calculus, probability, and Python that everything else stands on.
> The deep, phase-by-phase math syllabus lives in **[Maths for AI-ML](Maths%20for%20AI-ML/README.md)**.

**⭐ Start here:** [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) + [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr).

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and — where
present — a curated `.references.md` resource card (free, open courses · videos · papers · articles ·
books · cross-links). These are the **canonical homes** for the math the rest of the platform leans
on — Deep Learning, Supervised/Unsupervised Learning, and NLP cross-link *into* these cards rather
than re-deriving the math.
> **✅ ready.** New here? Start with the field overview above, then work top to bottom.

### Linear Algebra
1. ✅ [Vectors & Vector Spaces](01-Vectors-and-Vector-Spaces/01-Vectors-and-Vector-Spaces.md)
2. ✅ [Matrices & Matrix Operations](02-Matrices-and-Matrix-Operations/02-Matrices-and-Matrix-Operations.md)
3. ✅ [Norms, Inner Products & Orthogonality](03-Norms-Inner-Products-and-Orthogonality/03-Norms-Inner-Products-and-Orthogonality.md)
4. ✅ [Eigenvalues & Eigenvectors](04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md)
5. ✅ [Matrix Decompositions (LU · QR · Cholesky)](05-Matrix-Decompositions/05-Matrix-Decompositions.md)
6. ✅ [Singular Value Decomposition (SVD)](06-Singular-Value-Decomposition/06-Singular-Value-Decomposition.md)
7. ✅ [Principal Component Analysis (PCA) — the math](07-Principal-Component-Analysis-Math/07-Principal-Component-Analysis-Math.md)

### Calculus & Optimization
8. ✅ [Derivatives & Gradients](08-Derivatives-and-Gradients/08-Derivatives-and-Gradients.md)
9. ✅ [The Chain Rule (& Backpropagation)](09-The-Chain-Rule/09-The-Chain-Rule.md)
10. ✅ [Jacobian & Hessian](10-Jacobian-and-Hessian/10-Jacobian-and-Hessian.md)
11. ✅ [Taylor Expansion](11-Taylor-Expansion/11-Taylor-Expansion.md)
12. ✅ [Convexity & Convex Functions](12-Convexity/12-Convexity.md)
13. ✅ [Gradient Descent — theory & convergence](13-Gradient-Descent-Theory/13-Gradient-Descent-Theory.md)
14. ✅ [Lagrange Multipliers & Constrained Optimization](14-Lagrange-Multipliers-Constrained-Optimization/14-Lagrange-Multipliers-Constrained-Optimization.md)

### Probability & Statistics
15. ✅ [Probability & Bayes' Theorem](15-Probability-and-Bayes-Theorem/15-Probability-and-Bayes-Theorem.md)
16. ✅ [Random Variables & Distributions](16-Random-Variables-and-Distributions/16-Random-Variables-and-Distributions.md)
17. ✅ [Expectation, Variance & Covariance](17-Expectation-Variance-Covariance/17-Expectation-Variance-Covariance.md)
18. ✅ [Law of Large Numbers & the CLT](18-LLN-and-CLT/18-LLN-and-CLT.md)
19. ✅ [Maximum Likelihood Estimation (MLE)](19-Maximum-Likelihood-Estimation/19-Maximum-Likelihood-Estimation.md)
20. ✅ [Bayesian Inference (priors, posteriors, MAP)](20-Bayesian-Inference/20-Bayesian-Inference.md)
21. ✅ [Hypothesis Testing & Confidence Intervals](21-Hypothesis-Testing-and-Confidence-Intervals/21-Hypothesis-Testing-and-Confidence-Intervals.md)

### Information Theory
22. ✅ [Entropy](22-Entropy/22-Entropy.md)
23. ✅ [Cross-Entropy & KL Divergence](23-Cross-Entropy-and-KL-Divergence/23-Cross-Entropy-and-KL-Divergence.md)
24. ✅ [Mutual Information](24-Mutual-Information/24-Mutual-Information.md)

### Related concepts (covered in another section)
> These build *on* the math above but live where they're applied, to avoid repetition.
- **Applied optimizers** — Momentum · Adam · AdamW · learning-rate schedules → [Deep Learning](../05.%20Deep_Learning/README.md)
- **Regularization in practice** — L1/L2 weight decay · dropout → [Deep Learning](../05.%20Deep_Learning/README.md)
- **Dimensionality reduction for visualization** — t-SNE · UMAP → [Unsupervised Learning](../04.%20Unsupervised_Learning/README.md)
- **Clustering & EM** — k-means · Gaussian Mixture Models → [Unsupervised Learning](../04.%20Unsupervised_Learning/README.md)

## 🎓 Courses (free)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — the legendary linear-algebra course.
- [Harvard Stat 110: Probability](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein** — the gold-standard probability course.

## 🎥 Videos
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — **3Blue1Brown** — see what matrices *do*.
- [Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) — **StatQuest** — probability & stats, intuition-first.

## 📰 Articles
- [Stanford CS229 math review (linear algebra, probability)](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — the applied cheat-sheet.
- [Python Crash Course](https://realpython.com/python-crash-course/) — **Real Python** — NumPy/Pandas/Matplotlib that ML actually uses.

## 📚 Books (free)
- [Mathematics for Machine Learning](https://mml-book.github.io/) — **Deisenroth et al.** — the on-ramp to all of it.
- [Think Stats](https://greenteapress.com/wp/think-stats-2e/) — **Allen B. Downey** — free, computational statistics in Python.

## 🔗 In this platform
- The full math curriculum (phases + specializations): [Maths for AI-ML](Maths%20for%20AI-ML/README.md)
- Math-as-ML-concepts deep dives: [AI-ML-intuition Module 0–2](../../AI-ML-intuition/)

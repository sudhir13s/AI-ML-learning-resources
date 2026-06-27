---
id: "01-foundations/expectation-variance-covariance"
topic: "Expectation, Variance & Covariance"
parent: "01-foundations"
level: beginner
prereqs: ["01-foundations/random-variables-and-distributions"]
interview_frequency: high
updated: 2026-06-20
---

# Expectation, Variance & Covariance
> Expectation is the probability-weighted average (the "center"); variance measures spread;
> covariance measures how two variables move together, and the covariance matrix packages all
> pairwise covariances. These moments are the summary statistics behind loss functions, the
> bias–variance tradeoff, PCA, and the multivariate Gaussian.

**Why it matters:** expectation linearity, variance of sums, and the covariance matrix come up
constantly — from "why does averaging gradients reduce variance" to deriving PCA from the covariance
matrix to the bias–variance decomposition. Interviewers expect fluency with `E`, `Var`, `Cov`, and
their algebra (e.g. `Var(X) = E[X²] − E[X]²`).

**⭐ Start here — suggested path:**

1. **Expectation** — watch [StatQuest: Expected Values, Main Ideas](https://www.youtube.com/watch?v=KLs_7b7SKi4). *The probability-weighted-average intuition.*
2. **Variance & covariance** — watch [StatQuest: Covariance, Clearly Explained](https://www.youtube.com/watch?v=qtaqvPAeEJY). *How covariance encodes co-movement (and its sign).*
3. **The covariance matrix** — watch [ritvikmath: The Covariance Matrix](https://www.youtube.com/watch?v=152tSYtiQbw). *The object PCA and the multivariate Gaussian are built on.*
4. **Formalize** — read [MML Ch. 6.4 (Summary Statistics & Independence)](https://mml-book.github.io/book/mml-book.pdf). *Moments, covariance, and their algebra.*
5. **Connect to ML** — read [AI-ML-intuition 0.03 Expectation, Variance, Covariance](../../../AI-ML-intuition/Module_0_Foundations/0.03_Expectation_Variance_Covariance.md). *Where moments power ML reasoning.*

## 🎓 Courses (free)
- [Harvard Stat 110: Probability — expectation & variance](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein (Harvard)** — rigorous treatment of moments, linearity, and covariance.
- [Khan Academy — Random variables (expected value & variance)](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) — **Khan Academy** — expected value and variance with exercises.

## 🎥 Videos
- [Expected Values, Main Ideas](https://www.youtube.com/watch?v=KLs_7b7SKi4) — **StatQuest (Josh Starmer)** — expectation built from the ground up.
- [Covariance, Clearly Explained](https://www.youtube.com/watch?v=qtaqvPAeEJY) — **StatQuest (Josh Starmer)** — covariance intuition and its sign.
- [The Covariance Matrix: Data Science Basics](https://www.youtube.com/watch?v=152tSYtiQbw) — **ritvikmath** — the covariance matrix used in PCA & Gaussians.
- [Binomial distributions | part 1](https://www.youtube.com/watch?v=8idr1WZ1A7Q) — **3Blue1Brown** — expectation/variance of a concrete distribution.

## 📄 Key Papers
- [MML book — Ch. 6.4 (Summary Statistics & Independence)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — expectation, variance, covariance, and the covariance matrix.
- [CS229 Probability Review — moments & covariance](https://cs229.stanford.edu/section/cs229-prob.pdf) — **Stanford (Ng et al.)** — expectation, variance, covariance, and the multivariate Gaussian.

## 📰 Articles / Blogs (free, no paywall)
- [Seeing Theory — Ch. 4 (Frequentist Inference / expected value)](https://seeing-theory.brown.edu/frequentist-inference/index.html) — **Brown University** — interactive expectation and variance.
- [A geometric interpretation of the covariance matrix](https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/) — **Vincent Spruyt** — covariance as the shape/orientation of a data cloud (free).

## 📚 Books (free, with chapters)
- [Introduction to Probability — **Ch. 4 (Expectation), Ch. 7 (Covariance)**](http://probabilitybook.net/) — **Blitzstein & Hwang** — the Stat 110 textbook; 1st-edition PDF free.
- [Mathematics for Machine Learning — **Ch. 6.4 (Summary Statistics)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — moments and covariance for ML.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.03 Expectation, Variance, Covariance](../../../AI-ML-intuition/Module_0_Foundations/0.03_Expectation_Variance_Covariance.md) · [1.05 Spectral Methods (covariance → PCA)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Probability, row 3.3)](../Maths%20for%20AI-ML/README.md)
- Prereq: [16 Random Variables & Distributions](16-Random-Variables-and-Distributions.md) · Next: [18 LLN & CLT](18-LLN-and-CLT.md) · Related: [07 PCA — the math](07-Principal-Component-Analysis-Math.md)
</content>

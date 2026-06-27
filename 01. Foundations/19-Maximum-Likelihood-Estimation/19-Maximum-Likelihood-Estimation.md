---
id: "01-foundations/maximum-likelihood-estimation"
topic: "Maximum Likelihood Estimation (MLE)"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/random-variables-and-distributions", "01-foundations/derivatives-and-gradients"]
interview_frequency: very-high
updated: 2026-06-20
---

# Maximum Likelihood Estimation (MLE)
> MLE picks the parameters that make the observed data most probable: maximize the likelihood
> (usually the log-likelihood). It's the principle that *derives* most ML loss functions — squared
> error falls out of a Gaussian likelihood, cross-entropy out of a categorical/Bernoulli one — so it
> unifies "why this loss?" across models.

**Why it matters:** "derive the loss from a likelihood" is a top-tier ML interview move. You should
show that least squares = Gaussian MLE, logistic regression's loss = Bernoulli MLE, and softmax
cross-entropy = categorical MLE — and explain why we maximize the *log*-likelihood and how MLE
relates to MAP (add a prior).

**⭐ Start here — suggested path:**

1. **Likelihood ≠ probability** — watch [StatQuest: Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4). *Clears up the single most common confusion before any math.*
2. **The method** — watch [StatQuest: Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc). *Set up the likelihood, take logs, differentiate, solve.*
3. **Derive ML losses** — read [CS229 Lecture notes 1 (GLMs / MLE)](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf). *Least squares and logistic regression *as* MLE — the key interview connection.*
4. **Formalize** — read [MML Ch. 8.3 (Parameter Estimation: MLE & MAP)](https://mml-book.github.io/book/mml-book.pdf). *MLE, MAP, and where the prior enters.*
5. **Connect to ML** — read [AI-ML-intuition 3.04 Maximum Likelihood Estimation](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md). *The platform's deep dive, tying MLE to cross-entropy.*

## 🎓 Courses (free)
- [Stanford CS229 — Generalized Linear Models / MLE notes](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) — **Stanford (Ng et al.)** — derives regression losses from likelihoods.
- [Harvard Stat 110 → Stat 111 (Inference)](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein (Harvard)** — likelihood and estimation foundations, free.

## 🎥 Videos
- [In Statistics, Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) — **StatQuest (Josh Starmer)** — the crucial conceptual distinction.
- [Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) — **StatQuest (Josh Starmer)** — the method end to end.
- [Machine Learning: Maximum Likelihood Estimation](https://www.youtube.com/watch?v=sguol03tfWo) — **Boris Meinardus** — MLE framed directly for ML loss functions.
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) — **3Blue1Brown** — the prior/likelihood split that distinguishes MLE from MAP.

## 📄 Key Papers
- [MML book — Ch. 8.3 (Parameter Estimation: MLE & MAP)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — MLE, MAP, and the likelihood principle for ML.
- [CS229 Lecture Notes 1 (GLMs)](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) — **Stanford (Ng et al.)** — least squares & logistic regression derived as MLE.

## 📰 Articles / Blogs (free, no paywall)
- [A Gentle Introduction to Maximum Likelihood Estimation](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/) — **Jason Brownlee (ML Mastery)** — MLE for ML, free and accessible.
- [Maximum likelihood (StatLect)](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) — **Marco Taboga** — a rigorous, free reference treatment.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 8.3 (MLE & MAP)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — the canonical free chapter.
- [An Introduction to Statistical Learning — **Ch. 4 (Logistic Regression / likelihood)**](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — likelihood-based fitting, applied.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.04 Maximum Likelihood Estimation](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md) · [3.03 Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md)
- Curriculum context: [Maths for AI-ML — Phase 4 (Statistics, row 4.1)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [16 Random Variables & Distributions](16-Random-Variables-and-Distributions.md) · [08 Derivatives & Gradients](08-Derivatives-and-Gradients.md) · Next: [20 Bayesian Inference](20-Bayesian-Inference.md) · Related: [23 Cross-Entropy & KL Divergence](23-Cross-Entropy-and-KL-Divergence.md)
</content>

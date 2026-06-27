---
id: "01-foundations/bayesian-inference"
topic: "Bayesian Inference (priors, posteriors, MAP)"
parent: "01-foundations"
level: advanced
prereqs: ["01-foundations/probability-and-bayes-theorem", "01-foundations/maximum-likelihood-estimation"]
interview_frequency: high
updated: 2026-06-20
---

# Bayesian Inference — priors, posteriors, MAP
> Bayesian inference treats parameters as *random* and updates a prior into a posterior with data:
> `posterior ∝ likelihood × prior`. The **MAP** estimate is the posterior's peak; full Bayes keeps
> the whole posterior to quantify uncertainty. This is the basis for regularization-as-priors,
> Bayesian models, and the ELBO behind VAEs.

**Why it matters:** the MLE-vs-MAP-vs-full-Bayes distinction is a frequent interview topic, as is
"L2 regularization = Gaussian prior" and "L1 = Laplace prior." You should derive a posterior for a
simple model (e.g. Beta–Binomial), explain conjugacy, and articulate when uncertainty (full
posterior) matters versus a point estimate.

**⭐ Start here — suggested path:**

1. **The mindset** — watch [StataCorp: Introduction to Bayesian statistics, part 1](https://www.youtube.com/watch?v=0F0QoMCSKJ4) and revisit [3B1B: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM). *Priors → posteriors as belief updating.*
2. **MLE vs MAP vs Bayes** — read [MML Ch. 8.3–8.4 (MLE, MAP, Bayesian inference)](https://mml-book.github.io/book/mml-book.pdf). *The three estimators and where the prior enters.*
3. **A worked posterior** — read [Think Bayes Ch. 1–3](https://allendowney.github.io/ThinkBayes2/) (Beta–Binomial, conjugacy). *Compute a posterior in Python end to end.*
4. **Regularization as a prior** — connect L2/L1 penalties to Gaussian/Laplace priors (covered in CS229 / MML). *The cleanest "why does regularization work" answer.*
5. **Connect to ML** — read [AI-ML-intuition 5.02 Latent-Variable Models (ELBO/VAEs)](../../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md). *Where Bayesian inference scales to deep generative models.*

## 🎓 Courses (free)
- [Harvard Stat 110: Probability (Bayes & conditioning)](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein (Harvard)** — the probabilistic foundations of Bayesian reasoning, free.
- [Stanford CS229 — MAP / Bayesian methods notes](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) — **Stanford (Ng et al.)** — MAP estimation and priors as regularization.

## 🎥 Videos
- [Introduction to Bayesian statistics, part 1: the basic concepts](https://www.youtube.com/watch?v=0F0QoMCSKJ4) — **StataCorp** — priors, likelihoods, and posteriors clearly framed.
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) — **3Blue1Brown** — the update rule that drives all of it.
- [You Know I'm All About that Bayes (Crash Course Statistics #24)](https://www.youtube.com/watch?v=9TDjifpGj-k) — **CrashCourse** — Bayesian vs frequentist thinking, accessibly.
- [The Bayesian Trap](https://www.youtube.com/watch?v=R13BD8qKeTg) — **Veritasium** — why the prior (base rate) cannot be ignored.

## 📄 Key Papers
- [MML book — Ch. 8.3–8.4 (Parameter Estimation, Bayesian Inference)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — MLE/MAP/Bayes and the role of priors.
- [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2014)** — Bayesian inference scaled to deep models via the ELBO.

## 📰 Articles / Blogs (free, no paywall)
- [Think Bayes (online book)](https://allendowney.github.io/ThinkBayes2/) — **Allen B. Downey** — Bayesian inference in Python, fully free and computational.
- [Seeing Theory — Ch. 5 (Bayesian Inference)](https://seeing-theory.brown.edu/bayesian-inference/index.html) — **Brown University** — interactive priors/likelihoods/posteriors.

## 📚 Books (free, with chapters)
- [Think Bayes — **Ch. 1–5**](https://allendowney.github.io/ThinkBayes2/) — **Allen B. Downey** — Bayes, conjugacy, and estimation in code; free.
- [Mathematics for Machine Learning — **Ch. 8.3–8.4 (Estimation & Bayesian Inference)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — the canonical free chapter.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.01 Probability & Bayes' Theorem](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) · [5.02 Latent-Variable Models (ELBO/VAEs)](../../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Probability, row 3.6)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [15 Probability & Bayes' Theorem](15-Probability-and-Bayes-Theorem.md) · [19 Maximum Likelihood Estimation](19-Maximum-Likelihood-Estimation.md)
</content>

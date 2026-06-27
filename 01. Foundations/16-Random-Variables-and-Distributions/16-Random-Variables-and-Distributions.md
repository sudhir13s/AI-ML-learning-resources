---
id: "01-foundations/random-variables-and-distributions"
topic: "Random Variables & Distributions"
parent: "01-foundations"
level: beginner
prereqs: ["01-foundations/probability-and-bayes-theorem"]
interview_frequency: high
updated: 2026-06-20
---

# Random Variables & Distributions
> A random variable maps outcomes to numbers; its distribution (PMF/PDF/CDF) says how probability
> is spread over those numbers. Knowing the workhorse distributions — Bernoulli, binomial, Gaussian,
> categorical, Poisson, exponential — and especially the (multivariate) Gaussian is the entry fee for
> probabilistic ML.

**Why it matters:** models *are* distributions — a classifier outputs a categorical, a VAE assumes
Gaussians, a language model factorizes a joint. Interviewers ask the difference between PMF/PDF/CDF,
the parameters and shape of the Gaussian, why the multivariate Gaussian's covariance matters, and
when to reach for which distribution.

**⭐ Start here — suggested path:**

1. **PMF/PDF/CDF** — watch [zedstatistics: Probability Distribution Functions (PMF, PDF, CDF)](https://www.youtube.com/watch?v=YXLVjCKVP7U). *Nails the distinction people most often confuse.*
2. **Key discrete & continuous** — watch [3B1B: Binomial distributions](https://www.youtube.com/watch?v=8idr1WZ1A7Q) and [StatQuest: The Normal Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0). *The two most-used families.*
3. **Formalize** — read [MML Ch. 6 (Probability & Distributions)](https://mml-book.github.io/book/mml-book.pdf), incl. the Gaussian and multivariate Gaussian. *Definitions, moments, and the Gaussian's role in ML.*
4. **Reference** — keep [CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) and [Stat 110](https://projects.iq.harvard.edu/stat110/home) handy. *Distribution tables and properties.*
5. **Connect to ML** — read [AI-ML-intuition 0.02 Distributions & the Gaussian](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md). *Why Gaussians are everywhere in ML.*

## 🎓 Courses (free)
- [Harvard Stat 110: Probability](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein (Harvard)** — random variables and the full distribution zoo, with free lectures.
- [Khan Academy — Random variables](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) — **Khan Academy** — discrete/continuous RVs, expected value, and distributions with exercises.

## 🎥 Videos
- [Probability Distribution Functions (PMF, PDF, CDF)](https://www.youtube.com/watch?v=YXLVjCKVP7U) — **zedstatistics** — the clearest PMF/PDF/CDF explainer.
- [Binomial distributions | Probabilities of probabilities, part 1](https://www.youtube.com/watch?v=8idr1WZ1A7Q) — **3Blue1Brown** — the canonical discrete distribution, visually.
- [The Normal Distribution, Clearly Explained](https://www.youtube.com/watch?v=rzFX5NWojp0) — **StatQuest (Josh Starmer)** — the Gaussian and its parameters.
- [Why π is in the normal distribution](https://www.youtube.com/watch?v=cy8r7WSuT1I) — **3Blue1Brown** — deeper intuition for the Gaussian's form.

## 📄 Key Papers
- [MML book — Ch. 6 (Probability & Distributions)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — random variables, common distributions, and the Gaussian/multivariate Gaussian.
- [CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) — **Stanford (Ng et al.)** — random variables, distributions, and the (multivariate) Gaussian for ML.

## 📰 Articles / Blogs (free, no paywall)
- [Seeing Theory — Ch. 3 (Probability Distributions)](https://seeing-theory.brown.edu/probability-distributions/index.html) — **Brown University** — interactive PMF/PDF/CDF and common distributions.
- [Common Probability Distributions: a Field Guide](https://blog.cloudera.com/common-probability-distributions-the-data-scientists-crib-sheet/) — **Sean Owen (Cloudera)** — a free, practical cheat-sheet of when to use which.

## 📚 Books (free, with chapters)
- [Introduction to Probability — **Ch. 3–5 (Random Variables, Distributions)**](http://probabilitybook.net/) — **Blitzstein & Hwang** — the Stat 110 textbook; 1st-edition PDF free.
- [Think Stats — **Ch. 2–6 (Distributions)**](https://greenteapress.com/wp/think-stats-2e/) — **Allen B. Downey** — distributions computationally, in Python; free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.02 Distributions & the Gaussian](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md) · [1.10 Mahalanobis Distance (multivariate Gaussian)](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Probability, row 3.2)](../Maths%20for%20AI-ML/README.md)
- Prereq: [15 Probability & Bayes' Theorem](../15-Probability-and-Bayes-Theorem/15-Probability-and-Bayes-Theorem.md) · Next: [17 Expectation, Variance & Covariance](../17-Expectation-Variance-Covariance/17-Expectation-Variance-Covariance.md) · [18 LLN & CLT](../18-LLN-and-CLT/18-LLN-and-CLT.md)
</content>

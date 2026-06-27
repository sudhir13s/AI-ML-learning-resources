---
id: "01-foundations/hypothesis-testing-and-confidence-intervals"
topic: "Hypothesis Testing & Confidence Intervals"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/lln-and-clt"]
interview_frequency: high
updated: 2026-06-20
---

# Hypothesis Testing & Confidence Intervals
> A hypothesis test asks "could this result be chance?" — a p-value is the probability of data this
> extreme under the null. A confidence interval reports a range of plausible values with a stated
> coverage. Together they're how you decide whether a model improvement, an A/B test, or a metric
> difference is *real*.

**Why it matters:** comparing models honestly is a core ML-engineering skill, and A/B-test
reasoning shows up in interviews. You should explain what a p-value does (and does *not*) mean,
Type I vs Type II error and power, what "95% confidence" actually claims, and why multiple
comparisons inflate false positives.

**⭐ Start here — suggested path:**

1. **Hypothesis tests & null** — watch [StatQuest: Hypothesis Testing & the Null Hypothesis](https://www.youtube.com/watch?v=0oc49DyA3hU). *The framework before any formulas.*
2. **p-values, correctly** — watch [StatQuest: p-values, what they are and how to interpret them](https://www.youtube.com/watch?v=vemZtEM63GY). *Kills the most common misinterpretation.*
3. **Confidence intervals** — watch [StatQuest: Confidence Intervals, Clearly Explained](https://www.youtube.com/watch?v=TqOeMYtOc1w). *What the interval does and doesn't say.*
4. **Formalize** — read [OpenStax Introductory Statistics — Hypothesis Testing & CIs](https://openstax.org/books/introductory-statistics/pages/9-introduction). *Test statistics, errors, power, and interval construction.*
5. **Connect to ML** — read [AI-ML-intuition 0.05 Hypothesis Testing & Confidence Intervals](../../../AI-ML-intuition/Module_0_Foundations/0.05_Hypothesis_Testing_Confidence_Intervals.md). *Evaluating model comparisons honestly.*

## 🎓 Courses (free)
- [Harvard Stat 110 → inference](https://projects.iq.harvard.edu/stat110/home) — **Joe Blitzstein (Harvard)** — the probability foundations behind tests and intervals, free.
- [Khan Academy — Significance tests & confidence intervals](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample) — **Khan Academy** — full unit with exercises.

## 🎥 Videos
- [Hypothesis Testing and the Null Hypothesis, Clearly Explained](https://www.youtube.com/watch?v=0oc49DyA3hU) — **StatQuest (Josh Starmer)** — the testing framework.
- [p-values: what they are and how to interpret them](https://www.youtube.com/watch?v=vemZtEM63GY) — **StatQuest (Josh Starmer)** — the correct interpretation.
- [Confidence Intervals, Clearly Explained](https://www.youtube.com/watch?v=TqOeMYtOc1w) — **StatQuest (Josh Starmer)** — what coverage really means.
- [P-values and significance tests](https://www.youtube.com/watch?v=KS6KEWaoOOE) — **Khan Academy** — a complementary, exercise-oriented walkthrough.

## 📄 Key Papers
- [OpenStax Introductory Statistics — Ch. 9 (Hypothesis Testing with One Sample)](https://openstax.org/books/introductory-statistics/pages/9-introduction) — **OpenStax** — test statistics, Type I/II errors, and power, free and rigorous.
- [MML book — Ch. 6 (Probability)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the distribution machinery tests are built on.

## 📰 Articles / Blogs (free, no paywall)
- [Seeing Theory — Ch. 4 (Frequentist Inference)](https://seeing-theory.brown.edu/frequentist-inference/index.html) — **Brown University** — interactive p-values and confidence intervals.
- [Statistical significance & A/B testing](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) — **Evan Miller** — a classic free essay on testing pitfalls (peeking, multiple comparisons).

## 📚 Books (free, with chapters)
- [Introductory Statistics (OpenStax) — **Ch. 8 (Confidence Intervals), Ch. 9 (Hypothesis Testing)**](https://openstax.org/details/books/introductory-statistics) — **OpenStax** — free, thorough, with examples.
- [Think Stats — **Ch. 9 (Hypothesis Testing)**](https://greenteapress.com/wp/think-stats-2e/) — **Allen B. Downey** — computational hypothesis testing in Python; free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.05 Hypothesis Testing & Confidence Intervals](../../../AI-ML-intuition/Module_0_Foundations/0.05_Hypothesis_Testing_Confidence_Intervals.md)
- Curriculum context: [Maths for AI-ML — Phase 4 (Statistics, row 4.2)](../Maths%20for%20AI-ML/README.md)
- Prereq: [18 Law of Large Numbers & the CLT](18-LLN-and-CLT.md)
</content>

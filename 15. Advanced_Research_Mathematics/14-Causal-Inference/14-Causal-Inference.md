---
id: "15-advanced-math/causal-inference"
topic: "Causal Inference"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["probability", "statistics", "graphical-models"]
interview_frequency: medium
updated: 2026-06-20
---

# Causal Inference
> The mathematics of *cause*, not just correlation: structural causal models (SCMs), causal DAGs, the
> do-operator, the backdoor and frontdoor adjustment criteria, do-calculus, counterfactuals, and the
> potential-outcomes (Neyman–Rubin) framework. The tools that answer "what *would* happen if we
> intervened?" — a strictly harder question than prediction.

**Why it matters:** observational ML predicts P(Y | X); causal inference targets P(Y | do(X)) — the
basis of A/B-test reasoning, confounding control, off-policy evaluation in RL, fairness, and robust
ML under distribution shift. "Correlation isn't causation — so when *can* you infer causation from
data?" is exactly what the backdoor criterion and do-calculus answer, and it's a rising interview
topic for applied/research roles.

**⭐ Start here — suggested path:**

1. **Get oriented** — watch [A Brief Introduction to Causal Inference (Course Preview)](https://www.youtube.com/watch?v=DXBPtpBhGqo) (Brady Neal). *Why prediction ≠ intervention, and what the field is about.*
2. **See the two frameworks** — watch [VMLW 2021: A brief introduction to causal inference](https://www.youtube.com/watch?v=n8HFNel9xpU). *SCM/do-calculus vs potential outcomes, reconciled.*
3. **Read the course** — work [Brady Neal's Introduction to Causal Inference (free book)](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf). *DAGs, backdoor/frontdoor, do-calculus, identification — the best free ML-flavored text.*
4. **Master adjustment** — focus on the backdoor criterion and do-calculus chapters; read [Pearl's Causal Inference: an overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf). *Identification is the central technical skill.*
5. **Connect to ML & decisions** — watch [1.1 Intro & Outline](https://www.youtube.com/watch?v=CfzO4IEMVUk) and link to bandits/RL via the [Game Theory card](15-Game-Theory-and-Multi-Agent-Math.md). *Causality underpins off-policy evaluation and counterfactual learning.*

## 🎓 Courses (free)
- [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) — **Brady Neal** — a complete free course (videos + [book PDF](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf)) from an ML perspective: SCMs, do-calculus, identification.
- [Causal Inference: an overview & primer materials](https://bayes.cs.ucla.edu/PRIMER/) — **Judea Pearl (UCLA)** — the source: SCMs, the do-operator, and the Causal Inference in Statistics primer, free resources.
- [Bandit Algorithms (free book/course)](https://banditalgs.com/) — **Lattimore & Szepesvári** — the decision-theoretic neighbor (interventions over time), fully free.

## 🎥 Videos
- [A Brief Introduction to Causal Inference (Course Preview)](https://www.youtube.com/watch?v=DXBPtpBhGqo) — **Brady Neal** — the field in one accessible overview.
- [VMLW 2021: A brief introduction to causal inference](https://www.youtube.com/watch?v=n8HFNel9xpU) — **Brady Neal** — SCM/do-calculus and potential outcomes, reconciled.
- [1.1 — Intro and Outline of A Brief Introduction to Causal Inference](https://www.youtube.com/watch?v=CfzO4IEMVUk) — **Brady Neal** — the course roadmap (DAGs → backdoor → do-calculus → identification).
- [OAMLS — Generalization Theory](https://www.youtube.com/watch?v=Wr2yvPPIk6k) — **Peter Bartlett** — the generalization mindset that distribution-shift/causal-robustness arguments extend.

## 📄 Key Papers
- [Causal Inference in Statistics: An Overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) — **Judea Pearl (2009)** — the definitive survey: SCMs, do-calculus, identification, free PDF.
- [The Do-Calculus Revisited](https://arxiv.org/abs/1210.4852) — **Judea Pearl (2012)** — the three rules of do-calculus and what they buy you, free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Introduction to Causal Inference — free book](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf) — **Brady Neal** — the most readable free text bridging Pearl's SCMs and Rubin's potential outcomes.
- [Causal Inference: an overview (Pearl)](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) — **Judea Pearl** — the authoritative conceptual map, openly posted.

## 📚 Books (free, with chapters)
- [Introduction to Causal Inference — **Ch. 3 (backdoor), Ch. 4 (do-calculus), Ch. 6 (counterfactuals)**](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf) — **Brady Neal** — the free ML-first causal text.
- [Causal Inference: What If — **Part I (potential outcomes & confounding)**](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) — **Hernán & Robins (Harvard)** — the standard epidemiology/PO reference, free PDF.
- [Bandit Algorithms — **Ch. on stochastic bandits (interventional decisions)**](https://tor-lattimore.com/downloads/book/book.pdf) — **Lattimore & Szepesvári** — the sequential-decision complement, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.01 Probability & Bayes](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) · [6.04 MDPs & Exploration](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.04_MDPs_and_Exploration.md)
- Foundations (the basics this builds on): [Probability & Bayes' Theorem](../../01.%20Foundations/concepts/15-Probability-and-Bayes-Theorem.md) · [Bayesian Inference](../../01.%20Foundations/concepts/20-Bayesian-Inference.md) · [Hypothesis Testing & Confidence Intervals](../../01.%20Foundations/concepts/21-Hypothesis-Testing-and-Confidence-Intervals.md)
- Prerequisite & next: [01 Measure Theory & Probability](01-Measure-Theory-and-Probability-Foundations.md) · [15 Game Theory & Multi-Agent Math](15-Game-Theory-and-Multi-Agent-Math.md)
- Related domain (sequential decisions): [10. Reinforcement Learning](../../08.%20Reinforcement_Learning/concepts/README.md)
</content>

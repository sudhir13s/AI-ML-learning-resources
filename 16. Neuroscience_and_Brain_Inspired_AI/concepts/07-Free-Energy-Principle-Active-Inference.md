---
id: "16-neuroscience/free-energy-principle-active-inference"
topic: "The Free Energy Principle / Active Inference"
parent: "16-neuroscience"
level: advanced
prereqs: ["bayesian-inference", "variational-inference", "predictive-coding"]
interview_frequency: low
updated: 2026-06-20
---

# The Free Energy Principle / Active Inference
> Karl Friston's grand unifying theory: any self-organizing system that resists disorder must minimize
> **variational free energy** — an upper bound on surprise, the same quantity as the negative ELBO in
> machine learning. **Active inference** extends this to action: you don't just update beliefs to match
> the world, you also *act* to make the world match your predictions.

**Why it matters:** the free energy principle is where neuroscience and ML share the *exact same math*
— variational free energy **is** the negative evidence lower bound (ELBO) used to train VAEs.
Predictive coding falls out as a special case. The interview-relevant insight is that perception
(belief updating) and action (active inference) become one optimization, and it directly mirrors the
variational objectives you already know from generative modeling.

**⭐ Start here — suggested path:**

1. **Hear it from the source** — watch [Free Energy Principle — Karl Friston](https://www.youtube.com/watch?v=NIu_dJGyIQI). *Markov blankets, model evidence, and the core claim.*
2. **Get a careful intro** — watch [The Free Energy Principle and predictive processing](https://www.youtube.com/watch?v=UkH-7gZnrr4). *The clearest plain-language unpacking of free energy = surprise bound.*
3. **Read the unified-theory paper** — [The free-energy principle: a unified brain theory?](https://www.nature.com/articles/nrn2787). *Friston's accessible Nature review (open).*
4. **Connect to the ELBO** — recognize free energy as the negative ELBO from variational inference. *The single most useful equivalence for an ML reader.*
5. **Get the math** — [The free energy principle for action and perception: a mathematical review](https://arxiv.org/abs/1705.09156). *Buckley et al.'s rigorous, readable derivation.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — Bayesian-brain and variational-inference tutorials that ground the free-energy view.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the neural-circuit substrate active inference must be implemented on.

## 🎥 Videos
- [Free Energy Principle — Karl Friston](https://www.youtube.com/watch?v=NIu_dJGyIQI) — **Serious Science** — Friston himself on Markov blankets and model evidence.
- [#033 Karl Friston — The Free Energy Principle](https://www.youtube.com/watch?v=KkR24ieh5Ow) — **Machine Learning Street Talk** — long-form, ML-audience interview connecting FEP to learning.
- [The Free Energy Principle and predictive processing](https://www.youtube.com/watch?v=UkH-7gZnrr4) — **Shamil Chandaria** — the clearest conceptual walkthrough of free energy as a surprise bound.
- [Predictive Coding: Why Our Brain Is Constantly Predicting The Future](https://www.youtube.com/watch?v=5eSxcygk8UM) — **Psyched!** — predictive coding, the perception side of active inference.

## 📄 Key Papers
- [The free-energy principle: a unified brain theory?](https://www.nature.com/articles/nrn2787) — **Friston (2010)** — the canonical accessible statement (open Nature review).
- [A free energy principle for the brain](https://www.fil.ion.ucl.ac.uk/~karl/A%20free%20energy%20principle%20for%20the%20brain.pdf) — **Friston, Kilner & Harrison (2006)** — the early formal paper (author-hosted PDF).
- [The free energy principle for action and perception: a mathematical review](https://arxiv.org/abs/1705.09156) — **Buckley, Kim, McGregor & Seth (2017)** — the rigorous, ML-friendly derivation.

## 📰 Articles / Blogs (free, no paywall)
- [The Anticipating Brain Is Not a Scientist (Frontiers in Psychology)](https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00679/full) — **open access** — critical, readable discussion of free-energy/predictive-processing claims.
- [Friston — A free energy principle for the brain (full PDF)](https://www.fil.ion.ucl.ac.uk/~karl/A%20free%20energy%20principle%20for%20the%20brain.pdf) — **author-hosted** — the source paper, free to read.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 10 (Representational Learning)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — generative models and the variational machinery FEP rests on.
- [Neuronal Dynamics — **Ch. 19 (Synaptic Plasticity & Learning)**](https://neuronaldynamics.epfl.ch/online/Ch19.html) — **Gerstner et al.** — the local learning that free-energy minimization implies in circuits.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.01 Probability & Bayes' Theorem](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) · [5.01 Information Theory — Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) — the Bayesian + KL machinery free energy is built from.
- Prereqs in this section: [06 Predictive Coding](06-Predictive-Coding.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

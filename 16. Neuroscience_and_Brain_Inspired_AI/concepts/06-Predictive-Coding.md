---
id: "16-neuroscience/predictive-coding"
topic: "Predictive Coding"
parent: "16-neuroscience"
level: advanced
prereqs: ["bayesian-inference", "backpropagation", "visual-cortex"]
interview_frequency: low
updated: 2026-06-20
---

# Predictive Coding
> A theory of cortex as a **prediction machine**: higher layers send top-down predictions of what
> lower layers should be seeing, lower layers send back only the **prediction error** (the surprise).
> The brain learns by minimizing that error everywhere, locally — a strikingly backprop-like
> objective achieved without a global backward pass.

**Why it matters:** predictive coding is the most concrete *biologically-plausible alternative to
backprop*. Its punchline — proven mathematically — is that predictive-coding networks can
**approximate the backprop gradient using only local computations**. That makes it the bridge between
the brain's local learning and deep learning's global credit assignment, and a recurring topic where
neuroscience meaningfully informs ML.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Predictive Coding: Why Our Brain Is Constantly Predicting The Future](https://www.youtube.com/watch?v=5eSxcygk8UM). *The prediction-error mental model before any math.*
2. **Read the founding paper** — [Rao & Ballard (1999): Predictive coding in the visual cortex](https://www.cs.utexas.edu/~dana/nn.pdf). *Where the theory began; explains extra-classical receptive fields.*
3. **See the backprop link** — watch [Predictive Coding Approximates Backprop (paper explained)](https://www.youtube.com/watch?v=LB4B5FYvtdI). *The headline result, walked through.*
4. **Read that result** — [Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182). *Local error units recover the global gradient.*
5. **Get the full review** — [Predictive Coding: a Theoretical and Experimental Review](https://arxiv.org/abs/2107.12979). *Mechanisms, variants, and open questions in one place.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — Bayesian-inference and hierarchical-inference tutorials underpinning predictive coding.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the circuit and plasticity background predictive coding builds on.

## 🎥 Videos
- [Predictive Coding: Why Our Brain Is Constantly Predicting The Future](https://www.youtube.com/watch?v=5eSxcygk8UM) — **Psyched!** — clean conceptual intro to top-down prediction and bottom-up error.
- [Predictive Coding Approximates Backprop (paper explained)](https://www.youtube.com/watch?v=LB4B5FYvtdI) — **Yannic Kilcher** — the local-vs-global learning result, derivation included.
- [The Free Energy Principle and predictive processing](https://www.youtube.com/watch?v=UkH-7gZnrr4) — **Shamil Chandaria** — situates predictive coding inside the broader free-energy view.
- [The Modular Architecture of Intelligence](https://www.youtube.com/watch?v=-_OgW6KSGE4) — **Artem Kirsanov** — hierarchical, error-driven organization in neural systems.

## 📄 Key Papers
- [Predictive coding in the visual cortex (Rao & Ballard, 1999)](https://www.cs.utexas.edu/~dana/nn.pdf) — **Rao & Ballard** — the founding model; explains end-stopping and extra-classical receptive-field effects.
- [Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182) — **Millidge, Tschantz & Buckley (2020)** — local predictive coding recovers the backprop gradient.
- [Predictive Coding: a Theoretical and Experimental Review](https://arxiv.org/abs/2107.12979) — **Millidge, Seth & Buckley (2021)** — the comprehensive, free survey.

## 📰 Articles / Blogs (free, no paywall)
- [The free-energy principle: a unified brain theory?](https://www.nature.com/articles/nrn2787) — **Karl Friston (2010)** — predictive coding as a special case of free-energy minimization (open Nature review).
- [Rao & Ballard — full PDF](https://www.cs.utexas.edu/~dana/nn.pdf) — **author-hosted** — the original paper, free to read in full.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 10 (Representational Learning)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — generative models and hierarchical inference behind predictive coding.
- [Neuronal Dynamics — **Ch. 19 (Synaptic Plasticity & Learning)**](https://neuronaldynamics.epfl.ch/online/Ch19.html) — **Gerstner et al.** — the local learning rules predictive-coding circuits use.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation — The Chain Rule](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) — the global algorithm predictive coding approximates locally.
- Prereqs in this section: [08 Visual Cortex & CNN Inspiration](08-Visual-Cortex-and-CNN-Inspiration.md)
- Next concepts: [07 Free Energy Principle / Active Inference](07-Free-Energy-Principle-Active-Inference.md) · [14 Biologically-Plausible Backprop Alternatives](14-Biologically-Plausible-Backprop-Alternatives.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

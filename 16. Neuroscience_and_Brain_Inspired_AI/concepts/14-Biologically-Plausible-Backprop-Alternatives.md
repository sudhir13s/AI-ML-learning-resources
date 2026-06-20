---
id: "16-neuroscience/biologically-plausible-backprop-alternatives"
topic: "Biologically-Plausible Backprop Alternatives"
parent: "16-neuroscience"
level: advanced
prereqs: ["backpropagation", "hebbian-learning", "predictive-coding"]
interview_frequency: medium
updated: 2026-06-20
---

# Biologically-Plausible Backprop Alternatives
> Backprop works spectacularly — but the brain almost certainly doesn't do it. It requires a precise
> backward pass that reuses the forward weights (the **weight-transport problem**), global error
> signals, and separate forward/backward phases — none of which biology obviously supports. This card
> surveys the candidates: **feedback alignment, target propagation, predictive coding, equilibrium
> propagation, and the forward-forward algorithm**.

**Why it matters:** this is the intellectual heart of neuro-AI and a top-tier interview topic. You
should be able to state *exactly why* backprop is biologically implausible (weight transport, non-local
credit assignment, two distinct phases), and name how each alternative attacks one of those problems —
e.g. feedback alignment kills weight transport with random feedback, predictive coding makes credit
assignment local, forward-forward removes the backward pass entirely.

**⭐ Start here — suggested path:**

1. **Frame the problem** — read [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) (Lillicrap, Hinton et al.). *The definitive statement of why backprop is implausible and what could replace it.*
2. **Feedback alignment** — [Random synaptic feedback weights support error backpropagation](https://www.nature.com/articles/ncomms13276). *Random feedback weights still let learning work — no weight transport needed.*
3. **Predictive coding route** — watch [Predictive Coding Approximates Backprop (paper explained)](https://www.youtube.com/watch?v=LB4B5FYvtdI). *Local error units recover the backprop gradient.*
4. **Forward-Forward** — watch [This Algorithm Could Make a GPT-4 Toaster Possible](https://www.youtube.com/watch?v=rVzDRfO2sgs), then read [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345). *Two forward passes, no backward pass at all (Hinton, 2022).*
5. **See the local-learning contrast** — review [04 Hebbian Learning & STDP](04-Hebbian-Learning-and-STDP.md). *All these methods reach for local, biologically-realizable updates.*

## 🎓 Courses (free)
- [Neuromatch Academy — Deep Learning](https://deeplearning.neuromatch.io/) — **Neuromatch** — covers credit assignment and biologically-motivated learning rules.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the local plasticity rules these algorithms try to use as their learning signal.

## 🎥 Videos
- [Predictive Coding Approximates Backprop (paper explained)](https://www.youtube.com/watch?v=LB4B5FYvtdI) — **Yannic Kilcher** — how local predictive coding recovers the backprop gradient.
- [This Algorithm Could Make a GPT-4 Toaster Possible](https://www.youtube.com/watch?v=rVzDRfO2sgs) — **Edan Meyer** — clear walkthrough of Hinton's Forward-Forward algorithm.
- [Brain’s Hidden Learning Limits](https://www.youtube.com/watch?v=Ay3_D7VgzZs) — **Artem Kirsanov** — what local, biologically-realizable learning rules can achieve.
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — the neuron whose local signals these rules must work with.

## 📄 Key Papers
- [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) — **Lillicrap, Santoro, Marris, Akerman & Hinton (2020)** — the survey of plausibility problems and candidate solutions.
- [Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276) — **Lillicrap et al. (2016)** — feedback alignment; solves the weight-transport problem.
- [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) — **Hinton (2022)** — learning with two forward passes and no backprop.
- [Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182) — **Millidge, Tschantz & Buckley (2020)** — local predictive coding ≈ backprop gradient.

## 📰 Articles / Blogs (free, no paywall)
- [Random feedback weights support learning in deep neural networks (arXiv)](https://arxiv.org/abs/1411.0247) — **Lillicrap et al.** — the original feedback-alignment preprint, free in full.
- [Predictive Coding: a Theoretical and Experimental Review](https://arxiv.org/abs/2107.12979) — **Millidge, Seth & Buckley** — survey covering predictive-coding-based credit assignment.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 8 (Plasticity & Learning)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — the local learning rules these alternatives build on.
- [Neuronal Dynamics — **Ch. 19 (Synaptic Plasticity & Learning)**](https://neuronaldynamics.epfl.ch/online/Ch19.html) — **Gerstner et al.** — biologically-realizable plasticity, the target of all these methods.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation — The Chain Rule](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) — the algorithm these methods aim to replace with local, plausible learning.
- Prereqs in this section: [04 Hebbian Learning & STDP](04-Hebbian-Learning-and-STDP.md) · [06 Predictive Coding](06-Predictive-Coding.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

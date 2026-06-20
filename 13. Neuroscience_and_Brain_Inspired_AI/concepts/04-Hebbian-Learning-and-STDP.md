---
id: "13-neuroscience/hebbian-learning-and-stdp"
topic: "Hebbian Learning & STDP"
parent: "13-neuroscience"
level: advanced
prereqs: ["biological-neurons", "neural-coding", "linear-algebra"]
interview_frequency: low
updated: 2026-06-20
---

# Hebbian Learning & STDP
> "Cells that fire together, wire together." Hebbian learning is the brain's **local, unsupervised**
> plasticity rule: strengthen a synapse when pre- and post-synaptic activity coincide.
> **Spike-Timing-Dependent Plasticity (STDP)** is its precise, experimentally-measured form — the
> *order* and *millisecond gap* between pre- and post-spikes decides potentiation vs. depression.

**Why it matters:** this is the field's deepest contrast with deep learning. Hebbian/STDP is **local**
(each synapse uses only its own pre/post signals) and needs no global error signal — the opposite of
backprop's non-local credit assignment. Articulating that gap — local correlation-based plasticity vs.
the chain rule — is the single most important idea connecting neuroscience to ML learning theory.

**⭐ Start here — suggested path:**

1. **Get the principle** — read [Wikipedia: Hebbian theory](https://en.wikipedia.org/wiki/Hebbian_theory) for "fire together, wire together" and its instability (why you need normalization). *The one-line rule and its catch.*
2. **Get STDP precisely** — [Scholarpedia: Spike-Timing Dependent Plasticity](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity). *The asymmetric LTP/LTD timing window, authoritative and free.*
3. **Read the modern review** — [The spike timing dependence of plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3431193/) (Feldman, 2012). *Experimental evidence and what STDP does and doesn't explain.*
4. **See the credit-assignment contrast** — compare with [backprop](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md). *Local correlation vs. global gradient — the field's open analogy gap.*
5. **Get the math** — [Neuronal Dynamics Ch. 19 (Synaptic Plasticity & Learning)](https://neuronaldynamics.epfl.ch/online/Ch19.html). *Hebbian/STDP as differential equations you can simulate.*

## 🎓 Courses (free)
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the plasticity chapters formalize Hebbian rules and STDP, with the course free online.
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — synaptic-plasticity tutorials with runnable STDP simulations.

## 🎥 Videos
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — the spiking dynamics whose precise timing STDP reads out.
- [Brain’s Hidden Learning Limits](https://www.youtube.com/watch?v=Ay3_D7VgzZs) — **Artem Kirsanov** — what local plasticity rules can and cannot learn.
- [The Modular Architecture of Intelligence](https://www.youtube.com/watch?v=-_OgW6KSGE4) — **Artem Kirsanov** — how local learning composes into structured, modular computation.
- [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus) — **Artem Kirsanov** — dendritic context that shapes where and when synapses change.

## 📄 Key Papers
- [Spike-Timing Dependent Plasticity (Scholarpedia)](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity) — **Sjöström & Gerstner** — the authoritative, free reference on the STDP timing window.
- [The spike timing dependence of plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3431193/) — **Feldman (2012)** — open-access review: mechanisms, evidence, and limits.
- [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) — **Lillicrap, Santoro, Marris, Akerman & Hinton (2020)** — frames local plasticity (Hebbian/STDP) against backprop's credit assignment.

## 📰 Articles / Blogs (free, no paywall)
- [Wikipedia: Hebbian theory](https://en.wikipedia.org/wiki/Hebbian_theory) — clear statement of the rule, Oja's normalization, and BCM extensions.
- [Neuronal Dynamics — Ch. 19 (Synaptic Plasticity & Learning)](https://neuronaldynamics.epfl.ch/online/Ch19.html) — **Gerstner et al.** — Hebbian and STDP learning equations explained, free online.

## 📚 Books (free, with chapters)
- [Neuronal Dynamics — **Ch. 19 (Synaptic Plasticity & Learning)**](https://neuronaldynamics.epfl.ch/online/Ch19.html) — **Gerstner, Kistler, Naud & Paninski** — the canonical free treatment of Hebbian rules and STDP.
- [Theoretical Neuroscience — **Ch. 8 (Plasticity & Learning)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — Hebb rule, stability, Oja's rule, and unsupervised learning in neurons.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation — The Chain Rule](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) — the global learning rule Hebbian/STDP is contrasted against.
- Prereqs in this section: [01 Biological Neurons & Synapses](01-Biological-Neurons-and-Synapses.md) · [02 Neural Coding](02-Neural-Coding.md)
- Next concepts: [09 Dopamine & RL in the Brain](09-Dopamine-and-RL-in-the-Brain.md) · [14 Biologically-Plausible Backprop Alternatives](14-Biologically-Plausible-Backprop-Alternatives.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

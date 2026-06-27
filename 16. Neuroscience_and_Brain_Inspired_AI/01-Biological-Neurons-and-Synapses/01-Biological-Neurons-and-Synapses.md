---
id: "16-neuroscience/biological-neurons-and-synapses"
topic: "Biological Neurons & Synapses"
parent: "16-neuroscience"
level: intermediate
prereqs: ["differential-equations", "basic-circuits", "activation-functions"]
interview_frequency: low
updated: 2026-06-20
---

# Biological Neurons & Synapses
> The real computing element the artificial neuron is a cartoon of: a membrane that integrates
> incoming current, fires an all-or-none **spike** when its voltage crosses threshold, and talks to
> other neurons across chemical **synapses**. Understand the Hodgkin–Huxley and leaky
> integrate-and-fire models and you understand why a biological neuron is *not* a weighted sum.

**Why it matters:** this is the ground truth every brain-inspired idea (SNNs, STDP, neuromorphic
hardware) builds on. The interview-relevant payoff is the contrast: a `y = σ(Wx + b)` unit is a
rate-coded, stateless caricature of a leaky, threshold, spiking dynamical system — knowing exactly
where the analogy holds and breaks is the whole point of this domain.

**⭐ Start here — suggested path:**

1. **Get the picture** — watch [The Action Potential](https://www.youtube.com/watch?v=oa6rvUJlg7o). *A correct mental model of spike generation before any equations.*
2. **Get the core equation** — watch [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw). *Hodgkin–Huxley made visual — ion channels, gating, voltage.*
3. **Read the model math** — [Neuronal Dynamics Ch. 1 (Integrate-and-Fire)](https://neuronaldynamics.epfl.ch/online/Ch1.html) then [Ch. 2 (Hodgkin–Huxley)](https://neuronaldynamics.epfl.ch/online/Ch2.html). *The LIF and HH equations you can actually simulate.*
4. **See dendrites compute** — watch [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus). *Why a single neuron is closer to a small network than a single unit.*
5. **Simulate one** — work the [Neuromatch Computational Neuroscience](https://compneuro.neuromatch.io/) biological-neuron notebooks. *Implementing LIF cements the "leaky ReLU with state" intuition.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — notebook-driven curriculum; the biological-neuron-models week builds LIF and Hodgkin–Huxley from scratch.
- [Neuronal Dynamics (EPFL online course)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the canonical free course + book on single-neuron and synapse models.

## 🎥 Videos
- [The Action Potential](https://www.youtube.com/watch?v=oa6rvUJlg7o) — **Harvard Extension School** — clear walkthrough of depolarization, threshold, and the all-or-none spike.
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — the Hodgkin–Huxley model visualized: ion channels, gating variables, membrane voltage.
- [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus) — **Artem Kirsanov** — dendritic computation: why one neuron is far richer than a single artificial unit.
- [Python Simulation Of The Hodgkin-Huxley Model](https://www.youtube.com/watch?v=qzhctJxdYyI) — **Science With Tal** — short, code-along simulation that makes the equations concrete.

## 📄 Key Papers
- [A quantitative description of membrane current… (Hodgkin & Huxley, 1952)](https://www.ncbi.nlm.nih.gov/books/NBK11164/) — **Hodgkin & Huxley** — the Nobel-winning model of the action potential (background + the equations).
- [Neuronal cable theory](http://www.scholarpedia.org/article/Neuronal_cable_theory) — **Scholarpedia (Rall et al.)** — how voltage spreads through dendrites; the math behind "mini-brains."
- [Dendritic computation (review)](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) — **open-access review** — how single neurons perform nonlinear computation in their dendrites.

## 📰 Articles / Blogs (free, no paywall)
- [Neuronal Dynamics — Ch. 1: Integrate-and-Fire](https://neuronaldynamics.epfl.ch/online/Ch1.html) — **Gerstner et al.** — the LIF neuron derived and explained, fully free online.
- [Neuronal Dynamics — Ch. 2: Hodgkin–Huxley](https://neuronaldynamics.epfl.ch/online/Ch2.html) — **Gerstner et al.** — the biophysical spiking model in readable form.

## 📚 Books (free, with chapters)
- [Neuronal Dynamics — **Ch. 1–2 (LIF, Hodgkin–Huxley)** and **Ch. 4 (dimensionality, phase plane)**](https://neuronaldynamics.epfl.ch/online/Ch4.html) — **Gerstner, Kistler, Naud & Paninski** — the anchor text for single-neuron models, free in full online.
- [Neuroscience Online / NBK reference — **action potential & membrane chapters**](https://www.ncbi.nlm.nih.gov/books/NBK11164/) — **NCBI Bookshelf** — open biology reference for the underlying physiology.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md) — the artificial neuron a LIF neuron is a caricature of.
- Next concepts: [02 Neural Coding](../02-Neural-Coding/02-Neural-Coding.md) · [03 Spiking Neural Networks](../03-Spiking-Neural-Networks/03-Spiking-Neural-Networks.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/README.md)

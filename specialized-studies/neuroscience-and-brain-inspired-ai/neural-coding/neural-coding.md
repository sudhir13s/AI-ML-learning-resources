---
id: "20-neuroscience/neural-coding"
topic: "Neural Coding (rate · temporal · population)"
parent: "20-neuroscience"
level: intermediate
built_from: ["probability", "information-theory", "biological-neurons"]
interview_frequency: low
updated: 2026-06-20
tier: core
est_minutes: 10
title: "Neural Coding (rate · temporal · population)"
minutes: 10
category: neuroscience-and-brain-inspired-ai
---

# Neural Coding — Rate · Temporal · Population
> How does a spike train *mean* something? Neural coding is the study of the mapping from stimulus to
> spikes (encoding) and from spikes back to stimulus (decoding) — via **rate codes** (how many spikes),
> **temporal codes** (when each spike lands), and **population codes** (the pattern across many cells).
> This is information theory applied to the brain.

**Why it matters:** it's the most *transferable* neuroscience math for an ML person — tuning curves
are receptive fields, population codes are distributed representations, and the rate-vs-temporal
debate is exactly the question SNNs and neuromorphic encoders must answer. The Bayesian-brain /
efficient-coding view also rhymes directly with information bottleneck and entropy.

**⭐ Start here — suggested path:**

1. **Frame the question** — read [Scholarpedia: Spike-timing dependent plasticity](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity) intro for the spike-timing-matters case, then skim a tuning-curve overview. *Why timing and population pattern carry information beyond a mean rate.*
2. **Get the formal coding math** — [Theoretical Neuroscience Ch. 1–3](https://www.gatsby.ucl.ac.uk/~dayan/book/) (Dayan & Abbott). *Encoding/decoding, spike-triggered averages, Fisher information — the rigorous core.*
3. **Connect to information theory** — relate it to [entropy / mutual information](../../../../AI-ML-intuition/foundational-mental-models/information-and-dependence/entropy-and-kl-divergence-intuition.md). *Efficient coding = maximize information per spike given a metabolic budget.*
4. **See it in spiking nets** — read [snnTorch — Spike Encoding tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html). *Rate, latency, and delta encoders implemented in code.*
5. **Go deeper on population codes** — [Neuromatch Computational Neuroscience](https://compneuro.neuromatch.io/) coding notebooks. *Decode a stimulus from a population of tuning curves yourself.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — dedicated tutorials on neural coding, tuning curves, and population decoding.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — covers rate vs. spike-time coding and noise in the encoding chapters.

## 🎥 Videos
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — grounds coding in how a single neuron turns input current into spikes.
- [The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I) — **Artem Kirsanov** — the Bayesian / information-theoretic lens that "efficient coding" depends on.
- [The Action Potential](https://www.youtube.com/watch?v=oa6rvUJlg7o) — **Harvard Extension School** — the discrete spike event that all codes are built from.
- [Brain’s Hidden Learning Limits](https://www.youtube.com/watch?v=Ay3_D7VgzZs) — **Artem Kirsanov** — how representation and coding constrain what a circuit can learn.

## 📄 Key Papers
- [Spike-timing dependent plasticity (Scholarpedia)](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity) — **Sjöström & Gerstner** — why precise spike timing carries information the rate ignores.
- [The spike timing dependence of plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3431193/) — **Feldman (2012)** — open-access review tying temporal coding to learning.
- [Hodgkin & Huxley (1952) — quantitative description of membrane current](https://www.ncbi.nlm.nih.gov/books/NBK11164/) — **Hodgkin & Huxley** — the biophysics that sets the timescales coding lives on.

## 📰 Articles / Blogs (free, no paywall)
- [snnTorch — Spike Encoding tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html) — **Jason Eshraghian** — rate, latency, and delta-modulation encoders with runnable code.
- [Neuronal Dynamics — Ch. 7 (Variability of spike trains)](https://neuronaldynamics.epfl.ch/online/Ch7.html) — **Gerstner et al.** — noise, the Poisson model, and what "rate" really means.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 1–3 (Neural Encoding, Decoding, Information Theory)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — the definitive treatment of neural coding (author-hosted page).
- [Neuronal Dynamics — **Ch. 7 (spike-train statistics)**](https://neuronaldynamics.epfl.ch/online/Ch7.html) — **Gerstner et al.** — variability, renewal processes, and rate estimation, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.01 Information Theory — Entropy & KL](../../../../AI-ML-intuition/foundational-mental-models/information-and-dependence/entropy-and-kl-divergence-intuition.md) · [0.01 Probability & Bayes' Theorem](../../../../AI-ML-intuition/foundational-mental-models/probability-and-belief/probability-and-bayes-intuition.md)
- Next concepts: [03 Spiking Neural Networks](../spiking-neural-networks/spiking-neural-networks.md) · [04 Hebbian Learning & STDP](../hebbian-learning-and-stdp/hebbian-learning-and-stdp.md)
- Related domain: [05. Deep Learning](../../../deep-learning/README.md)

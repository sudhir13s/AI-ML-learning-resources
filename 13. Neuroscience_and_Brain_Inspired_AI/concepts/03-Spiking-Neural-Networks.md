---
id: "13-neuroscience/spiking-neural-networks"
topic: "Spiking Neural Networks (SNNs)"
parent: "13-neuroscience"
level: advanced
prereqs: ["biological-neurons", "backpropagation", "neural-coding"]
interview_frequency: low
updated: 2026-06-20
---

# Spiking Neural Networks (SNNs)
> Networks of model neurons that communicate with discrete **spikes** in continuous time, not smooth
> activations every layer-step. They are the "third generation" of neural nets — event-driven, sparse,
> and the natural software for neuromorphic hardware — but the spike's non-differentiability makes
> training the central challenge, solved today with **surrogate gradients**.

**Why it matters:** SNNs are where biology and deep learning collide on the engineering bench. The
interview-relevant ideas are concrete: why spikes break backprop (zero/undefined gradient at the
threshold), how surrogate gradients fix it, and why event-driven sparsity promises orders-of-magnitude
energy savings — the efficiency argument that ties this card to quantization and neuromorphic chips.

**⭐ Start here — suggested path:**

1. **See why spikes matter** — watch [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw). *The LIF/HH dynamics each SNN unit runs.*
2. **Encode inputs as spikes** — [snnTorch — Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html). *Rate/latency encoding is step one of any SNN.*
3. **Train one end-to-end** — [snnTorch — Training SNNs with backprop](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html). *Surrogate gradients in practice — the key trick.*
4. **Read the bridge paper** — [Training SNNs Using Lessons From Deep Learning](https://arxiv.org/abs/2109.12894). *The definitive tutorial-paper connecting DL and SNNs.*
5. **Understand the gradient trick** — [Surrogate Gradient Learning in SNNs](https://arxiv.org/abs/1901.09948). *Why and how you replace the spike's derivative.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — builds spiking neuron models and networks from the ground up in notebooks.
- [snnTorch tutorial series](https://snntorch.readthedocs.io/en/latest/tutorials/index.html) — **Jason Eshraghian** — a full, free, runnable course on building and training SNNs in PyTorch.

## 🎥 Videos
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — the spiking-neuron dynamics that make a network an SNN.
- [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus) — **Artem Kirsanov** — dendritic nonlinearity that richer spiking models try to capture.
- [How neuromorphic computing will change our world](https://www.youtube.com/watch?v=N9C3kJE7G-Q) — **Intel** — why event-driven spiking computation is energy-efficient on the right hardware.
- [Neuromorphic Computing on Intel's Loihi 2](https://www.youtube.com/watch?v=CxXTiHExBNc) — **AutoML Freiburg/Hannover/Tübingen** — running real SNNs on neuromorphic silicon.

## 📄 Key Papers
- [Training Spiking Neural Networks Using Lessons From Deep Learning](https://arxiv.org/abs/2109.12894) — **Eshraghian et al. (2023)** — the canonical tutorial-paper; SNNs through a deep-learning lens.
- [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948) — **Neftci, Mostafa & Zenke (2019)** — how to backprop through non-differentiable spikes.
- [Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182) — **Millidge, Tschantz & Buckley (2020)** — local, spike-compatible learning that matches backprop.

## 📰 Articles / Blogs (free, no paywall)
- [snnTorch — Training SNNs with backprop](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) — **Jason Eshraghian** — surrogate-gradient training walked through in code.
- [snnTorch on GitHub](https://github.com/jeshraghian/snntorch) — **Jason Eshraghian** — the open-source library, examples, and population-coding tutorials.

## 📚 Books (free, with chapters)
- [Neuronal Dynamics — **Ch. 1–2 (spiking neuron models)** and **Ch. 4 (network dynamics)**](https://neuronaldynamics.epfl.ch/online/Ch4.html) — **Gerstner et al.** — the model neurons SNNs are made of, free online.
- [Theoretical Neuroscience — **Ch. 5–7 (model neurons & network models)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — the network-level dynamics behind spiking circuits.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation — The Chain Rule](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) — what surrogate gradients have to approximate.
- Prereqs in this section: [01 Biological Neurons & Synapses](01-Biological-Neurons-and-Synapses.md) · [02 Neural Coding](02-Neural-Coding.md)
- Next concepts: [05 Neuromorphic Computing](05-Neuromorphic-Computing.md) · [14 Biologically-Plausible Backprop Alternatives](14-Biologically-Plausible-Backprop-Alternatives.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

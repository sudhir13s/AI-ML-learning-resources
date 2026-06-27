---
id: "16-neuroscience/neuromorphic-computing"
topic: "Neuromorphic Computing"
parent: "16-neuroscience"
level: advanced
prereqs: ["spiking-neural-networks", "computer-architecture", "neural-coding"]
interview_frequency: low
updated: 2026-06-20
---

# Neuromorphic Computing
> Hardware that computes the way the brain does: **event-driven, massively parallel, in-memory**, with
> spiking neurons and synapses on-chip instead of a von-Neumann CPU shuttling data to and from RAM.
> Think Intel Loihi, IBM TrueNorth, SpiNNaker — chips where computation happens only when a spike
> arrives, slashing energy for sparse, temporal workloads.

**Why it matters:** this is the *why now* of the whole brain-inspired program — the energy wall.
A GPU burns power on every dense matmul; a neuromorphic chip spends energy only on spikes that fire.
The interview-relevant framing is the same efficiency argument behind quantization and sparsity,
pushed all the way down to the silicon: co-design the algorithm (SNNs) and the substrate.

**⭐ Start here — suggested path:**

1. **Get the vision** — watch [How neuromorphic computing will change our world](https://www.youtube.com/watch?v=N9C3kJE7G-Q). *The energy/sparsity case from Intel's neuromorphic lead.*
2. **See real silicon** — watch [Neuromorphic Computing on Intel's Loihi 2](https://www.youtube.com/watch?v=CxXTiHExBNc). *How SNNs map onto an actual neuromorphic chip.*
3. **Read the landscape** — [Opportunities for neuromorphic computing algorithms and applications](https://www.nature.com/articles/s41928-020-00475-8). *The definitive open survey of where the field is and is going.*
4. **Connect to SNNs** — review [03 Spiking Neural Networks](03-Spiking-Neural-Networks.md). *Neuromorphic hardware exists to run SNNs efficiently.*
5. **Tie it to ML efficiency** — relate to [quantization](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.05_Quantization.md). *Both chase energy-per-inference; neuromorphic does it via events, quantization via bits.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — the spiking-network foundations neuromorphic hardware accelerates.
- [snnTorch tutorial series](https://snntorch.readthedocs.io/en/latest/tutorials/index.html) — **Jason Eshraghian** — train the SNNs that deploy to neuromorphic chips, free and runnable.

## 🎥 Videos
- [How neuromorphic computing will change our world](https://www.youtube.com/watch?v=N9C3kJE7G-Q) — **Intel (Mike Davies)** — the energy and sparsity argument for brain-like hardware.
- [Neuromorphic Computing on Intel's Loihi 2](https://www.youtube.com/watch?v=CxXTiHExBNc) — **AutoML Freiburg/Hannover/Tübingen** — a technical deep dive on programming Loihi 2.
- [The Core Equation Of Neuroscience](https://www.youtube.com/watch?v=zOmhHE2xctw) — **Artem Kirsanov** — the neuron dynamics neuromorphic circuits emulate in silicon.
- [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus) — **Artem Kirsanov** — the dendritic, in-memory computation neuromorphic design aspires to.

## 📄 Key Papers
- [Opportunities for neuromorphic computing algorithms and applications](https://www.nature.com/articles/s41928-020-00475-8) — **Schuman et al. (2022)** — comprehensive survey of neuromorphic hardware, algorithms, and use cases.
- [Training Spiking Neural Networks Using Lessons From Deep Learning](https://arxiv.org/abs/2109.12894) — **Eshraghian et al. (2023)** — how to get accurate SNNs onto neuromorphic chips.
- [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948) — **Neftci, Mostafa & Zenke (2019)** — the training method behind deployable neuromorphic models.

## 📰 Articles / Blogs (free, no paywall)
- [Frontiers in Neuroscience — Neuromorphic engineering](https://www.frontiersin.org/articles/10.3389/fnins.2011.00073/full) — **open access** — foundational perspective on analog/event-driven neuromorphic systems.
- [snnTorch on GitHub](https://github.com/jeshraghian/snntorch) — **Jason Eshraghian** — open library bridging PyTorch SNNs to neuromorphic deployment.

## 📚 Books (free, with chapters)
- [Neuronal Dynamics — **Ch. 1–2 (spiking neuron models)**](https://neuronaldynamics.epfl.ch/online/Ch1.html) — **Gerstner et al.** — the model neurons neuromorphic circuits implement, free online.
- [Theoretical Neuroscience — **Ch. 5–6 (model neurons)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — the biophysical models neuromorphic hardware abstracts.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 7.05 Quantization](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.05_Quantization.md) — the ML-side energy-efficiency lever, parallel to event-driven computation.
- Prereqs in this section: [03 Spiking Neural Networks](03-Spiking-Neural-Networks.md)
- Next concepts: [14 Biologically-Plausible Backprop Alternatives](14-Biologically-Plausible-Backprop-Alternatives.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

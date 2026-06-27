---
id: "16-neuroscience/connectomics"
topic: "Connectomics"
parent: "16-neuroscience"
level: intermediate
prereqs: ["biological-neurons", "graph-theory"]
interview_frequency: low
updated: 2026-06-20
---

# Connectomics
> Mapping the brain's complete wiring diagram — the **connectome** — neuron by neuron, synapse by
> synapse. From the 302-neuron worm (*C. elegans*, the first complete connectome) to the 2024
> 140,000-neuron adult fruit-fly brain, connectomics reconstructs neural circuits from electron
> microscopy using large-scale ML for image segmentation and synapse detection.

**Why it matters:** connectomics is both a data-engineering / computer-vision triumph (petabyte EM
volumes segmented by deep nets) and the structural ground truth for everything else in this section.
The interview-relevant angles: (1) it's a massive ML application — 3D segmentation at scale; (2) the
graph it produces is what brain-inspired architectures abstract; (3) "wiring ≠ function" is the
honest caveat — a static map doesn't give you the dynamics.

**⭐ Start here — suggested path:**

1. **Get the vision** — watch [I am my connectome (TED)](https://www.youtube.com/watch?v=L74TEED6fzw). *Why a wiring diagram might matter — Sebastian Seung's case.*
2. **Go deeper** — watch [Connectome: How the Brain's Wiring Makes Us Who We Are](https://www.youtube.com/watch?v=qS6nTA3DUuY). *The science and the ML behind reconstructing circuits.*
3. **See the first connectome** — [White et al. 1986 — C. elegans connectome (OpenWorm)](http://openworm.org/ConnectomeToolbox/White_1986/). *The 302-neuron worm: where connectomics began.*
4. **See the frontier** — [Neuronal wiring diagram of an adult brain (FlyWire, 2024)](https://www.nature.com/articles/s41586-024-07558-y). *The full fruit-fly connectome — 140k neurons, 50M synapses.*
5. **Explore the data** — browse [FlyWire](https://flywire.ai/) or [MICrONS Explorer](https://www.microns-explorer.org/). *Interact with real reconstructed circuits.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — network/graph analysis of neural circuits, the analytical side of connectomics.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — how circuit structure shapes the dynamics a connectome alone can't reveal.

## 🎥 Videos
- [I am my connectome (TED)](https://www.youtube.com/watch?v=L74TEED6fzw) — **Sebastian Seung** — the foundational argument for mapping the connectome.
- [Connectome: How the Brain's Wiring Makes Us Who We Are](https://www.youtube.com/watch?v=qS6nTA3DUuY) — **Microsoft Research (Sebastian Seung)** — the science and ML of circuit reconstruction.
- [Your Brain Is 150,000 Mini-Brains](https://www.youtube.com/watch?v=Dykkubb-Qus) — **Artem Kirsanov** — the neuron complexity a connectome must capture per node.
- [The Modular Architecture of Intelligence](https://www.youtube.com/watch?v=-_OgW6KSGE4) — **Artem Kirsanov** — reading structure (modules, motifs) off the wiring graph.

## 📄 Key Papers
- [Neuronal wiring diagram of an adult brain (FlyWire)](https://www.nature.com/articles/s41586-024-07558-y) — **Dorkenwald et al. (2024)** — the complete adult Drosophila connectome.
- [Random synaptic feedback weights support error backpropagation](https://www.nature.com/articles/ncomms13276) — **Lillicrap et al. (2016)** — how learning could work without symmetric wiring (relevant to what connectivity implies).
- [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) — **Lillicrap et al. (2020)** — connecting structural wiring to plausible learning algorithms.

## 📰 Articles / Blogs (free, no paywall)
- [OpenWorm — C. elegans connectome (White et al. 1986)](http://openworm.org/ConnectomeToolbox/White_1986/) — **OpenWorm** — the first complete connectome with open data and tools.
- [WormAtlas](https://www.wormatlas.org/) — **WormAtlas** — the canonical open reference for C. elegans anatomy and wiring.
- [MICrONS Explorer](https://www.microns-explorer.org/) — **MICrONS** — open mammalian cortical connectome data, browseable in 3D.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 7 (Network Models)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — how connectivity structure determines network computation.
- [Neuronal Dynamics — **Ch. 12 (Neuronal Populations)**](https://neuronaldynamics.epfl.ch/online/Ch12.html) — **Gerstner et al.** — population/circuit structure and its dynamics, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.04 Graph Representations](../../../AI-ML-intuition/Module_1_Representation/1.04_Graph_Representations.md) — the graph formalism a connectome is analyzed with.
- Prereqs in this section: [01 Biological Neurons & Synapses](01-Biological-Neurons-and-Synapses.md)
- Related domains: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md) · [07. Computer Vision](../../07.%20Computer%20Vision/concepts/README.md)

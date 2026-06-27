---
id: "16-neuroscience/attention-and-working-memory-biological"
topic: "Attention & Working Memory (biological)"
parent: "16-neuroscience"
level: advanced
prereqs: ["neural-coding", "memory-systems", "attention-mechanism"]
interview_frequency: medium
updated: 2026-06-20
---

# Attention & Working Memory (biological)
> Two intertwined cognitive controllers. **Selective attention** is the brain biasing competition
> between stimuli — boosting the gain of behaviorally relevant signals (the biased-competition model).
> **Working memory** is holding information "online" for seconds, classically via **persistent
> neural activity** in prefrontal cortex (and, recent work argues, partly via activity-silent synaptic traces).

**Why it matters:** the name "attention" in transformers is a *loose* analogy to biological attention,
and knowing exactly how loose is a sharp interview signal. Biological attention is gain modulation and
competition, not a softmax over key-query dot products — and biological working memory is recurrent
persistent activity, not a fixed context window. Articulating where the metaphor helps and misleads is
the whole value of this card.

**⭐ Start here — suggested path:**

1. **Get working memory** — watch [The Neurobiology of Working Memory](https://www.youtube.com/watch?v=9lzbEb-7_NI). *Persistent prefrontal activity as the substrate of "holding in mind."*
2. **Read the mechanism** — [Role of Prefrontal Persistent Activity in Working Memory](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4700146/). *The classic delay-period firing account, open access.*
3. **Get selective attention** — [Neuronal Mechanisms of Visual Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/). *Biased competition and gain modulation in cortex.*
4. **Contrast with ML attention** — review [4.08 Multi-Head Attention Routing](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md). *See precisely how the transformer mechanism differs from biological attention.*
5. **See the memory-attention link** — watch [A Brain-Inspired Algorithm For Memory](https://www.youtube.com/watch?v=1WPJdAW-sFo). *Modern Hopfield networks tie associative memory to attention math.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — recurrent-dynamics and attractor tutorials underlying persistent activity.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the recurrent-network models that produce persistent working-memory activity.

## 🎥 Videos
- [The Neurobiology of Working Memory](https://www.youtube.com/watch?v=9lzbEb-7_NI) — **Paul Merritt** — prefrontal persistent activity and the delay-period account of working memory.
- [A Brain-Inspired Algorithm For Memory](https://www.youtube.com/watch?v=1WPJdAW-sFo) — **Artem Kirsanov** — modern Hopfield networks: associative memory that *is* attention mathematically.
- [Theta rhythm: A Memory Clock](https://www.youtube.com/watch?v=5CxSoFK5tOQ) — **Artem Kirsanov** — temporal organization that coordinates attention and memory.
- [The Modular Architecture of Intelligence](https://www.youtube.com/watch?v=-_OgW6KSGE4) — **Artem Kirsanov** — control and routing of information across brain modules.

## 📄 Key Papers
- [Role of Prefrontal Persistent Activity in Working Memory](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4700146/) — **Riley & Constantinidis (2016)** — the persistent-activity model, open access.
- [Neuronal Mechanisms of Visual Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/) — **open access review** — biased competition, gain modulation, and attentional selection in cortex.
- [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) — **Lillicrap et al. (2020)** — context for how recurrent dynamics could support credit assignment over time.

## 📰 Articles / Blogs (free, no paywall)
- [Role of Prefrontal Persistent Activity in Working Memory (Frontiers)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2015.00181/full) — **open access** — readable review of the working-memory persistent-activity debate.
- [Neuronal Mechanisms of Visual Attention (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/) — **open access** — full review of attentional modulation, free to read.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 7 (Network Models: persistent activity & attractors)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — the recurrent dynamics behind working memory.
- [Neuronal Dynamics — **Ch. 17 (Memory & Attractor Dynamics)**](https://neuronaldynamics.epfl.ch/online/Ch17.html) — **Gerstner et al.** — persistent-activity working memory in recurrent networks, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.08 Multi-Head Attention Routing](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md) · [4.07 Gating Mechanisms (LSTM/GRU)](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.07_Gating_Mechanisms_LSTM_GRU.md) — the ML attention/memory mechanisms biological attention is contrasted with.
- Prereqs in this section: [10 Memory Systems (hippocampus · replay)](10-Memory-Systems-Hippocampus-Replay.md) · [02 Neural Coding](02-Neural-Coding.md)
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)

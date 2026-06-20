---
id: "13-neuroscience/memory-systems-hippocampus-replay"
topic: "Memory Systems (hippocampus · replay · consolidation)"
parent: "13-neuroscience"
level: advanced
prereqs: ["biological-neurons", "hebbian-learning", "reinforcement-learning"]
interview_frequency: low
updated: 2026-06-20
---

# Memory Systems — Hippocampus · Replay · Consolidation
> How the brain forms, stores, and stabilizes memories. The **hippocampus** rapidly binds experiences
> (place cells, grid cells form a cognitive map), then during rest and sleep it **replays** compressed
> spike sequences in **sharp-wave ripples**, gradually consolidating memories into neocortex. This is
> the brain's experience replay — and yes, it directly inspired replay buffers in deep RL.

**Why it matters:** the parallels to ML are vivid and exam-worthy: hippocampal replay ↔ experience
replay in DQN; systems consolidation ↔ avoiding catastrophic forgetting; complementary learning
systems (fast hippocampus + slow cortex) ↔ the fast/slow weight ideas in continual learning. It's the
clearest case where a memory mechanism crossed from neuroscience into a working ML algorithm.

**⭐ Start here — suggested path:**

1. **Get the building blocks** — watch [Building Blocks of Memory in the Brain](https://www.youtube.com/watch?v=X5trRLX7PQY). *Engrams, place cells, and how a memory is physically stored.*
2. **See replay & consolidation** — watch [Memory Consolidation: Time Machine of the Brain](https://www.youtube.com/watch?v=NteHQv0ceN4). *Replay during sleep moving memories hippocampus → cortex.*
3. **Read the cognitive map** — [Place Cells, Grid Cells, and Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC4315928/). *The spatial code (O'Keefe / Mosers, 2014 Nobel), open access.*
4. **Read the replay mechanism** — [The hippocampal sharp wave-ripple in memory retrieval & consolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/). *How ripples select and replay experiences.*
5. **Connect to ML** — link replay to [experience replay](../../10.%20Reinforcement_Learning/concepts/README.md) and continual learning. *Same idea: replay past experience to learn stably.*

## 🎓 Courses (free)
- [Neuromatch Academy — Computational Neuroscience](https://compneuro.neuromatch.io/) — **Neuromatch** — attractor networks, Hopfield memory, and hippocampal modeling tutorials.
- [Neuronal Dynamics (EPFL)](https://neuronaldynamics.epfl.ch/online/index.html) — **Gerstner et al.** — the recurrent-network and attractor models behind memory storage.

## 🎥 Videos
- [Building Blocks of Memory in the Brain](https://www.youtube.com/watch?v=X5trRLX7PQY) — **Artem Kirsanov** — engrams, place cells, and the physical substrate of memory.
- [Memory Consolidation: Time Machine of the Brain](https://www.youtube.com/watch?v=NteHQv0ceN4) — **Artem Kirsanov** — replay and the transfer of memories during sleep.
- [Theta rhythm: A Memory Clock](https://www.youtube.com/watch?v=5CxSoFK5tOQ) — **Artem Kirsanov** — phase coding and the temporal organization of hippocampal memory.
- [A Brain-Inspired Algorithm For Memory](https://www.youtube.com/watch?v=1WPJdAW-sFo) — **Artem Kirsanov** — modern Hopfield networks: associative memory that links to attention.

## 📄 Key Papers
- [Place Cells, Grid Cells, and Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC4315928/) — **Moser, Rowland & Moser (2015)** — the hippocampal-entorhinal cognitive map, open access.
- [The hippocampal sharp wave-ripple in memory retrieval and consolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/) — **Joo & Frank (2018)** — how replay events select and consolidate experience.
- [Sharp wave-ripples and sequence replay emerge from structured synaptic interactions](https://elifesciences.org/articles/71850) — **eLife** — a computational model of replay generation, fully open.

## 📰 Articles / Blogs (free, no paywall)
- [The Nobel Prize in Physiology or Medicine 2014 — O'Keefe, Moser & Moser](https://www.nobelprize.org/prizes/medicine/2014/okeefe/facts/) — **Nobel Foundation** — the brain's GPS: place and grid cells.
- [Place Cells, Grid Cells, and Memory (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4315928/) — **Moser et al.** — accessible full review, free to read.

## 📚 Books (free, with chapters)
- [Theoretical Neuroscience — **Ch. 7 (Network Models: attractors & memory)**](https://www.gatsby.ucl.ac.uk/~dayan/book/) — **Dayan & Abbott** — Hopfield/attractor memory, the model of associative recall.
- [Neuronal Dynamics — **Ch. 17 (Memory & Attractor Dynamics)**](https://neuronaldynamics.epfl.ch/online/Ch17.html) — **Gerstner et al.** — attractor networks and working/long-term memory, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md) — the value learning that experience replay (inspired by hippocampal replay) stabilizes.
- Prereqs in this section: [04 Hebbian Learning & STDP](04-Hebbian-Learning-and-STDP.md) · [09 Dopamine & RL in the Brain](09-Dopamine-and-RL-in-the-Brain.md)
- Next concepts: [11 Attention & Working Memory (biological)](11-Attention-and-Working-Memory-Biological.md)
- Related domain: [10. Reinforcement Learning](../../10.%20Reinforcement_Learning/concepts/README.md)

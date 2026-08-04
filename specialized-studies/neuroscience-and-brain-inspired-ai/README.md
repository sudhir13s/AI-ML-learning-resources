---
id: "20-neuroscience-and-brain-inspired-ai"
topic: "Neuroscience & Brain-Inspired AI"
level: advanced
built_from: ["foundations", "deep-learning"]
updated: 2026-06-27
---

# 🧠 Neuroscience & Brain-Inspired AI — Curriculum (Specialization)
> Elective deep-dive track, absorbed and expanded from the retired `math-for-AIML-Q5`
> neuroscience specialization. Format: what to study → why → best resources → connections
> back to [AI-ML-intuition](../../../AI-ML-intuition/).

**Goal:** the mathematics of biological computation — neuron dynamics, plasticity, neural
coding — and the two-way street between neuroscience and modern AI (what the brain inspired,
and where the analogies break).

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its resource card — a short
guided learning path plus the best **free, open** courses, videos, papers, articles, and books for that
topic.
> **✅ ready.** New to the field? Start with the field overview & study-order map below.

### The biological substrate
1. ✅ [Biological Neurons & Synapses](biological-neurons-and-synapses/biological-neurons-and-synapses.md)
2. ✅ [Neural Coding (rate · temporal · population)](neural-coding/neural-coding.md)
3. ✅ [Spiking Neural Networks (SNNs)](spiking-neural-networks/spiking-neural-networks.md)
4. ✅ [Hebbian Learning & STDP](hebbian-learning-and-stdp/hebbian-learning-and-stdp.md)

### Brain-inspired computation & hardware
5. ✅ [Neuromorphic Computing](neuromorphic-computing/neuromorphic-computing.md)
6. ✅ [Predictive Coding](predictive-coding/predictive-coding.md)
7. ✅ [The Free Energy Principle / Active Inference](free-energy-principle-active-inference/free-energy-principle-active-inference.md)
8. ✅ [Visual Cortex & CNN Inspiration](visual-cortex-and-cnn-inspiration/visual-cortex-and-cnn-inspiration.md)

### Learning, reward & memory in the brain
9. ✅ [Dopamine & Reinforcement Learning in the Brain](dopamine-and-rl-in-the-brain/dopamine-and-rl-in-the-brain.md)
10. ✅ [Memory Systems (hippocampus · replay · consolidation)](memory-systems-hippocampus-replay/memory-systems-hippocampus-replay.md)
11. ✅ [Attention & Working Memory (biological)](attention-and-working-memory-biological/attention-and-working-memory-biological.md)

### Interfaces, maps & the backprop question
12. ✅ [Brain-Computer Interfaces (BCIs)](brain-computer-interfaces/brain-computer-interfaces.md)
13. ✅ [Connectomics](connectomics/connectomics.md)
14. ✅ [Biologically-Plausible Backprop Alternatives](biologically-plausible-backprop-alternatives/biologically-plausible-backprop-alternatives.md)

### Related concepts (covered in another section)
> These topics are the *machine-learning* counterparts of the biology above — the brain inspired
> them, so the deep dives live in their home sections and we link out to avoid repetition.
- **Convolutional Neural Networks** — the architecture inspired by the visual cortex's hierarchy → [05. Deep Learning](../../deep-learning/README.md)
- **Attention (ML mechanism)** — the routing operation loosely analogous to biological attention → [05. Deep Learning](../../deep-learning/README.md)
- **RL & Temporal-Difference learning** — the algorithmic side of the dopamine-as-TD-error story → [08. Reinforcement Learning](../../core-machine-learning/reinforcement-learning/README.md)

## Core resource backbone
- **Neuronal Dynamics** (Gerstner et al., EPFL) — [free book + course](https://neuronaldynamics.epfl.ch/) (the anchor)
- **Theoretical Neuroscience** (Dayan & Abbott) — the classic text
- **Neuromatch Academy** — [free computational neuroscience curriculum](https://compneuro.neuromatch.io/) (notebook-driven, excellent)
- **Artem Kirsanov** — [YouTube](https://www.youtube.com/@ArtemKirsanov) — beautiful visual explainers on neuro-AI topics

## Study order & what each module unlocks

| Module | Key sub-topics | Best resources | Connections |
| :--- | :--- | :--- | :--- |
| **N1. Dynamical-systems foundations** | neural state variables, linear stability, attractors, noise | Neuromatch W2D2; Strogatz *Nonlinear Dynamics* ch. 1–6 | the continuous-time view behind [6.04's MDP loop](../../../AI-ML-intuition/decision-making-and-control/mdps-and-environments/mdps-and-exploration-intuition.md) |
| **N2. Single-neuron models** | leaky integrate-and-fire, Hodgkin-Huxley intuition, firing rates vs spikes | Gerstner ch. 1–2, 4; Neuromatch W2D3 | the *real* neuron vs the [4.14 artificial one](../../../AI-ML-intuition/architectural-mechanisms/nonlinear-transformation/activation-functions-and-softmax-intuition.md) (a LIF neuron ≈ leaky ReLU with state) |
| **N3. Synapses & plasticity** | Hebbian learning ("fire together, wire together"), STDP, homeostasis | Gerstner ch. 19; Kirsanov's STDP video | contrast with [backprop](../../../AI-ML-intuition/learning-and-optimization/gradients-and-credit-assignment/chain-rule-and-backpropagation-intuition.md): local vs global credit assignment — the field's deepest open analogy gap |
| **N4. Neural coding** | rate/temporal/population codes, tuning curves, **information theory in the brain**, Bayesian brain | Dayan & Abbott ch. 1–3; Neuromatch W1D5 | [5.01 Entropy/MI](../../../AI-ML-intuition/foundational-mental-models/information-and-dependence/entropy-and-kl-divergence-intuition.md), [0.01 Bayes](../../../AI-ML-intuition/foundational-mental-models/probability-and-belief/probability-and-bayes-intuition.md) |
| **N5. Recurrent circuits & memory** | attractor networks, Hopfield nets, working memory | Gerstner ch. 17; [Hopfield is all you need](https://arxiv.org/abs/2008.02217) | [4.07 gating/LSTM](../../../AI-ML-intuition/architectural-mechanisms/memory-and-gating/lstm-and-gru-gates-intuition.md); modern Hopfield ↔ [attention](../../../AI-ML-intuition/architectural-mechanisms/attention-and-routing/multi-head-attention-intuition.md) |
| **N6. Decision & reward** | evidence accumulation (drift-diffusion), **dopamine = TD error**, predictive coding | Neuromatch W3D2; Schultz's reward-prediction papers | the famous bridge: dopamine neurons compute the [Bellman/TD error](../../../AI-ML-intuition/decision-making-and-control/value-learning/bellman-equation-and-q-learning-intuition.md) — RL's biological vindication |
| **N7. Spiking networks & neuromorphic** | SNNs, event-driven computation, neuromorphic hardware overview | Gerstner ch. 6–8; Intel Loihi / SpiNNaker overviews | the energy argument vs [7.05 quantization](../../../AI-ML-intuition/scaling-adaptation-and-efficiency/compression/quantization-intuition.md)-style efficiency |
| **N8. Brain ↔ modern AI** | local learning vs backprop, energy/predictive-coding views, memory & attention parallels | [Backprop and the brain (Hinton/Lillicrap)](https://www.nature.com/articles/s41583-020-0277-3); Kirsanov | the honest scorecard: where the analogy works (RL, attention-ish) and where it doesn't (backprop, scale) |

### Suggested first pass
1. N1 → N2 (the dynamics language), then N4 (coding — the most transferable math).
2. N6 is the highest-ROI module for an ML person: TD-learning-in-the-brain makes Module 6
   click at a deeper level.
3. N3 + N8 as a pair: the credit-assignment contrast is the intellectually richest thread.

**Completion target:** simulate a leaky integrate-and-fire neuron, explain STDP vs backprop
credit assignment, and tell the dopamine-as-TD-error story accurately.

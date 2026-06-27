---
id: "08-rl/q-learning"
topic: "Q-Learning"
parent: "08-reinforcement-learning"
level: intermediate
prereqs: ["temporal-difference-learning", "bellman-equations"]
interview_frequency: very-high
updated: 2026-06-20
---

# Q-Learning
> The most famous RL algorithm: learn the optimal action-value `Q*(s,a)` directly with a TD update
> whose target uses a **`max` over next actions** — `Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]`.
> Because the target is the greedy action regardless of what was actually played, Q-learning is
> **off-policy**: it can learn the optimal policy from exploratory or even logged behavior.

**Why it matters:** the canonical control algorithm and a near-certain interview question — write the
update, explain why the `max` makes it off-policy, contrast with SARSA's on-policy `Q(s',a')`, state
the convergence conditions (visit every state-action infinitely, decaying α), and explain the
maximization bias that motivates Double Q-learning and DQN.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4). *Derives Q-learning and SARSA side by side; the cleanest on-vs-off-policy explanation.*
2. **Read the chapter** — [Sutton & Barto §6.5 "Q-learning"](http://incompleteideas.net/book/RLbook2020.pdf). *The update rule, off-policy property, and the cliff-walking comparison with SARSA.*
3. **See it implemented** — [deeplizard: Train a Q-Learning Agent](https://www.youtube.com/watch?v=HGeI30uATws). *Building the Q-table + ε-greedy loop makes the update concrete.*
4. **Scale the intuition** — [Pieter Abbeel L2: Deep Q-Learning](https://www.youtube.com/watch?v=Psrhxy88zww). *How tabular Q-learning generalizes once Q becomes a neural net — the bridge to DQN.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 5: Model-Free Control](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — Q-learning vs SARSA, GLIE, and convergence.
- [Stanford CS234 — Model-Free Control](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — Q-learning's off-policy property and convergence guarantees.
- [Spinning Up — Intro to RL (value-based methods)](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — situates Q-learning among the algorithm families.

## 🎥 Videos
- [RL Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) — **David Silver (DeepMind)** — the definitive Q-learning vs SARSA derivation.
- [Q-Learning Explained — RL Tutorial](https://www.youtube.com/watch?v=kEGAMppyWkQ) — **deeplizard** — step-by-step build of the Q-table and the update rule.
- [Train a Q-Learning Agent with Python](https://www.youtube.com/watch?v=HGeI30uATws) — **deeplizard** — hands-on tabular Q-learning from scratch.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — concise framing of value-based control including Q-learning.

## 📄 Key Papers
- [Q-learning](https://link.springer.com/article/10.1007/BF00992698) — **Watkins & Dayan (1992)** — the paper that introduced Q-learning and proved its convergence.
- [Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — **Hado van Hasselt (2010)** — fixes Q-learning's maximization bias (precursor to Double DQN).

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into RL — Q-learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — the off-policy TD-control update with the SARSA contrast.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — where Q-learning fits among model-free methods.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **§6.5 "Q-learning" & §6.6 "Expected SARSA"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the update, off-policy property, maximization bias, and Double Q-learning (§6.7).
- [Algorithms for Reinforcement Learning — **§3.2 (TD control)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the convergence theory for Q-learning.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [05 Temporal-Difference Learning](../05-Temporal-Difference-Learning/05-Temporal-Difference-Learning.md) · Contrast: [07 SARSA](../07-SARSA/07-SARSA.md) · Scale up: [08 Deep Q-Networks](../08-Deep-Q-Networks-DQN/08-Deep-Q-Networks-DQN.md)

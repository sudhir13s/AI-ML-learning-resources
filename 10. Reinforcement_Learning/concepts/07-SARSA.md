---
id: "10-rl/sarsa"
topic: "SARSA"
parent: "10-reinforcement-learning"
level: intermediate
prereqs: ["temporal-difference-learning", "q-learning"]
interview_frequency: high
updated: 2026-06-20
---

# SARSA (on-policy TD control)
> Q-learning's on-policy twin. The name is the transition it uses — **S**tate, **A**ction, **R**eward,
> next **S**tate, next **A**ction — and the update bootstraps on the action *actually taken next*:
> `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') − Q(s,a)]`. Because it evaluates the policy it follows
> (exploration included), SARSA learns a *safer* policy near hazards than the optimistic Q-learning.

**Why it matters:** the other half of the "on-policy vs off-policy" interview question. You'll be
asked why SARSA uses `Q(s',a')` instead of `max_{a'} Q(s',a')`, what the classic cliff-walking example
shows (SARSA takes the safe path, Q-learning the risky optimal one), and what Expected SARSA changes.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4). *Derives SARSA right next to Q-learning; the clearest on-vs-off-policy framing.*
2. **Read the chapter** — [Sutton & Barto §6.4–6.6](http://incompleteideas.net/book/RLbook2020.pdf). *SARSA, Expected SARSA, and the cliff-walking experiment that distinguishes it from Q-learning.*
3. **See the cliff-walk** — [Lilian Weng: RL overview (SARSA section)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). *Why an on-policy agent learns to keep a safety margin.*
4. **Code it** — adapt a tabular agent to use the next-action target instead of the `max`. *One line changes — and the learned path changes — which cements the distinction.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 5: Model-Free Control](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — SARSA, GLIE, and the SARSA-vs-Q-learning comparison.
- [Stanford CS234 — Model-Free Control](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — on-policy TD control and its convergence.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — frames on-policy vs off-policy learning.

## 🎥 Videos
- [RL Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) — **David Silver (DeepMind)** — SARSA derived alongside Q-learning, with the cliff-walking intuition.
- [RL Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA) — **David Silver (DeepMind)** — the TD(0) target SARSA control builds on.
- [Q-Learning Explained — RL Tutorial](https://www.youtube.com/watch?v=kEGAMppyWkQ) — **deeplizard** — the value-update mechanics SARSA shares with Q-learning (swap the target).
- [Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) — **StatQuest (Josh Starmer)** — gentle grounding for the policy-following value update.

## 📄 Key Papers
- [Convergence Results for Single-Step On-Policy RL Algorithms](https://link.springer.com/content/pdf/10.1023/A:1007678930559.pdf) — **Singh, Jaakkola, Littman & Szepesvári (2000)** — proves SARSA's convergence and formalizes the on-policy control setting (SARSA originated in Rummery & Niranjan's 1994 report).
- [A Theoretical and Empirical Analysis of Expected SARSA](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf) — **van Seijen et al. (2009)** — Expected SARSA's lower-variance update and where it sits between SARSA and Q-learning.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into RL — SARSA](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — the on-policy update and the safe-path behavior, illustrated.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — the on-policy/off-policy axis SARSA exemplifies.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **§6.4 "SARSA" & §6.6 "Expected SARSA"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the on-policy TD-control algorithm and the cliff-walking comparison.
- [Algorithms for Reinforcement Learning — **§3.2 (TD control)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — on-policy vs off-policy TD control theory.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [05 Temporal-Difference Learning](05-Temporal-Difference-Learning.md) · Contrast: [06 Q-Learning](06-Q-Learning.md)

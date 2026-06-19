---
id: "10-rl/temporal-difference-learning"
topic: "Temporal-Difference Learning"
parent: "10-reinforcement-learning"
level: intermediate
prereqs: ["monte-carlo-methods", "bellman-equations"]
interview_frequency: very-high
updated: 2026-06-20
---

# Temporal-Difference Learning
> The idea that makes RL *online*: update a value estimate toward a **bootstrapped target** —
> `V(s) ← V(s) + α[r + γV(s') − V(s)]` — using the very next step instead of waiting for the episode
> to end. The bracketed quantity is the **TD error**, the single most important signal in RL (it even
> matches dopamine in the brain). TD(λ) and n-step methods interpolate smoothly between TD(0) and Monte Carlo.

**Why it matters:** TD is the engine under Q-learning, SARSA, and every actor-critic. The interview
staple is "MC vs TD vs DP" (sampling vs bootstrapping vs both), what the TD error is, why bootstrapping
trades bias for lower variance, and how the λ in TD(λ) and eligibility traces unify the family.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA). *Introduces the TD update and the TD error, directly against MC.*
2. **Read the chapter** — [Sutton & Barto Ch. 6 "Temporal-Difference Learning"](http://incompleteideas.net/book/RLbook2020.pdf). *TD(0), the TD error, batch TD vs MC, and why TD often learns faster.*
3. **Get TD(λ) & traces** — [Sutton & Barto Ch. 7 & 12](http://incompleteideas.net/book/RLbook2020.pdf). *n-step TD and eligibility traces — the spectrum from one-step TD to full-return MC.*
4. **Place it on the map** — [Lilian Weng: RL overview (TD section)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). *Where bootstrapping sits and why it powers Q-learning/SARSA next.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 4: Model-Free Prediction](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — TD(0), TD(λ), and the MC-vs-TD comparison.
- [Stanford CS234 — Model-Free Policy Evaluation](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — TD learning, bootstrapping, and convergence.
- [Spinning Up — Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — value functions and the bootstrap target TD uses.

## 🎥 Videos
- [RL Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA) — **David Silver (DeepMind)** — TD(0), the TD error, n-step, and TD(λ).
- [RL Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) — **David Silver (DeepMind)** — TD control (SARSA/Q-learning) built on the TD error.
- [Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) — **StatQuest (Josh Starmer)** — intuitive grounding for bootstrapped value updates.
- [Reinforcement Learning 2: Markov Decision Processes](https://www.youtube.com/watch?v=RmOdTQYQqmQ) — **DeepMind x UCL** — sets up the Bellman target that TD samples.

## 📄 Key Papers
- [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009) — **Richard Sutton (1988)** — the paper that introduced TD learning and TD(λ).
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári (2010)** — convergence of TD methods and stochastic approximation.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into RL — Temporal-Difference Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — TD prediction/control and eligibility traces, well illustrated.
- [Spinning Up — Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — the value/return framing behind the TD target.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 6 "Temporal-Difference Learning"** (+ Ch. 7 n-step, Ch. 12 eligibility traces)](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the canonical treatment of TD(0), the TD error, n-step TD, and TD(λ).
- [Algorithms for Reinforcement Learning — **§3 (TD prediction)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the stochastic-approximation view of TD convergence.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [04 Monte Carlo Methods](04-Monte-Carlo-Methods.md) · Next: [06 Q-Learning](06-Q-Learning.md) · [07 SARSA](07-SARSA.md)

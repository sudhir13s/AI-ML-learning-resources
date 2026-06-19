---
id: "10-rl/multi-agent-rl"
topic: "Multi-Agent RL"
parent: "10-reinforcement-learning"
level: advanced
prereqs: ["policy-gradients-reinforce", "actor-critic", "continuous-control"]
interview_frequency: medium
updated: 2026-06-20
---

# Multi-Agent RL (MARL)
> Many agents learn *simultaneously* in a shared environment — cooperating, competing, or both. The
> twist that breaks single-agent RL: from any one agent's view the environment is **non-stationary**,
> because the other agents are changing their policies too, so the "MDP" keeps shifting under your feet.
> Solutions include **centralized training with decentralized execution** (CTDE — e.g. **MADDPG**, where
> a centralized critic sees all agents but each actor runs alone), and **self-play** (an agent improves
> by playing copies of itself — the engine behind AlphaGo and OpenAI Five).

**Why it matters:** the frontier for games, markets, robotics swarms, and autonomous driving, and an
increasingly common interview topic. Be ready to explain the non-stationarity problem, the CTDE paradigm
and why MADDPG's centralized critic helps, the difference between cooperative/competitive/mixed settings
(Nash equilibria), and why self-play produces an automatic curriculum.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Emergent Tool Use from Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528). *A vivid demonstration (hide-and-seek) of multi-agent self-play producing emergent strategies.*
2. **Get the core method** — [MADDPG paper](https://arxiv.org/abs/1706.02275). *Centralized-critic, decentralized-actor (CTDE) — the standard MARL algorithm.*
3. **Get the survey** — [A Survey and Critique of Multi-Agent Deep RL](https://arxiv.org/abs/1810.05587). *Non-stationarity, credit assignment, and the MARL algorithm families, in one open reference.*
4. **See self-play at scale** — [OpenAI Five](https://arxiv.org/abs/1912.06680) / [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z). *Self-play + RL beating top humans at Dota 2 and StarCraft II.*

## 🎓 Courses (free)
- [Berkeley CS285 — Multi-Agent / Advanced Topics](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — multi-agent settings and the non-stationarity problem.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — the single-agent actor-critic basis MARL extends.
- [Stanford CS234 — Advanced Topics](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — game-theoretic equilibria and multi-agent learning.

## 🎥 Videos
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — the actor-critic foundation MADDPG extends to many agents.
- [RL Lecture 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc) — **David Silver (DeepMind)** — MCTS/self-play planning, the backbone of AlphaGo-style MARL.
- [RL Series: Overview of Methods](https://www.youtube.com/watch?v=i7q8bISGwMQ) — **Steve Brunton** — situates multi-agent and game-playing RL in the landscape.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — the single-agent framing that MARL generalizes.

## 📄 Key Papers
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — **Lowe et al. (2017)** — centralized-critic, decentralized-actor (CTDE).
- [A Survey and Critique of Multiagent Deep RL](https://arxiv.org/abs/1810.05587) — **Hernandez-Leal, Kartal & Taylor (2019)** — the open reference on MARL challenges and methods.
- [Emergent Tool Use from Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) — **Baker et al. (2020)** — self-play produces emergent, increasingly complex strategies.
- [Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — **Foerster et al. (2018)** — a centralized critic solving multi-agent credit assignment.

## 📰 Articles / Blogs (free, no paywall)
- [Emergent Tool Use (hide-and-seek)](https://arxiv.org/abs/1909.07528) — **OpenAI** — multi-agent self-play and emergent strategy (open arXiv paper).
- [A (Long) Peek into RL](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — the single-agent foundations MARL builds on.

## 📚 Books (free, with chapters)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches](https://www.marl-book.com/) — **Albrecht, Christianos & Schäfer (2024)** — a dedicated free MARL textbook (full PDF online).
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 13 (policy gradients) & §1.5 (the agent–environment interface)**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the single-agent backbone MARL extends.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.02 Policy Gradients (REINFORCE)](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Prereq: [10 Actor-Critic](10-Actor-Critic-A2C-A3C.md) · [13 Continuous Control (DDPG/TD3/SAC)](13-Continuous-Control-DDPG-TD3-SAC.md)

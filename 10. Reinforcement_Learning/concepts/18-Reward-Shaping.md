---
id: "10-rl/reward-shaping"
topic: "Reward Shaping"
parent: "10-reinforcement-learning"
level: intermediate
prereqs: ["markov-decision-processes", "q-learning"]
interview_frequency: medium
updated: 2026-06-20
---

# Reward Shaping
> Sparse rewards ("+1 only at the goal") make learning agonizingly slow — the agent rarely stumbles
> onto reward to learn from. **Reward shaping** adds an auxiliary signal to guide it. The key theorem:
> **potential-based reward shaping** `F(s,s') = γΦ(s') − Φ(s)` adds dense guidance *without changing the
> optimal policy* — any other shaping risks **reward hacking** (the agent optimizes the proxy, not the
> task). Alternatives to hand-shaping: **Hindsight Experience Replay** (relabel failures as successes
> for whatever they did achieve) and intrinsic curiosity.

**Why it matters:** the practical "my agent won't learn — the reward is too sparse" problem, and a
classic trap question. Be ready to state why arbitrary shaping changes the optimum, why *potential-based*
shaping provably doesn't (Ng, Harada & Russell), what reward hacking is, and how HER sidesteps shaping
entirely in goal-conditioned tasks.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Arxiv Insights: RL with sparse rewards](https://www.youtube.com/watch?v=0Ey02HT_1Ho). *Why sparse reward is hard and the shaping / HER / curiosity options.*
2. **Get the theorem** — read [Ng, Harada & Russell: Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf). *The potential-based shaping result — the one fact to know cold.*
3. **Get HER** — [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). *Relabel goals so every episode yields learnable reward — no manual shaping.*
4. **See the failure mode** — [DeepMind: Specification gaming (reward hacking)](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/). *Vivid examples of agents maximizing the proxy reward, not the intended task.*

## 🎓 Courses (free)
- [Berkeley CS285 — Reward & Exploration](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — sparse rewards, shaping, and intrinsic motivation.
- [Stanford CS234 — Reward Design](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — reward specification and its pitfalls.
- [Spinning Up — Key Concepts (reward & return)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — the reward signal shaping modifies.

## 🎥 Videos
- [Reinforcement Learning with Sparse Rewards](https://www.youtube.com/watch?v=0Ey02HT_1Ho) — **Arxiv Insights** — shaping, HER, and curiosity for sparse-reward problems.
- [RL Lecture 9: Exploration & Exploitation](https://www.youtube.com/watch?v=sGuiWX07sKw) — **David Silver (DeepMind)** — the exploration problem sparse rewards intensify.
- [RL Lecture 1: Introduction to RL](https://www.youtube.com/watch?v=2pWv7GOvuf0) — **David Silver (DeepMind)** — the reward hypothesis: all goals as reward maximization.
- [RL Series: Overview of Methods](https://www.youtube.com/watch?v=i7q8bISGwMQ) — **Steve Brunton** — context for reward design across RL methods.

## 📄 Key Papers
- [Policy Invariance Under Reward Transformations (potential-based shaping)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) — **Ng, Harada & Russell (1999)** — the theorem that shaping with a potential preserves the optimal policy.
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495) — **Andrychowicz et al. (2017)** — goal relabeling for sparse-reward, goal-conditioned RL.
- [Curiosity-driven Exploration by Self-supervised Prediction (ICM)](https://arxiv.org/abs/1705.05363) — **Pathak et al. (2017)** — an intrinsic reward replacing hand-shaping.

## 📰 Articles / Blogs (free, no paywall)
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — **Lilian Weng** — a thorough open survey of reward hacking, its causes, and mitigations.
- [Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) — **Lilian Weng** — intrinsic-reward alternatives to manual shaping.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **§3.2 "Goals and Rewards" & §17.4 "Designing Reward Signals"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the reward hypothesis and the pitfalls of reward design.
- [Algorithms for Reinforcement Learning — **§2 (the MDP reward model)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the formal role of the reward function.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.04 MDPs & Exploration](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.04_MDPs_and_Exploration.md)
- Related: [14 Exploration vs Exploitation](14-Exploration-vs-Exploitation.md) · [01 Markov Decision Processes](01-Markov-Decision-Processes.md)

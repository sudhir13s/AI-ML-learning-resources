---
id: "08-rl/model-based-rl"
topic: "Model-Based RL"
parent: "08-reinforcement-learning"
level: advanced
prereqs: ["dynamic-programming-value-and-policy-iteration", "temporal-difference-learning"]
interview_frequency: medium
updated: 2026-06-20
---

# Model-Based RL
> Instead of learning values/policies purely from real experience, **learn (or use) a model** of the
> environment's dynamics `P(s'|s,a)` and reward, then **plan** with it. This buys dramatic sample
> efficiency — you can imagine rollouts instead of acting them. The spectrum runs from **Dyna**
> (interleave real learning with planning on a learned model), through **MPC** (replan a short horizon
> each step), to **MuZero** (learn a *latent* model and plan with MCTS — superhuman without being told
> the rules).

**Why it matters:** the "sample efficiency" axis of RL and a frequent senior-level question. Be ready to
contrast model-free vs model-based (sample efficiency vs model-bias / compounding error), explain Dyna's
planning–acting loop, what MPC does, and why MuZero's learned latent model + MCTS is a landmark.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Steve Brunton: Model-Based RL — Policy & Value Iteration](https://www.youtube.com/watch?v=sJIFUTITfBc). *Planning with a known model — the foundation model-based RL learns.*
2. **Get Dyna** — [Sutton & Barto Ch. 8 "Planning and Learning with Tabular Methods"](http://incompleteideas.net/book/RLbook2020.pdf). *Dyna-Q: interleaving model learning, planning, and acting.*
3. **Get the lecture** — [David Silver Lecture 8: Integrating Learning & Planning](https://www.youtube.com/watch?v=ItMutbeOHtc). *Model learning, Dyna, and Monte-Carlo tree search.*
4. **See the frontier** — [MuZero (DeepMind blog)](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/). *Learning a latent dynamics model and planning with MCTS, no rules given.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 8: Integrating Learning and Planning](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — model learning, Dyna, and MCTS.
- [Berkeley CS285 — Model-Based RL & Planning](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — learned dynamics, MPC, and uncertainty-aware models.
- [Spinning Up — Kinds of RL Algorithms (model-based)](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — the model-free vs model-based taxonomy.

## 🎥 Videos
- [Model-Based RL: Policy Iteration, Value Iteration](https://www.youtube.com/watch?v=sJIFUTITfBc) — **Steve Brunton** — planning with a model, the core of model-based RL.
- [RL Lecture 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc) — **David Silver (DeepMind)** — Dyna and MCTS, derived.
- [RL Series: Overview of Methods](https://www.youtube.com/watch?v=i7q8bISGwMQ) — **Steve Brunton** — where model-based methods sit in the landscape.
- [L1: MDPs & Exact Solution Methods](https://www.youtube.com/watch?v=2GwBez0D20A) — **Pieter Abbeel (Foundations of Deep RL)** — planning with a known model (the model-based ideal).

## 📄 Key Papers
- [Integrated Architectures for Learning, Planning, and Reacting (Dyna)](http://www.incompleteideas.net/papers/sutton-90.pdf) — **Richard Sutton (1990)** — the Dyna framework unifying learning and planning.
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://arxiv.org/abs/1911.08265) — **Schrittwieser et al. (2020)** — latent-model planning with MCTS, superhuman without rules.
- [When to Trust Your Model: Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) — **Janner et al. (2019)** — short model rollouts for sample-efficient, stable model-based RL.

## 📰 Articles / Blogs (free, no paywall)
- [MuZero: Mastering Go, chess, shogi and Atari without rules](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/) — **DeepMind** — the intuition behind learned-model planning.
- [A (Long) Peek into RL — Model-based vs model-free](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — where planning fits in the RL landscape.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 8 "Planning and Learning with Tabular Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — Dyna-Q, prioritized sweeping, MCTS, and the planning–learning unification.
- [Algorithms for Reinforcement Learning — **§2 (planning in MDPs)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the planning foundations.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [03 Dynamic Programming](03-Dynamic-Programming-Value-and-Policy-Iteration.md) · Contrast: [06 Q-Learning](06-Q-Learning.md) (model-free)

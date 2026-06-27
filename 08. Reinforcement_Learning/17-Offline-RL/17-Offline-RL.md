---
id: "08-rl/offline-rl"
topic: "Offline RL"
parent: "08-reinforcement-learning"
level: advanced
prereqs: ["q-learning", "policy-gradients-reinforce", "continuous-control"]
interview_frequency: medium
updated: 2026-06-20
---

# Offline RL (batch RL)
> Learn a policy from a **fixed, pre-collected dataset** — no further interaction with the environment.
> This is how RL becomes usable where exploration is costly or dangerous (healthcare, robotics, recommendation).
> The core difficulty is **distribution shift**: the learned policy queries `Q` at actions the dataset
> never covered, where the value estimates are wildly overoptimistic and there's no way to correct them.
> Methods like **CQL** (conservatively lower-bound out-of-distribution Q-values) and policy-constraint
> approaches (**BCQ/BEAR**) keep the policy close to the data.

**Why it matters:** a hot research and applied area, and a sharp senior interview question. Be ready to
explain why naively running Q-learning/SAC on logged data fails (bootstrapping off OOD actions →
extrapolation error), what distributional shift is, and how CQL and behavior-constraint methods fix it.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [BAIR Blog: Offline RL](https://bair.berkeley.edu/blog/2020/12/07/offline/). *Why offline RL is hard (distribution shift) and the families of fixes, by the survey's authors.*
2. **Read the survey** — [Offline RL: Tutorial, Review, and Perspectives (Levine et al.)](https://arxiv.org/abs/2005.01643). *The definitive reference: problem formulation, error sources, and methods.*
3. **Get the lecture** — [NeurIPS 2020 Offline RL Tutorial](https://sites.google.com/view/offlinerltutorial-neurips2020/home). *Levine & Kumar's video tutorial with slides — the clearest spoken walkthrough.*
4. **Get the key method** — [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779). *How conservatively lower-bounding Q on OOD actions prevents overestimation.*

## 🎓 Courses (free)
- [Berkeley CS285 — Offline Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — the definitive lectures (Levine's group authored CQL and the survey).
- [NeurIPS 2020 Tutorial: Offline RL](https://sites.google.com/view/offlinerltutorial-neurips2020/home) — **Levine & Kumar** — full video tutorial + slides, free.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — off-policy learning, the basis offline RL pushes to its limit.

## 🎥 Videos
- [Offline Reinforcement Learning Tutorial (NeurIPS 2020)](https://sites.google.com/view/offlinerltutorial-neurips2020/home) — **Sergey Levine & Aviral Kumar** — the canonical spoken walkthrough (linked video).
- [L2: Deep Q-Learning](https://www.youtube.com/watch?v=Psrhxy88zww) — **Pieter Abbeel (Foundations of Deep RL)** — the off-policy Q-learning offline RL builds on (and where OOD actions bite).
- [RL Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps) — **David Silver (DeepMind)** — the deadly triad / extrapolation error that distribution shift amplifies.
- [RL Series: Overview of Methods](https://www.youtube.com/watch?v=i7q8bISGwMQ) — **Steve Brunton** — situates batch/offline learning in the RL landscape.

## 📄 Key Papers
- [Offline RL: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643) — **Levine, Kumar, Tucker & Fu (2020)** — the definitive survey.
- [Conservative Q-Learning for Offline RL (CQL)](https://arxiv.org/abs/2006.04779) — **Kumar et al. (2020)** — lower-bounds OOD Q-values; a standard offline baseline.
- [Off-Policy Deep RL without Exploration (BCQ)](https://arxiv.org/abs/1812.02900) — **Fujimoto, Meger & Precup (2019)** — names extrapolation error and constrains to in-distribution actions.

## 📰 Articles / Blogs (free, no paywall)
- [Offline (Batch) Reinforcement Learning](https://bair.berkeley.edu/blog/2020/12/07/offline/) — **BAIR Blog (Levine et al.)** — the problem, why it's hard, and the method families.
- [D4RL: Datasets for Deep Data-Driven RL](https://arxiv.org/abs/2004.07219) — **Fu et al. (2020)** — the standard offline-RL benchmark suite (open).

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 11 "Off-policy Methods with Approximation"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the off-policy / function-approximation instabilities offline RL must overcome.
- [Algorithms for Reinforcement Learning — **§3 (off-policy value estimation)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the theory behind off-policy estimation error.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [06 Q-Learning](../06-Q-Learning/06-Q-Learning.md) · [13 Continuous Control (DDPG/TD3/SAC)](../13-Continuous-Control-DDPG-TD3-SAC/13-Continuous-Control-DDPG-TD3-SAC.md)

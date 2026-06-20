---
id: "08-rl/continuous-control"
topic: "Continuous Control — DDPG · TD3 · SAC"
parent: "08-reinforcement-learning"
level: advanced
prereqs: ["deep-q-networks", "actor-critic", "policy-gradients-reinforce"]
interview_frequency: high
updated: 2026-06-20
---

# Continuous Control — DDPG · TD3 · SAC
> When actions are continuous (motor torques, joint angles), you can't take a `max_a Q(s,a)` over an
> infinite action set. The off-policy actor-critic family solves this: **DDPG** learns a deterministic
> actor whose gradient is `∇_a Q` (deterministic policy gradient) plus a DQN-style critic and replay;
> **TD3** fixes DDPG's overestimation with twin critics, delayed actor updates, and target-policy
> smoothing; **SAC** adds **maximum-entropy** RL — maximize reward *and* policy entropy — for far more
> stable, sample-efficient, exploration-friendly learning. SAC is today's default for continuous control.

**Why it matters:** the go-to robotics/control interview cluster. Be ready to explain why discrete-action
Q-learning fails in continuous spaces, the deterministic policy gradient, TD3's three tricks and the
overestimation they cure, and SAC's entropy term (why maximizing entropy improves exploration and
robustness, and how the temperature α is auto-tuned).

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Spinning Up: DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html). *Deterministic policy gradient + DQN-style critic and replay, explained cleanly.*
2. **Get the fixes** — [Spinning Up: TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html). *The three additions (twin critics, delayed updates, target smoothing) and the overestimation they address.*
3. **Get the entropy view** — [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) + [Lilian Weng: Policy Gradient (SAC section)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). *Maximum-entropy RL and why it's the modern default.*
4. **Read the sources** — [DDPG](https://arxiv.org/abs/1509.02971) → [TD3](https://arxiv.org/abs/1802.09477) → [SAC](https://arxiv.org/abs/1801.01290). *The progression of fixes that produced today's best continuous-control methods.*

## 🎓 Courses (free)
- [Spinning Up — DDPG / TD3 / SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) — **OpenAI** — the three algorithms with shared notation, pseudocode, and code.
- [Berkeley CS285 — Actor-Critic & Q-Learning for Continuous Actions](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — DDPG/SAC and max-entropy RL in depth (Levine co-authored SAC).
- [Hugging Face Deep RL Course — Actor-Critic / SAC](https://huggingface.co/learn/deep-rl-course) — **Hugging Face** — hands-on continuous-control training.

## 🎥 Videos
- [An Introduction to Policy Gradient Methods](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — the policy-based foundation these actor-critics extend.
- [Deep RL Bootcamp L5: Natural PG, TRPO, PPO](https://www.youtube.com/watch?v=xvRrgxcpaHY) — **John Schulman (Berkeley)** — context on policy optimization that frames off-policy actor-critics.
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — the deterministic-policy-gradient lineage (Silver authored DPG).
- [L2: Deep Q-Learning](https://www.youtube.com/watch?v=Psrhxy88zww) — **Pieter Abbeel (Foundations of Deep RL)** — the DQN critic + replay machinery DDPG/TD3 reuse.

## 📄 Key Papers
- [Continuous Control with Deep RL (DDPG)](https://arxiv.org/abs/1509.02971) — **Lillicrap et al. (2016)** — deterministic actor-critic for continuous actions.
- [Addressing Function Approximation Error in Actor-Critic (TD3)](https://arxiv.org/abs/1802.09477) — **Fujimoto et al. (2018)** — twin critics, delayed updates, target smoothing.
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) — **Haarnoja et al. (2018)** — maximum-entropy off-policy actor-critic.
- [SAC: Algorithms and Applications](https://arxiv.org/abs/1812.05905) — **Haarnoja et al. (2018)** — automatic temperature tuning and real-robot results.
- [Deterministic Policy Gradient Algorithms (DPG)](https://proceedings.mlr.press/v32/silver14.html) — **Silver et al. (2014)** — the deterministic-policy-gradient theorem DDPG builds on.

## 📰 Articles / Blogs (free, no paywall)
- [Policy Gradient Algorithms — DDPG, TD3, SAC](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng** — all three derived in one open survey, with the entropy view.
- [Spinning Up — SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) — **OpenAI** — the max-entropy objective and implementation, free.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 13 (policy gradients) & Ch. 10 (on-policy control with approximation)**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the policy-gradient and function-approximation foundations these methods extend.
- [Algorithms for Reinforcement Learning — **§4 (policy search)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the policy-gradient framing for continuous actions.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.02 Policy Gradients (REINFORCE)](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Prereq: [08 Deep Q-Networks](08-Deep-Q-Networks-DQN.md) · [10 Actor-Critic](10-Actor-Critic-A2C-A3C.md) · [12 PPO](12-Proximal-Policy-Optimization-PPO.md)

---
id: "08-rl/actor-critic"
topic: "Actor-Critic (A2C / A3C)"
parent: "08-reinforcement-learning"
level: advanced
prereqs: ["policy-gradients-reinforce", "temporal-difference-learning"]
interview_frequency: high
updated: 2026-06-20
---

# Actor-Critic (A2C · A3C · GAE)
> The marriage of the two RL families: an **actor** (a policy `π(a|s;θ)` updated by policy gradient)
> and a **critic** (a value function `V(s;w)` learned by TD) that supplies a low-variance baseline.
> The policy gradient now multiplies `∇θ log π` by the **advantage** `A(s,a) = Q(s,a) − V(s)` — the
> critic tells the actor how much *better than average* an action was, slashing the variance that
> cripples plain REINFORCE.

**Why it matters:** the template under A2C, A3C, TRPO, PPO, and SAC. Interviewers ask why a critic
reduces variance (baseline subtraction), what the advantage is and how **GAE** trades bias for
variance with λ, and what A3C's asynchronous parallel actors buy you (decorrelation without a replay buffer).

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Lilian Weng: Policy Gradient (Actor-Critic section)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). *Where the critic/baseline comes from and why it lowers variance.*
2. **Get the lecture** — [David Silver Lecture 7: Policy Gradient → Actor-Critic](https://www.youtube.com/watch?v=KHZVXao4qXs). *The transition from REINFORCE to actor-critic, derived.*
3. **Read the sources** — [A3C paper](https://arxiv.org/abs/1602.01783) then [GAE paper](https://arxiv.org/abs/1506.02438). *Asynchronous advantage actor-critic, and generalized advantage estimation (the λ knob).*
4. **Code it** — follow [Spinning Up: Vanilla Policy Gradient (with value baseline)](https://spinningup.openai.com/en/latest/algorithms/vpg.html). *VPG is actor-critic with a GAE critic — implementing it makes A2C concrete.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 7: Policy Gradient](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — actor-critic derived from the policy-gradient theorem.
- [Berkeley CS285 — Actor-Critic Algorithms](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — advantage estimation, bias/variance, and GAE in depth.
- [Spinning Up — Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — **OpenAI** — actor-critic with a GAE value baseline, with pseudocode and code.

## 🎥 Videos
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — REINFORCE → actor-critic, baselines, and the advantage.
- [Deep RL Bootcamp Lecture 4A: Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44) — **Pieter Abbeel (Berkeley)** — the baseline/advantage construction that defines actor-critic.
- [An Introduction to Policy Gradient Methods](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — visual intuition for the critic reducing gradient variance.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — situates actor-critic between value-based and policy-based methods.

## 📄 Key Papers
- [Asynchronous Methods for Deep RL (A3C)](https://arxiv.org/abs/1602.01783) — **Mnih et al. (2016)** — asynchronous advantage actor-critic; parallel actors replace the replay buffer.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) — **Schulman et al. (2016)** — the λ-controlled advantage estimator used by A2C/PPO.
- [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — **Konda & Tsitsiklis (2000)** — the foundational convergence theory.

## 📰 Articles / Blogs (free, no paywall)
- [Policy Gradient Algorithms — Actor-Critic, A2C, A3C](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng** — the actor-critic family derived in one open survey.
- [Spinning Up — Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — **OpenAI** — actor-critic with GAE, implemented.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **§13.5 "Actor–Critic Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the actor-critic update with a TD critic and the advantage view.
- [Algorithms for Reinforcement Learning — **§4 (policy search & actor-critic)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the estimation theory behind the critic.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.02 Policy Gradients (REINFORCE)](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Prereq: [09 Policy Gradients (REINFORCE)](../09-Policy-Gradients-REINFORCE/09-Policy-Gradients-REINFORCE.md) · Next: [11 TRPO](../11-Trust-Region-Policy-Optimization-TRPO/11-Trust-Region-Policy-Optimization-TRPO.md) · [12 PPO](../12-Proximal-Policy-Optimization-PPO/12-Proximal-Policy-Optimization-PPO.md)

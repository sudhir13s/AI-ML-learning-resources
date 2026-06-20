---
id: "08-rl/policy-gradients-reinforce"
topic: "Policy Gradients (REINFORCE)"
parent: "08-reinforcement-learning"
level: advanced
prereqs: ["markov-decision-processes", "expectation", "log-derivative-trick", "gradient-ascent"]
interview_frequency: very-high
updated: 2026-06-20
---

# Policy Gradients (REINFORCE)
> Instead of learning values and acting greedily, **parameterize the policy** `π(a|s;θ)` directly and
> push θ up the reward gradient. The policy-gradient theorem gives an estimator you can compute from
> sampled trajectories: `∇θ J = E[ ∇θ log π(a|s) · G_t ]` — the **log-derivative trick** turns an
> expectation's gradient into an expectation of a gradient. REINFORCE is the simplest instance.

**Why it matters:** the foundation of every modern policy method (A2C, TRPO, PPO, and RLHF). Be ready
to derive the policy-gradient theorem, explain the log-derivative ("REINFORCE") trick, why a **baseline**
(e.g. a value function) reduces variance without adding bias, and why policy methods handle continuous
and stochastic actions where value-based methods struggle.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Arxiv Insights: Intro to Policy Gradient methods](https://www.youtube.com/watch?v=5P7I-xPq8u8). *The clearest visual intuition for "increase the log-prob of good actions."*
2. **Derive it** — read [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). *The policy-gradient theorem, baselines, and the whole family derived in one place.*
3. **Get the lecture** — [David Silver Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs). *REINFORCE, baselines, and the move to actor-critic.*
4. **Read the source + chapter** — [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) and [Sutton & Barto Ch. 13](http://incompleteideas.net/book/RLbook2020.pdf). *The full derivation with the reward-to-go and baseline variants you'll implement.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 7: Policy Gradient](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — the policy-gradient theorem, REINFORCE, and actor-critic.
- [Spinning Up — Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — **OpenAI** — derives the policy gradient, reward-to-go, and baselines, with code.
- [Berkeley CS285 — Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — the rigorous derivation and variance-reduction toolkit.

## 🎥 Videos
- [An Introduction to Policy Gradient Methods](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — vivid intuition for the log-prob push and the gradient estimator.
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — REINFORCE, the policy-gradient theorem, baselines, actor-critic.
- [Deep RL Bootcamp Lecture 4A: Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44) — **Pieter Abbeel (Berkeley)** — the full policy-gradient derivation and practical variance reduction.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — places policy-based methods against value-based ones.

## 📄 Key Papers
- [Policy Gradient Methods for RL with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — **Sutton, McAllester, Singh & Mansour (2000)** — the policy-gradient theorem.
- [Simple Statistical Gradient-Following Algorithms (REINFORCE)](https://link.springer.com/article/10.1007/BF00992696) — **Ronald Williams (1992)** — the original REINFORCE estimator.

## 📰 Articles / Blogs (free, no paywall)
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng** — the definitive open survey: theorem, baselines, A2C/A3C, TRPO, PPO, SAC.
- [Spinning Up — Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — **OpenAI** — the gradient derived and implemented, with the baseline trick.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 13 "Policy Gradient Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the policy-gradient theorem, REINFORCE, baselines, and actor-critic.
- [Algorithms for Reinforcement Learning — **§4 (policy search)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the gradient-estimation view.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.02 Policy Gradients (REINFORCE)](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Prereq: [01 Markov Decision Processes](01-Markov-Decision-Processes.md) · Next: [10 Actor-Critic](10-Actor-Critic-A2C-A3C.md) · [11 TRPO](11-Trust-Region-Policy-Optimization-TRPO.md) · [12 PPO](12-Proximal-Policy-Optimization-PPO.md)

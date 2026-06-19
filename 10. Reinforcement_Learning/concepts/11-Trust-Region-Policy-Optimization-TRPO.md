---
id: "10-rl/trpo"
topic: "Trust Region Policy Optimization (TRPO)"
parent: "10-reinforcement-learning"
level: advanced
prereqs: ["policy-gradients-reinforce", "actor-critic", "kl-divergence"]
interview_frequency: medium
updated: 2026-06-20
---

# Trust Region Policy Optimization (TRPO)
> Plain policy gradients take a step in parameter space and can fall off a cliff — one bad update
> collapses the policy. TRPO instead takes the **largest improving step that stays within a trust
> region**: maximize the surrogate advantage objective subject to a hard constraint that the new
> policy's KL divergence from the old one is small. This *monotonic-improvement* guarantee is what
> made deep policy optimization reliable — and PPO is its cheaper approximation.

**Why it matters:** the conceptual parent of PPO and the standard "why constrain the policy update?"
question. Be ready to explain the surrogate objective, the KL trust-region constraint, why it gives
(approximate) monotonic improvement, and why the natural-gradient / conjugate-gradient + Fisher-vector
machinery is expensive — which is exactly the pain PPO removes with a clipped objective.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Spinning Up: TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html). *The surrogate objective and KL constraint explained without drowning in the natural-gradient algebra.*
2. **Get the derivation** — [Lilian Weng: Policy Gradient (TRPO section)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). *The monotonic-improvement bound and where the KL trust region comes from.*
3. **Watch the lecture** — [Deep RL Bootcamp L5: Natural PG, TRPO, PPO](https://www.youtube.com/watch?v=xvRrgxcpaHY). *John Schulman (the author) walks through natural gradients → TRPO → PPO.*
4. **Read the source** — [TRPO paper](https://arxiv.org/abs/1502.05477). *The theorem, the practical approximations (conjugate gradient, line search), and the experiments.*

## 🎓 Courses (free)
- [Spinning Up — Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html) — **OpenAI** — the cleanest exposition of the surrogate objective and KL constraint, with pseudocode.
- [Berkeley CS285 — Advanced Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — natural gradients, trust regions, and the monotonic-improvement bound.
- [Stanford CS234 — Policy Search](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — places TRPO among policy-optimization methods.

## 🎥 Videos
- [Deep RL Bootcamp L5: Natural Policy Gradients, TRPO, PPO](https://www.youtube.com/watch?v=xvRrgxcpaHY) — **John Schulman (Berkeley)** — the author derives TRPO from the natural gradient.
- [An Introduction to Policy Gradient Methods](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — the step-size problem TRPO solves, visually.
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — the policy-gradient foundation TRPO constrains.
- [Deep RL Bootcamp Lecture 4A: Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44) — **Pieter Abbeel (Berkeley)** — the surrogate-objective groundwork.

## 📄 Key Papers
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — **Schulman et al. (2015)** — the monotonic-improvement theorem and the practical TRPO algorithm.
- [Approximately Optimal Approximate RL (CPI)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) — **Kakade & Langford (2002)** — conservative policy iteration; the improvement bound TRPO builds on.

## 📰 Articles / Blogs (free, no paywall)
- [Policy Gradient Algorithms — TRPO](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng** — the surrogate objective, KL constraint, and improvement bound derived.
- [Spinning Up — TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html) — **OpenAI** — the algorithm and its trade-offs vs PPO, with code.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 13 "Policy Gradient Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the policy-gradient backbone TRPO constrains (the trust-region machinery is in the papers above).
- [Algorithms for Reinforcement Learning — **§4 (policy search)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the policy-search framing.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.03 PPO & RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md)
- Prereq: [10 Actor-Critic](10-Actor-Critic-A2C-A3C.md) · Successor: [12 PPO](12-Proximal-Policy-Optimization-PPO.md)

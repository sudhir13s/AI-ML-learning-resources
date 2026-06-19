---
id: "10-rl/ppo"
topic: "Proximal Policy Optimization (PPO)"
parent: "10-reinforcement-learning"
level: advanced
prereqs: ["policy-gradients-reinforce", "actor-critic", "trpo"]
interview_frequency: very-high
updated: 2026-06-20
---

# Proximal Policy Optimization (PPO)
> TRPO's reliability without TRPO's machinery. PPO keeps the policy update inside a trust region using
> a **clipped surrogate objective**: it multiplies the advantage by the probability ratio
> `r(θ) = π_new/π_old`, then *clips* that ratio to `[1−ε, 1+ε]` so the update can't move the policy too
> far in one step — all with plain first-order SGD. Simple, stable, sample-efficient enough to reuse a
> batch for several epochs. It is the default deep-RL algorithm and the RL engine inside RLHF.

**Why it matters:** the single most-asked deep-RL interview topic. Write the clipped objective, explain
why clipping approximates TRPO's KL trust region, how the min(clipped, unclipped) prevents over-large
updates in both directions, the roles of the value loss and entropy bonus, and — crucially — how PPO is
the optimizer in **RLHF** (the policy is the LLM, the reward comes from a learned reward model, plus a
KL penalty to the reference model).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Arxiv Insights: Policy Gradients & PPO](https://www.youtube.com/watch?v=5P7I-xPq8u8). *The clearest visual story of the step-size problem and the clip that fixes it.*
2. **Read the canonical explainer** — [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html). *The clipped objective, GAE advantages, and the full update loop with pseudocode.*
3. **Get the derivation** — [Lilian Weng: Policy Gradient (PPO section)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/). *PPO as a first-order approximation to TRPO's trust region.*
4. **Read the source + lecture** — [PPO paper](https://arxiv.org/abs/1707.06347) and [Deep RL Bootcamp L5 (Schulman)](https://www.youtube.com/watch?v=xvRrgxcpaHY). *The author's own framing, TRPO → PPO.*

## 🎓 Courses (free)
- [Spinning Up — Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — **OpenAI** — the reference exposition: clipped objective, GAE, code.
- [Berkeley CS285 — Advanced Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — PPO/TRPO derived from the trust-region view.
- [Hugging Face Deep RL Course — PPO unit](https://huggingface.co/learn/deep-rl-course/unit8/introduction) — **Hugging Face** — hands-on: implement and train PPO yourself.

## 🎥 Videos
- [An Introduction to Policy Gradient Methods (PPO)](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — the intuition behind the clip and why PPO is stable.
- [Deep RL Bootcamp L5: Natural PG, TRPO, PPO](https://www.youtube.com/watch?v=xvRrgxcpaHY) — **John Schulman (Berkeley)** — the author's TRPO → PPO walkthrough.
- [RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind)** — the policy-gradient foundation PPO constrains.
- [Deep RL Bootcamp Lecture 4A: Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44) — **Pieter Abbeel (Berkeley)** — surrogate objectives and advantage estimation.

## 📄 Key Papers
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — **Schulman et al. (2017)** — the clipped-surrogate PPO and the adaptive-KL variant.
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — **Schulman et al. (2015)** — the trust-region principle PPO approximates first-order.
- [High-Dimensional Continuous Control with GAE](https://arxiv.org/abs/1506.02438) — **Schulman et al. (2016)** — the advantage estimator PPO uses in practice.

## 📰 Articles / Blogs (free, no paywall)
- [Policy Gradient Algorithms — PPO](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng** — PPO derived as first-order TRPO, with the clipping math.
- [Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — **OpenAI** — the definitive open implementation guide.
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — **Huang et al. (ICLR blog)** — the practical tricks that make PPO actually work.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 13 "Policy Gradient Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the policy-gradient backbone; PPO's clip is detailed in the papers above.
- [Dive into Deep Learning — **Ch. 17 (Reinforcement Learning)**](https://d2l.ai/chapter_reinforcement-learning/index.html) — **Zhang et al.** — runnable RL foundations leading to policy optimization.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.03 PPO & RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md)
- Prereq: [11 TRPO](11-Trust-Region-Policy-Optimization-TRPO.md) · [10 Actor-Critic](10-Actor-Critic-A2C-A3C.md)
- **RLHF for LLMs** (PPO applied to language-model alignment, with reward models + DPO) lives in [08. LLMs](../../08.%20LLMs/README.md) — this card owns the PPO *mechanics* it relies on.

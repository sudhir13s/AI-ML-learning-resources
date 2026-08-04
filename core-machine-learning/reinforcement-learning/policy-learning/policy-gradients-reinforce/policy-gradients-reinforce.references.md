---
id: "08-rl/policy-gradients-reinforce/references"
topic: "Policy Gradients (REINFORCE) — References"
parent: "08-rl/policy-gradients-reinforce"
type: references
updated: 2026-07-03
---

# Policy Gradients (REINFORCE) — references and further reading

> Companion link library for **[Policy Gradients (REINFORCE)](policy-gradients-reinforce.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on *policy gradients specifically* (the policy-gradient theorem, the log-derivative / score-function trick, baselines and variance reduction, and the on-policy REINFORCE estimator), not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch ⭐ [RL Course, Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) by **David Silver (DeepMind)**. *Derives the policy-gradient theorem, REINFORCE, the baseline, and the bridge to actor-critic — the cleanest single lecture on the topic.*
2. **Read the source chapter** — [Sutton & Barto, Chapter 13 "Policy Gradient Methods"](http://incompleteideas.net/book/RLbook2020.pdf) (free PDF). *The policy-gradient theorem, REINFORCE (§13.3), the baseline (§13.4), and actor-critic (§13.5) — the derivation this page follows.*
3. **See it implemented** — read [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) (**OpenAI**, Josh Achiam). *Derives the gradient, the reward-to-go form, and the baseline, then shows the exact PyTorch loss `-(logp * weights).mean()` this chapter's module uses.*
4. **Cement the derivation** — read [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) (**Lilian Weng, OpenAI**). *The theorem, the log-derivative trick, baselines, and the whole A2C/TRPO/PPO family derived in one place.*
5. **Make it concrete** — run this chapter's [notebook](code/policy-gradients-reinforce.ipynb): a from-scratch policy network trained by REINFORCE on real Gymnasium `CartPole-v1`, with the score-function gradient asserted equal to the analytic and finite-difference gradients, and the baseline's variance reduction measured.

**Videos**:
- [RL Course, Lecture 7 — Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs) — **David Silver (DeepMind / UCL)** — the policy-gradient theorem, the score-function trick, baselines, and the move to actor-critic, from a co-author of DQN.
- [Deep RL Bootcamp, Lecture 4A: Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44) — **Pieter Abbeel (UC Berkeley)** — the full derivation of the estimator and the practical variance-reduction toolkit (reward-to-go, baselines).
- [Deep RL from Pong Pixels — Policy Gradients](https://www.youtube.com/watch?v=tqrcjHuNdmQ) — **Andrej Karpathy** — the intuition and code for training a policy by "make the actions that led to reward more likely," building directly on his classic write-up.
- [CS285 Lecture 5: Policy Gradients](https://www.youtube.com/watch?v=GKoKNYaBvM0) — **Sergey Levine (UC Berkeley)** — a rigorous, from-first-principles derivation of the policy gradient, causality/reward-to-go, and baselines.
- [Reinforcement Learning: Policy Gradient Methods](https://www.youtube.com/watch?v=e20EY4tFC_Q) — **Steve Brunton (University of Washington)** — a crisp, geometric walk through gradient ascent on expected return.

**Courses (free)**:
- [UCL Course on RL — Lecture 7: Policy Gradient](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — the theorem, REINFORCE, baselines, and actor-critic, with slides.
- [Spinning Up in Deep RL — Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — **OpenAI** — derives the policy gradient, the reward-to-go and baseline variants, then the vanilla-policy-gradient implementation.
- [Berkeley CS285 — Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — the rigorous derivation, the variance-reduction toolkit, and the connection to actor-critic.
- [Hugging Face Deep RL Course — Unit 4: Policy Gradient with PyTorch](https://huggingface.co/learn/deep-rl-course/unit4/introduction) — **Thomas Simonini (Hugging Face)** — a free hands-on unit that codes REINFORCE from scratch and trains it on CartPole, the same environment this page solves.

**Articles / blogs (free, no paywall)**:
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **Lilian Weng (OpenAI)** — the definitive open survey: the policy-gradient theorem, the log-derivative trick, baselines, and the entire modern family (A2C, A3C, TRPO, PPO, SAC), derived cleanly.
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) — **Andrej Karpathy** — the classic, intuition-first introduction to policy gradients: "run a policy, then increase the probability of the actions that led to reward."
- [An Intuitive Explanation of Policy Gradient](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c/) — **Adrien Lucas Ecoffet** — the REINFORCE estimator and the baseline built up step by step.
- [Gymnasium documentation — CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — **Farama Foundation** — the exact environment this chapter trains on, with its observation, action, reward, and the `reward_threshold = 475` "solved" bar.

**Key papers** (the source and the theorem):
- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)](https://link.springer.com/article/10.1007/BF00992696) — **Ronald J. Williams (1992), Machine Learning 8** — the original paper that introduced the REINFORCE estimator and the log-derivative-based update this page derives.
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — **Sutton, McAllester, Singh & Mansour (2000), NeurIPS** — the **policy-gradient theorem**: the formal result that the transition dynamics drop out of the gradient.
- [Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783) — **Mnih et al. (2016), ICML** — where the learned value baseline (the critic) turns REINFORCE into actor-critic at scale.
- [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347) — **Schulman et al. (2017)** — the policy-gradient method that dominates today (and powers RLHF), replacing REINFORCE's single step with a clipped, multi-epoch surrogate.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) — **Schulman et al. (2016), ICLR** — the modern advantage estimator that generalizes the baseline used here.

**Books (free chapters)**:
- [Reinforcement Learning: An Introduction (2nd ed.) — Chapter 13 "Policy Gradient Methods"](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the canonical treatment: the policy-gradient theorem, REINFORCE (§13.3), REINFORCE with baseline (§13.4), actor-critic (§13.5), and the Short-Corridor example that motivates stochastic policies. The whole book is free.
- [Algorithms for Reinforcement Learning — §4 (policy search)](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the gradient-estimation view of policy search, concise and rigorous (free PDF).
- [Grokking Deep Reinforcement Learning — policy-based and actor-critic methods](https://github.com/mimoralea/gdrl) — **Miguel Morales** — the accompanying code (notebooks) is fully open and builds REINFORCE, baselines, and A2C from scratch.

**In this platform**:
- Concept page (full explanation): [Policy Gradients (REINFORCE)](policy-gradients-reinforce.md)
- Runnable code: [the module `reinforce.py`](code/reinforce.py) · [the step-by-step notebook](code/policy-gradients-reinforce.ipynb) — a from-scratch policy network trained by REINFORCE on real Gymnasium `CartPole-v1`, with the score-function gradient asserted equal to the analytic and finite-difference gradients, the baseline's unbiasedness confirmed, and its variance reduction measured; every number computed, none fabricated.
- The value-based sibling (contrast): [06 Q-Learning](../../value-based-learning/q-learning/q-learning.md) — learns $Q(s,a)$ and acts greedily (value-based, off-policy); this page learns the *policy* $\pi_\theta(a\mid s)$ directly (policy-based, on-policy). The two pillars of model-free control.
- The framework it lives in: [01 Markov Decision Processes](../../foundations/markov-decision-processes/markov-decision-processes.md) — states, actions, transitions, reward, and $\gamma$, the language the objective $J(\theta)=\mathbb{E}_\tau[R(\tau)]$ is written in.
- The values it builds on: [02 Bellman Equations](../../foundations/bellman-equations/bellman-equations.md) — the state-value $V(s)$ that becomes the *learned baseline / critic* in the next chapter.
- The gradient engine: [05 Deep Learning · Backpropagation and Computational Graphs](../../../../deep-learning/neural-network-foundations/backpropagation-and-computational-graphs/backpropagation-and-computational-graphs.md) — the policy gradient is computed by ordinary backprop through $\log\pi_\theta(a_t\mid s_t)$; `loss.backward()` *is* the policy gradient.
- Where the baseline becomes a critic: [10 Actor-Critic (A2C / A3C)](../actor-critic-a2c-a3c/actor-critic-a2c-a3c.md) — replace the mean-return baseline with a learned value function $V_\phi(s)$ and bootstrap it; REINFORCE-with-baseline is the seed of actor-critic.
- The modern successor: [12 Proximal Policy Optimization (PPO)](../proximal-policy-optimization-ppo/proximal-policy-optimization-ppo.md) — fixes REINFORCE's on-policy sample inefficiency with a clipped surrogate and importance sampling; the algorithm behind RLHF.
- The exploration view: [14 Exploration vs Exploitation](../../foundations/exploration-vs-exploitation/exploration-vs-exploitation.md) — a stochastic policy explores *by construction*; the entropy of $\pi_\theta$ is the exploration knob here.
- Field overview: [Reinforcement Learning](../../README.md)

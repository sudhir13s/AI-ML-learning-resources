---
id: "08-reinforcement-learning"
topic: "Reinforcement Learning"
level: advanced
built_from: ["probability", "deep-learning"]
updated: 2026-06-27
---

# Reinforcement Learning
> Learning by trial and reward — MDPs, value & policy methods, deep RL, and the RLHF that
> aligns modern LLMs.

**⭐ Start here:** [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — free, hands-on, you train agents.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page — a short guided
learning path plus the best **free, open** courses, videos, papers, articles, and books for that topic.
> **✅ ready.** New to RL? Start with the field overview below, then work top to bottom.

### Foundations — the formalism
1. ✅ [Markov Decision Processes (states · actions · rewards · transitions)](foundations/markov-decision-processes/markov-decision-processes.md)
2. ✅ [Bellman Equations (expectation & optimality)](foundations/bellman-equations/bellman-equations.md)
3. ✅ [Dynamic Programming — Value & Policy Iteration](foundations/dynamic-programming-value-and-policy-iteration/dynamic-programming-value-and-policy-iteration.md)

### Tabular model-free learning
4. ✅ [Monte Carlo Methods (first-visit · every-visit · MC control)](value-based-learning/monte-carlo-methods/monte-carlo-methods.md)
5. ✅ [Temporal-Difference Learning (TD(0) · TD(λ) · n-step)](value-based-learning/temporal-difference-learning/temporal-difference-learning.md)
6. ✅ [Q-Learning (off-policy TD control)](value-based-learning/q-learning/q-learning.md)
7. ✅ [SARSA (on-policy TD control)](value-based-learning/sarsa/sarsa.md)

### Deep RL — value-based
8. ✅ [Deep Q-Networks (DQN + Double · Dueling · Prioritized · Rainbow)](value-based-learning/deep-q-networks-dqn/deep-q-networks-dqn.md)

### Deep RL — policy-based & actor-critic
9. ✅ [Policy Gradients (REINFORCE)](policy-learning/policy-gradients-reinforce/policy-gradients-reinforce.md)
10. ✅ [Actor-Critic (A2C · A3C · GAE)](policy-learning/actor-critic-a2c-a3c/actor-critic-a2c-a3c.md)
11. ✅ [Trust Region Policy Optimization (TRPO)](policy-learning/trust-region-policy-optimization-trpo/trust-region-policy-optimization-trpo.md)
12. ✅ [Proximal Policy Optimization (PPO)](policy-learning/proximal-policy-optimization-ppo/proximal-policy-optimization-ppo.md)
13. ✅ [Continuous Control — DDPG · TD3 · SAC](policy-learning/continuous-control-ddpg-td3-sac/continuous-control-ddpg-td3-sac.md)

### Exploration & decision-making
14. ✅ [Exploration vs Exploitation (ε-greedy · UCB · Thompson · intrinsic)](foundations/exploration-vs-exploitation/exploration-vs-exploitation.md)
15. ✅ [Multi-Armed Bandits (stochastic · contextual)](foundations/multi-armed-bandits/multi-armed-bandits.md)

### Advanced paradigms
16. ✅ [Model-Based RL (Dyna · MPC · MuZero)](model-based-reinforcement-learning/model-based-rl/model-based-rl.md)
17. ✅ [Offline RL (batch RL · CQL · distribution shift)](offline-reinforcement-learning/offline-rl/offline-rl.md)
18. ✅ [Reward Shaping (potential-based · sparse rewards · HER)](foundations/reward-shaping/reward-shaping.md)
19. ✅ [Multi-Agent RL (MADDPG · self-play · cooperation/competition)](multi-agent-reinforcement-learning/multi-agent-rl/multi-agent-rl.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **RLHF / alignment for LLMs** — reward models, PPO-on-language, DPO → [09. LLMs](../../09.%20LLMs/README.md). *RL owns the policy-gradient / PPO **mechanics** ([12 PPO](policy-learning/proximal-policy-optimization-ppo/proximal-policy-optimization-ppo.md)); the LLM-alignment RLHF card lives in the LLMs section and links back here.*
- **Deep learning prerequisites** — backprop, optimizers, function approximation → [05. Deep Learning](../../deep-learning/README.md)

## 🎓 Courses (free)
- [DeepMind x UCL: RL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) — **David Silver / DeepMind** — the classic, by an AlphaGo author.
- [CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — the definitive deep-RL course.

## 🎥 Videos
- [RL & Q-Learning (StatQuest)](https://www.youtube.com/watch?v=qhRNvCVVZaA) — **Josh Starmer** — gentle, visual intro.
- [Policy Gradients & PPO](https://www.youtube.com/watch?v=5P7I-xPq8u8) — **Arxiv Insights** — the modern policy-method intuition.

## 📄 Key Papers
- [Playing Atari with Deep RL (DQN)](https://arxiv.org/abs/1312.5602) — **Mnih et al. (2013)** — deep RL is born.
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — **Schulman et al. (2017)** — the workhorse behind RLHF.

## 📚 Books (free)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — **Sutton & Barto** — free; *the* RL textbook.

## 🔗 In this platform
- Math: [AI-ML-intuition Module 6 (RL & Alignment)](../../../AI-ML-intuition/decision-making-and-control/)

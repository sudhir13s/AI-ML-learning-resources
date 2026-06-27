---
id: "08-reinforcement-learning"
topic: "Reinforcement Learning"
level: advanced
prereqs: ["probability", "deep-learning"]
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
1. ✅ [Markov Decision Processes (states · actions · rewards · transitions)](01-Markov-Decision-Processes/01-Markov-Decision-Processes.md)
2. ✅ [Bellman Equations (expectation & optimality)](02-Bellman-Equations/02-Bellman-Equations.md)
3. ✅ [Dynamic Programming — Value & Policy Iteration](03-Dynamic-Programming-Value-and-Policy-Iteration/03-Dynamic-Programming-Value-and-Policy-Iteration.md)

### Tabular model-free learning
4. ✅ [Monte Carlo Methods (first-visit · every-visit · MC control)](04-Monte-Carlo-Methods/04-Monte-Carlo-Methods.md)
5. ✅ [Temporal-Difference Learning (TD(0) · TD(λ) · n-step)](05-Temporal-Difference-Learning/05-Temporal-Difference-Learning.md)
6. ✅ [Q-Learning (off-policy TD control)](06-Q-Learning/06-Q-Learning.md)
7. ✅ [SARSA (on-policy TD control)](07-SARSA/07-SARSA.md)

### Deep RL — value-based
8. ✅ [Deep Q-Networks (DQN + Double · Dueling · Prioritized · Rainbow)](08-Deep-Q-Networks-DQN/08-Deep-Q-Networks-DQN.md)

### Deep RL — policy-based & actor-critic
9. ✅ [Policy Gradients (REINFORCE)](09-Policy-Gradients-REINFORCE/09-Policy-Gradients-REINFORCE.md)
10. ✅ [Actor-Critic (A2C · A3C · GAE)](10-Actor-Critic-A2C-A3C/10-Actor-Critic-A2C-A3C.md)
11. ✅ [Trust Region Policy Optimization (TRPO)](11-Trust-Region-Policy-Optimization-TRPO/11-Trust-Region-Policy-Optimization-TRPO.md)
12. ✅ [Proximal Policy Optimization (PPO)](12-Proximal-Policy-Optimization-PPO/12-Proximal-Policy-Optimization-PPO.md)
13. ✅ [Continuous Control — DDPG · TD3 · SAC](13-Continuous-Control-DDPG-TD3-SAC/13-Continuous-Control-DDPG-TD3-SAC.md)

### Exploration & decision-making
14. ✅ [Exploration vs Exploitation (ε-greedy · UCB · Thompson · intrinsic)](14-Exploration-vs-Exploitation/14-Exploration-vs-Exploitation.md)
15. ✅ [Multi-Armed Bandits (stochastic · contextual)](15-Multi-Armed-Bandits/15-Multi-Armed-Bandits.md)

### Advanced paradigms
16. ✅ [Model-Based RL (Dyna · MPC · MuZero)](16-Model-Based-RL/16-Model-Based-RL.md)
17. ✅ [Offline RL (batch RL · CQL · distribution shift)](17-Offline-RL/17-Offline-RL.md)
18. ✅ [Reward Shaping (potential-based · sparse rewards · HER)](18-Reward-Shaping/18-Reward-Shaping.md)
19. ✅ [Multi-Agent RL (MADDPG · self-play · cooperation/competition)](19-Multi-Agent-RL/19-Multi-Agent-RL.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **RLHF / alignment for LLMs** — reward models, PPO-on-language, DPO → [09. LLMs](../09.%20LLMs/README.md). *RL owns the policy-gradient / PPO **mechanics** ([12 PPO](12-Proximal-Policy-Optimization-PPO/12-Proximal-Policy-Optimization-PPO.md)); the LLM-alignment RLHF card lives in the LLMs section and links back here.*
- **Deep learning prerequisites** — backprop, optimizers, function approximation → [05. Deep Learning](../05.%20Deep_Learning/README.md)

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
- Math: [AI-ML-intuition Module 6 (RL & Alignment)](../../AI-ML-intuition/Module_6_Reinforcement_Learning/)

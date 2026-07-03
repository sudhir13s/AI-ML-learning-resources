---
id: "08-rl/q-learning/references"
topic: "Q-Learning — References"
parent: "08-rl/q-learning"
type: references
updated: 2026-07-03
---

# Q-Learning — references and further reading

> Companion link library for **[Q-Learning](06-Q-Learning.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on *Q-learning specifically* (the off-policy TD-control update, its Bellman-optimality derivation, convergence, and the SARSA contrast), not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch ⭐ [RL Course, Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) by **David Silver (DeepMind)**. *Derives Q-learning and SARSA side by side — the cleanest on-policy-vs-off-policy explanation anywhere.*
2. **Read the source chapter** — [Sutton & Barto, §6.5 "Q-learning: Off-policy TD Control"](http://incompleteideas.net/book/RLbook2020.pdf) (free PDF). *The update rule, the off-policy property, and the CliffWalking-vs-SARSA comparison (Example 6.6) this page reproduces.*
3. **See it implemented** — read [An introduction to Q-Learning: reinforcement learning](https://huggingface.co/learn/deep-rl-course/unit2/q-learning) (**Hugging Face Deep RL Course**, Thomas Simonini). *Builds the Q-table + ε-greedy loop step by step, then trains on FrozenLake — the same environment this page proves optimal.*
4. **Cement the derivation** — read the [Q-learning](https://link.springer.com/article/10.1007/BF00992698) paper (**Watkins & Dayan, 1992**). *Short, readable, and the origin of both the algorithm and its convergence proof.*
5. **Make it concrete** — run this chapter's [notebook](code/06-Q-Learning.ipynb): from-scratch tabular Q-learning on real Gymnasium `FrozenLake-v1` and `CliffWalking-v1`, with the learned policy asserted equal to the value-iteration optimum.

**Videos**:
- [RL Course, Lecture 5 — Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) — **David Silver (DeepMind / UCL)** — the definitive Q-learning vs SARSA derivation, GLIE, and convergence, from a co-author of DQN.
- [DeepMind × UCL RL Lecture Series — Model-Free Control](https://www.youtube.com/watch?v=t9uf9cuogBo) — **Hado van Hasselt (DeepMind)** — the modern DeepMind lecture on TD control and off-policy learning, by the author of Double Q-learning.
- [Reinforcement Learning: Q-Learning](https://www.youtube.com/watch?v=0iqz4tcKN58) — **Steve Brunton (University of Washington)** — a crisp, geometric walk through the Bellman optimality equation and the Q-learning update.
- [Reinforcement Learning Series](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-Mmh4nOB5jYlMGBjr) — **Mutual Information** — clear, visual episodes on value functions, TD learning, and Q-learning with excellent animations.
- [Q-Learning Explained — a hands-on RL tutorial](https://www.youtube.com/watch?v=qhRNvCVVJaA) — **deeplizard** — a step-by-step build of the Q-table and the ε-greedy update loop, code-first.
- [Reinforcement Learning: Machine Learning Meets Control Theory](https://www.youtube.com/watch?v=0MNVhXEX9to) — **Steve Brunton** — situates Q-learning among value- and policy-based methods, for the bigger picture.

**Courses (free)**:
- [UCL Course on RL — Lecture 5: Model-Free Control](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — Q-learning vs SARSA, GLIE exploration, and convergence, with slides.
- [Hugging Face Deep RL Course — Unit 2: Q-Learning](https://huggingface.co/learn/deep-rl-course/unit2/introduction) — **Thomas Simonini (Hugging Face)** — a free, hands-on unit that trains a Q-table on FrozenLake and Taxi, then bridges to Deep Q-Learning.
- [Stanford CS234 — Reinforcement Learning](https://web.stanford.edu/class/cs234/) — **Emma Brunskill (Stanford)** — the model-free-control lectures cover Q-learning's off-policy property and convergence guarantees rigorously.
- [Spinning Up in Deep RL — Intro to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — situates value-based methods (Q-learning, DQN) among the RL algorithm families.

**Articles / blogs (free, no paywall)**:
- [A (Long) Peek into Reinforcement Learning — Q-Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng (OpenAI)** — the off-policy TD-control update with the SARSA contrast, derived cleanly.
- [An introduction to Q-Learning](https://huggingface.co/learn/deep-rl-course/unit2/q-learning) — **Hugging Face** — the Q-table, ε-greedy, and the Bellman update, with runnable FrozenLake code.
- [Diving deeper into Reinforcement Learning with Q-Learning](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/) — **Thomas Simonini (freeCodeCamp)** — a well-illustrated first build of a Q-learning agent from scratch.
- [Gymnasium documentation — FrozenLake & CliffWalking](https://gymnasium.farama.org/environments/toy_text/) — **Farama Foundation** — the exact environments this chapter runs, with their dynamics, rewards, and the `is_slippery` option documented.

**Key papers** (the source and the fixes):
- [Q-learning](https://link.springer.com/article/10.1007/BF00992698) — **Watkins & Dayan (1992), Machine Learning 8** — the paper that introduced Q-learning and proved its convergence to $Q^*$.
- [Learning from Delayed Rewards](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) — **Christopher Watkins (1989), PhD thesis** — where Q-learning first appears, in full.
- [Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — **Hado van Hasselt (2010), NeurIPS** — fixes Q-learning's maximization bias; the precursor to Double DQN.
- [Human-level control through deep reinforcement learning (DQN)](https://www.nature.com/articles/nature14236) — **Mnih et al. (2015), Nature** — Q-learning with a neural network, experience replay, and a target network; where tabular Q-learning goes to scale ([author-hosted PDF](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)).
- [On the Convergence of Stochastic Iterative Dynamic Programming Algorithms](https://www.mit.edu/~jnt/Papers/J052-94-jjt-td.pdf) — **Jaakkola, Jordan & Singh (1994)** — the stochastic-approximation convergence analysis that underlies the Q-learning guarantee.

**Books (free chapters)**:
- [Reinforcement Learning: An Introduction (2nd ed.) — §6.5 "Q-learning" & §6.6 "Expected SARSA"](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the canonical treatment: the update, off-policy property, Example 6.6 (CliffWalking vs SARSA), maximization bias, and Double Q-learning (§6.7). The whole book is free.
- [Algorithms for Reinforcement Learning — §3.2 (TD control)](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the convergence theory for Q-learning, concise and rigorous (free PDF).
- [Grokking Deep Reinforcement Learning — value-based methods](https://github.com/mimoralea/gdrl) — **Miguel Morales** — the accompanying code (notebooks) is fully open and builds Q-learning, SARSA, and Double Q-learning from scratch.

**In this platform**:
- Concept page (full explanation): [Q-Learning](06-Q-Learning.md)
- Runnable code: [the module `q_learning.py`](code/q_learning.py) · [the step-by-step notebook](code/06-Q-Learning.ipynb) — from-scratch tabular Q-learning on real Gymnasium `FrozenLake-v1` and `CliffWalking-v1`, with value iteration as ground truth and the learned greedy policy asserted equal to the DP optimum (gap 0; greedy return −13 = V* on the cliff); every number measured, none fabricated.
- The framework it lives in: [01 Markov Decision Processes](../01-Markov-Decision-Processes/01-Markov-Decision-Processes.md) — states, actions, transitions, reward, and $\gamma$, the language Q-learning is written in.
- The equations it approximates: [02 Bellman Equations](../02-Bellman-Equations/02-Bellman-Equations.md) — the Bellman optimality equation for $Q^*$ that the Q-learning target samples.
- The ground truth it is checked against: [03 Dynamic Programming (Value & Policy Iteration)](../03-Dynamic-Programming-Value-and-Policy-Iteration/03-Dynamic-Programming-Value-and-Policy-Iteration.md) — the model-based optimum this page proves Q-learning recovers.
- The learning update it uses: [05 Temporal-Difference Learning](../05-Temporal-Difference-Learning/05-Temporal-Difference-Learning.md) — TD prediction; Q-learning is TD *control* with a greedy target.
- The on-policy contrast: [07 SARSA](../07-SARSA/07-SARSA.md) — the same loop with an on-policy target; the CliffWalking Example 6.6 counterpart.
- The exploration knob: [14 Exploration vs Exploitation](../14-Exploration-vs-Exploitation/14-Exploration-vs-Exploitation.md) — ε-greedy and the trade-off that satisfies Q-learning's "visit everything" convergence condition.
- Where it scales next: [08 Deep Q-Networks (DQN)](../08-Deep-Q-Networks-DQN/08-Deep-Q-Networks-DQN.md) — Q-learning with a neural network replacing the table, for large or continuous state spaces.
- Field overview: [Reinforcement Learning](../README.md)

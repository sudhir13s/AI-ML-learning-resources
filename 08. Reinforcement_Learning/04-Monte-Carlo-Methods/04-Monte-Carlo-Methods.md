---
id: "08-rl/monte-carlo-methods"
topic: "Monte Carlo Methods"
parent: "08-reinforcement-learning"
level: intermediate
prereqs: ["markov-decision-processes", "expectation", "law-of-large-numbers"]
interview_frequency: high
updated: 2026-06-20
---

# Monte Carlo Methods
> Learn values by **averaging complete returns** from sampled episodes — no model needed. Play an
> episode to the end, look at the actual return that followed each state, and average over many
> episodes. It's the most direct embodiment of "value = expected return," and the natural contrast
> to the bootstrapping of TD learning.

**Why it matters:** the first model-free method, and the classic "MC vs TD" interview question —
MC is unbiased but high-variance and needs episodes to terminate; TD is biased (bootstraps) but
lower-variance and works online. Also where first-visit vs every-visit, exploring starts, and
on- vs off-policy (importance sampling) first appear.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA). *Introduces MC prediction as averaging sampled returns, then contrasts with TD.*
2. **Read the chapter** — [Sutton & Barto Ch. 5 "Monte Carlo Methods"](http://incompleteideas.net/book/RLbook2020.pdf). *First-visit vs every-visit MC, MC control with exploring starts, off-policy MC via importance sampling.*
3. **See MC control** — [David Silver Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4). *How MC estimates of Q drive ε-greedy policy improvement.*
4. **Place it on the map** — [Lilian Weng: RL overview (MC section)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). *Where MC sits between exact DP and bootstrapping TD.*

## 🎓 Courses (free)
- [UCL Course on RL — Lectures 4 & 5](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — MC prediction and MC control with exploring starts and ε-greedy.
- [Stanford CS234 — Model-Free Policy Evaluation](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — MC estimation and its variance, bias, and convergence.
- [Spinning Up — Key Concepts (returns & value estimation)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — grounds the return that MC averages.

## 🎥 Videos
- [RL Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA) — **David Silver (DeepMind)** — MC vs TD prediction from sampled returns.
- [RL Lecture 5: Model-Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4) — **David Silver (DeepMind)** — MC control: GLIE, exploring starts, ε-greedy improvement.
- [Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) — **StatQuest (Josh Starmer)** — gentle grounding in returns and value estimation.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — intuitive overview placing sampling-based value estimation in context.

## 📄 Key Papers
- [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166) — **Kaelbling, Littman & Moore (1996)** — covers Monte Carlo value estimation among model-free methods.
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári (2010)** — the bias/variance and convergence analysis of MC vs bootstrapping.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into RL — Monte Carlo](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — MC prediction/control and the MC-vs-TD contrast, clearly illustrated.
- [Spinning Up — Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — the return and value definitions MC samples.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 5 "Monte Carlo Methods"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — first/every-visit MC, exploring starts, off-policy MC, importance sampling.
- [Algorithms for Reinforcement Learning — **§3 (value prediction)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the estimation theory behind MC.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [03 Dynamic Programming](../03-Dynamic-Programming-Value-and-Policy-Iteration/03-Dynamic-Programming-Value-and-Policy-Iteration.md) · Contrast with: [05 Temporal-Difference Learning](../05-Temporal-Difference-Learning/05-Temporal-Difference-Learning.md)

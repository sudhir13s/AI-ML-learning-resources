---
id: "08-rl/multi-armed-bandits"
topic: "Multi-Armed Bandits"
parent: "08-reinforcement-learning"
level: intermediate
prereqs: ["probability", "expectation", "exploration-vs-exploitation"]
interview_frequency: high
updated: 2026-06-20
---

# Multi-Armed Bandits
> The simplest RL setting — one state, `k` actions ("arms"), each with an unknown reward distribution.
> No transitions, no credit assignment over time: pure **exploration vs exploitation**. Pull arms,
> learn their means, and minimize **regret** (the gap to always pulling the best arm). **Contextual
> bandits** add features per round (the model behind ad/recommendation/clinical-trial allocation),
> bridging to full RL.

**Why it matters:** the cleanest place to *prove* exploration results, and a direct interview/industry
topic (A/B testing, recommendations, ad allocation). Be ready to define regret, derive/state UCB1's
logarithmic regret, explain Thompson sampling (Bayesian posterior sampling), and the difference between
context-free and contextual bandits — and why a bandit is "an MDP with one state."

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Lilian Weng: The Multi-Armed Bandit Problem](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/). *ε-greedy, UCB1, and Thompson sampling with regret, all in one place.*
2. **Get the chapter** — [Sutton & Barto Ch. 2 "Multi-armed Bandits"](http://incompleteideas.net/book/RLbook2020.pdf). *Action-value estimation, ε-greedy, optimistic init, UCB, gradient bandits.*
3. **Watch the lecture** — [David Silver Lecture 9: Exploration & Exploitation](https://www.youtube.com/watch?v=sGuiWX07sKw). *Frames bandits as the regret-minimization core of exploration.*
4. **Go rigorous** — [Bandit Algorithms (Lattimore & Szepesvári)](https://tor-lattimore.com/downloads/book/book.pdf). *The definitive free book: stochastic, adversarial, and contextual bandits with proofs.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 9: Exploration & Exploitation](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — bandits, regret, UCB, Thompson sampling.
- [Stanford CS234 — Exploration & Bandits](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — bandit regret bounds and the link to RL exploration.
- [Berkeley CS285 — Exploration](https://rail.eecs.berkeley.edu/deeprlcourse/) — **UC Berkeley (Sergey Levine)** — bandit-style exploration scaled to deep RL.

## 🎥 Videos
- [RL Lecture 9: Exploration & Exploitation](https://www.youtube.com/watch?v=sGuiWX07sKw) — **David Silver (DeepMind)** — bandits, regret, UCB, and Thompson sampling derived.
- [RL Series: Overview of Methods](https://www.youtube.com/watch?v=i7q8bISGwMQ) — **Steve Brunton** — where bandits sit relative to full sequential RL.
- [Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) — **StatQuest (Josh Starmer)** — gentle grounding in reward estimation and action selection.
- [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0) — **Arxiv Insights** — the exploration–exploitation framing bandits isolate.

## 📄 Key Papers
- [Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)](https://link.springer.com/article/10.1023/A:1013689704352) — **Auer, Cesa-Bianchi & Fischer (2002)** — UCB1 and its logarithmic regret.
- [An Empirical Evaluation of Thompson Sampling](https://proceedings.neurips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) — **Chapelle & Li (2011)** — Thompson sampling's strong empirical performance for (contextual) bandits.
- [A Contextual-Bandit Approach to Personalized News (LinUCB)](https://arxiv.org/abs/1003.0146) — **Li et al. (2010)** — contextual bandits in production (news recommendation).

## 📰 Articles / Blogs (free, no paywall)
- [The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/) — **Lilian Weng** — the canonical open tutorial: ε-greedy, UCB, Thompson, regret.
- [Spinning Up — Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — situates bandits as the single-state special case of RL.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 2 "Multi-armed Bandits"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — action-value estimation, ε-greedy, optimistic init, UCB, gradient bandits.
- [Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf) — **Lattimore & Szepesvári** — the definitive free reference: stochastic, adversarial, contextual bandits, with regret proofs.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.04 MDPs & Exploration](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.04_MDPs_and_Exploration.md)
- Closely related: [14 Exploration vs Exploitation](../14-Exploration-vs-Exploitation/14-Exploration-vs-Exploitation.md) · Generalizes to: [01 Markov Decision Processes](../01-Markov-Decision-Processes/01-Markov-Decision-Processes.md)

---
id: "10-rl/markov-decision-processes"
topic: "Markov Decision Processes"
parent: "10-reinforcement-learning"
level: intermediate
prereqs: ["probability", "expectation", "discounting"]
interview_frequency: very-high
updated: 2026-06-20
---

# Markov Decision Processes (MDPs)
> The mathematical frame for sequential decision-making: an agent in a **state** picks an **action**,
> the environment returns a **reward** and a (stochastic) **next state**, and this repeats. The
> "Markov" property says the next state depends only on the current state and action — the past is
> already summarized in the present. Every RL algorithm is, at bottom, solving an MDP.

**Why it matters:** the very first RL interview question — define an MDP `(S, A, P, R, γ)`, state the
Markov property, explain return vs reward, why we discount with γ, and the difference between a policy,
a value function, and the model. Get this wrong and nothing downstream (Bellman, Q-learning, PPO) makes sense.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 2: MDPs](https://www.youtube.com/watch?v=lfHX2hHRMVQ). *The canonical lecture; sets up states, returns, value, and policy precisely.*
2. **See it formalized** — read [Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). *Clean, modern definitions of state, action, reward, return, policy, value — the vocabulary you'll reuse everywhere.*
3. **Get the textbook treatment** — [Sutton & Barto Ch. 3 "Finite MDPs"](http://incompleteideas.net/book/RLbook2020.pdf). *Defines the MDP, the Markov property, returns, and the value functions you'll derive Bellman equations over next.*
4. **Cement the formalism** — work the grid-world examples in [Lilian Weng: A (Long) Peek into RL](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). *Ties the pieces together with worked notation before you move on to Bellman.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 2: MDPs](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — slides + video; the reference treatment of the MDP formalism.
- [Spinning Up — Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — the cleanest modern intro to states, actions, returns, policies, and value functions.
- [Stanford CS234 — Reinforcement Learning](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — early lectures formalize MDPs and Markov reward processes.

## 🎥 Videos
- [RL Lecture 2: Markov Decision Processes](https://www.youtube.com/watch?v=lfHX2hHRMVQ) — **David Silver (DeepMind)** — the definitive walkthrough of the MDP, return, and value.
- [Reinforcement Learning 2: Markov Decision Processes](https://www.youtube.com/watch?v=RmOdTQYQqmQ) — **DeepMind x UCL** — a second, complementary derivation of the same formalism.
- [Markov Decision Processes (MDPs) — Structuring an RL Problem](https://www.youtube.com/watch?v=my207WNoeyA) — **deeplizard** — short, visual, beginner-friendly intro to the components.
- [L1: MDPs & Exact Solution Methods](https://www.youtube.com/watch?v=2GwBez0D20A) — **Pieter Abbeel (Foundations of Deep RL)** — concise modern framing of MDPs leading into value/policy iteration.

## 📄 Key Papers
- [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166) — **Kaelbling, Littman & Moore (1996)** — the classic open-access survey that frames RL as solving MDPs (states, actions, value, policy).
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári (2010)** — a free monograph whose §2 is a rigorous, self-contained formal treatment of MDPs.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — rigorous, well-illustrated tour of the MDP formalism and the algorithm landscape.
- [Spinning Up — Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — **OpenAI** — definitions you'll reuse in every later card, with crisp notation.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 3 "Finite Markov Decision Processes"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the canonical chapter: agent–environment interface, returns, episodic vs continuing, value functions.
- [Algorithms for Reinforcement Learning — **§2 "Markov Decision Processes"**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — compact, mathematically precise free monograph.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.04 MDPs & Exploration](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.04_MDPs_and_Exploration.md)
- Next concepts: [02 Bellman Equations](02-Bellman-Equations.md) · [03 Dynamic Programming](03-Dynamic-Programming-Value-and-Policy-Iteration.md)

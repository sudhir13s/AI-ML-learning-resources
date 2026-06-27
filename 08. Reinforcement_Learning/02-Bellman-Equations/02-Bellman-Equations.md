---
id: "08-rl/bellman-equations"
topic: "Bellman Equations"
parent: "08-reinforcement-learning"
level: intermediate
prereqs: ["markov-decision-processes", "expectation", "recursion"]
interview_frequency: very-high
updated: 2026-06-20
---

# Bellman Equations
> The recursive heart of RL: the value of a state equals the immediate reward plus the discounted
> value of where you land next. This self-consistency turns an infinite sum over the future into a
> single one-step equation — `V(s) = E[r + γ V(s')]` — and its **optimality** form
> `V*(s) = max_a E[r + γ V*(s')]` defines the best possible behavior.

**Why it matters:** the equation every value-based method (DP, TD, Q-learning, DQN) is secretly
solving. Interviewers ask you to write the Bellman expectation vs optimality equations, explain the
`max` vs expectation, why it's a contraction (so iteration converges), and how the *Bellman error*
becomes the TD target.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 2 (value & Bellman)](https://www.youtube.com/watch?v=lfHX2hHRMVQ). *Derives the Bellman expectation equation directly from the definition of return.*
2. **Get the optimality form** — read [Sutton & Barto §3.5–3.6](http://incompleteideas.net/book/RLbook2020.pdf). *The Bellman optimality equations for V\* and Q\*, with the backup-diagram pictures.*
3. **See why iteration converges** — [Lilian Weng: RL overview (Bellman section)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). *The γ-contraction argument — why repeatedly applying the Bellman operator finds the fixed point.*
4. **Make it concrete** — trace the equation through a grid-world in [David Silver Lecture 3 (DP)](https://www.youtube.com/watch?v=Nd1-UUMVfz4). *Watch value iteration apply the Bellman optimality backup until it converges.*

## 🎓 Courses (free)
- [UCL Course on RL — Lectures 2 & 3](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — derives the Bellman expectation and optimality equations and uses them in DP.
- [Spinning Up — Bellman Equations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations) — **OpenAI** — compact statement of all four Bellman equations (on-policy / optimal × V / Q).
- [Stanford CS234 — Lecture 2 (MDPs & Bellman)](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — the contraction-mapping view and convergence.

## 🎥 Videos
- [RL Lecture 2: MDPs (value functions & Bellman)](https://www.youtube.com/watch?v=lfHX2hHRMVQ) — **David Silver (DeepMind)** — derives the Bellman expectation equation from first principles.
- [RL Lecture 3: Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4) — **David Silver (DeepMind)** — the Bellman optimality backup in action via value/policy iteration.
- [Reinforcement Learning 2: Markov Decision Processes](https://www.youtube.com/watch?v=RmOdTQYQqmQ) — **DeepMind x UCL** — a clean second derivation of the Bellman equations.
- [Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) — **StatQuest (Josh Starmer)** — gentle visual grounding for value and the one-step recursion.

## 📄 Key Papers
- [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166) — **Kaelbling, Littman & Moore (1996)** — presents the Bellman equations as the basis for value-based RL.
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári (2010)** — rigorous treatment of the Bellman operator as a γ-contraction and its fixed point.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into Reinforcement Learning — Bellman equations](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — clear derivation plus the contraction/convergence intuition.
- [Spinning Up — Bellman Equations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations) — **OpenAI** — the four Bellman equations stated precisely in one place.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **§3.5–3.6 & Ch. 4 "Dynamic Programming"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the Bellman expectation and optimality equations, backup diagrams, and policy-evaluation iteration.
- [Algorithms for Reinforcement Learning — **§2 (the Bellman operators)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the operator-theoretic view that proves convergence.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [01 Markov Decision Processes](../01-Markov-Decision-Processes/01-Markov-Decision-Processes.md) · Next: [03 Dynamic Programming](../03-Dynamic-Programming-Value-and-Policy-Iteration/03-Dynamic-Programming-Value-and-Policy-Iteration.md)

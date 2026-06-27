---
id: "08-rl/dynamic-programming"
topic: "Dynamic Programming — Value & Policy Iteration"
parent: "08-reinforcement-learning"
level: intermediate
prereqs: ["bellman-equations", "markov-decision-processes"]
interview_frequency: high
updated: 2026-06-20
---

# Dynamic Programming — Value & Policy Iteration
> When you **know the model** (transitions and rewards), you can solve an MDP exactly by repeatedly
> applying the Bellman backup. **Policy iteration** alternates evaluating a policy and greedily
> improving it; **value iteration** folds both into one `max` backup. These are the ground-truth
> planners that every model-free method approximates by sampling.

**Why it matters:** the bridge from Bellman equations to algorithms. Interviewers ask the difference
between policy iteration and value iteration, why both converge (contraction + policy-improvement
theorem), their complexity, and why DP needs a known model — which motivates Monte Carlo and TD.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [David Silver Lecture 3: Planning by DP](https://www.youtube.com/watch?v=Nd1-UUMVfz4). *The definitive walkthrough of policy evaluation, policy iteration, and value iteration.*
2. **Read the chapter** — [Sutton & Barto Ch. 4 "Dynamic Programming"](http://incompleteideas.net/book/RLbook2020.pdf). *Policy-evaluation sweeps, the policy-improvement theorem, generalized policy iteration (GPI).*
3. **Compare the two** — [Steve Brunton: Model-Based RL — Policy & Value Iteration](https://www.youtube.com/watch?v=sJIFUTITfBc). *Side-by-side of the two algorithms and when each is cheaper.*
4. **Code it** — implement value iteration on a grid-world following the [Spinning Up intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html). *Watch values propagate from the goal outward until the greedy policy stops changing.*

## 🎓 Courses (free)
- [UCL Course on RL — Lecture 3: Planning by Dynamic Programming](https://www.davidsilver.uk/teaching/) — **David Silver (DeepMind)** — the canonical lecture on policy evaluation, policy iteration, and value iteration.
- [Stanford CS234 — Planning & DP](https://web.stanford.edu/class/cs234/) — **Stanford (Emma Brunskill)** — convergence proofs and the contraction argument.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — situates model-based DP against model-free learning.

## 🎥 Videos
- [RL Lecture 3: Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4) — **David Silver (DeepMind)** — policy evaluation → policy iteration → value iteration, fully derived.
- [Model-Based RL: Policy Iteration, Value Iteration](https://www.youtube.com/watch?v=sJIFUTITfBc) — **Steve Brunton** — clear, intuitive contrast of the two exact methods.
- [L1: MDPs & Exact Solution Methods](https://www.youtube.com/watch?v=2GwBez0D20A) — **Pieter Abbeel (Foundations of Deep RL)** — value iteration and policy iteration as the exact solvers.
- [Reinforcement Learning 2: Markov Decision Processes](https://www.youtube.com/watch?v=RmOdTQYQqmQ) — **DeepMind x UCL** — sets up the Bellman backups that DP iterates.

## 📄 Key Papers
- [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166) — **Kaelbling, Littman & Moore (1996)** — frames DP (value/policy iteration) as the exact MDP solvers.
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári (2010)** — convergence of value and policy iteration via the Bellman operator.

## 📰 Articles / Blogs (free, no paywall)
- [A (Long) Peek into RL — Dynamic Programming](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) — **Lilian Weng** — policy iteration and value iteration with worked notation.
- [Spinning Up — Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) — **OpenAI** — where exact DP sits relative to model-free and model-based learning.

## 📚 Books (free, with chapters)
- [Reinforcement Learning: An Introduction (2nd ed.) — **Ch. 4 "Dynamic Programming"**](http://incompleteideas.net/book/RLbook2020.pdf) — **Sutton & Barto** — the definitive chapter: iterative policy evaluation, policy/value iteration, GPI, asynchronous DP.
- [Algorithms for Reinforcement Learning — **§2 (planning in MDPs)**](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) — **Csaba Szepesvári** — the math behind why DP converges.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 6.01 Bellman Optimality & Q-Learning](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.01_Bellman_Optimality_Q-Learning.md)
- Prereq: [02 Bellman Equations](../02-Bellman-Equations/02-Bellman-Equations.md) · Next: [04 Monte Carlo Methods](../04-Monte-Carlo-Methods/04-Monte-Carlo-Methods.md) · [05 Temporal-Difference Learning](../05-Temporal-Difference-Learning/05-Temporal-Difference-Learning.md)

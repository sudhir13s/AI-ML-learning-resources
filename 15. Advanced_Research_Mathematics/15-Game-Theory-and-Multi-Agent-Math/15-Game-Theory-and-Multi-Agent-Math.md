---
id: "15-advanced-math/game-theory-multi-agent"
topic: "Game Theory & Multi-Agent Math"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["probability", "convex-analysis-duality", "optimization"]
interview_frequency: medium
updated: 2026-06-20
---

# Game Theory & Multi-Agent Math
> The mathematics of strategic interaction among rational agents: normal- and extensive-form games,
> dominant strategies, **Nash equilibrium** (and its existence via fixed-point theorems), mixed
> strategies, minimax/zero-sum duality, correlated equilibria, and the learning dynamics
> (no-regret, fictitious play) by which agents *reach* equilibria. The decision-theoretic backbone of
> multi-agent RL, mechanism design, and adversarial training.

**Why it matters:** GAN training is a two-player minimax game (and its instabilities are
equilibrium-finding instabilities); adversarial robustness is a game against a perturbing adversary;
multi-agent RL, self-play (AlphaGo/AlphaZero), and RLHF reward modeling all live here. Minimax duality
ties straight back to convex duality (card 4), and "what is a Nash equilibrium, and why does one
always exist?" is a recurring interview question.

**⭐ Start here — suggested path:**

1. **Get the five core ideas** — watch [Yale ECON-159, Lecture 1: five first lessons](https://www.youtube.com/watch?v=nM3rTU927io). *Dominance, best response, and why rational play can be self-defeating.*
2. **Nail Nash equilibrium** — watch [Game Theory 101 (#5): What Is a Nash Equilibrium?](https://www.youtube.com/watch?v=5TcYV6CZ7mI). *The single most important solution concept, cleanly.*
3. **See mixed strategies** — watch [What Is a Nash Equilibrium? (Stoplight Game)](https://www.youtube.com/watch?v=0i7p9DNvtjk). *Why randomizing can be optimal — the bridge to minimax.*
4. **Read the reference** — work [Multiagent Systems (Shoham & Leyton-Brown), Ch. 3–5](https://www.masfoundations.org/mas.pdf). *Normal/extensive games, equilibria, and computation, the free standard text.*
5. **Connect to ML** — read [GANs](https://arxiv.org/abs/1406.2661) and link to [bandits/RL](../../08.%20Reinforcement_Learning/concepts/README.md). *Minimax games, self-play, and learning dynamics in modern ML.*

## 🎓 Courses (free)
- [Game Theory (Open Yale ECON-159)](https://oyc.yale.edu/economics/econ-159) — **Ben Polak (Yale)** — the legendary full course: dominance, Nash, mixed strategies, signaling, free video + transcripts.
- [Multiagent Systems — free book & course](https://www.masfoundations.org/) — **Shoham & Leyton-Brown** — game theory, mechanism design, and multi-agent learning, fully free.
- [Bandit Algorithms — free book](https://banditalgs.com/) — **Lattimore & Szepesvári** — the single-agent decision theory that multi-agent learning generalizes, free.

## 🎥 Videos
- [Yale ECON-159 — Lecture 1: Introduction, five first lessons](https://www.youtube.com/watch?v=nM3rTU927io) — **Ben Polak (Yale)** — dominance and best response, the foundations.
- [Game Theory — full Open Yale lecture](https://www.youtube.com/watch?v=M3oWYHYoBvk) — **Yale University (Ben Polak)** — the complete classroom treatment of strategic reasoning.
- [Game Theory 101 (#5): What Is a Nash Equilibrium?](https://www.youtube.com/watch?v=5TcYV6CZ7mI) — **William Spaniel** — the cleanest short explanation of Nash equilibrium.
- [Game Theory 101: What Is a Nash Equilibrium? (Stoplight Game)](https://www.youtube.com/watch?v=0i7p9DNvtjk) — **William Spaniel** — mixed strategies and the intuition behind randomization.

## 📄 Key Papers
- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — **Goodfellow et al. (2014)** — generative modeling as a two-player minimax game, the flagship ML application.
- [Game Theory and Multi-Agent Reinforcement Learning: From Nash Equilibria to Evolutionary Dynamics](https://arxiv.org/abs/2412.20523) — **(survey, 2024)** — the modern bridge from equilibria to MARL, free on arXiv.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — **Arjovsky, Chintala & Bottou (2017)** — reframes the GAN game with a better-behaved minimax objective.

## 📰 Articles / Blogs (free, no paywall)
- [Multiagent Systems — full free book](https://www.masfoundations.org/mas.pdf) — **Shoham & Leyton-Brown** — equilibria, computation of Nash, mechanism design, openly posted.
- [Open Yale Game Theory — lecture transcripts](https://oyc.yale.edu/economics/econ-159) — **Ben Polak** — the full course in readable transcript form, free.

## 📚 Books (free, with chapters)
- [Multiagent Systems — **Ch. 3 (normal-form games), Ch. 5 (extensive-form), Ch. 7 (learning in games)**](https://www.masfoundations.org/mas.pdf) — **Shoham & Leyton-Brown** — the free standard text.
- [Bandit Algorithms — **Ch. on adversarial bandits & game-theoretic learning**](https://tor-lattimore.com/downloads/book/book.pdf) — **Lattimore & Szepesvári** — no-regret learning as equilibrium-finding, free PDF.
- [Convex Optimization — **§5.4–5.8 (duality, minimax, saddle points)**](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — the minimax-duality math behind zero-sum games, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.04 GANs & WGAN](../../../AI-ML-intuition/Module_5_Generation/5.04_GANs_and_WGAN.md) · [6.04 MDPs & Exploration](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.04_MDPs_and_Exploration.md)
- Prerequisite & related: [04 Convex Analysis & Duality (minimax)](04-Convex-Analysis-and-Duality.md) · [14 Causal Inference (interventions)](14-Causal-Inference.md) · [09 Optimal Transport (WGAN)](09-Optimal-Transport-Wasserstein.md)
- Related domain (multi-agent decisions): [10. Reinforcement Learning](../../08.%20Reinforcement_Learning/concepts/README.md)
</content>

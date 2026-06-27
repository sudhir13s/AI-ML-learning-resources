---
id: "01-foundations/gradient-descent-theory/references"
topic: "Gradient Descent — References"
parent: "01-foundations/gradient-descent-theory"
type: references
updated: 2026-06-21
---

# Gradient Descent — references and further reading

> Companion link library for **[Gradient Descent — theory & convergence](13-Gradient-Descent-Theory.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Intuition** — watch [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) (**StatQuest**) and [how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (**3Blue1Brown**). *The "roll downhill" picture and the role of the step size.*
2. **The update & batch vs stochastic** — read [CS231n Optimization notes](https://cs231n.github.io/optimization-1/). *Full-batch, mini-batch, and SGD, with practical step-size guidance.*
3. **Feel the conditioning** — play with [Why Momentum Really Works](https://distill.pub/2017/momentum/) (**Distill**). *Dial the condition number and watch the zig-zag.*
4. **The convergence theory** — read [Boyd & Vandenberghe Ch. 9](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). *Step-size rules, convex/smooth convergence, conditioning.*
5. **Why SGD works** — read [Bottou, Curtis & Nocedal](https://arxiv.org/abs/1606.04838). *The rigorous treatment of stochastic-gradient convergence.*

**Videos**:
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — from-scratch intuition with a worked example.
- [Gradient descent, how neural networks learn (Ch. 2)](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — gradient descent on a real loss surface.
- [Convex Optimization I — Lecture 1](https://www.youtube.com/watch?v=McLq1hEq3UY) — **Stephen Boyd (Stanford)** — sets up the descent-method framework.
- [Gradients and Partial Derivatives](https://www.youtube.com/watch?v=GkB4vW16QHI) — **Eugene Khutoryansky** — the steepest-descent direction visualized.

**Interactive & visual**:
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — **Goh (Distill)** — interactive: change the condition number and watch gradient descent zig-zag, then watch momentum fix it.

**Courses (free)**:
- [Stanford CS231n — Optimization](https://cs231n.github.io/optimization-1/) — **Stanford** — gradient descent, SGD, and update rules, ML-focused.
- [Stanford EE364a — Convex Optimization I (descent methods)](https://web.stanford.edu/class/ee364a/) — **Stephen Boyd (Stanford)** — convergence theory for gradient/descent methods.

**Articles / blogs (free, no paywall)**:
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — batch/stochastic/mini-batch and the lineage of optimizers.

**Key papers**:
- [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) — **Bottou, Curtis & Nocedal (2018)** — the definitive survey of SGD theory and convergence.
- [Identifying and attacking the saddle point problem](https://arxiv.org/abs/1406.2572) — **Dauphin et al. (2014)** — why saddle points, not local minima, are the real obstacle in high-dimensional non-convex optimization.

**Books (free chapters)**:
- [Convex Optimization — Ch. 9 (Unconstrained Minimization)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — gradient-descent convergence, step sizes, and conditioning (free PDF).
- [Mathematics for Machine Learning — Ch. 7.1 (Gradient Descent)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — gradient descent and step-size intuition for ML (free PDF).

**In this platform**:
- Concept page (full explanation): [Gradient Descent — theory & convergence](13-Gradient-Descent-Theory.md)
- Concept depth (the *why*): [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)
- Applied next layer: [Optimizers](../../05.%20Deep_Learning/concepts/07-Optimizers.md) (momentum, Adam, AdamW, LR schedules) — what fixes the zig-zag
- Prereqs: [Derivatives & Gradients](08-Derivatives-and-Gradients.md) · [Convexity](12-Convexity.md)
- Curriculum context: [Maths for AI-ML — Phase 5 (Optimization)](../Maths%20for%20AI-ML/README.md)

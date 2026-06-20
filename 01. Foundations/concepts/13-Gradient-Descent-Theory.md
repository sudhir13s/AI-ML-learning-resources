---
id: "01-foundations/gradient-descent-theory"
topic: "Gradient Descent — theory & convergence"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/derivatives-and-gradients", "01-foundations/convexity"]
interview_frequency: very-high
updated: 2026-06-20
---

# Gradient Descent — theory & convergence
> Repeatedly step against the gradient — `θ ← θ − η∇L(θ)` — and you descend the loss surface. The
> *theory* answers the questions that matter: when does it converge, how fast, how does the step
> size (and curvature/conditioning) control everything, and what changes when the gradient is a
> noisy stochastic estimate.

**Why it matters:** this is the algorithm that trains essentially every model. Interviewers probe
the theory: the convergence rate on convex/smooth functions, why the step size must respect the
Lipschitz/smoothness constant, how the condition number sets the rate, and why SGD's noisy
gradients still converge (in expectation) and even help escape saddles.

**⭐ Start here — suggested path:**

1. **Intuition** — watch [StatQuest: Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) and [3B1B: how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w). *The "roll downhill" picture and the role of the step size.*
2. **The update & batch vs stochastic** — read [CS231n Optimization notes](https://cs231n.github.io/optimization-1/). *Full-batch, mini-batch, and SGD, with practical step-size guidance.*
3. **The convergence theory** — read [Boyd & Vandenberghe Ch. 9.2–9.4 (Descent methods, gradient descent)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). *Step-size rules, convergence on convex/smooth functions, conditioning.*
4. **Why SGD works** — read [Bottou, Curtis & Nocedal: Optimization Methods for Large-Scale ML](https://arxiv.org/abs/1606.04838). *The rigorous treatment of stochastic gradient convergence.*
5. **Connect to ML** — read [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md). *The platform's deep dive, including why mini-batches work.*

## 🎓 Courses (free)
- [Stanford CS231n — Optimization](https://cs231n.github.io/optimization-1/) — **Stanford** — gradient descent, SGD, and update rules, ML-focused.
- [Stanford EE364a — Convex Optimization I (descent methods)](https://web.stanford.edu/class/ee364a/) — **Stephen Boyd (Stanford)** — convergence theory for gradient/descent methods.

## 🎥 Videos
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — from-scratch intuition with a worked example.
- [Gradient descent, how neural networks learn | Deep Learning Ch. 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — gradient descent on a real loss surface.
- [Convex Optimization I — Lecture 1](https://www.youtube.com/watch?v=McLq1hEq3UY) — **Stephen Boyd (Stanford)** — sets up the descent-method framework.
- [Gradients and Partial Derivatives](https://www.youtube.com/watch?v=GkB4vW16QHI) — **Eugene Khutoryansky** — the steepest-descent direction visualized.

## 📄 Key Papers
- [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) — **Bottou, Curtis & Nocedal (2018)** — the definitive survey of SGD theory and convergence.
- [Convex Optimization — Ch. 9 (Unconstrained Minimization)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — gradient-descent convergence, step sizes, and conditioning (free PDF).

## 📰 Articles / Blogs (free, no paywall)
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — batch/stochastic/mini-batch and the lineage of optimizers, fully free.
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — **Goh (Distill)** — an interactive look at how curvature/conditioning shapes convergence.

## 📚 Books (free, with chapters)
- [Convex Optimization — **Ch. 9 (Unconstrained Minimization)**](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — the convergence theory (free PDF).
- [Mathematics for Machine Learning — **Ch. 7.1 (Gradient Descent)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — gradient descent and step-size intuition for ML.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)
- Curriculum context: [Maths for AI-ML — Phase 5 (Optimization for ML/DL)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [08 Derivatives & Gradients](08-Derivatives-and-Gradients.md) · [12 Convexity](12-Convexity.md)
- Applied optimizers (momentum, Adam, AdamW, LR schedules) → [Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
</content>

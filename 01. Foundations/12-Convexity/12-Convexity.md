---
id: "01-foundations/convexity"
topic: "Convexity & Convex Functions"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/derivatives-and-gradients", "01-foundations/jacobian-and-hessian"]
interview_frequency: high
updated: 2026-06-20
---

# Convexity & Convex Functions
> A set is convex if the line between any two of its points stays inside it; a function is convex
> if it sits below its chords (equivalently, its Hessian is positive semidefinite). Convexity is
> the property that makes optimization *easy* — any local minimum is global — which is why so much
> ML theory is framed around it.

**Why it matters:** convexity is the dividing line between "guaranteed-solvable" (linear/logistic
regression, SVMs, LASSO) and "no global guarantees" (deep nets). Interviewers ask whether a loss is
convex, how to prove convexity (PSD Hessian, Jensen's inequality, composition rules), and why
convexity guarantees a unique global optimum.

**⭐ Start here — suggested path:**

1. **What & why** — watch [Visually Explained: Convexity & Duality](https://www.youtube.com/watch?v=d0CF3d5aEGc) and [What is Mathematical Optimization?](https://www.youtube.com/watch?v=AM6BY4btj-M). *The geometric "below-the-chord" picture and why it matters.*
2. **The definitions** — read [Boyd & Vandenberghe Ch. 2 (Convex Sets) & Ch. 3 (Convex Functions)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). *The canonical free source: convex sets/functions, first- and second-order conditions.*
3. **The full course** — watch [Stephen Boyd: Convex Optimization I, Lecture 1](https://www.youtube.com/watch?v=McLq1hEq3UY). *Boyd's own framing of why convexity is the key dividing line.*
4. **ML connection** — read MML Ch. 7 (Continuous Optimization). *Convexity, gradient descent, and constrained problems in an ML context.*
5. **Tie to losses** — note which ML losses are convex in their parameters (linear/logistic regression, SVM hinge) vs not (neural nets). *This framing recurs in interviews.*

## 🎓 Courses (free)
- [Stanford EE364a — Convex Optimization I](https://web.stanford.edu/class/ee364a/) — **Stephen Boyd (Stanford)** — the definitive course; lectures + slides + the free textbook.
- [MML book companion — Ch. 7 (Continuous Optimization)](https://mml-book.github.io/) — **Deisenroth et al.** — convexity and optimization scoped for ML.

## 🎥 Videos
- [Convexity and The Principle of Duality](https://www.youtube.com/watch?v=d0CF3d5aEGc) — **Visually Explained** — crisp geometric intuition for convex functions.
- [What Is Mathematical Optimization?](https://www.youtube.com/watch?v=AM6BY4btj-M) — **Visually Explained** — frames why convex problems are tractable.
- [Convex Optimization I — Lecture 1](https://www.youtube.com/watch?v=McLq1hEq3UY) — **Stephen Boyd (Stanford)** — the authoritative introduction.
- [Gradient descent, how neural networks learn | Ch. 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — contrasts the convex bowl with the non-convex landscapes of deep nets.

## 📄 Key Papers
- [Convex Optimization — Ch. 2–3 (Convex Sets & Functions)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — the canonical reference; definitions, conditions, and operations preserving convexity (free PDF).
- [MML book — Ch. 7 (Continuous Optimization)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — convexity, gradient descent, and constrained optimization for ML.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Optimization notes](https://cs231n.github.io/optimization-1/) — **Stanford** — convex vs non-convex loss landscapes in deep learning.
- [EE364a — Convex sets & functions lecture slides](https://web.stanford.edu/class/ee364a/lectures.html) — **Stephen Boyd (Stanford)** — concise free slides for self-study.

## 📚 Books (free, with chapters)
- [Convex Optimization — **Ch. 2–5**](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — convex sets, functions, problems, and duality (free PDF).
- [Mathematics for Machine Learning — **Ch. 7 (Continuous Optimization)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — convexity in the ML optimization chapter.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)
- Curriculum context: [Maths for AI-ML — Phase 5 (Optimization for ML/DL)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [10 Jacobian & Hessian](10-Jacobian-and-Hessian.md) (PSD Hessian ⇒ convex) · Next: [13 Gradient Descent — theory](13-Gradient-Descent-Theory.md) · [14 Lagrange Multipliers](14-Lagrange-Multipliers-Constrained-Optimization.md)
</content>

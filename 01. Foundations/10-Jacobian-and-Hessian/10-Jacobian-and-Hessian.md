---
id: "01-foundations/jacobian-and-hessian"
topic: "Jacobian & Hessian"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/derivatives-and-gradients", "01-foundations/chain-rule"]
interview_frequency: high
updated: 2026-06-20
---

# Jacobian & Hessian
> The **Jacobian** is the matrix of all first-order partials of a vector-valued function — the
> local linear map (and the object autograd multiplies in reverse mode). The **Hessian** is the
> matrix of second-order partials — local curvature, which tells you whether a critical point is a
> min, max, or saddle, and underlies Newton's method.

**Why it matters:** the Jacobian is how gradients propagate through vector functions (the "Jacobian
of the layer"); the Hessian governs curvature, conditioning, and second-order optimization.
Interviewers ask how the Jacobian relates to the chain rule, what a positive-definite Hessian
implies, why saddle points stall training, and what Newton's step does.

**⭐ Start here — suggested path:**

1. **Jacobian intuition** — watch [Khan: The Jacobian matrix](https://www.youtube.com/watch?v=bohL918kXQk). *The Jacobian as the local linear approximation of a multivariable map.*
2. **Hessian & curvature** — watch [Khan: The Hessian matrix](https://www.youtube.com/watch?v=LbBcuZukCAw). *Second derivatives, curvature, and the second-derivative test.*
3. **Formalize** — read [MML Ch. 5.3–5.7 (Vector Calculus)](https://mml-book.github.io/book/mml-book.pdf): Jacobians, higher-order derivatives, Taylor. *The precise definitions and shapes.*
4. **Why it matters for optimization** — read [Boyd & Vandenberghe Ch. 9.5 (Newton's method)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). *Where the Hessian becomes the step direction.*
5. **Connect to ML** — read [AI-ML-intuition 2.03 Jacobian & Hessian](../../../AI-ML-intuition/Module_2_Optimization/2.03_Jacobian_and_Hessian.md). *How both show up in training and curvature analysis.*

## 🎓 Courses (free)
- [Khan Academy — Multivariable derivatives (Jacobian, Hessian)](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) — **Khan Academy** — full unit covering both matrices with exercises.
- [Stanford CS231n — Backprop / Jacobian notes](https://cs231n.github.io/optimization-2/) — **Stanford** — Jacobian-vector products as backprop primitives.

## 🎥 Videos
- [The Jacobian matrix](https://www.youtube.com/watch?v=bohL918kXQk) — **Khan Academy** — the Jacobian as a local linear map.
- [The Hessian matrix](https://www.youtube.com/watch?v=LbBcuZukCAw) — **Khan Academy** — curvature and the second-derivative test.
- [What is the Jacobian?](https://www.youtube.com/watch?v=wCZ1VEmVjVo) — **Mathemaniac** — the right way to think about Jacobians in derivatives and integrals.
- [Backpropagation calculus | Deep Learning Ch. 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — **3Blue1Brown** — where Jacobian-style products appear in gradient flow.

## 📄 Key Papers
- [MML book — Ch. 5.3–5.7 (Vector Calculus)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — Jacobians, Hessians, and higher-order derivatives.
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/) — **Parr & Howard** — Jacobians of vector functions, the practical backprop view.

## 📰 Articles / Blogs (free, no paywall)
- [Identifying and attacking the saddle-point problem](https://arxiv.org/abs/1406.2572) — **Dauphin et al. (2014)** — why Hessian saddle points (not local minima) dominate deep-net loss surfaces.
- [The Matrix Cookbook — derivatives & Hessians](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — **Petersen & Pedersen** — the reference identities for Jacobians and Hessians.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 5 (Vector Calculus)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — Jacobian and Hessian with ML examples.
- [Convex Optimization — **Ch. 9.5 (Newton's method)**](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — the Hessian in second-order optimization (free PDF).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.03 Jacobian & Hessian](../../../AI-ML-intuition/Module_2_Optimization/2.03_Jacobian_and_Hessian.md)
- Curriculum context: [Maths for AI-ML — Phase 2 (Calculus, row 2.3)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [08 Derivatives & Gradients](08-Derivatives-and-Gradients.md) · [09 The Chain Rule](09-The-Chain-Rule.md) · Related: [11 Taylor Expansion](11-Taylor-Expansion.md) · [12 Convexity](12-Convexity.md)
</content>

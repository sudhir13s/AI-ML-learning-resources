---
id: "01-foundations/taylor-expansion"
topic: "Taylor Expansion"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/derivatives-and-gradients", "01-foundations/jacobian-and-hessian"]
interview_frequency: medium
updated: 2026-06-20
---

# Taylor Expansion
> A Taylor expansion approximates a function near a point by a polynomial built from its
> derivatives there: value + slope·Δ + ½·curvature·Δ² + …. It's the lens through which optimization
> "sees" the loss locally — gradient descent uses the first-order term, Newton's method the
> second-order (Hessian) term.

**Why it matters:** the first- and second-order Taylor expansions are the formal justification for
gradient descent and Newton's method, and they underpin convergence analysis and curvature
arguments. Interviewers ask you to write the quadratic Taylor model of a loss and read off the
optimal step — pure Taylor.

**⭐ Start here — suggested path:**

1. **Intuition** — watch [3B1B: Taylor series](https://www.youtube.com/watch?v=3d6DsjIBzJ4). *Why successive derivatives reconstruct a function, with the best visual treatment available.*
2. **Single-variable formalism** — read [Paul's Online Notes: Taylor Series](https://tutorial.math.lamar.edu/Classes/CalcII/TaylorSeries.aspx). *The expansion, remainder term, and worked examples.*
3. **Multivariable + the quadratic model** — read [MML Ch. 5.8 (Linearization & Multivariate Taylor)](https://mml-book.github.io/book/mml-book.pdf). *The gradient + Hessian quadratic form used in optimization.*
4. **See it drive optimization** — read [Boyd & Vandenberghe Ch. 9.5 (Newton's method)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). *The second-order Taylor model is exactly the Newton step.*

## 🎓 Courses (free)
- [MIT 18.01 Single Variable Calculus — Taylor series unit](https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/) — **MIT OCW** — derivation and convergence of Taylor/Maclaurin series.
- [Khan Academy — Taylor & Maclaurin series](https://www.khanacademy.org/math/ap-calculus-bc/bc-series-new/bc-10-11/v/maclaurin-and-taylor-series-intuition) — **Khan Academy** — guided build-up with exercises.

## 🎥 Videos
- [Taylor series | Essence of Calculus Ch. 11](https://www.youtube.com/watch?v=3d6DsjIBzJ4) — **3Blue1Brown** — the definitive visual intuition.
- [Taylor Series and Maclaurin Series](https://www.youtube.com/watch?v=LDBnS4c7YbA) — **The Organic Chemistry Tutor** — building the polynomial term by term, with worked examples.
- [The essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) — **3Blue1Brown** — the derivative groundwork Taylor stands on.
- [The paradox of the derivative | Ch. 2](https://www.youtube.com/watch?v=9vKqVkMQHKk) — **3Blue1Brown** — local linear approximation, the order-1 Taylor term.

## 📄 Key Papers
- [MML book — Ch. 5.8 (Linearization & Multivariate Taylor Series)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the multivariable Taylor expansion with gradient and Hessian.
- [Convex Optimization — Ch. 9.5 (Newton's method)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — **Boyd & Vandenberghe** — the quadratic Taylor model as the basis of Newton's method (free PDF).

## 📰 Articles / Blogs (free, no paywall)
- [Paul's Online Notes — Taylor Series](https://tutorial.math.lamar.edu/Classes/CalcII/TaylorSeries.aspx) — **Paul Dawkins** — the single-variable expansion and remainder, with examples.
- [CS231n — Optimization notes (local approximation)](https://cs231n.github.io/optimization-1/) — **Stanford** — how the first-order Taylor model justifies a gradient step.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 5.8 (Taylor)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — multivariate Taylor for ML.
- [Calculus (OpenStax) — **Vol. 2, Ch. 6 (Power & Taylor Series)**](https://openstax.org/details/books/calculus-volume-2) — **OpenStax** — free, thorough single-variable treatment.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.03 Jacobian & Hessian](../../../AI-ML-intuition/Module_2_Optimization/2.03_Jacobian_and_Hessian.md) · [2.05 Gradient Descent & SGD](../../../AI-ML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)
- Curriculum context: [Maths for AI-ML — Phase 2 (Calculus)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [08 Derivatives & Gradients](../08-Derivatives-and-Gradients/08-Derivatives-and-Gradients.md) · [10 Jacobian & Hessian](../10-Jacobian-and-Hessian/10-Jacobian-and-Hessian.md) · Next: [12 Convexity](../12-Convexity/12-Convexity.md) · [13 Gradient Descent — theory](../13-Gradient-Descent-Theory/13-Gradient-Descent-Theory.md)
</content>

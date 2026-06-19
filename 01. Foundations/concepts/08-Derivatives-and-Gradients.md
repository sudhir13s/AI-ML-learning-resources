---
id: "01-foundations/derivatives-and-gradients"
topic: "Derivatives & Gradients"
parent: "01-foundations"
level: beginner
prereqs: ["01-foundations/vectors-and-vector-spaces"]
interview_frequency: very-high
updated: 2026-06-20
---

# Derivatives & Gradients
> A derivative is a local rate of change; a gradient bundles all the partial derivatives into the
> vector that points in the direction of steepest ascent. Every training step in ML is "compute the
> gradient of the loss, step against it" — so this is the most operationally important math you'll
> own.

**Why it matters:** gradients drive all of gradient-based learning. Interviewers check that you
know the gradient points uphill (so you descend against it), that it's perpendicular to level sets,
what a directional derivative is, and why a zero gradient marks a stationary point — the setup for
optimization.

**⭐ Start here — suggested path:**

1. **Single-variable intuition** — watch [3B1B: The essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) and [The paradox of the derivative](https://www.youtube.com/watch?v=9vKqVkMQHKk). *Derivative as instantaneous rate, built from the ground up.*
2. **Go multivariable** — watch [Khan: Gradient and graphs](https://www.youtube.com/watch?v=_-02ze7tf08). *Partial derivatives → the gradient vector and why it points uphill.*
3. **Formalize** — read [MML Ch. 5 (Vector Calculus)](https://mml-book.github.io/book/mml-book.pdf), §5.1–5.3. *Partial derivatives, gradients, and gradient rules.*
4. **Reference & practice** — use [Paul's Online Notes: The Gradient Vector](https://tutorial.math.lamar.edu/Classes/CalcIII/GradientVector.aspx) and Khan exercises. *Solidify computation and the steepest-ascent property.*
5. **Connect to ML** — read [AI-ML-intuition 2.01 Partial Derivatives & the Gradient](../../../AI-ML-intuition/Module_2_Optimization/2.01_Partial_Derivatives_and_the_Gradient.md). *Where the gradient becomes the training signal.*

## 🎓 Courses (free)
- [MIT 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/) — **MIT OCW** — derivatives from first principles.
- [Khan Academy — Multivariable derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) — **Khan Academy** — partial derivatives, gradient, directional derivatives, with exercises.

## 🎥 Videos
- [The essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) — **3Blue1Brown** — the whole subject's intuition in one video.
- [The paradox of the derivative | Ch. 2](https://www.youtube.com/watch?v=9vKqVkMQHKk) — **3Blue1Brown** — what a derivative really measures.
- [Gradient and graphs](https://www.youtube.com/watch?v=_-02ze7tf08) — **Khan Academy** — the gradient as steepest-ascent direction.
- [Gradients and Partial Derivatives](https://www.youtube.com/watch?v=GkB4vW16QHI) — **Eugene Khutoryansky** — vivid 3D visualization of partials and the gradient.

## 📄 Key Papers
- [MML book — Ch. 5 "Vector Calculus" (§5.1–5.3)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — derivatives, partial derivatives, and gradients for ML.
- [CS231n — Optimization & gradients notes](https://cs231n.github.io/optimization-1/) — **Stanford** — numerical vs analytic gradients and gradient checking in practice.

## 📰 Articles / Blogs (free, no paywall)
- [Paul's Online Notes — The Gradient Vector](https://tutorial.math.lamar.edu/Classes/CalcIII/GradientVector.aspx) — **Paul Dawkins** — the gradient and directional derivatives, with worked examples.
- [Paul's Online Notes — Definition of the Derivative](https://tutorial.math.lamar.edu/Classes/CalcI/DefnOfDerivative.aspx) — **Paul Dawkins** — the limit definition, carefully done.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 5 (Vector Calculus)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — gradients, gradient rules, and the build-up to backprop.
- [Calculus (OpenStax) — **Vol. 3, Ch. 4 (Differentiation of Functions of Several Variables)**](https://openstax.org/details/books/calculus-volume-3) — **OpenStax** — free, thorough multivariable derivatives and the gradient.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.01 Partial Derivatives & the Gradient](../../../AI-ML-intuition/Module_2_Optimization/2.01_Partial_Derivatives_and_the_Gradient.md)
- Curriculum context: [Maths for AI-ML — Phase 2 (Calculus, row 2.1)](../Maths%20for%20AI-ML/README.md)
- Prereq: [01 Vectors & Vector Spaces](01-Vectors-and-Vector-Spaces.md) · Next: [09 The Chain Rule](09-The-Chain-Rule.md) · [10 Jacobian & Hessian](10-Jacobian-and-Hessian.md) · [13 Gradient Descent — theory](13-Gradient-Descent-Theory.md)
</content>

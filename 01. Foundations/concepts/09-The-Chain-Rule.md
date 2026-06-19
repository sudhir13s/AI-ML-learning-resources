---
id: "01-foundations/chain-rule"
topic: "The Chain Rule (& Backpropagation)"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/derivatives-and-gradients"]
interview_frequency: very-high
updated: 2026-06-20
---

# The Chain Rule (& Backpropagation)
> The chain rule differentiates a composition: the rate of change of `f(g(x))` is the product of
> the local rates. Stack it across the layers of a network and you get backpropagation — the
> reverse-mode accumulation of gradients that makes deep learning trainable at all.

**Why it matters:** backprop *is* the multivariable chain rule applied over a computational graph.
Interviewers ask you to differentiate composed functions by hand, explain forward vs reverse mode
(why reverse is cheap for scalar losses), and derive the gradient through a small network — all
chain rule.

**⭐ Start here — suggested path:**

1. **Single-variable chain rule** — watch [3B1B: Visualizing the chain rule and product rule](https://www.youtube.com/watch?v=YG15m2VwSjA). *See where the product of derivatives comes from.*
2. **From chain rule to backprop** — watch [3B1B: Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) then [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8). *The exact moment the chain rule becomes a learning algorithm.*
3. **Build it yourself** — watch [Karpathy: micrograd — the spelled-out intro to backprop](https://www.youtube.com/watch?v=VMj-3S1tku0). *Implementing reverse-mode autodiff cements it forever.*
4. **Matrix/vector form** — read [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/). *The vector chain rule and Jacobian-product view used in real frameworks.*
5. **Connect to ML** — read [AI-ML-intuition 2.02 Backpropagation (The Chain Rule)](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md). *The platform's deep dive on credit assignment.*

## 🎓 Courses (free)
- [Stanford CS231n — Backpropagation notes](https://cs231n.github.io/optimization-2/) — **Stanford** — the chain rule on computational graphs, with worked gradient flows.
- [Khan Academy — Multivariable chain rule](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/multivariable-chain-rule/v/multivariable-chain-rule) — **Khan Academy** — the multivariable version with exercises.

## 🎥 Videos
- [Visualizing the chain rule and product rule | Ch. 4](https://www.youtube.com/watch?v=YG15m2VwSjA) — **3Blue1Brown** — the geometric why behind the chain rule.
- [Backpropagation, intuitively | Deep Learning Ch. 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — **3Blue1Brown** — chain rule as gradient flow through a network.
- [Backpropagation calculus | Deep Learning Ch. 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — **3Blue1Brown** — the explicit chain-rule derivation.
- [The spelled-out intro to backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — **Andrej Karpathy** — build reverse-mode autodiff from scratch.

## 📄 Key Papers
- [MML book — Ch. 5.2–5.6 (Vector Calculus / chain rule, backprop)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the chain rule, gradients of compositions, and automatic differentiation.
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/) — **Parr & Howard** — the vector/matrix chain rule needed for real backprop.

## 📰 Articles / Blogs (free, no paywall)
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) — **Christopher Olah** — the clearest free essay on chain rule over graphs.
- [Paul's Online Notes — Chain Rule](https://tutorial.math.lamar.edu/Classes/CalcI/ChainRule.aspx) — **Paul Dawkins** — mechanics and worked examples.

## 📚 Books (free, with chapters)
- [Mathematics for Machine Learning — **Ch. 5 (Vector Calculus)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — chain rule and automatic differentiation.
- [Dive into Deep Learning — **§2.5 (Automatic Differentiation), §5.3 (Backprop)**](https://d2l.ai/chapter_preliminaries/autograd.html) — **Zhang et al.** — chain rule, autograd, and backprop with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation (The Chain Rule)](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) · [2.04 Computational Graphs / Autograd](../../../AI-ML-intuition/Module_2_Optimization/2.04_Computational_Graphs_Autograd.md)
- Curriculum context: [Maths for AI-ML — Phase 2 (Calculus, row 2.2)](../Maths%20for%20AI-ML/README.md)
- Prereq: [08 Derivatives & Gradients](08-Derivatives-and-Gradients.md) · Next: [10 Jacobian & Hessian](10-Jacobian-and-Hessian.md)
</content>

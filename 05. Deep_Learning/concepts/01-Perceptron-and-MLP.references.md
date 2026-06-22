---
id: "05-deep-learning/perceptron-mlp/references"
topic: "Perceptron & MLP — References"
parent: "05-deep-learning/perceptron-mlp"
type: references
updated: 2026-06-22
---

# Perceptron and MLP — references and further reading

> Companion link library for **[Perceptron and MLP](01-Perceptron-and-MLP.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (the original papers) or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (**3Blue1Brown**). *See neurons, weights, and layers as a function before any math.*
2. **See the limit, then the fix** — play with [TensorFlow Playground](https://playground.tensorflow.org/). *Watch a linear model fail on XOR, add one hidden layer + a nonlinearity, and watch it separate the classes — block 2 vs block 3 of the page, animated.*
3. **Read it from first principles** — [Neural Networks and Deep Learning, Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) (**Nielsen**). *Perceptrons → sigmoid neurons → a working MLP, derived gently.*
4. **See universal approximation visually** — [A Visual Proof that Neural Nets Can Compute Any Function](http://neuralnetworksanddeeplearning.com/chap4.html) (**Nielsen**). *The bump-construction proof, interactive.*
5. **Go to the source** — read [the perceptron and its learning rule](https://en.wikipedia.org/wiki/Perceptron) (**Rosenblatt 1958**, summarized with the original citation) and the [XOR limitation in *Perceptrons*](https://en.wikipedia.org/wiki/Perceptrons_(book)) (**Minsky & Papert 1969**). *The work that started — and nearly ended — the field.*

**Videos**:
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — **3Blue1Brown** — the definitive visual intro to neurons, weights, biases, and layers as a function.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — **StatQuest (Josh Starmer)** — from-scratch intuition for how a small network bends straight lines into a fit.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — how an MLP's weights get tuned to minimize loss (the bridge to backprop).
- [Neural Networks from Scratch — the perceptron](https://www.youtube.com/watch?v=Wo5dMEP_BbI) — **sentdex** — codes a single neuron with no frameworks, making the weighted-sum-then-activate math tangible.

**Courses (free)**:
- [Stanford CS231n — Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/) — **Stanford (Karpathy / Li / Johnson)** — the canonical lecture notes on the neuron model, layers, and architecture.
- [MIT 6.S191 — Intro to Deep Learning](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — opens with the perceptron and builds to deep feedforward nets in one lecture.

**Articles / blogs (free, no paywall)**:
- [A Visual Proof that Neural Nets Can Compute Any Function](http://neuralnetworksanddeeplearning.com/chap4.html) — **Michael Nielsen** — the interactive, intuitive proof of universal approximation (the bump construction on the page).
- [CS231n — Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) — **Stanford CS231n** — neuron model, activation functions, and how layers compose.
- [The Perceptron — convergence theorem with the margin/radius proof](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html) — **Cornell CS4780 (Kilian Weinberger)** — the mistake bound $\le(R/\gamma)^2$ derived step by step.
- [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) — **Wikipedia** — precise statements of the Cybenko and Hornik versions, plus the depth-separation results, with full citations.

**Papers (the foundational five — read at least the abstracts)**:
- [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) — **McCulloch & Pitts (1943)** — the first formal threshold neuron; networks of them are universal for logic.
- [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://en.wikipedia.org/wiki/Perceptron) — **Rosenblatt (1958)** — the perceptron and its learning rule (summarized with the original citation); the start of trainable neural nets.
- [Perceptrons (the XOR limitation)](https://en.wikipedia.org/wiki/Perceptrons_(book)) — **Minsky & Papert (1969)** — proved single-layer units can't compute non-linearly-separable functions; triggered the first AI winter.
- [Approximation by Superpositions of a Sigmoidal Function](https://link.springer.com/article/10.1007/BF02551274) — **Cybenko (1989)** — the universal approximation theorem for sigmoidal one-hidden-layer nets.
- [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) — **Rumelhart, Hinton & Williams (1986)** — backpropagation: the algorithm that finally trained the hidden layers Minsky & Papert said no one could.

**Papers (deeper — approximation and depth)**:
- [Multilayer Feedforward Networks are Universal Approximators](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/notes/Sonia_Hornik.pdf) — **Hornik, Stinchcombe & White (1989)** — the general UAT, beyond Cybenko's sigmoid-specific proof.
- [Benefits of depth in neural networks](https://arxiv.org/abs/1602.04485) — **Telgarsky (2016)** — proves deep nets need exponentially fewer units than shallow ones for some functions (the depth-separation result).
- [The Power of Depth for Feedforward Neural Networks](https://arxiv.org/abs/1512.03965) — **Eldan & Shamir (2016)** — a function easy for depth-3 nets but provably hard for any depth-2 net of reasonable width.

**Books (free, with chapters)**:
- [Neural Networks and Deep Learning — Ch. 1 "Using neural nets to recognize handwritten digits"](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — perceptrons → sigmoid neurons → a working MLP, from first principles.
- [Dive into Deep Learning — Ch. 5 "Multilayer Perceptrons"](https://d2l.ai/chapter_multilayer-perceptrons/index.html) — **Zhang et al.** — MLPs, activations, and the forward pass with runnable code.
- [Deep Learning — Ch. 6 "Deep Feedforward Networks"](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of feedforward nets and why they're universal approximators.

**In this platform**:
- Concept page (full explanation): [Perceptron and MLP](01-Perceptron-and-MLP.md)
- The smooth cousin (single sigmoid neuron): [Logistic Regression](../../03.%20Supervised_Learning/concepts/02-Logistic-Regression.md)
- How an MLP actually learns: [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- The nonlinearity that makes it work: [Activation Functions](03-Activation-Functions.md)
- Why deep MLPs are hard to train: [Vanishing & Exploding Gradients](06-Vanishing-Exploding-Gradients.md)
- The MLP at scale (the transformer's FFN is an MLP): [Transformer Architecture](16-Transformer-Architecture.md)
- The *why* behind the math: [AI-ML-intuition 2.01 Partial Derivatives & the Gradient](../../../AI-ML-intuition/Module_2_Optimization/2.01_Partial_Derivatives_and_the_Gradient.md) · [4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md)
- Field overview: [Deep Learning](../README.md)

---
id: "05-deep-learning/perceptron-mlp"
topic: "Perceptron & MLP (Feedforward Networks)"
parent: "05-deep-learning"
level: beginner
prereqs: ["linear-algebra", "calculus"]
interview_frequency: high
updated: 2026-06-19
---

# Perceptron & MLP (Feedforward Networks)
> The starting point of deep learning: a **perceptron** is a single linear unit with a threshold; a
> **multilayer perceptron (MLP)** stacks layers of these units with nonlinear activations so the
> network can approximate any function. Information flows one way — input → hidden layers → output —
> which is why these are called *feedforward* networks.

**Why it matters:** the foundational interview warm-up — define a perceptron and why a single one
can't solve XOR, explain how stacking layers + a **nonlinearity** breaks that linear limit (the
universal approximation theorem), describe forward propagation as repeated `affine → activation`,
and articulate why depth and nonlinearity together are what make a network expressive.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk). *See neurons, weights, and layers as a function before any math.*
2. **See why depth needs nonlinearity** — play with ⭐ [TensorFlow Playground](https://playground.tensorflow.org/). *Watch a linear model fail on XOR, then add a hidden layer and a nonlinearity and watch it separate the classes.*
3. **Get the math** — read [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/). *Neuron model, layer math, and why nonlinear activations matter.*
4. **Read from first principles** — [Neural Networks and Deep Learning, Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) (**Nielsen**). *Perceptrons → sigmoid neurons → an MLP that recognizes digits.*
5. **Make it concrete** — code a tiny MLP from scratch following [d2l Ch. 5](https://d2l.ai/chapter_multilayer-perceptrons/index.html). *Implementing the forward pass cements the affine→activation pattern.*

## 🎓 Courses (free)
- [Stanford CS231n — Neural Networks](https://cs231n.github.io/) — **Stanford (Karpathy / Li / Johnson)** — the canonical lecture notes on the neuron model, layers, and architecture.
- [MIT 6.S191 — Intro to Deep Learning](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — opens with the perceptron and builds to deep feedforward nets in one lecture.

## 🎥 Videos
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — **3Blue1Brown** — the definitive visual intro to neurons, weights, biases, and layers.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — **StatQuest (Josh Starmer)** — from-scratch intuition for how a small network bends lines into a fit.
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — how an MLP's weights get tuned to minimize loss.
- [Create a Simple Neural Network in Python from Scratch](https://www.youtube.com/watch?v=kft1AJ9WVDk) — **Polycode** — codes a perceptron with no frameworks, making the math tangible.

## 📄 Key Papers
- [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) — **Rumelhart, Hinton & Williams (1986)** — showed multilayer perceptrons could be trained, reviving neural nets.
- [Deep Learning (Nature review)](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (2015)** — situates feedforward nets within the broader deep-learning story.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) — **Stanford CS231n** — neuron model, activation functions, and how layers compose.
- [A Visual Proof that Neural Nets Can Compute Any Function](http://neuralnetworksanddeeplearning.com/chap4.html) — **Michael Nielsen** — an interactive, intuitive proof of universal approximation.
- [CS231n — Neural Networks Part 2: Data and Loss](https://cs231n.github.io/neural-networks-2/) — **Stanford CS231n** — preprocessing, initialization, and loss for feedforward nets.

## 📚 Books (free, with chapters)
- [Neural Networks and Deep Learning — **Ch. 1 "Using neural nets to recognize handwritten digits"**](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — perceptrons → sigmoid neurons → a working MLP, from first principles.
- [Dive into Deep Learning — **Ch. 5 "Multilayer Perceptrons"**](https://d2l.ai/chapter_multilayer-perceptrons/index.html) — **Zhang et al.** — MLPs, activations, and the forward pass with runnable code.
- [Deep Learning — **Ch. 6 "Deep Feedforward Networks"**](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of feedforward nets and why they're universal approximators.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 2.01 Partial Derivatives & the Gradient](../../../AI-ML-intuition/Module_2_Optimization/2.01_Partial_Derivatives_and_the_Gradient.md) · [4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md)
- Next concept: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md) (how an MLP actually learns)
- Field overview: [Deep Learning](../README.md)
- Related domain: [01. Foundations · Maths for AI-ML](../../01.%20Foundations/concepts/README.md) (linear algebra & calculus behind the layers)

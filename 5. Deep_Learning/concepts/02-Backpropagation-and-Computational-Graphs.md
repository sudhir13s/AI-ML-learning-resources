---
id: "05-deep-learning/backpropagation"
topic: "Backpropagation & Computational Graphs"
parent: "05-deep-learning"
level: intermediate
prereqs: ["calculus", "linear-algebra", "feedforward-networks"]
interview_frequency: very-high
updated: 2026-06-19
---

# Backpropagation & Computational Graphs
> The algorithm that trains neural networks: represent the network as a graph of operations, then
> apply the chain rule **backward** through it to compute the gradient of the loss with respect to
> every parameter in one efficient pass — reusing intermediate results instead of recomputing them.

**Why it matters:** backprop is the single most-asked deep-learning fundamental — you should be able
to derive it from the chain rule, explain why a computational graph makes reverse-mode
differentiation cheap (one backward pass for *all* parameters), distinguish the **forward** pass
(cache activations) from the **backward** pass (accumulate gradients), and reason about what
autograd is doing under the hood when you call `.backward()`.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [3Blue1Brown: What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U). *See the gradient as "nudges" to weights before any symbols appear.*
2. **See why it works** — read ⭐ [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) (**Chris Olah**). *The clearest explanation of why reverse-mode gets all derivatives in one pass.*
3. **Get the math** — [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) + [CS231n notes: Backprop](https://cs231n.github.io/optimization-2/). *The chain rule on a graph, with the local-gradient bookkeeping you'll be asked to do.*
4. **Read the source** — [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (**Rumelhart, Hinton & Williams, 1986**). *The paper that popularized backprop.*
5. **Make it concrete** — build it from scratch with [Karpathy's micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0), then map it to [PyTorch autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). *Implementing a tiny autograd engine makes the graph + chain rule click permanently.*

## 🎓 Courses (free)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — opens by building a full autograd engine (micrograd) by hand; the best way to *internalize* backprop.
- [Stanford CS231n — Backpropagation, Neural Networks](https://cs231n.github.io/) — **Stanford (Karpathy / Li / Johnson)** — the definitive lecture notes with worked gradient examples and circuit intuition.

## 🎥 Videos
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — **3Blue1Brown** — visual intuition for how gradients adjust weights, before any calculus.
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — **3Blue1Brown** — the chain-rule derivation, layer by layer.
- [The spelled-out intro to backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — **Andrej Karpathy** — builds a working autograd engine from scratch (≈2.5 hrs, deep).
- [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — **3Blue1Brown (Deep Learning Ch. 3)** — pairs the "nudge" intuition with the network picture.

## 📄 Key Papers
- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — **Rumelhart, Hinton & Williams (1986)** — the paper that brought backprop to neural networks.
- [Deep Learning (Nature review)](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (2015)** — situates backprop within the broader deep-learning story.

## 📰 Articles / Blogs (free, no paywall)
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) — **Chris Olah** — the gold-standard explanation of forward- vs reverse-mode differentiation on a graph.
- [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — **Stanford CS231n** — staged worked examples (the "circuit" view of local gradients).
- [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — **PyTorch** — how a real framework records the graph and runs the backward pass.

## 📚 Books (free, with chapters)
- [Neural Networks and Deep Learning — **Ch. 2 "How the backpropagation algorithm works"**](http://neuralnetworksanddeeplearning.com/chap2.html) — **Michael Nielsen** — the clearest from-first-principles derivation of the four backprop equations.
- [Dive into Deep Learning — **§5.3 "Forward Propagation, Backward Propagation, and Computational Graphs"**](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) — **Zhang et al.** — backprop on an explicit computational graph, with code.
- [Deep Learning — **Ch. 6.5 "Back-Propagation and Other Differentiation Algorithms"**](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment (general computational graphs, reverse-mode autodiff).

## 🔗 In this platform
- Field overview: [Deep Learning](../README.md)
- Next concepts: [13 CNNs & Convolution](13-CNNs-and-Convolution.md) · [14 RNN / LSTM / GRU](14-RNN-LSTM-GRU.md)
- Related domain: [11. Tools & Frameworks](../../11.%20Tools_and_Frameworks/README.md) (PyTorch / autograd in practice)

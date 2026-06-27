---
id: "05-deep-learning/backpropagation/references"
topic: "Backpropagation & Computational Graphs — References"
parent: "05-deep-learning/backpropagation"
type: references
updated: 2026-06-22
---

# Backpropagation & Computational Graphs — references and further reading

> Companion link library for **[Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) (**3Blue1Brown**). *Gradients as "nudges" to weights, before any symbols.*
2. **See why it works** — read [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) (**Chris Olah**). *The clearest explanation of why reverse mode gets all derivatives in one pass.*
3. **Get the math** — watch [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) (**3Blue1Brown**) + read [CS231n: Backprop](https://cs231n.github.io/optimization-2/). *The chain rule on a graph, with the local-gradient bookkeeping.*
4. **Read the source** — [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (**Rumelhart, Hinton & Williams, 1986**). *The paper that popularized backprop.*
5. **Make it concrete** — build it from scratch with [Karpathy's micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0), then map it to [PyTorch autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). *Implementing a tiny autograd engine makes the graph + chain rule click permanently.*

**Videos**:
- [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — **3Blue1Brown** — visual intuition for how gradients adjust weights, before any calculus.
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — **3Blue1Brown** — the chain-rule derivation, layer by layer.
- [The spelled-out intro to backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — **Andrej Karpathy** — builds a working autograd engine from scratch, node by node.
- [Neural Networks Pt. 2: Backpropagation Main Ideas](https://www.youtube.com/watch?v=IN2XmBhILt4) — **StatQuest (Josh Starmer)** — gentle, gated walk through the backprop computation.

**Interactive & visual**:
- [TensorFlow Playground](https://playground.tensorflow.org/) — **Google** — build a small net in the browser and watch backprop train it (weights + boundary updating live).

**Courses (free)**:
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — opens by building a full autograd engine (micrograd) by hand; the best way to *internalize* backprop.
- [Stanford CS231n — Backpropagation, Neural Networks](https://cs231n.github.io/) — **Stanford (Karpathy / Li / Johnson)** — the definitive lecture notes with worked gradient examples and circuit intuition.

**Articles / blogs (free, no paywall)**:
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) — **Chris Olah** — the gold-standard explanation of forward- vs reverse-mode differentiation on a graph.
- [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — **Stanford CS231n** — staged worked examples (the "circuit" view of local gradients).
- [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — **PyTorch** — how a real framework records the graph and runs the backward pass.
- [Autodiff Cookbook (JAX)](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) — **JAX team (Google)** — VJPs (`vjp`), JVPs (`jvp`), and forward- vs reverse-mode composition, with code; the cleanest modern treatment of the VJP view.

**Key papers**:
- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — **Rumelhart, Hinton & Williams (1986)** — the paper that brought backprop to neural networks (hidden layers learn representations; backprop is how).
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) — **Baydin et al. (2015)** — forward vs reverse mode and how autograd engines work, rigorously (the VJP/JVP formalism).
- [Applications of advances in nonlinear sensitivity analysis (backprop's origin)](https://link.springer.com/chapter/10.1007/BFb0006203) — **Paul Werbos (1982; thesis 1974)** — the early derivation of reverse-mode gradients for ordered systems, predating the 1986 NN paper.
- [Deep Learning (Nature review)](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (2015)** — situates backprop within the broader deep-learning story.
- [Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432) — **Bengio, Léonard & Courville (2013)** — the straight-through estimator for non-differentiable ops.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — **Chen et al. (2016)** — gradient checkpointing: trade compute for activation memory.

**Books (free chapters)**:
- [Neural Networks and Deep Learning — Ch. 2 "How the backpropagation algorithm works"](http://neuralnetworksanddeeplearning.com/chap2.html) — **Michael Nielsen** — the clearest from-first-principles derivation of the four backprop equations.
- [Dive into Deep Learning — §5.3 "Forward Propagation, Backward Propagation, and Computational Graphs"](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) — **Zhang et al.** — backprop on an explicit computational graph, with code.
- [Deep Learning — Ch. 6.5 "Back-Propagation and Other Differentiation Algorithms"](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment (general computational graphs, reverse-mode autodiff).

**In this platform**:
- Concept page (full explanation): [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md)
- Concept depth (the *why*): [AI-ML-intuition 2.02 Backpropagation / Chain Rule](../../../AI-ML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) · [2.04 Computational Graphs & Autograd](../../../AI-ML-intuition/Module_2_Optimization/2.04_Computational_Graphs_Autograd.md)
- The network being differentiated: [01 Perceptron & MLP](../01-Perceptron-and-MLP/01-Perceptron-and-MLP.md)
- The activations whose derivatives gate the flow: [03 Activation Functions](../03-Activation-Functions/03-Activation-Functions.md)
- The consequence of the backward product: [06 Vanishing & Exploding Gradients](../06-Vanishing-Exploding-Gradients/06-Vanishing-Exploding-Gradients.md)
- What uses the gradient: [07 Optimizers](../07-Optimizers/07-Optimizers.md) (turns the gradient into a weight update)
- Where it's applied: [13 CNNs & Convolution](../13-CNNs-and-Convolution/13-CNNs-and-Convolution.md) · [14 RNN / LSTM / GRU](../14-RNN-LSTM-GRU/14-RNN-LSTM-GRU.md) (backprop through time)
- Field overview: [Deep Learning](../README.md)

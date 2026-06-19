---
id: "05-deep-learning/residual-skip-connections"
topic: "Residual / Skip Connections"
parent: "05-deep-learning"
level: intermediate
prereqs: ["backpropagation", "cnns", "vanishing-exploding-gradients"]
interview_frequency: high
updated: 2026-06-19
---

# Residual / Skip Connections
> Instead of asking a block to learn a full mapping `H(x)`, a residual connection has it learn the
> **residual** `F(x) = H(x) − x` and adds the input back: `output = F(x) + x`. That identity shortcut
> gives gradients a direct path backward, making it possible to train networks hundreds of layers
> deep. Residuals (ResNet) and their relatives are now in nearly every deep architecture, including
> the transformer.

**Why it matters:** a recurring "how do we train *very* deep nets?" question — explain the
degradation problem (deeper plain nets train *worse*, not just overfit), why the identity path lets
gradients flow without vanishing, why learning a residual near zero is easy, and where skip
connections appear beyond ResNet (DenseNet, U-Net, transformer blocks).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) (**Andrew Ng**). *Why adding the input back makes deeper strictly safe.*
2. **See the gradient highway** — read ⭐ [d2l: Residual Networks (ResNet)](https://d2l.ai/chapter_convolutional-modern/resnet.html). *How the identity branch preserves gradient flow, with code.*
3. **Get the math** — read [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (**He et al., 2016**). *Why a clean identity path (pre-activation) is optimal.*
4. **Read the source** — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (**He et al., 2015**). *The paper; the degradation problem and the fix.*
5. **Make it concrete** — implement a residual block following [d2l ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html). *Coding `F(x)+x` makes the shortcut and its gradient path obvious.*

## 🎓 Courses (free)
- [Stanford CS231n — CNN Architectures (ResNet)](https://cs231n.github.io/convolutional-networks/) — **Stanford (Karpathy / Li / Johnson)** — residual blocks within the architecture progression.
- [Dive into Deep Learning — Residual Networks (ResNet)](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — the residual block built and trained from scratch.

## 🎥 Videos
- [Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) — **DeepLearningAI (Andrew Ng)** — the cleanest argument for why identity shortcuts help.
- [Why Residual Connections (ResNet) Work](https://www.youtube.com/watch?v=Gey9CG6R6w8) — **DataMListic** — the degradation problem and the gradient-highway intuition.
- [Residual Networks and Skip Connections (DL 15)](https://www.youtube.com/watch?v=Q1JCrG1bJ-A) — **Professor Bryce** — lecture-style derivation of the residual block.
- [Deep Residual Learning for Image Recognition (Paper Explained)](https://www.youtube.com/watch?v=GWt6Fu05voI) — **Yannic Kilcher** — a careful read-through of the ResNet paper.

## 📄 Key Papers
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the residual block; trains 100+ layer nets and won ImageNet 2015.
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) — **He et al. (2016)** — pre-activation residuals and why a clean identity path is best.
- [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993) — **Huang et al. (2016)** — connect every layer to every later layer; a skip-connection extreme.

## 📰 Articles / Blogs (free, no paywall)
- [d2l — Residual Networks (ResNet) and ResNeXt](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — the residual block, gradient flow, and code.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — where residual + LayerNorm sit inside each transformer block.
- [CS231n — Convolutional Networks (architectures)](https://cs231n.github.io/convolutional-networks/) — **Stanford CS231n** — ResNet in the CNN-architecture lineage.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§8.6 "Residual Networks (ResNet) and ResNeXt"**](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — the residual block built, with the gradient-flow argument.
- [Deep Learning — **Ch. 8 (optimization)** — gradient flow & shortcut connections](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — why skip connections ease optimization.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.06 Residual / Skip Connections](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.06_Residual_Skip_Connections.md)
- Prerequisites: [06 Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) · [13 CNNs & Convolution](13-CNNs-and-Convolution.md)
- Related concept: [16 Transformer Architecture](16-Transformer-Architecture.md) (residuals around attention + FFN)
- Field overview: [Deep Learning](../README.md)

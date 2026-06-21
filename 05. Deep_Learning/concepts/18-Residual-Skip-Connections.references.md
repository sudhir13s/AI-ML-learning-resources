---
id: "05-deep-learning/residual-skip-connections/references"
topic: "Residual / Skip Connections — References"
parent: "05-deep-learning/residual-skip-connections"
type: references
updated: 2026-06-21
---

# Residual / Skip Connections — references and further reading

> Companion link library for **[Residual / Skip Connections](18-Residual-Skip-Connections.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) (**Andrew Ng**). *Why adding the input back makes deeper strictly safe.*
2. **See the gradient highway** — read & run [d2l: Residual Networks (ResNet)](https://d2l.ai/chapter_convolutional-modern/resnet.html). *How the identity branch preserves gradient flow, with code.*
3. **Get the math** — read [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (**He et al. 2016**). *Why a clean identity path (pre-activation) is optimal.*
4. **Read the source** — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (**He et al. 2015**). *The paper; the degradation problem and the fix.*
5. **Make it concrete** — implement a residual block following [d2l ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html).

**Videos**:
- [Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) — **DeepLearningAI (Andrew Ng)** — the cleanest argument for why identity shortcuts help.
- [Why Residual Connections (ResNet) Work](https://www.youtube.com/watch?v=Gey9CG6R6w8) — **DataMListic** — the degradation problem and the gradient-highway intuition.
- [Residual Networks and Skip Connections (DL 15)](https://www.youtube.com/watch?v=Q1JCrG1bJ-A) — **Professor Bryce** — lecture-style derivation of the residual block.
- [Deep Residual Learning for Image Recognition (Paper Explained)](https://www.youtube.com/watch?v=GWt6Fu05voI) — **Yannic Kilcher** — a careful read-through of the ResNet paper.

**Interactive & visual**:
- [Dive into Deep Learning — Residual Networks (runnable)](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — build and train a residual block in-browser (Colab/notebook); see the gradient-flow argument with code.

**Courses (free)**:
- [Stanford CS231n — CNN Architectures (ResNet)](https://cs231n.github.io/convolutional-networks/) — **Stanford (Karpathy / Li / Johnson)** — residual blocks within the architecture progression.
- [Dive into Deep Learning — Residual Networks (ResNet)](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — the residual block built and trained from scratch.

**Articles / blogs (free, no paywall)**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — where residual + LayerNorm sit inside each transformer block.
- [CS231n — Convolutional Networks (architectures)](https://cs231n.github.io/convolutional-networks/) — **Stanford CS231n** — ResNet in the CNN-architecture lineage.

**Key papers**:
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the residual block; trains 100+ layer nets and won ImageNet 2015.
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) — **He et al. (2016)** — pre-activation residuals and why a clean identity path is best.
- [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993) — **Huang et al. (2016)** — connect every layer to every later layer; a skip-connection extreme.

**Books (free chapters)**:
- [Dive into Deep Learning — §8.6 "Residual Networks (ResNet) and ResNeXt"](https://d2l.ai/chapter_convolutional-modern/resnet.html) — **Zhang et al.** — the residual block built, with the gradient-flow argument.
- [Deep Learning — Ch. 8 (optimization) — gradient flow & shortcut connections](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — why skip connections ease optimization.

**In this platform**:
- Concept page (full explanation): [Residual / Skip Connections](18-Residual-Skip-Connections.md)
- Concept depth (the *why*): [AI-ML-intuition 4.06 Residual / Skip Connections](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.06_Residual_Skip_Connections.md)
- Prerequisites: [Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md) · [CNNs & Convolution](13-CNNs-and-Convolution.md)
- Related: [Transformer Architecture](16-Transformer-Architecture.md) (residuals around attention + FFN) · [Normalization](11-Normalization.md) (pre-norm pairs with residuals)
- Field overview: [Deep Learning](../README.md)

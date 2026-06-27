---
id: "05-deep-learning/normalization/references"
topic: "Normalization — References"
parent: "05-deep-learning/normalization"
type: references
updated: 2026-06-22
---

# Normalization — references and further reading

> Companion link library for **[Normalization](11-Normalization.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Normalizing Activations in a Network](https://www.youtube.com/watch?v=tNIpEZLv_eg) (**Andrew Ng**). *Why centering/scaling activations stabilizes training.*
2. **See the variants** — watch [All About Normalizations — Batch, Layer, Instance, Group](https://www.youtube.com/watch?v=1JmZ5idFcVI) (**ChiDotPhi**). *Exactly which axes each norm reduces over.*
3. **Get the math** — read [Understanding the backward pass through Batch Norm](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) (**Kratzert**). *The full forward + backward derivation.*
4. **Read the sources** — [BatchNorm](https://arxiv.org/abs/1502.03167) → [LayerNorm](https://arxiv.org/abs/1607.06450) → [GroupNorm](https://arxiv.org/abs/1803.08494) → [RMSNorm](https://arxiv.org/abs/1910.07467). *In order; note what each changes.*
5. **Make it concrete** — implement BatchNorm forward/backward following [d2l §8.5](https://d2l.ai/chapter_convolutional-modern/batch-norm.html). *Coding the running stats clarifies the train/eval split.*

**Videos**:
- [Normalizing Activations in a Network (C2W3L04)](https://www.youtube.com/watch?v=tNIpEZLv_eg) — **DeepLearningAI (Andrew Ng)** — why normalizing activations smooths and speeds training.
- [Fitting Batch Norm Into Neural Networks (C2W3L05)](https://www.youtube.com/watch?v=em6dfRxYkYU) — **DeepLearningAI (Andrew Ng)** — where BN sits in a layer and how γ/β work.
- [All About Normalizations — Batch, Layer, Instance, Group](https://www.youtube.com/watch?v=1JmZ5idFcVI) — **ChiDotPhi** — the clearest comparison of which axes each norm reduces.
- [Batch Normalization — EXPLAINED!](https://www.youtube.com/watch?v=DtEq44FTPM4) — **CodeEmporium** — intuition for internal covariate shift and the BN computation.

**Interactive & visual**:
- [How Does Batch Normalization Help Optimization? (paper + figures)](https://arxiv.org/abs/1805.11604) — **Santurkar et al. (2018)** — the loss-landscape visualizations are the clearest picture of *why* normalization helps (smoother landscape, not covariate shift).
- [Understanding the backward pass through Batch Norm](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) — **Frederik Kratzert** — the computation graph drawn out, node by node, for the BN backward pass.

**Courses (free)**:
- [Stanford CS231n — Training Neural Networks (Batch Normalization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — BN placement, train/test behavior, and benefits.
- [Dive into Deep Learning — Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) — **Zhang et al.** — the method and its backward pass with runnable code.

**Articles / blogs (free, no paywall)**:
- [Batch vs Layer Normalization](https://www.pinecone.io/learn/batch-layer-normalization/) — **Pinecone** — clean side-by-side of the two main norms and when to use each.
- [Layer Normalization explained](https://leimao.github.io/blog/Layer-Normalization/) — **Lei Mao** — LayerNorm math with the gradient, the transformer default.

**Key papers**:
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — **Ioffe & Szegedy (2015)** — the original; normalize per mini-batch to accelerate training.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — **Ba, Kiros & Hinton (2016)** — per-example normalization; the transformer/RNN choice.
- [Group Normalization](https://arxiv.org/abs/1803.08494) — **Wu & He (2018)** — batch-size-independent normalization for small batches.
- [Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467) — **Zhang & Sennrich (2019)** — drop the mean-centering; the modern-LLM default.
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) — **Ulyanov, Vedaldi & Lempitsky (2016)** — per-image, per-channel normalization; the style-transfer norm (and AdaIN's basis).
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) — **Santurkar et al. (2018)** — BN works by smoothing the loss landscape, not "covariate shift."
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — **Xiong et al. (2020)** — the pre-norm vs post-norm analysis; why pre-LN trains stably.
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) — **Wang et al. (2022)** — DeepNorm, a post-norm variant with residual up-weighting for extreme depth.
- [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) — **Zhang, Dauphin & Ma (2019)** — train very deep ResNets with scaled init and no norm layers.
- [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887) — **Bachlechner et al. (2020)** — a single zero-initialized residual gate per branch; deep training without normalization.
- [High-Performance Large-Scale Image Recognition Without Normalization (NFNets)](https://arxiv.org/abs/2102.06171) — **Brock et al. (2021)** — match BatchNorm on ImageNet via weight standardization + adaptive gradient clipping.

**Books (free chapters)**:
- [Dive into Deep Learning — §8.5 "Batch Normalization"](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) — **Zhang et al.** — BN forward/backward and its effect, with code.
- [Deep Learning — §8.7.1 "Batch Normalization"](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the optimization view of normalization.

**In this platform**:
- Concept page (full explanation): [Normalization](11-Normalization.md)
- Concept depth (the *why*): [AI-ML-intuition 4.01 Batch Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.01_Batch_Normalization.md) · [4.02 Layer Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.02_Layer_Normalization.md) · [4.03 Group Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.03_Group_Normalization.md) · [4.05 RMSNorm](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.05_RMSNorm.md)
- Prerequisite: [Vanishing / Exploding Gradients](../06-Vanishing-Exploding-Gradients/06-Vanishing-Exploding-Gradients.md)
- Related: [Dropout](../10-Dropout/10-Dropout.md) (often chosen alongside or instead of normalization) · [Transformer Architecture](../16-Transformer-Architecture/16-Transformer-Architecture.md) (where LayerNorm/RMSNorm + pre-norm live)
- Field overview: [Deep Learning](../README.md)

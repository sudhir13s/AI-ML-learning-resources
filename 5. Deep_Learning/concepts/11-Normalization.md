---
id: "05-deep-learning/normalization"
topic: "Normalization (Batch · Layer · Group)"
parent: "05-deep-learning"
level: intermediate
prereqs: ["feedforward-networks", "backpropagation", "vanishing-exploding-gradients"]
interview_frequency: very-high
updated: 2026-06-19
---

# Normalization (Batch · Layer · Group)
> Layers that re-center and re-scale activations to keep them in a stable range, smoothing the loss
> landscape and letting you train deeper nets with higher learning rates. **BatchNorm** normalizes
> over the batch dimension (great for CNNs); **LayerNorm** normalizes per-example over features (the
> transformer/RNN default); **GroupNorm** splits channels into groups (robust at small batch sizes).

**Why it matters:** a very-high-frequency architecture question — explain what statistics each norm
computes and over which axes, why BatchNorm depends on batch size (and its train-vs-eval running-
stats behavior) while LayerNorm doesn't, why transformers use LayerNorm, the role of the learnable
**γ/β**, and the modern pre-norm vs post-norm placement debate.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Normalizing Activations in a Network (C2W3L04)](https://www.youtube.com/watch?v=tNIpEZLv_eg) (**Andrew Ng**). *Why centering/scaling activations stabilizes training.*
2. **See the variants** — watch ⭐ [All About Normalizations — Batch, Layer, Instance, Group](https://www.youtube.com/watch?v=1JmZ5idFcVI) (**ChiDotPhi**). *Exactly which axes each norm reduces over.*
3. **Get the math** — read [Understanding the backward pass through Batch Norm](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) (**Kratzert**). *The full forward + backward derivation.*
4. **Read the sources** — [BatchNorm](https://arxiv.org/abs/1502.03167) → [LayerNorm](https://arxiv.org/abs/1607.06450) → [GroupNorm](https://arxiv.org/abs/1803.08494). *The three papers, in order.*
5. **Make it concrete** — implement BatchNorm forward/backward following [d2l §8.5](https://d2l.ai/chapter_convolutional-modern/batch-norm.html). *Coding the running stats clarifies the train/eval split.*

## 🎓 Courses (free)
- [Stanford CS231n — Training Neural Networks (Batch Normalization)](https://cs231n.github.io/neural-networks-2/) — **Stanford (Karpathy / Li / Johnson)** — BatchNorm placement, train/test behavior, and benefits.
- [Dive into Deep Learning — Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) — **Zhang et al.** — the method and its backward pass with runnable code.

## 🎥 Videos
- [Normalizing Activations in a Network (C2W3L04)](https://www.youtube.com/watch?v=tNIpEZLv_eg) — **DeepLearningAI (Andrew Ng)** — why normalizing activations smooths and speeds training.
- [Fitting Batch Norm Into Neural Networks (C2W3L05)](https://www.youtube.com/watch?v=em6dfRxYkYU) — **DeepLearningAI (Andrew Ng)** — where BN sits in a layer and how γ/β work.
- [All About Normalizations — Batch, Layer, Instance, Group](https://www.youtube.com/watch?v=1JmZ5idFcVI) — **ChiDotPhi** — the clearest comparison of which axes each norm reduces.
- [Batch Normalization — EXPLAINED!](https://www.youtube.com/watch?v=DtEq44FTPM4) — **CodeEmporium** — intuition for internal covariate shift and the BN computation.

## 📄 Key Papers
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — **Ioffe & Szegedy (2015)** — the original; normalize per mini-batch to accelerate training.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — **Ba, Kiros & Hinton (2016)** — per-example normalization; the transformer/RNN choice.
- [Group Normalization](https://arxiv.org/abs/1803.08494) — **Wu & He (2018)** — batch-size-independent normalization for small batches.
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) — **Santurkar et al. (2018)** — shows BN works by smoothing the loss landscape, not "covariate shift."

## 📰 Articles / Blogs (free, no paywall)
- [Understanding the backward pass through Batch Norm](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) — **Frederik Kratzert** — the canonical step-by-step BN backprop derivation.
- [Batch vs Layer Normalization](https://www.pinecone.io/learn/batch-layer-normalization/) — **Pinecone** — clean side-by-side of the two main norms and when to use each.
- [Layer Normalization explained](https://leimao.github.io/blog/Layer-Normalization/) — **Lei Mao** — LayerNorm math with the gradient, the transformer default.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **§8.5 "Batch Normalization"**](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) — **Zhang et al.** — BN forward/backward and its effect, with code.
- [Deep Learning — **§8.7.1 "Batch Normalization"**](https://www.deeplearningbook.org/contents/optimization.html) — **Goodfellow, Bengio & Courville** — the optimization view of normalization.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.01 Batch Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.01_Batch_Normalization.md) · [4.02 Layer Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.02_Layer_Normalization.md) · [4.03 Group Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.03_Group_Normalization.md) · [4.05 RMSNorm](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.05_RMSNorm.md)
- Prerequisite: [06 Vanishing / Exploding Gradients](06-Vanishing-Exploding-Gradients.md)
- Related concept: [10 Dropout](10-Dropout.md) (often chosen alongside or instead of normalization)
- Field overview: [Deep Learning](../README.md)

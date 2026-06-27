---
id: "07-computer-vision/pooling-and-receptive-fields"
topic: "Pooling & Receptive Fields"
parent: "07-computer-vision"
level: beginner
prereqs: ["convolution", "cnns"]
interview_frequency: high
updated: 2026-06-20
---

# Pooling & Receptive Fields
> **Pooling** downsamples a feature map (max or average over a window) to shrink spatial size, add a
> little translation invariance, and cut compute. The **receptive field** is the region of the input
> pixels that a single deep-layer activation actually "sees" — it grows with depth, kernel size, stride,
> and dilation. Together they explain how a CNN goes from local edges to whole-object reasoning.

**Why it matters:** a very common CNN interview thread — max vs average pooling, why modern nets often
replace pooling with strided convs, how to *compute* the receptive field of a layer, and the difference
between the theoretical and **effective** receptive field. You'll be asked to reason about output sizes
and "how much context does this neuron have?"

**⭐ Start here — suggested path:**

1. **See pooling in action** — watch [Pooling Layers (Andrew Ng)](https://www.youtube.com/watch?v=8oOgPUO-TBY). *Max/average pooling, window, stride, and why it downsamples.*
2. **Build the mental model** — watch [Max Pooling Explained](https://www.youtube.com/watch?v=ZjM_XQa5s6s) (deeplizard). *What pooling keeps and throws away, with feature-map visuals.*
3. **Understand receptive fields** — watch ⭐ [CNN Receptive Field (Deepia)](https://www.youtube.com/watch?v=ip2HYPC_T9Q), then [Why 3×3 conv is best?](https://www.youtube.com/watch?v=lxpQZRvfnCc). *How depth and kernel size grow the field — the reason VGG stacks 3×3 convs.*
4. **Get the math** — read [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/) (pooling + spatial arrangement). *Output-size and receptive-field arithmetic you'll be quizzed on.*
5. **Go rigorous** — read the ⭐ [Distill: Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/). *Closed-form receptive-field math with interactive visuals.*

## 🎓 Courses (free)
- [Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/) — **Stanford** — the definitive notes on pooling layers, strides, and spatial dimension arithmetic.
- [Dive into Deep Learning — Pooling](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html) — **Zhang et al.** — free chapter with runnable max/average pooling code.

## 🎥 Videos
- [Pooling Layers (C4W1L09)](https://www.youtube.com/watch?v=8oOgPUO-TBY) — **DeepLearning.AI (Andrew Ng)** — crisp intro to max/average pooling and downsampling.
- [Max Pooling in CNNs explained](https://www.youtube.com/watch?v=ZjM_XQa5s6s) — **deeplizard** — visual, intuitive treatment of what pooling preserves.
- [CNN Receptive Field](https://www.youtube.com/watch?v=ip2HYPC_T9Q) — **Deepia** — clean animation of how the receptive field grows with depth.
- [Receptive Fields: Why 3×3 conv layer is the best?](https://www.youtube.com/watch?v=lxpQZRvfnCc) — **Soroush Mehraban** — connects receptive field to the VGG 3×3-stacking design choice.
- [Receptive field & its impact on CNN performance](https://www.youtube.com/watch?v=AIrXjAJ1a1k) — **Bionic Algorithm** — how receptive-field size affects what a network can model.

## 📄 Key Papers
- [Understanding the Effective Receptive Field in Deep CNNs](https://arxiv.org/abs/1701.04128) — **Luo et al. (2016)** — shows the *effective* receptive field is Gaussian and far smaller than the theoretical one.
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) — **Zeiler & Fergus (2014)** — deconv visualizations of what units across the receptive-field hierarchy respond to.

## 📰 Articles / Blogs (free, no paywall)
- [Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/) — **Distill (Araujo et al.)** — the definitive, interactive derivation of receptive-field math.
- [CS231n notes — pooling & spatial arrangement](https://cs231n.github.io/convolutional-networks/) — **Stanford** — output-size formulas and pooling mechanics, free.
- [Pooling layers (d2l)](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html) — **Zhang et al.** — explanation plus code, free.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 7.5 (Pooling)** + **Ch. 7.3 (Padding & Stride)**](https://d2l.ai/chapter_convolutional-neural-networks/index.html) — **Zhang et al.** — pooling and the stride/padding math behind receptive fields, with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.13 Convolution](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md) — receptive fields are a direct consequence of stacking convolutions.
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Next concepts: [03 Classic CNN Architectures](03-Classic-CNN-Architectures.md) · [08 Semantic Segmentation](08-Semantic-Segmentation.md)

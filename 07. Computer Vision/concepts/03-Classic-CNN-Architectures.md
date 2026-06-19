---
id: "07-computer-vision/classic-cnn-architectures"
topic: "Classic CNN Architectures (LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet)"
parent: "07-computer-vision"
level: intermediate
prereqs: ["cnns", "pooling-and-receptive-fields", "batch-normalization"]
interview_frequency: very-high
updated: 2026-06-20
---

# Classic CNN Architectures — LeNet · AlexNet · VGG · ResNet · Inception · EfficientNet
> The lineage of image-classification backbones: **LeNet** (1998) proved convnets work, **AlexNet**
> (2012) sparked the deep-learning era on ImageNet, **VGG** went deep with stacked 3×3 convs,
> **Inception** added multi-scale parallel paths, **ResNet** made 100+ layers trainable via skip
> connections, and **EfficientNet** balanced depth/width/resolution with compound scaling.

**Why it matters:** the single most-asked vision question — "walk me through the ImageNet architectures
and what each one fixed." You should be able to explain why AlexNet's ReLU+dropout mattered, why VGG
stacks 3×3s, what Inception modules do, why residual connections solve degradation, and what compound
scaling buys EfficientNet. These backbones are also the default feature extractors for transfer learning.

**⭐ Start here — suggested path:**

1. **Why CNNs for vision** — watch [Computer Vision (Andrew Ng)](https://www.youtube.com/watch?v=ArPaAX_PhIs). *Sets up why we need specialized architectures for images.*
2. **The full lineage** — watch ⭐ [CS231n Lec 9: CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y). *AlexNet → VGG → GoogLeNet → ResNet in one authoritative lecture.*
3. **Master the key idea** — watch [ResNet explained in 10 min](https://www.youtube.com/watch?v=o_3mboe1jYI), then read the ⭐ [ResNet paper](https://arxiv.org/abs/1512.03385). *Residual learning is the most important architectural idea — and a guaranteed interview topic.*
4. **Read the sources** — [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) → [VGG](https://arxiv.org/abs/1409.1556) → [GoogLeNet/Inception](https://arxiv.org/abs/1409.4842) → [EfficientNet](https://arxiv.org/abs/1905.11946). *The progression of ideas that each won/advanced ImageNet.*
5. **Make it concrete** — work through [d2l: Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html). *Implement AlexNet/VGG/ResNet blocks to internalize the design.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — Lecture 9 is the definitive tour of ImageNet architectures and their trade-offs.
- [Dive into Deep Learning — Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html) — **Zhang et al.** — free chapters implementing AlexNet, VGG, NiN, GoogLeNet, ResNet, DenseNet.

## 🎥 Videos
- [Computer Vision (C4W1L01)](https://www.youtube.com/watch?v=ArPaAX_PhIs) — **DeepLearning.AI (Andrew Ng)** — why images need convolutional architectures.
- [CS231n Lecture 9 — CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — **Stanford** — the authoritative AlexNet → VGG → GoogLeNet → ResNet walkthrough.
- [ResNet (actually) explained in under 10 minutes](https://www.youtube.com/watch?v=o_3mboe1jYI) — **rupert ai** — crisp intuition for residual blocks and the degradation problem.
- [Deep Residual Learning (Paper Explained)](https://www.youtube.com/watch?v=GWt6Fu05voI) — **Yannic Kilcher** — a deeper read of the ResNet paper itself.
- [The CNN Evolution: AlexNet vs VGG vs ResNet](https://www.youtube.com/watch?v=FX1sGcGa9Mw) — **Ghanasyam PB** — a quick side-by-side comparison of what each architecture changed.

## 📄 Key Papers
- [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — **LeCun et al. (1998)** — the original convnet.
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — **Krizhevsky et al. (2012)** — the deep-learning big bang.
- [Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556) — **Simonyan & Zisserman (2014)** — depth via stacked 3×3 convs.
- [Going Deeper with Convolutions (GoogLeNet/Inception)](https://arxiv.org/abs/1409.4842) — **Szegedy et al. (2014)** — multi-scale Inception modules.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — skip connections enable very deep nets; the most-cited.

## 📰 Articles / Blogs (free, no paywall)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — **Georgia Tech (Polo Club)** — interactive in-browser CNN you can poke layer by layer.
- [CS231n notes — Convolutional Networks](https://cs231n.github.io/convolutional-networks/) — **Stanford** — layer-pattern conventions and architecture case studies, free.
- [EfficientNet (TF Hub / Keras docs)](https://keras.io/api/applications/efficientnet/) — **Keras** — the compound-scaling family with ready-to-use weights.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 8 (Modern Convolutional Neural Networks)**](https://d2l.ai/chapter_convolutional-modern/index.html) — **Zhang et al.** — AlexNet, VGG, NiN, GoogLeNet, BatchNorm, ResNet, DenseNet, with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 5 (Deep Learning)**](https://szeliski.org/Book/) — **Richard Szeliski** — the architectures in context, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.06 Residual / Skip Connections](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.06_Residual_Skip_Connections.md) · [4.01 Batch Normalization](../../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.01_Batch_Normalization.md)
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md) · [Deep Learning › Residual / Skip Connections](../../05.%20Deep_Learning/concepts/18-Residual-Skip-Connections.md)
- Next concepts: [04 Image Classification](04-Image-Classification.md) · [05 Transfer Learning for Vision](05-Transfer-Learning-for-Vision.md)

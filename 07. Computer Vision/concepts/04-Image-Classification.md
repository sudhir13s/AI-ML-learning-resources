---
id: "07-computer-vision/image-classification"
topic: "Image Classification"
parent: "07-computer-vision"
level: beginner
prereqs: ["cnns", "softmax", "cross-entropy"]
interview_frequency: very-high
updated: 2026-06-20
---

# Image Classification
> The canonical vision task: assign a whole image to one of `K` categories. A CNN (or ViT) extracts
> features, a final linear layer produces logits, **softmax** turns them into class probabilities, and
> **cross-entropy** trains against the label. ImageNet (1.000 classes, 1.2M images) is the benchmark
> that defined the field, and classification is the foundation every other vision task builds on.

**Why it matters:** the entry point to applied vision and a frequent interview warm-up — the full
pipeline (data → augmentation → backbone → softmax head → cross-entropy → top-1/top-5 accuracy), why
the nearest-neighbor baseline fails, the train/val/test split discipline, and how classification
backbones get reused (transfer learning) for detection and segmentation.

**⭐ Start here — suggested path:**

1. **The task & naive baselines** — watch ⭐ [CS231n Lec 2: Image Classification](https://www.youtube.com/watch?v=OoUX-nOEjG0). *Nearest neighbor, the data-driven approach, and why we need learned features.*
2. **Build intuition for CNN classifiers** — watch [Image Classification with CNNs (StatQuest)](https://www.youtube.com/watch?v=HGwBXDKFk9I). *How a CNN turns pixels into a class prediction, from scratch.*
3. **Read the reference notes** — [CS231n: Image Classification](https://cs231n.github.io/classification/). *The data-driven pipeline, k-NN, and the train/val/test discipline.*
4. **Understand the benchmark** — read about [ImageNet](https://en.wikipedia.org/wiki/ImageNet) + the [ILSVRC paper](https://arxiv.org/abs/1409.0575). *Top-1/top-5 accuracy and the challenge that drove every architecture advance.*
5. **Make it concrete** — train a classifier with the [PyTorch CIFAR-10 tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). *End-to-end: load data → CNN → cross-entropy → evaluate.*

## 🎓 Courses (free)
- [Stanford CS231n — Image Classification](https://cs231n.github.io/classification/) — **Stanford** — the definitive intro to the classification task and data-driven approach.
- [Dive into Deep Learning — Linear & Softmax Classification](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — softmax regression and cross-entropy from the ground up, with code.

## 🎥 Videos
- [CS231n Lecture 2 — Image Classification](https://www.youtube.com/watch?v=OoUX-nOEjG0) — **Stanford** — the task, nearest-neighbor baselines, and the data-driven paradigm.
- [Image Classification with CNNs](https://www.youtube.com/watch?v=HGwBXDKFk9I) — **StatQuest (Josh Starmer)** — gentle, from-scratch CNN classifier walkthrough.
- [How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA) — **Brandon Rohrer** — pixels → features → class, explained visually.
- [Computer Vision (C4W1L01)](https://www.youtube.com/watch?v=ArPaAX_PhIs) — **DeepLearning.AI (Andrew Ng)** — framing the classification problem for images.

## 📄 Key Papers
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://arxiv.org/abs/1409.0575) — **Russakovsky et al. (2015)** — the dataset, metrics, and challenge that defined classification.
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — **Krizhevsky et al. (2012)** — the result that made CNN classification dominant.

## 📰 Articles / Blogs (free, no paywall)
- [ImageNet](https://en.wikipedia.org/wiki/ImageNet) — **Wikipedia** — the dataset, top-1/top-5 metrics, and historical context.
- [Image Classification leaderboard](https://paperswithcode.com/task/image-classification) — **Papers with Code** — current SOTA models and benchmarks, free.
- [CIFAR-10 / CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) — **Krizhevsky (Toronto)** — the standard small-scale classification benchmarks.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 4 (Linear Classification)** + **Ch. 8 (Modern CNNs)**](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — softmax/cross-entropy then deep classifiers, with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6 (Recognition)**](https://szeliski.org/Book/) — **Richard Szeliski** — classification in the broader recognition landscape, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.14 Activation Functions & Softmax](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.14_Activation_Functions_and_Softmax.md) · [3.05 Precision / Recall / F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Foundation: [Deep Learning › Loss Functions](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) · [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Next concepts: [05 Transfer Learning for Vision](05-Transfer-Learning-for-Vision.md) · [06 Data Augmentation](06-Data-Augmentation.md)

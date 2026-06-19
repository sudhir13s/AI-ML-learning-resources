---
id: "07-computer-vision/transfer-learning-for-vision"
topic: "Transfer Learning for Vision"
parent: "07-computer-vision"
level: intermediate
prereqs: ["cnns", "classic-cnn-architectures", "image-classification"]
interview_frequency: very-high
updated: 2026-06-20
---

# Transfer Learning for Vision
> Instead of training a CNN from scratch, start from a backbone **pretrained on ImageNet** and adapt it
> to your task — either as a frozen **feature extractor** (replace only the head) or by **fine-tuning**
> (unfreeze some/all layers at a low learning rate). Early conv layers learn generic edges/textures that
> transfer everywhere; later layers are task-specific. This is how most real-world vision models get built.

**Why it matters:** an extremely common applied-ML interview topic — feature extraction vs fine-tuning,
which layers to freeze, how learning rates and dataset size drive the choice, why ImageNet features
transfer, and when transfer hurts (domain gap). It's also the practical answer to "how would you build
a classifier with only 500 labeled images?"

**⭐ Start here — suggested path:**

1. **Build intuition** — watch ⭐ [Transfer Learning (Andrew Ng, C3W2L07)](https://www.youtube.com/watch?v=yofjFQddwHE). *Why reusing pretrained features works and when to use it.*
2. **Feature extraction vs fine-tuning** — watch [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) (deeplizard). *The two regimes and how to choose between them.*
3. **Read the reference notes** — [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/). *The decision table (dataset size × similarity) you'll be quizzed on.*
4. **Read the sources** — [How transferable are features?](https://arxiv.org/abs/1411.1792) + [CNN Features off-the-shelf](https://arxiv.org/abs/1403.6382). *The empirical evidence for layer transferability.*
5. **Make it concrete** — run the [PyTorch Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). *Freeze a ResNet, swap the head, then fine-tune — end to end.*

## 🎓 Courses (free)
- [Stanford CS231n — Transfer Learning](https://cs231n.github.io/transfer-learning/) — **Stanford** — the canonical guide with the dataset-size × similarity decision matrix.
- [Dive into Deep Learning — Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html) — **Zhang et al.** — free chapter that fine-tunes a pretrained model end to end.

## 🎥 Videos
- [Transfer Learning (C3W2L07)](https://www.youtube.com/watch?v=yofjFQddwHE) — **DeepLearning.AI (Andrew Ng)** — the clearest intuition for why and when transfer learning works.
- [Transfer Learning (C4W2L09)](https://www.youtube.com/watch?v=FQM13HkEfBk) — **DeepLearning.AI (Andrew Ng)** — transfer learning in the CNN context specifically.
- [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — **deeplizard** — feature extraction vs fine-tuning, clearly distinguished.
- [Transfer Learning (Deep Learning Tutorial 27)](https://www.youtube.com/watch?v=LsdxvjLWkIY) — **codebasics** — a hands-on Keras/TensorFlow walkthrough.

## 📄 Key Papers
- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) — **Yosinski et al. (2014)** — quantifies layer-by-layer transferability; the foundational study.
- [CNN Features off-the-shelf: an Astounding Baseline](https://arxiv.org/abs/1403.6382) — **Razavian et al. (2014)** — frozen ImageNet features beat hand-crafted ones across tasks.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the most common transfer-learning backbone.

## 📰 Articles / Blogs (free, no paywall)
- [CS231n notes — Transfer Learning](https://cs231n.github.io/transfer-learning/) — **Stanford** — when to fine-tune vs freeze, with the decision rules, free.
- [PyTorch — Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — **PyTorch** — runnable feature-extraction + fine-tuning code.
- [Fine-Tuning (d2l)](https://d2l.ai/chapter_computer-vision/fine-tuning.html) — **Zhang et al.** — explanation plus code, free.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.2 (Fine-Tuning)**](https://d2l.ai/chapter_computer-vision/fine-tuning.html) — **Zhang et al.** — transfer learning with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 5.4 (Transfer / pretraining)**](https://szeliski.org/Book/) — **Richard Szeliski** — pretraining and transfer in context, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 7.03 Transfer Learning & Fine-Tuning](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.03_Transfer_Learning_and_Fine_Tuning.md) · [7.04 Knowledge Distillation](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.04_Knowledge_Distillation.md)
- Foundation: [Classic CNN Architectures](03-Classic-CNN-Architectures.md) (the backbones you transfer)
- Next concepts: [06 Data Augmentation](06-Data-Augmentation.md) · [12 Self-Supervised Vision](12-Self-Supervised-Vision.md)

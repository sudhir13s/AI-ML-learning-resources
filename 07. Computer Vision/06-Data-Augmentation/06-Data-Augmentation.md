---
id: "07-computer-vision/data-augmentation"
topic: "Data Augmentation"
parent: "07-computer-vision"
level: beginner
prereqs: ["image-classification", "regularization"]
interview_frequency: high
updated: 2026-06-20
---

# Data Augmentation
> Synthetically expand a training set by applying label-preserving transforms — flips, crops, rotations,
> color jitter — plus modern mixing strategies like **Mixup**, **CutMix**, and **Cutout**, or learned
> policies like **AutoAugment / RandAugment**. Augmentation is the cheapest, most reliable regularizer
> in vision: it teaches invariances, fights overfitting, and is essential when labeled data is scarce.

**Why it matters:** a guaranteed practical question — name the standard transforms, explain *why*
augmentation regularizes (it enlarges the data manifold and enforces invariances), how Mixup/CutMix
differ from classic transforms, and which augmentations are safe for which task (e.g. don't vertical-flip
digits). It's also half the answer to "my model overfits — what do you do?"

**⭐ Start here — suggested path:**

1. **Build intuition** — watch ⭐ [Data Augmentation explained](https://www.youtube.com/watch?v=rfM4DaLTkMs) (deeplizard). *What augmentation is and why it reduces overfitting.*
2. **See it in code** — watch [Data Augmentation with Keras API](https://www.youtube.com/watch?v=WSvpLUietIM) (deeplizard). *The standard transform pipeline applied to real images.*
3. **University treatment** — watch [Data Augmentation (DLFVC)](https://www.youtube.com/watch?v=MmhnmTnI_8A) (Peter Wonka). *A more formal lecture covering the rationale and techniques.*
4. **Read the modern sources** — [Mixup](https://arxiv.org/abs/1710.09412) → [Cutout](https://arxiv.org/abs/1708.04552) → [CutMix](https://arxiv.org/abs/1905.04899) → [RandAugment](https://arxiv.org/abs/1909.13719). *The mixing/policy methods that became default in strong baselines.*
5. **Make it concrete** — work through the [torchvision transforms](https://pytorch.org/vision/stable/transforms.html) + [d2l: Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html). *Build an augmentation pipeline and see the accuracy lift.*

## 🎓 Courses (free)
- [Dive into Deep Learning — Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html) — **Zhang et al.** — free chapter building an augmentation pipeline with code.
- [Stanford CS231n](https://cs231n.github.io/neural-networks-2/) — **Stanford** — covers augmentation as a regularization technique alongside preprocessing.

## 🎥 Videos
- [Data Augmentation explained](https://www.youtube.com/watch?v=rfM4DaLTkMs) — **deeplizard** — clear conceptual intro to why augmentation works.
- [Data Augmentation with Keras API](https://www.youtube.com/watch?v=WSvpLUietIM) — **deeplizard** — the standard transforms applied in code.
- [Data Augmentation (DLFVC 13)](https://www.youtube.com/watch?v=MmhnmTnI_8A) — **Peter Wonka (KAUST)** — a university lecture on the rationale and methods.
- [Image classification vs Object detection vs Segmentation](https://www.youtube.com/watch?v=taC5pMCm70U) — **codebasics** — frames why augmentation strategy must respect the task/label.

## 📄 Key Papers
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) — **Zhang et al. (2017)** — convex combinations of images and labels; strong regularizer.
- [Improved Regularization with Cutout](https://arxiv.org/abs/1708.04552) — **DeVries & Taylor (2017)** — mask out random patches to force redundancy.
- [CutMix](https://arxiv.org/abs/1905.04899) — **Yun et al. (2019)** — paste patches across images with proportional label mixing.
- [RandAugment](https://arxiv.org/abs/1909.13719) — **Cubuk et al. (2019)** — a simple, search-free automated augmentation policy.

## 📰 Articles / Blogs (free, no paywall)
- [torchvision.transforms documentation](https://pytorch.org/vision/stable/transforms.html) — **PyTorch** — the full catalog of standard and modern image transforms.
- [Albumentations documentation](https://albumentations.ai/docs/) — **Albumentations** — fast, open augmentation library with examples per task.
- [Image Augmentation (d2l)](https://d2l.ai/chapter_computer-vision/image-augmentation.html) — **Zhang et al.** — explanation plus runnable code, free.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.1 (Image Augmentation)**](https://d2l.ai/chapter_computer-vision/image-augmentation.html) — **Zhang et al.** — augmentation with runnable code and measured impact.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 5.3 (Training, regularization & augmentation)**](https://szeliski.org/Book/) — **Richard Szeliski** — augmentation as regularization, free.

## 🔗 In this platform
- Foundation: [Deep Learning › Regularization](../../05.%20Deep_Learning/09-Regularization/09-Regularization.md) · [Deep Learning › Dropout](../../05.%20Deep_Learning/10-Dropout/10-Dropout.md) — augmentation is a data-side regularizer.
- Related: [Image Classification](../04-Image-Classification/04-Image-Classification.md) · [Transfer Learning for Vision](../05-Transfer-Learning-for-Vision/05-Transfer-Learning-for-Vision.md)
- Next concepts: [11 Vision Transformers](../11-Vision-Transformers/11-Vision-Transformers.md) · [12 Self-Supervised Vision](../12-Self-Supervised-Vision/12-Self-Supervised-Vision.md) (augmentation defines the contrastive views)

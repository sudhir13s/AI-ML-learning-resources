---
id: "07-computer-vision/self-supervised-vision"
topic: "Self-Supervised Vision (SimCLR, MAE, DINO)"
parent: "07-computer-vision"
level: advanced
prereqs: ["cnns", "vision-transformers", "data-augmentation", "contrastive-learning"]
interview_frequency: high
updated: 2026-06-20
---

# Self-Supervised Vision — SimCLR · MAE · DINO
> Learn strong visual representations from **unlabeled** images by inventing a pretext task. Three
> families dominate: **contrastive** (SimCLR pulls two augmented views of the same image together,
> pushes others apart via InfoNCE), **masked image modeling** (MAE masks ~75% of patches and
> reconstructs them with a ViT), and **self-distillation** (DINO matches student/teacher outputs,
> yielding emergent segmentation). The pretrained features then transfer with few or no labels.

**Why it matters:** the modern representation-learning question — why self-supervision matters when
labels are expensive, how contrastive learning's positive/negative pairs and InfoNCE work, the role of
augmentation in defining "views," why MAE's high masking ratio works, and how DINO produces
object-segmentation attention for free. Central to foundation-model and multimodal interviews.

**⭐ Start here — suggested path:**

1. **Build intuition (contrastive)** — watch ⭐ [Contrastive Learning — SimCLR illustrated](https://www.youtube.com/watch?v=YZgeWsuyRH8) (AI Bites). *Positive/negative pairs, augmentation views, and the contrastive loss.*
2. **Go a bit deeper** — watch [Can Contrastive Learning Work? — SimCLR Explained](https://www.youtube.com/watch?v=7Id8SPH31UE). *Why big batches, projection heads, and strong augmentation matter.*
3. **Masked image modeling** — watch [Masked Autoencoders (MAE) Paper Explained](https://www.youtube.com/watch?v=-EBqzYIJRaQ), then read the ⭐ [MAE paper](https://arxiv.org/abs/2111.06377). *Mask 75% of patches, reconstruct with an asymmetric ViT encoder/decoder.*
4. **Read the sources** — [SimCLR](https://arxiv.org/abs/2002.05709) → [BYOL](https://arxiv.org/abs/2006.07733) (no negatives) → [DINO](https://arxiv.org/abs/2104.14294) (self-distillation). *The three paradigms and their key tricks.*
5. **See it work** — read [Meta AI: DINO](https://ai.meta.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/). *Emergent segmentation maps from purely self-supervised ViTs.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — recent editions cover self-supervised and contrastive representation learning.
- [CS224W / d2l contrastive-learning material](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — the augmentation + representation foundations self-supervision builds on, free.

## 🎥 Videos
- [Contrastive Learning — SimCLR (illustrated)](https://www.youtube.com/watch?v=YZgeWsuyRH8) — **AI Bites** — positive/negative pairs and the contrastive objective, visualized.
- [Can Contrastive Learning Work? — SimCLR Explained](https://www.youtube.com/watch?v=7Id8SPH31UE) — **Boris Meinardus** — why batch size, projection head, and augmentation matter.
- [Masked Autoencoders (MAE) Paper Explained](https://www.youtube.com/watch?v=-EBqzYIJRaQ) — **Soroush Mehraban** — the asymmetric encoder/decoder and high masking ratio.
- [ViT — An Image Is Worth 16×16 Words](https://www.youtube.com/watch?v=8phM16htKbU) — **Uygar Kurt** — the ViT backbone MAE/DINO build on.

## 📄 Key Papers
- [A Simple Framework for Contrastive Learning (SimCLR)](https://arxiv.org/abs/2002.05709) — **Chen et al. (2020)** — augmentation views + InfoNCE + projection head; the contrastive benchmark.
- [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) — **Grill et al. (2020)** — strong self-supervised features without negative pairs.
- [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377) — **He et al. (2021)** — mask 75% of patches, reconstruct; scalable ViT pretraining.
- [Emerging Properties in Self-Supervised ViTs (DINO)](https://arxiv.org/abs/2104.14294) — **Caron et al. (2021)** — self-distillation; emergent segmentation attention.

## 📰 Articles / Blogs (free, no paywall)
- [Meta AI — DINO: self-supervised ViTs](https://ai.meta.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/) — **Meta AI** — the DINO results and emergent segmentation, free.
- [The Illustrated SimCLR Framework](https://amitness.com/posts/simclr) — **Amit Chaudhary** — a clear, fully open visual explanation of SimCLR.
- [Lil'Log — Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) — **Lilian Weng** — the definitive open survey of pretext tasks and contrastive methods.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.8 (Region-based / representation pretraining)** + **Ch. 16 (NLP pretraining, contrastive analog)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — the augmentation/pretraining machinery, with code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 5.4 (Self-supervised & contrastive pretraining)**](https://szeliski.org/Book/) — **Richard Szeliski** — self-supervision in context, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.13 Contrastive Learning (SimCLR · InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md) · [1.14 Triplet Loss](../../../AI-ML-intuition/Module_1_Representation/1.14_Triplet_Loss.md)
- Foundation: [Vision Transformers](../11-Vision-Transformers/11-Vision-Transformers.md) (MAE/DINO backbone) · [Data Augmentation](../06-Data-Augmentation/06-Data-Augmentation.md) (defines contrastive views)
- Related domain: [Deep Learning › Autoencoders](../../05.%20Deep_Learning/19-Autoencoders/19-Autoencoders.md) (MAE is a masked autoencoder)

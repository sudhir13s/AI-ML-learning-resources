---
id: "04-unsupervised-learning/contrastive-self-supervised"
topic: "Contrastive / Self-Supervised Learning"
parent: "04-unsupervised-learning"
level: advanced
prereqs: ["neural-networks", "embeddings", "cross-entropy", "data-augmentation"]
interview_frequency: high
updated: 2026-06-19
---

# Contrastive / Self-Supervised Learning
> Learn useful representations from *unlabeled* data by inventing the labels: pull two augmented views
> of the same example together in embedding space and push different examples apart (contrastive), or
> predict masked/withheld parts of the input (predictive). The engine behind modern foundation models.

**Why it matters:** the bridge from classic unsupervised learning to today's pretraining. Interviews
probe the **pretext-task** idea (create supervision for free), the contrastive recipe — augment → encode
→ project → **InfoNCE** loss with positives vs negatives — and the practical levers: why **SimCLR**
needs large batches (many negatives), how **MoMo/MoCo** uses a momentum-encoded queue instead, and how
**BYOL / SimSiam** avoid negatives entirely (and the collapse problem they must dodge). You should also
contrast it with **masked** self-supervision (BERT, MAE). Knowing *why InfoNCE works* (mutual-information
lower bound) and *why representation collapse is the central risk* is the bar.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [SimCLR (paper illustrated)](https://www.youtube.com/watch?v=YZgeWsuyRH8). *Augment one image two ways → make those embeddings agree, everything else disagree.*
2. **See the framework** — [UvA DL Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://www.youtube.com/watch?v=waVZDFR-06U). *The encoder + projection head + InfoNCE pipeline, with code.*
3. **Get the loss** — read [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/). *InfoNCE, SimCLR, MoCo, BYOL, SimSiam, and the collapse problem — the single best written survey.*
4. **Read the sources** — [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) and [InfoNCE / CPC (Oord et al., 2018)](https://arxiv.org/abs/1807.03748). *The framework and the loss that the field standardized on.*
5. **Make it concrete** — work through the [illustrated SimCLR walkthrough](https://amitness.com/2020/03/illustrated-simclr/) and the [UvA SimCLR notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html). *Coding the augmentation + NT-Xent loss cements it.*

## 🎓 Courses (free)
- [UvA Deep Learning — Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html) — **University of Amsterdam** — a full, runnable notebook building SimCLR end to end.
- [Dive into Deep Learning — representation learning context](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; how encoders and embeddings (the substrate for contrastive learning) are built and trained.

## 🎥 Videos
- [Contrastive Learning of Visual Representations — SimCLR (paper illustrated)](https://www.youtube.com/watch?v=YZgeWsuyRH8) — **Yannic Kilcher** — a clear, critical walkthrough of the SimCLR paper.
- [Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://www.youtube.com/watch?v=waVZDFR-06U) — **Phillip Lippe (UvA)** — the framework plus the accompanying code notebook.
- [Contrastive Learning with SimCLR V1/V2 and Some Intriguing Properties](https://www.youtube.com/watch?v=-6jb1v1v0vc) — **Connor Shorten** — SimCLRv2, distillation, and what the learned features capture.
- [PyData: PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes** — context for how learned representations are visualized and inspected.

## 📄 Key Papers
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) — **Chen, Kornblith, Norouzi & Hinton (2020)** — augmentations, the projection head, and the NT-Xent loss.
- [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) — **van den Oord, Li & Vinyals (2018)** — the InfoNCE loss and its mutual-information justification.
- [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) — **Grill et al. (2020)** — contrastive-quality representations *without negatives*; the collapse-avoidance trick.

## 📰 Articles / Blogs (free, no paywall)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) — **Lilian Weng (OpenAI)** — the definitive free survey: InfoNCE, SimCLR, MoCo, BYOL, SimSiam, collapse.
- [The Illustrated SimCLR Framework](https://amitness.com/2020/03/illustrated-simclr/) — **Amit Chaudhary** — a visual, step-by-step build of SimCLR and the NT-Xent loss.
- [Advancing Self-Supervised and Semi-Supervised Learning with SimCLR](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr/) — **Google Research** — the authors' own accessible overview and results.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **representation learning & embeddings**](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; the encoder/embedding machinery that contrastive objectives train.
- [Understanding Deep Learning — **representation learning context**](https://udlbook.github.io/udlbook/) — **Simon J.D. Prince** — free PDF; how deep networks build the representations self-supervision shapes.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.13 Contrastive Learning (SimCLR / InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md)
- Related representation pages: [1.14 Triplet Loss](../../../AI-ML-intuition/Module_1_Representation/1.14_Triplet_Loss.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
- Loss foundations: [AI-ML-intuition 5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) — InfoNCE is a cross-entropy / MI-bound loss
- Where it leads: [Autoencoders & deep representation learning → Deep Learning](../../5.%20Deep_Learning/README.md)
- Field overview: [4. Unsupervised Learning](../README.md)

---
id: "04-unsupervised-learning/contrastive-self-supervised/references"
topic: "Contrastive / Self-Supervised Learning — References"
parent: "04-unsupervised-learning/contrastive-self-supervised"
type: references
updated: 2026-06-22
---

# Contrastive / Self-Supervised Learning — references and further reading

> Companion link library for **[Contrastive / Self-Supervised Learning](12-Contrastive-Self-Supervised-Learning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Contrastive Learning of Visual Representations — SimCLR (paper illustrated)](https://www.youtube.com/watch?v=YZgeWsuyRH8) (**Yannic Kilcher**). *Augment one image two ways → make those embeddings agree, everything else disagree.*
2. **See the framework + code** — watch [Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://www.youtube.com/watch?v=waVZDFR-06U) (**Phillip Lippe, UvA**). *The encoder + projection head + NT-Xent pipeline, built in a runnable notebook.*
3. **Get the loss + the whole family** — read [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) (**Lilian Weng**). *InfoNCE, SimCLR, MoCo, BYOL, SimSiam, and the collapse problem — the single best written survey.*
4. **Read the sources** — [SimCLR (Chen et al. 2020)](https://arxiv.org/abs/2002.05709) and [InfoNCE / CPC (Oord et al. 2018)](https://arxiv.org/abs/1807.03748). *The framework, and the loss the field standardized on.*
5. **Make it concrete** — work the [Illustrated SimCLR](https://amitness.com/2020/03/illustrated-simclr/) walkthrough and the [UvA SimCLR notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html). *Coding the augmentations + NT-Xent loss cements it.*

**Videos**:
- [Contrastive Learning of Visual Representations — SimCLR (paper illustrated)](https://www.youtube.com/watch?v=YZgeWsuyRH8) — **Yannic Kilcher** — a clear, critical walkthrough of the SimCLR paper and its augmentation/projection-head findings.
- [Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://www.youtube.com/watch?v=waVZDFR-06U) — **Phillip Lippe (University of Amsterdam)** — the framework plus the accompanying code notebook, end to end.
- [SimCLR Explained!](https://www.youtube.com/watch?v=APki8LmdJwY) — **AI Coffee Break with Letitia** — a concise, visual explainer of the SimCLR pipeline and the NT-Xent loss.
- [Contrastive Learning in PyTorch — Part 1: Introduction](https://www.youtube.com/watch?v=u-X_nZRsn5M) — **DeepFindr** — builds contrastive learning and the InfoNCE loss hands-on in PyTorch.
- [OpenAI CLIP: Connecting Text and Images (paper explained)](https://www.youtube.com/watch?v=T9XSU0pKX2E) — **Yannic Kilcher** — how the same contrastive objective scales across modalities for zero-shot classification.

**Interactive & visual**:
- [The Illustrated SimCLR Framework](https://amitness.com/2020/03/illustrated-simclr/) — **Amit Chaudhary** — a visual, step-by-step build of SimCLR and the NT-Xent loss, with annotated figures.
- [A Visual Notebook to Using SimCLR (Exploring SimCLR)](https://sthalles.github.io/simple-self-supervised-learning/) — **Thalles Silva** — a runnable, visual walk through a SimCLR implementation.

**Courses (free)**:
- [UvA Deep Learning — Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html) — **University of Amsterdam** — a full, runnable notebook building SimCLR end to end.
- [Stanford CS231n — Self-Supervised Learning](https://cs231n.stanford.edu/) — **Stanford** — the self-supervised / contrastive lecture within the canonical vision course.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — context for contrastive text embeddings (SimCSE) within the modern LLM stack.

**Articles / blogs (free, no paywall)**:
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) — **Lilian Weng (OpenAI)** — the definitive free survey: InfoNCE, SimCLR, MoCo, BYOL, SimSiam, and collapse.
- [Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) — **Lilian Weng (OpenAI)** — the companion survey covering pretext tasks (rotation, jigsaw, colorization) that led to contrastive learning.
- [Advancing Self-Supervised and Semi-Supervised Learning with SimCLR](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr/) — **Google Research** — the authors' own accessible overview and results.
- [Self-Supervised Learning: The Dark Matter of Intelligence](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/) — **Yann LeCun & Ishan Misra (Meta AI)** — the big-picture case for why self-supervision (and joint-embedding methods) matters.
- [Understanding Contrastive Learning](https://ai.stanford.edu/blog/understanding-contrastive-learning/) — **Stanford AI Lab** — the alignment-and-uniformity view of what the contrastive loss actually optimizes.

**Key papers**:
- [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) — **van den Oord, Li & Vinyals (2018)** — the InfoNCE loss and its mutual-information lower-bound justification.
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) — **Chen, Kornblith, Norouzi & Hinton (2020)** — augmentations, the projection head, and the NT-Xent loss.
- [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/abs/1911.05722) — **He, Fan, Wu, Xie & Girshick (2020)** — the momentum (EMA) key encoder + queue that decouples negatives from batch size.
- [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) — **Grill et al. (2020)** — contrastive-quality representations *without negatives*; the predictor + EMA + stop-gradient collapse-avoidance.
- [Exploring Simple Siamese Representation Learning (SimSiam)](https://arxiv.org/abs/2011.10566) — **Chen & He (2021)** — the ablation proving **stop-gradient** is the essential anti-collapse ingredient.
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) — **Zbontar, Jing, Misra, LeCun & Deny (2021)** — push the feature cross-correlation toward the identity (decorrelation) instead of using negatives.
- [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906) — **Bardes, Ponce & LeCun (2022)** — the most explicit anti-collapse formulation: one term per force.
- [Understanding Contrastive Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) — **Wang & Isola (2020)** — the two limiting losses (alignment + uniformity) that decompose InfoNCE.
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — **Radford et al. (2021, OpenAI)** — image-text contrastive learning; the bridge to multimodal foundation models.
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) — **Gao, Yao & Chen (2021)** — dropout-as-augmentation for unsupervised sentence embeddings.
- [Unsupervised Representation Learning by Predicting Image Rotations (RotNet)](https://arxiv.org/abs/1803.07728) — **Gidaris, Singh & Komodakis (2018)** — the rotation pretext task in the lineage that preceded contrastive learning.
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — **Oquab et al. (2023, Meta AI)** — where self-supervised vision representations are today; a strong end point for the lineage.

**Books (free chapters)**:
- [Dive into Deep Learning — representation learning & embeddings](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; the encoder/embedding machinery that contrastive objectives train.
- [Understanding Deep Learning — representation learning context](https://udlbook.github.io/udlbook/) — **Simon J.D. Prince** — free PDF; how deep networks build the representations self-supervision shapes.
- [An Introduction to Statistical Learning](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free; the bias–variance and similarity foundations under embedding-based methods.

**In this platform**:
- Concept page (full explanation): [Contrastive / Self-Supervised Learning](12-Contrastive-Self-Supervised-Learning.md)
- Foundations (the *why*): [AI-ML-intuition 1.13 Contrastive Learning (SimCLR / InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md) · [1.14 Triplet Loss](../../../AI-ML-intuition/Module_1_Representation/1.14_Triplet_Loss.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
- Loss foundations: [AI-ML-intuition 5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) — InfoNCE is a cross-entropy / MI-bound loss
- Evaluating the representation: [k-Nearest Neighbors](../../03.%20Supervised_Learning/concepts/04-k-Nearest-Neighbors.md) — the k-NN probe for self-supervised embeddings
- Where it leads: [4. Unsupervised Learning](../README.md) · [Deep Learning concepts](../../05.%20Deep_Learning/concepts/README.md)

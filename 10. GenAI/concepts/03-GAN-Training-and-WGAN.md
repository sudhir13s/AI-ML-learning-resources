---
id: "10-generative-ai/gan-training-wgan"
topic: "GAN Training Pathologies & WGAN"
parent: "10-generative-ai"
level: advanced
prereqs: ["gans-dcgan", "kl-divergence", "optimization", "lipschitz-continuity"]
interview_frequency: high
updated: 2026-06-20
---

# GAN Training Pathologies & WGAN
> GANs are notoriously hard to train: **mode collapse** (the generator emits a few samples that fool
> `D`), **vanishing gradients** when `D` gets too good, and oscillation/non-convergence. The
> **Wasserstein GAN** replaces the JS-divergence objective with the **Earth-Mover (Wasserstein-1)
> distance**, which gives smooth, non-vanishing gradients — enforced by a **Lipschitz** constraint
> (weight clipping in WGAN, a **gradient penalty** in WGAN-GP).

**Why it matters:** the "why won't my GAN train?" interview, and a clean test of whether you
understand *divergences*. Be ready to explain why JS divergence saturates when supports don't overlap
(zero gradient), what the Wasserstein distance measures and why it stays informative, the
Kantorovich–Rubinstein duality that turns it into a maximization over **1-Lipschitz critics**, why
weight clipping is a crude Lipschitz hack, and how **WGAN-GP** fixes it with a gradient-norm penalty.
Also: practical stabilizers (spectral normalization, TTUR, label smoothing) and how to *diagnose*
mode collapse.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Common Problems with GANs](https://developers.google.com/machine-learning/gan/problems) — **Google**. *Names and shows mode collapse, vanishing gradients, and non-convergence first.*
2. **See why it works** — read [Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html) — **Alex Irpan**. *The clearest "why JS fails and EM distance saves it" explanation.*
3. **Get the math** — read [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) (WGAN section) — **Lilian Weng** + watch [Nuts and Bolts of WGANs](https://www.youtube.com/watch?v=31mqB4yGgQY) — **Crazymuse**. *Earth-Mover distance, Kantorovich–Rubinstein duality, the Lipschitz constraint.*
4. **Read the sources** — [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — **Arjovsky et al. (2017)** → [Improved Training of WGANs (WGAN-GP)](https://arxiv.org/abs/1704.00028) — **Gulrajani et al. (2017)**. *The distance, then the gradient-penalty fix for clipping.*
5. **Make it concrete** — implement it with [WGAN-GP from scratch](https://www.youtube.com/watch?v=pG0QZ7OddX4) — **Aladdin Persson**. *Coding the critic + gradient penalty makes the Lipschitz constraint tangible.*

## 🎓 Courses (free)
- [Google — GANs: Common Problems](https://developers.google.com/machine-learning/gan/problems) — **Google** — free, concise catalog of GAN failure modes and the loss-function fixes.
- [Stanford CS231n — Generative Models notes](https://cs231n.github.io/) — **Stanford** — frames GAN training difficulty and divergence-based objectives in the generative-models module.

## 🎥 Videos
- [Nuts and Bolts of WGANs (Kantorovich–Rubinstein, Earth-Mover distance)](https://www.youtube.com/watch?v=31mqB4yGgQY) — **Crazymuse** — the duality and Lipschitz argument behind WGAN, derived clearly.
- [WGAN implementation from scratch (with gradient penalty)](https://www.youtube.com/watch?v=pG0QZ7OddX4) — **Aladdin Persson** — codes WGAN-GP in PyTorch; the critic, clipping vs penalty, and training loop.
- [The Math Behind GANs, Clearly Explained](https://www.youtube.com/watch?v=Gib_kiXgnvA) — **Normalized Nerd** — the JS-divergence objective whose failure motivates the Wasserstein distance.
- [GANs explained](https://www.youtube.com/watch?v=_qB4B6ttXk8) — **AI Coffee Break (Letitia)** — compact overview of the game and where training instability comes from.

## 📄 Key Papers
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — **Arjovsky, Chintala & Bottou (2017)** — replaces JS with the Earth-Mover distance; a meaningful loss curve and fewer collapses.
- [Improved Training of Wasserstein GANs (WGAN-GP)](https://arxiv.org/abs/1704.00028) — **Gulrajani et al. (2017)** — the gradient penalty that replaces weight clipping for the Lipschitz constraint.

## 📰 Articles / Blogs (free, no paywall)
- [Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html) — **Alex Irpan** — the most intuitive account of why JS divergence vanishes and EM distance does not.
- [From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) — **Lilian Weng** — the rigorous reference: divergences, EM distance, Kantorovich–Rubinstein duality, WGAN-GP.
- [Wasserstein GAN and the Kantorovich-Rubinstein Duality](https://agustinus.kristia.de/blog/wasserstein-gan/) — **Agustinus Kristiadi** — a focused, math-first derivation of the duality, fully open.

## 📚 Books (free, with chapters)
- [Deep Learning — **§20.10.4 (GANs) & §3.13 (KL/divergences)**](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — GAN training and the divergence background, free online.
- [Dive into Deep Learning — **Ch. 20 "Generative Adversarial Networks"**](https://d2l.ai/chapter_generative-adversarial-networks/index.html) — **Zhang et al.** — free, with runnable GAN code to experiment with instability.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.04 GANs & WGAN](../../../AI-ML-intuition/Module_5_Generation/5.04_GANs_and_WGAN.md) · [5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Prereq: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) (the base game this card fixes)
- Next concepts: [04 Conditional Generation & Classifier-Free Guidance](04-Conditional-Generation-and-Classifier-Free-Guidance.md) · [12 Evaluation of Generative Models](12-Evaluation-of-Generative-Models.md) (how to measure collapse)
- Field overview: [9. Generative AI](../README.md)

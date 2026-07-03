---
id: "10-generative-ai/vae/references"
topic: "Variational Autoencoders (VAE · ELBO) — References"
parent: "10-generative-ai/vae"
type: references
updated: 2026-07-03
---

# Variational Autoencoders (VAE · ELBO) — references and further reading

> Companion link library for **[Variational Autoencoders (VAE · ELBO)](01-Variational-Autoencoders-VAE-ELBO.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on *VAEs specifically* (the ELBO, the reparameterization trick, the closed-form Gaussian KL, and the β-VAE / posterior-collapse story), not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch ⭐ [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) by **Arxiv Insights (Xander Steenbrugge)**. *The clearest first picture: a continuous, sampleable latent space instead of a fixed code, and why that lets you generate.*
2. **See why it works** — read [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) by **Lilian Weng (OpenAI)**. *Walks the ELBO, the reparameterization trick, the closed-form KL, and β-VAE carefully — the canonical open write-up.*
3. **Get the math** — read [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) by **Doersch (2016)**. *The most readable end-to-end ELBO derivation and the variational-inference framing you may be asked to reproduce.*
4. **Read the source** — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by **Kingma & Welling (2013)**. *The paper that introduced the VAE and the reparameterized ELBO estimator.*
5. **Make it concrete** — run this chapter's [notebook](code/01-Variational-Autoencoders-VAE-ELBO.ipynb): a from-scratch VAE trained on real MNIST, with the closed-form Gaussian KL asserted equal to a Monte-Carlo estimate of KL(q‖p), the reparameterization trick's gradient flow proven, and real reconstructions, prior samples, and a 2-D latent manifold.

**Videos**:
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights (Xander Steenbrugge)** — the best gentle first watch: latent space, sampling, and why a VAE is generative where an autoencoder is not.
- [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=3G5hWM6jqPk) — **Alexander Amini (MIT)** — the lecture treatment: ELBO, the KL term, the reparameterization trick, then GANs, from the free MIT intro-to-deep-learning course.
- [Variational Autoencoders | Generative AI Animated](https://www.youtube.com/watch?v=qJeaCHQ1k2w) — **Deepia** — a clean animated derivation of the ELBO and the reparameterization trick, 3Blue1Brown-style.
- [Variational Autoencoder Paper Walkthrough](https://www.youtube.com/watch?v=5bA6gwo36Cw) — **Aladdin Persson** — reads the Kingma & Welling paper line by line, then codes a VAE in PyTorch — the implementer's view of everything on this page.

**Courses (free)**:
- [MIT 6.S191 — Intro to Deep Learning](https://introtodeeplearning.com/) — **MIT (Alexander Amini)** — the "Deep Generative Modeling" lecture covers VAEs end to end (ELBO, reparameterization, β), slides + video free.
- [Stanford CS231n — Generative Models](https://cs231n.github.io/) — **Stanford** — the generative-models module places VAEs alongside GANs and autoregressive models, with the latent-variable framing.
- [University of Waterloo STAT 946 / Deep Learning — Variational Autoencoders](https://uwaterloo.ca/data-analytics/teaching) — **Ali Ghodsi** — his lectures derive the ELBO and the reparameterization trick from a probabilistic-modeling angle (lecture videos are on YouTube under his course).
- [Hugging Face Diffusion / Generative course material](https://huggingface.co/learn) — **Hugging Face** — hands-on units that use the VAE as the latent-space autoencoder inside latent diffusion (the VAE's role in Stable Diffusion).

**Articles / blogs (free, no paywall)**:
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) — **Lilian Weng (OpenAI)** — the definitive open survey: the ELBO, the reparameterization trick, the closed-form KL, β-VAE, and VQ-VAE, derived cleanly.
- [What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) — **Jaan Altosaar** — the deep-learning view and the probabilistic-model view placed side by side; excellent for the "why a lower bound?" question.
- [Variational Autoencoders (with code)](https://avandekleut.github.io/vae/) — **Alexander Van de Kleut** — a minimal, runnable PyTorch VAE with the ELBO loss explained term by term.
- [Reparameterization trick](https://gregorygundersen.com/blog/2018/04/29/reparameterization/) — **Gregory Gundersen** — a focused, careful explanation of the single idea that makes the encoder trainable.

**Key papers** (the source and the extensions):
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2013/2014), ICLR** — the original VAE: amortized variational inference + the reparameterized ELBO estimator. The source for everything on this page.
- [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082) — **Rezende, Mohamed & Wierstra (2014), ICML** — the equivalent stochastic-backpropagation formulation of the reparameterized gradient, developed independently and concurrently.
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — **Doersch (2016)** — the most readable long-form VAE derivation; treat it as the extended explanation of the math here.
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — **Higgins et al. (2017), ICLR** — the KL-weighting knob (β) and the disentanglement it buys, the trade-off measured in this chapter.
- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) — **Kingma & Welling (2019)** — the authors' own book-length modern treatment (free on arXiv), covering the ELBO, reparameterization, and extensions.

**Books (free chapters)**:
- [Deep Learning — §20.10.3 "Variational Autoencoders"](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — VAEs within the deep generative-models chapter, free online.
- [Probabilistic Machine Learning: Advanced Topics — VAE chapter](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — the modern, rigorous treatment of the ELBO, amortized inference, and VAEs (free PDF).
- [Mathematics for Machine Learning — Ch. 8–9 (latent-variable models, density estimation)](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — the variational-inference and MLE background under the ELBO, free online.

**In this platform**:
- Concept page (full explanation): [Variational Autoencoders (VAE · ELBO)](01-Variational-Autoencoders-VAE-ELBO.md)
- Runnable code: [the module `vae.py`](code/vae.py) · [the step-by-step notebook](code/01-Variational-Autoencoders-VAE-ELBO.ipynb) — a from-scratch VAE trained on real MNIST, with the closed-form Gaussian KL asserted equal to a Monte-Carlo estimate of KL(q‖p), the reparameterization trick's gradient flow proven (a raw `.sample()` carries no gradient), and real reconstructions, prior samples, a 2-D latent manifold, an interpolation, and the β trade-off; every number computed, none fabricated.
- The deterministic precursor: [05 Deep Learning · 19 Autoencoders](../../05.%20Deep_Learning/19-Autoencoders/19-Autoencoders.md) — compresses and reconstructs, but its latent space is a bag of holes; the VAE adds the probabilistic, sampleable latent.
- The same ELBO, fit by EM: [04 Unsupervised Learning · 04 Gaussian Mixture Models & EM](../../04.%20Unsupervised_Learning/04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md) — the ELBO / KL-gap identity this page reuses; EM climbs it by a per-datapoint E-step, a VAE by an *amortized* encoder network.
- The closed-form KL: [01 Foundations · 23 Cross-Entropy and KL Divergence](../../01.%20Foundations/23-Cross-Entropy-and-KL-Divergence/23-Cross-Entropy-and-KL-Divergence.md) — where the two-Gaussian KL (the VAE regulariser) is derived; the VAE specializes it to a unit-Gaussian prior.
- The gradient engine: [05 Deep Learning · 02 Backpropagation and Computational Graphs](../../05.%20Deep_Learning/02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md) — the reparameterization trick exists so that ordinary backprop can reach the encoder's μ and log σ².
- The variance contrast: [08 Reinforcement Learning · 09 Policy Gradients (REINFORCE)](../../08.%20Reinforcement_Learning/09-Policy-Gradients-REINFORCE/09-Policy-Gradients-REINFORCE.md) — the *score-function* estimator, the high-variance alternative you must use when you *cannot* reparameterize; the VAE's pathwise estimator is the low-variance counterpart.
- The family contrast: [02 GANs and DCGAN](../02-GANs-and-DCGAN.md) — adversarial, sharp, unstable, no likelihood; the VAE is likelihood-based, stable, and blurry.
- Where it goes next: [05 Diffusion Models (DDPM)](../05-Diffusion-Models-DDPM.md) — a many-step hierarchical VAE trained by a per-step ELBO; the VAE is its one-step conceptual seed.
- Where the VAE lives in production: [07 Latent Diffusion / Stable Diffusion](../07-Latent-Diffusion-Stable-Diffusion.md) — diffusion runs in the compact latent space of a pretrained VAE; encode → diffuse → decode.
- Field overview: [10 Generative AI](../README.md)

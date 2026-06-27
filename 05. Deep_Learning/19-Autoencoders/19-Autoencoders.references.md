---
id: "05-deep-learning/autoencoders/references"
topic: "Autoencoders — References"
parent: "05-deep-learning/autoencoders"
type: references
updated: 2026-06-22
---

# Autoencoders — references and further reading

> Companion link library for **[Autoencoders](19-Autoencoders.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Autoencoders Explained Easily](https://www.youtube.com/watch?v=xwrzh4e8DLs) (**Valerio Velardo, The Sound of AI**). *Encoder → bottleneck → decoder, and what compression buys.*
2. **See the generative jump** — watch [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) (**Arxiv Insights**). *Why a VAE samples a latent distribution instead of a point.*
3. **Get the math** — read [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) (**Lilian Weng**). *Reconstruction loss, the ELBO, and the reparameterization trick, derived.*
4. **Read the source** — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (**Kingma & Welling, 2013**). *The paper behind variational autoencoders.*
5. **Make it concrete** — implement a small (V)AE following the [Deep Learning Book Ch. 14](https://www.deeplearningbook.org/contents/autoencoders.html) and Lilian Weng's write-up. *Coding the bottleneck and KL term makes the latent space tangible.*

**Videos**:
- [Autoencoders Explained Easily](https://www.youtube.com/watch?v=xwrzh4e8DLs) — **Valerio Velardo (The Sound of AI)** — the encoder/bottleneck/decoder structure, clearly.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights** — the clearest intuition anywhere for the VAE's latent distribution and KL term.
- [Variational Autoencoders (VAE) — clearly explained](https://www.youtube.com/watch?v=fcvYpzHmhvA) — **CodeEmporium** — the ELBO and reparameterization trick walked through step by step.
- [Autoencoder Explained — Deep Neural Networks](https://www.youtube.com/watch?v=q222maQaPYo) — **AIEngineering** — a practical walk-through with reconstruction and use cases.
- [What is an Autoencoder? (Two Minute Papers #86)](https://www.youtube.com/watch?v=Rdpbnd0pCiI) — **Two Minute Papers** — a crisp two-minute conceptual primer.

**Courses (free)**:
- [Stanford CS231n — Generative Models (Autoencoders & VAEs)](https://cs231n.github.io/) — **Stanford (Li / Johnson / Karpathy)** — autoencoders and VAEs in the generative-models lecture.
- [MIT 6.S191 — Deep Generative Modeling](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — autoencoders → VAEs → GANs in one current lecture.
- [UFLDL — Sparse Autoencoder](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/) — **Stanford (Andrew Ng et al.)** — the original tutorial that derives the KL sparsity penalty.

**Articles / blogs (free, no paywall)**:
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) — **Lilian Weng (OpenAI)** — the canonical write-up: AE → VAE → β-VAE, with the ELBO and reparameterization derived.
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — **Carl Doersch (CMU)** — the most-cited gentle-but-rigorous VAE tutorial; the ELBO and reparameterization from first principles.
- [Variational Autoencoders](https://www.jeremyjordan.me/variational-autoencoders/) — **Jeremy Jordan** — why the KL term makes the latent space continuous and samplable, with clear figures.
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html) — **François Chollet (Keras)** — plain, sparse, denoising, and variational autoencoders in runnable code.
- [Variational Autoencoders from scratch (PyTorch)](https://avandekleut.github.io/vae/) — **Alexander Van de Kleut** — derives the ELBO and reparameterization trick alongside a minimal, runnable implementation.

**Key papers**:
- [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf) — **Hinton & Salakhutdinov (2006, Science)** — deep autoencoders that beat PCA at dimensionality reduction; the paper that revived the idea.
- [Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima](https://doi.org/10.1016/0893-6080(89)90014-2) — **Baldi & Hornik (1989, Neural Networks)** — proves a linear autoencoder's only stable optimum spans the PCA subspace (the result derived on the page). The [Deep Learning Book Ch. 14](https://www.deeplearningbook.org/contents/autoencoders.html) summarizes it.
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) — **Vincent et al. (2008)** — the denoising autoencoder for robust representation learning.
- [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) — **Vincent (2011)** — proves denoising ≈ learning the score $\nabla\log p(\mathbf{x})$, the bridge to diffusion.
- [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](https://icml.cc/2011/papers/455_icmlpaper.pdf) — **Rifai et al. (2011)** — the encoder-Jacobian penalty and its tie to denoising.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2013)** — the variational autoencoder, the ELBO, and the reparameterization trick.
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — **Higgins et al. (2017)** — weighting the KL term for disentangled representations.
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — **He et al. (2022)** — mask 75% of patches and reconstruct; modern self-supervised vision pretraining.
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — **Rombach et al. (2022)** — Stable Diffusion; a VAE compresses images so diffusion runs in a small latent space.

**Books (free chapters)**:
- [Deep Learning — Ch. 14 "Autoencoders"](https://www.deeplearningbook.org/contents/autoencoders.html) — **Goodfellow, Bengio & Courville** — undercomplete / denoising / sparse / contractive autoencoders, rigorously, with the manifold view.
- [Dive into Deep Learning](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — encoder–decoder representations and latent codes with runnable code.

**In this platform**:
- Concept page (full explanation): [Autoencoders](19-Autoencoders.md)
- Prerequisites: [02 Backpropagation & Computational Graphs](../02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md) · [04 Loss Functions](../04-Loss-Functions/04-Loss-Functions.md) · [03 Activation Functions](../03-Activation-Functions/03-Activation-Functions.md)
- The linear baseline (PCA): [04. Unsupervised — Dimensionality Reduction Overview](../../04.%20Unsupervised_Learning/06-Dimensionality-Reduction-Overview/06-Dimensionality-Reduction-Overview.md)
- Visualization neighbors: [t-SNE](../../04.%20Unsupervised_Learning/07-t-SNE/07-t-SNE.md) · [UMAP](../../04.%20Unsupervised_Learning/08-UMAP/08-UMAP.md)
- Puts it to work: [Anomaly / Outlier Detection (reconstruction error)](../../04.%20Unsupervised_Learning/09-Anomaly-Outlier-Detection/09-Anomaly-Outlier-Detection.md) · [Contrastive / Self-Supervised Learning](../../04.%20Unsupervised_Learning/12-Contrastive-Self-Supervised-Learning/12-Contrastive-Self-Supervised-Learning.md)
- The generative deep-dive: [10. GenAI — Variational Autoencoders (VAE · ELBO)](../../10.%20GenAI/01-Variational-Autoencoders-VAE-ELBO/01-Variational-Autoencoders-VAE-ELBO.md) · [Latent Diffusion / Stable Diffusion](../../10.%20GenAI/07-Latent-Diffusion-Stable-Diffusion/07-Latent-Diffusion-Stable-Diffusion.md)
- Field overview: [Deep Learning](../README.md)

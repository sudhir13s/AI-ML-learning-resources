---
id: "05-deep-learning/autoencoders"
topic: "Autoencoders"
parent: "05-deep-learning"
level: intermediate
prereqs: ["feedforward-networks", "backpropagation", "loss-functions"]
interview_frequency: medium
updated: 2026-06-19
---

# Autoencoders
> A network trained to reconstruct its own input through a **bottleneck**: an *encoder* compresses the
> input to a low-dimensional latent code, and a *decoder* rebuilds it. By forcing reconstruction
> through a narrow layer, the model learns a compact representation. Variants — **denoising**,
> **sparse**, and **variational (VAE)** — turn this into representation learning, anomaly detection,
> and generative modeling.

**Why it matters:** a representation-learning and generative-modeling staple — explain the
encoder–bottleneck–decoder structure and the reconstruction loss, why an undercomplete autoencoder
learns useful features (and how it relates to PCA), what denoising/sparse variants add, and how a
**VAE** differs by learning a *distribution* over the latent (reparameterization + KL term, the ELBO).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Autoencoders Explained Easily](https://www.youtube.com/watch?v=xwrzh4e8DLs) (**Valerio Velardo**). *Encoder → bottleneck → decoder, and what compression buys.*
2. **See the generative jump** — watch ⭐ [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) (**Arxiv Insights**). *Why a VAE samples a latent distribution instead of a point.*
3. **Get the math** — read [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) (**Lilian Weng**). *Reconstruction loss, the ELBO, and the reparameterization trick.*
4. **Read the source** — [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) (**Kingma & Welling, 2013**). *The paper behind variational autoencoders.*
5. **Make it concrete** — implement a small (V)AE following [d2l / the Annotated VAE](https://lilianweng.github.io/posts/2018-08-12-vae/). *Coding the bottleneck and KL term makes the latent space tangible.*

## 🎓 Courses (free)
- [Stanford CS231n — Generative Models (Autoencoders & VAEs)](https://cs231n.github.io/) — **Stanford (Karpathy / Li / Johnson)** — autoencoders and VAEs in the generative-models lecture.
- [MIT 6.S191 — Deep Generative Modeling](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — autoencoders → VAEs → GANs in one current lecture.

## 🎥 Videos
- [Autoencoders Explained Easily](https://www.youtube.com/watch?v=xwrzh4e8DLs) — **Valerio Velardo (The Sound of AI)** — the encoder/bottleneck/decoder structure, clearly.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights** — the clearest intuition for the VAE's latent distribution and KL term.
- [Autoencoder Explained — Deep Neural Networks](https://www.youtube.com/watch?v=q222maQaPYo) — **AIEngineering** — practical walk-through with reconstruction and use cases.
- [What is an Autoencoder? (Two Minute Papers #86)](https://www.youtube.com/watch?v=Rdpbnd0pCiI) — **Two Minute Papers** — a crisp two-minute conceptual primer.

## 📄 Key Papers
- [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2013)** — the variational autoencoder and reparameterization trick.
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) — **Vincent et al. (2008)** — denoising autoencoders for robust representation learning.
- [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf) — **Hinton & Salakhutdinov (2006)** — deep autoencoders that beat PCA at dimensionality reduction.

## 📰 Articles / Blogs (free, no paywall)
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) — **Lilian Weng** — the canonical write-up: AE → VAE → β-VAE, with the ELBO.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — encoder–decoder framing (useful contrast for the AE structure).
- [Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/) — **Distill** — encoder–decoder representations in sequence models.

## 📚 Books (free, with chapters)
- [Deep Learning — **Ch. 14 "Autoencoders"**](https://www.deeplearningbook.org/contents/autoencoders.html) — **Goodfellow, Bengio & Courville** — undercomplete/denoising/sparse autoencoders, rigorously.
- [Dive into Deep Learning — Generative models / latent representations](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — encoder–decoder representations and latent codes with code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.02 Latent-Variable Models / ELBO / VAEs](../../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md) · [1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Prerequisites: [02 Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs.md) · [04 Loss Functions](04-Loss-Functions.md)
- Field overview: [Deep Learning](../README.md)
- Related domain: [09. GenAI](../../09.%20GenAI/concepts/README.md) (VAEs, diffusion, GANs in depth)

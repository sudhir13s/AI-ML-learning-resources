---
id: "10-generative-ai/vae"
topic: "Variational Autoencoders (VAE · ELBO)"
parent: "10-generative-ai"
level: advanced
prereqs: ["autoencoders", "kl-divergence", "multivariate-gaussian", "maximum-likelihood", "neural-networks"]
interview_frequency: high
updated: 2026-06-20
---

# Variational Autoencoders — VAE · ELBO · Reparameterization
> Turn a plain autoencoder into a *generative* model by making the latent code **probabilistic**:
> the encoder outputs a distribution `q(z|x)`, we regularize it toward a prior `p(z)=N(0,I)`, and
> we train by maximizing the **Evidence Lower BOund (ELBO)** — reconstruction quality minus a KL
> penalty. The **reparameterization trick** (`z = μ + σ⊙ε`) lets gradients flow through the sampling.

**Why it matters:** the canonical "derive the ELBO" / "explain the reparameterization trick" question,
and the conceptual root of diffusion (a VAE is a one-step latent-variable model; diffusion is many
steps). Interviews probe: why we maximize a *lower bound* on `log p(x)` instead of the likelihood
directly, what the KL term *does* (regularizes the posterior toward the prior, enabling sampling),
why we can't backprop through a raw `sample()` (and how reparameterization fixes it), and the failure
modes — **posterior collapse** and blurry samples from the Gaussian likelihood.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights**. *The clearest first picture: a continuous, sampleable latent space instead of a fixed code.*
2. **See why it works** — read [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) — **Lilian Weng**. *Walks the ELBO, the reparameterization trick, and the KL term carefully.*
3. **Get the math** — watch [MIT 6.S191: Deep Generative Modeling](https://www.youtube.com/watch?v=3G5hWM6jqPk) + read [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — **Doersch**. *The ELBO derivation and the variational inference framing you may be asked to reproduce.*
4. **Read the source** — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2013)**. *The paper that introduced the VAE and the reparameterized estimator.*
5. **Make it concrete** — implement one with the [Keras VAE example](https://keras.io/examples/generative/vae/) or [d2l-style PyTorch](https://avandekleut.github.io/vae/). *Coding the encoder/decoder + ELBO loss cements the trick.*

## 🎓 Courses (free)
- [MIT 6.S191 — Intro to Deep Learning](https://introtodeeplearning.com/) — **MIT (Amini)** — the "Deep Generative Modeling" lecture covers VAEs end to end, slides + video free.
- [Stanford CS231n — Generative Models notes](https://cs231n.github.io/) — **Stanford** — the generative-models module places VAEs alongside GANs and autoregressive models.

## 🎥 Videos
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights** — the best gentle first watch: latent space, sampling, and why it's generative.
- [MIT 6.S191 (2023): Deep Generative Modeling](https://www.youtube.com/watch?v=3G5hWM6jqPk) — **Alexander Amini (MIT)** — the lecture treatment: ELBO, KL term, reparameterization, then GANs.
- [Variational Autoencoders | Generative AI Animated](https://www.youtube.com/watch?v=qJeaCHQ1k2w) — **Deepia** — clean animated derivation of the ELBO and the reparameterization trick.
- [Variational AutoEncoder Paper Walkthrough](https://www.youtube.com/watch?v=5bA6gwo36Cw) — **Aladdin Persson** — reads the Kingma & Welling paper line by line, then codes it.

## 📄 Key Papers
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — **Kingma & Welling (2013)** — the original VAE: amortized variational inference + the reparameterized ELBO estimator.
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — **Doersch (2016)** — the most readable VAE derivation; treat it as the long-form explanation.

## 📰 Articles / Blogs (free, no paywall)
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) — **Lilian Weng** — the canonical math walkthrough (ELBO, KL, β-VAE, VQ-VAE), fully open.
- [Tutorial — What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) — **Jaan Altosaar** — the deep-learning and probabilistic-model views side by side.
- [Variational Autoencoders (with code)](https://avandekleut.github.io/vae/) — **Alexander Van de Kleut** — minimal PyTorch VAE you can run, with the loss explained.

## 📚 Books (free, with chapters)
- [Deep Learning — **§20.10.3 "Variational Autoencoders"**](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — VAEs within the deep generative-models chapter, free online.
- [Mathematics for Machine Learning — **Ch. 8–9 (latent-variable models, density estimation)**](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — the variational-inference and MLE background under the ELBO.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.02 Latent-Variable Models · ELBO · VAEs](../../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md) · [5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Prereq: [Deep Learning — Autoencoders](../../05.%20Deep_Learning/concepts/19-Autoencoders.md) (the deterministic precursor)
- Compare with: [04 GMMs & EM](../../04.%20Unsupervised_Learning/concepts/04-Gaussian-Mixture-Models-and-EM.md) (latent-variable model fit by EM rather than amortized inference)
- Next concepts: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) · [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md)
- Field overview: [9. Generative AI](../README.md)

---
id: "09-generative-ai/concepts"
topic: "Generative AI — Concept Index"
parent: "09-generative-ai"
level: advanced
updated: 2026-06-20
---

# Generative AI — Concept Index
> Pick a concept to open its resource card — a short guided learning path plus the best **free, open**
> courses, videos, papers, articles, and books for that topic.
> **✅ ready · ⬜ coming soon.** New to generative modeling? Start with the [field overview](../README.md).

## Likelihood-based models (VAEs, flows, autoregressive)

1. ✅ [Variational Autoencoders (VAE · ELBO · reparameterization)](01-Variational-Autoencoders-VAE-ELBO.md)
8. ⬜ [Normalizing Flows (RealNVP · Glow · exact likelihood)](08-Normalizing-Flows.md)
10. ⬜ [Autoregressive Image Generation (PixelRNN · PixelCNN)](10-Autoregressive-Image-Generation-PixelCNN.md)

## Adversarial models (GANs)

2. ✅ [GANs & DCGAN (the adversarial game)](02-GANs-and-DCGAN.md)
3. ✅ [GAN Training Pathologies & WGAN (mode collapse · Wasserstein)](03-GAN-Training-and-WGAN.md)
4. ✅ [Conditional Generation & Classifier-Free Guidance (cGAN · CFG)](04-Conditional-Generation-and-Classifier-Free-Guidance.md)

## Diffusion & score-based models

5. ✅ [Diffusion Models — DDPM (forward/reverse process)](05-Diffusion-Models-DDPM.md)
6. ✅ [Score-Based & SDE Diffusion (score matching · probability-flow ODE)](06-Score-Based-and-SDE-Diffusion.md)
7. ✅ [Latent Diffusion & Stable Diffusion (VAE + U-Net + text)](07-Latent-Diffusion-Stable-Diffusion.md)

## Energy-based models & systems

9. ⬜ [Energy-Based Models (EBM · contrastive divergence)](09-Energy-Based-Models.md)
11. ⬜ [Text-to-Image Systems (DALL·E · Imagen · CLIP guidance)](11-Text-to-Image-Systems.md)

## Sampling, guidance & evaluation

12. ⬜ [Evaluation of Generative Models (FID · Inception Score · precision/recall)](12-Evaluation-of-Generative-Models.md)
13. ⬜ [Sampling & Guidance Techniques (DDIM · ancestral · guidance scale)](13-Sampling-and-Guidance-Techniques.md)

## Related concepts (covered in another section)
> These topics are foundational or live in another domain, so they're kept in one place to avoid repetition.

- **Autoencoders (plain / denoising / sparse)** — the deterministic precursor to the VAE → [Deep Learning · Autoencoders](../../05.%20Deep_Learning/concepts/19-Autoencoders.md)
- **LLM text generation & autoregressive language models** — GPT-style next-token generation → [LLMs](../../08.%20LLMs/README.md)
- **Decoding strategies for text** (greedy · beam · top-k · top-p) → [NLP · Decoding Strategies](../../06.%20NLP/concepts/17-Decoding-Strategies.md)
- **Gaussian Mixture Models & the EM algorithm** — the classic latent-variable model → [Unsupervised Learning · GMM & EM](../../04.%20Unsupervised_Learning/concepts/04-Gaussian-Mixture-Models-and-EM.md)
- **Information theory** (entropy · cross-entropy · KL divergence) — the objective under every likelihood model → [Foundations](../../01.%20Foundations/README.md)

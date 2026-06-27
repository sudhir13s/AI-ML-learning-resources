---
id: "10-generative-ai"
topic: "Generative AI"
level: advanced
prereqs: ["deep-learning"]
updated: 2026-06-27
---

# Generative AI
> Models that *create* — images, audio, video, 3D — via VAEs, GANs, and (today) diffusion.
> (Text generation lives under [LLMs](../09.%20LLMs/README.md).)

**⭐ Start here:** [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) — **Ari Seff** — the clearest intro to the method behind Stable Diffusion / DALL·E.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** New to generative modeling? Start with the field overview below, then work top to bottom.

### Likelihood-based models (VAEs, flows, autoregressive)
1. ✅ [Variational Autoencoders (VAE · ELBO · reparameterization)](01-Variational-Autoencoders-VAE-ELBO/01-Variational-Autoencoders-VAE-ELBO.md)
8. ✅ [Normalizing Flows (RealNVP · Glow · exact likelihood)](08-Normalizing-Flows/08-Normalizing-Flows.md)
10. ✅ [Autoregressive Image Generation (PixelRNN · PixelCNN)](10-Autoregressive-Image-Generation-PixelCNN/10-Autoregressive-Image-Generation-PixelCNN.md)

### Adversarial models (GANs)
2. ✅ [GANs & DCGAN (the adversarial game)](02-GANs-and-DCGAN/02-GANs-and-DCGAN.md)
3. ✅ [GAN Training Pathologies & WGAN (mode collapse · Wasserstein)](03-GAN-Training-and-WGAN/03-GAN-Training-and-WGAN.md)
4. ✅ [Conditional Generation & Classifier-Free Guidance (cGAN · CFG)](04-Conditional-Generation-and-Classifier-Free-Guidance/04-Conditional-Generation-and-Classifier-Free-Guidance.md)

### Diffusion & score-based models
5. ✅ [Diffusion Models — DDPM (forward/reverse process)](05-Diffusion-Models-DDPM/05-Diffusion-Models-DDPM.md)
6. ✅ [Score-Based & SDE Diffusion (score matching · probability-flow ODE)](06-Score-Based-and-SDE-Diffusion/06-Score-Based-and-SDE-Diffusion.md)
7. ✅ [Latent Diffusion & Stable Diffusion (VAE + U-Net + text)](07-Latent-Diffusion-Stable-Diffusion/07-Latent-Diffusion-Stable-Diffusion.md)

### Energy-based models & systems
9. ✅ [Energy-Based Models (EBM · contrastive divergence)](09-Energy-Based-Models/09-Energy-Based-Models.md)
11. ✅ [Text-to-Image Systems (DALL·E · Imagen · CLIP guidance)](11-Text-to-Image-Systems/11-Text-to-Image-Systems.md)

### Sampling, guidance & evaluation
12. ✅ [Evaluation of Generative Models (FID · Inception Score · precision/recall)](12-Evaluation-of-Generative-Models/12-Evaluation-of-Generative-Models.md)
13. ✅ [Sampling & Guidance Techniques (DDIM · ancestral · guidance scale)](13-Sampling-and-Guidance-Techniques/13-Sampling-and-Guidance-Techniques.md)

### Related concepts (covered in another section)
> These topics are foundational or live in another domain, so they're kept in one place to avoid repetition.
- **Autoencoders (plain / denoising / sparse)** — the deterministic precursor to the VAE → [Deep Learning · Autoencoders](../05.%20Deep_Learning/19-Autoencoders/19-Autoencoders.md)
- **LLM text generation & autoregressive language models** — GPT-style next-token generation → [LLMs](../09.%20LLMs/README.md)
- **Decoding strategies for text** (greedy · beam · top-k · top-p) → [NLP · Decoding Strategies](../06.%20NLP/17-Decoding-Strategies/17-Decoding-Strategies.md)
- **Gaussian Mixture Models & the EM algorithm** — the classic latent-variable model → [Unsupervised Learning · GMM & EM](../04.%20Unsupervised_Learning/04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md)
- **Information theory** (entropy · cross-entropy · KL divergence) — the objective under every likelihood model → [Foundations](../01.%20Foundations/README.md)

## 🎓 Courses (free)
- [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) — **DeepLearning.AI** — free short course, build one.
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course) — **Hugging Face** — free, code-first.

## 🎥 Videos
- [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** — implement DDPM end to end.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — **Arxiv Insights** — the best VAE intro.

## 📄 Key Papers
- [DDPM (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239) — **Ho et al. (2020)** — the modern diffusion formulation.
- [High-Resolution Image Synthesis with Latent Diffusion](https://arxiv.org/abs/2112.10752) — **Rombach et al. (2022)** — Stable Diffusion.
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — **Goodfellow et al. (2014)** — the GAN that started it.

## 📰 Articles
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** — the canonical math walkthrough.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face** — runnable code + math.

## 🔗 In this platform
- Math: [AI-ML-intuition Module 5 (Generation)](../../AI-ML-intuition/Module_5_Generation/)

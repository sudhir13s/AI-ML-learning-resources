---
id: "10-generative-ai/latent-diffusion"
topic: "Latent Diffusion & Stable Diffusion"
parent: "10-generative-ai"
level: advanced
built_from: ["diffusion-ddpm", "vae", "conditional-cfg", "attention", "clip"]
interview_frequency: high
updated: 2026-06-20
tier: core
est_minutes: 10
title: "Latent Diffusion & Stable Diffusion"
minutes: 10
category: diffusion-models
---

# Latent Diffusion & Stable Diffusion
> Pixel-space diffusion is expensive. **Latent Diffusion Models (LDM)** first compress images into a
> small latent space with a pretrained **VAE**, run the whole diffusion process *there*, then decode
> back to pixels — orders of magnitude cheaper. Add **cross-attention** to a text encoder and you get
> **Stable Diffusion**: a U-Net denoiser conditioned on a CLIP text embedding, steered by
> classifier-free guidance.

**Why it matters:** the architecture behind the open-source image-generation explosion, and the
go-to systems-design answer for "how does Stable Diffusion work?" Interviews probe: *why* diffuse in
latent space (perceptual compression vs. semantic detail; the VAE handles the high-frequency pixels so
the U-Net models semantics), the three components (VAE encoder/decoder, conditioning text encoder,
denoising U-Net with cross-attention), where **CFG** plugs in, and the trade-offs vs pixel-space
diffusion (speed and memory vs a slight quality ceiling from the autoencoder).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [How AI Image Generators Work (Stable Diffusion / DALL·E)](https://www.youtube.com/watch?v=1CIpzeNxIhU) — **Computerphile**. *The clearest plain-language tour of the full pipeline.*
2. **See why it works** — read [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — **Jay Alammar**. *Visual, component-by-component: VAE, text encoder, U-Net, cross-attention.*
3. **Get the math** — watch [How does Stable Diffusion work? — Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — **AI Coffee Break (Letitia)** + read [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion). *Why latent space, and how the pieces connect.*
4. **Read the source** — [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — **Rombach et al. (2022)**. *The LDM paper; perceptual compression + cross-attention conditioning.*
5. **Make it concrete** — run the [Hugging Face `diffusers` quickstart](https://huggingface.co/docs/diffusers/index), or watch [Coding Stable Diffusion from scratch](https://www.youtube.com/watch?v=ZBKpAp_6TGI) — **Umar Jamil**. *Generating from a prompt and tweaking the guidance scale cements it.*

## 🎓 Courses (free)
- [Hugging Face — Diffusion Models Course (Stable Diffusion unit)](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first; build conditioned latent diffusion with `diffusers`.
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; situates latent diffusion within the broader generative-models map.

## 🎥 Videos
- [How AI Image Generators Work (Stable Diffusion / DALL·E)](https://www.youtube.com/watch?v=1CIpzeNxIhU) — **Computerphile** — the best plain-language overview of the whole pipeline.
- [How does Stable Diffusion work? — Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — **AI Coffee Break (Letitia)** — why latent space, and the VAE + U-Net + text-encoder design.
- [Stable Diffusion — What, Why, How?](https://www.youtube.com/watch?v=ltLNYA3lWAQ) — **Edan Meyer** — a thorough, accessible walkthrough of each component and the sampling loop.
- [Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI) — **Umar Jamil** — builds the full system (VAE, CLIP, U-Net, sampler) line by line.

## 📄 Key Papers
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — **Rombach et al. (2022)** — Stable Diffusion: diffuse in VAE latent space with cross-attention conditioning.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — **Ho & Salimans (2022)** — the guidance mechanism that gives Stable Diffusion its prompt adherence.

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — **Jay Alammar** — the canonical visual explainer; VAE, text encoder, U-Net, cross-attention.
- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion) — **Hugging Face** — the architecture plus runnable code for each stage, free.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face** — the U-Net denoiser internals that latent diffusion reuses, free.

## 📚 Books (free, with chapters)
- [Understanding Deep Learning — **Ch. 18 "Diffusion models"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF; conditioning and latent diffusion in the diffusion chapter.
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 25 "Diffusion models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; latent and conditional diffusion in the modern treatment.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.03 Diffusion Models](../../../../AI-ML-intuition/generation/diffusion-and-score-models/diffusion-forward-and-reverse-process-intuition.md) · [5.02 ELBO & VAEs](../../../../AI-ML-intuition/generation/latent-variable-generation/latent-variable-models-and-elbo-intuition.md)
- Prereq: [05 Diffusion Models (DDPM)](../diffusion-models-ddpm/diffusion-models-ddpm.md) · [01 Variational Autoencoders](../../generative-models/variational-autoencoders-vae-elbo/variational-autoencoders-vae-elbo.md) (the latent compressor) · [04 Conditional Generation & CFG](../conditional-generation-and-classifier-free-guidance/conditional-generation-and-classifier-free-guidance.md)
- Related: [Deep Learning — Attention Mechanism](../../../deep-learning/attention-and-transformers/attention-mechanism/attention-mechanism.md) (cross-attention conditions the U-Net)
- Next concepts: [11 Text-to-Image Systems](../text-to-image-systems/text-to-image-systems.md) · [13 Sampling & Guidance Techniques](../sampling-and-guidance-techniques/sampling-and-guidance-techniques.md)
- Field overview: [9. Generative AI](../../generative-models/README.md)

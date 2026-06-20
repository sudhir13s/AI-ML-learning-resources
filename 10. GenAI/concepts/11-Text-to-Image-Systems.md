---
id: "10-generative-ai/text-to-image"
topic: "Text-to-Image Systems"
parent: "10-generative-ai"
level: advanced
prereqs: ["diffusion-ddpm", "latent-diffusion", "conditional-cfg", "clip", "transformers"]
interview_frequency: high
updated: 2026-06-20
---

# Text-to-Image Systems — DALL·E · Imagen · Stable Diffusion
> The systems behind "type a prompt, get an image." They share a recipe: a **text encoder** (CLIP or
> a frozen T5 LLM) turns the prompt into an embedding, a **diffusion** model generates conditioned on
> it, and **classifier-free guidance** sharpens prompt adherence. They differ in *where* they diffuse
> and *how* they condition: **DALL·E 2** (unCLIP) diffuses to a CLIP image embedding then decodes;
> **Imagen** uses a frozen T5 + cascaded super-resolution; **Stable Diffusion** diffuses in a VAE
> latent with cross-attention.

**Why it matters:** the flagship application of generative AI and a favorite ML-systems-design
question. Interviews probe: the shared text-encoder → diffusion → CFG pipeline; the three design
choices that distinguish DALL·E 2 / Imagen / Stable Diffusion (CLIP-image-embedding prior vs. frozen
LLM text encoder vs. latent-space diffusion); why **Imagen found a big frozen LLM text encoder matters
more than image-model size**; CLIP's role in connecting text and images; and evaluation (FID for
fidelity, CLIP-score for prompt alignment) plus failure modes (compositionality, counting, text
rendering).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [How AI Image Generators Work (Stable Diffusion / DALL·E)](https://www.youtube.com/watch?v=1CIpzeNxIhU) — **Computerphile**. *The shared prompt → image pipeline in plain language.*
2. **See why it works** — read [How DALL·E 2 Actually Works](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/) — **AssemblyAI**. *CLIP + diffusion prior + decoder (unCLIP), clearly.*
3. **Get the math** — watch [How does DALL·E 2 actually work?](https://www.youtube.com/watch?v=F1X4fHzF4mQ) — **AssemblyAI** + read [How Imagen Actually Works](https://www.assemblyai.com/blog/how-imagen-actually-works/). *Contrast the unCLIP prior with Imagen's frozen-T5 + cascade.*
4. **Read the sources** — [CLIP](https://arxiv.org/abs/2103.00020) → [GLIDE](https://arxiv.org/abs/2112.10741) → [DALL·E 2 / unCLIP](https://arxiv.org/abs/2204.06125) → [Imagen](https://arxiv.org/abs/2205.11487). *The shared building blocks, then the two flagship systems.*
5. **Make it concrete** — run [Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion) — **Hugging Face**, and vary the prompt + guidance scale. *Generating and steering images cements the pipeline.*

## 🎓 Courses (free)
- [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first; build text-conditioned diffusion with `diffusers`.
- [Stanford CS231n — Generative Models notes](https://cs231n.github.io/) — **Stanford** — situates text-to-image within the broader generative-models map.

## 🎥 Videos
- [How AI Image Generators Work (Stable Diffusion / DALL·E)](https://www.youtube.com/watch?v=1CIpzeNxIhU) — **Computerphile** — the best plain-language tour of the prompt → image pipeline.
- [How does DALL·E 2 actually work?](https://www.youtube.com/watch?v=F1X4fHzF4mQ) — **AssemblyAI** — the unCLIP design: CLIP embeddings, diffusion prior, and decoder.
- [DALL·E 2 Explained](https://www.youtube.com/watch?v=qTgPSKKjfVg) — **OpenAI** — the official short overview of capabilities and the two-stage architecture.
- [Diffusion models explained: how does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) — **AI Coffee Break (Letitia)** — text conditioning + guidance in the model that seeded DALL·E 2.

## 📄 Key Papers
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — **Radford et al. (2021)** — the text–image embedding space every system relies on.
- [Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL·E 2 / unCLIP)](https://arxiv.org/abs/2204.06125) — **Ramesh et al. (2022)** — diffusion prior to a CLIP image embedding + decoder.
- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)](https://arxiv.org/abs/2205.11487) — **Saharia et al. (2022)** — frozen T5 text encoder + cascaded super-resolution.
- [GLIDE: Text-Guided Diffusion with CLIP/Classifier-Free Guidance](https://arxiv.org/abs/2112.10741) — **Nichol et al. (2022)** — the text-guided-diffusion precursor that established CFG for images.

## 📰 Articles / Blogs (free, no paywall)
- [How DALL·E 2 Actually Works](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/) — **AssemblyAI** — the clearest unCLIP walkthrough: CLIP, prior, decoder, upsamplers.
- [How Imagen Actually Works](https://www.assemblyai.com/blog/how-imagen-actually-works/) — **AssemblyAI** — frozen T5 encoder, diffusion, and the super-resolution cascade.
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — **Jay Alammar** — the open-source system's components, visually.

## 📚 Books (free, with chapters)
- [Understanding Deep Learning — **Ch. 18 "Diffusion models"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF; conditional/text-guided diffusion in the diffusion chapter.
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 25 "Diffusion models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; conditional generation and guidance.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.03 Diffusion Models](../../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md)
- Prereq: [07 Latent Diffusion & Stable Diffusion](07-Latent-Diffusion-Stable-Diffusion.md) · [04 Conditional Generation & CFG](04-Conditional-Generation-and-Classifier-Free-Guidance.md) · [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md)
- Related: [Deep Learning — Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) (the text encoders) · [LLMs](../../09.%20LLMs/concepts/README.md)
- Next concepts: [12 Evaluation of Generative Models](12-Evaluation-of-Generative-Models.md) · [13 Sampling & Guidance Techniques](13-Sampling-and-Guidance-Techniques.md)
- Field overview: [9. Generative AI](../README.md)

---
id: "11-diffusion"
topic: "Diffusion Models"
level: advanced
built_from: ["generative-ai", "deep-learning"]
leads_to: ["multimodal", "video-understanding"]
updated: 2026-07-14
---

# Diffusion Models
> The dedicated home for diffusion *beyond the foundations* — modern architectures, fast
> sampling, control and editing, video/3D, and production serving. The foundational chapters
> (DDPM, score-based/SDE, latent diffusion, sampling & guidance) live in
> [GenAI](../generative-models/README.md) and are prerequisites here, not duplicated.

**⭐ Start here:** [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** — the canonical math walkthrough; read it before anything in this topic.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** Foundations first: work through the GenAI diffusion chapters, then top to bottom here.

### Modern architectures
1. ⬜ Diffusion Transformers (DiT · U-Net → transformer backbones)
2. ⬜ Flow Matching & Rectified Flow (the post-DDPM formulation)

### Fast generation
3. ⬜ Consistency Models & Few-Step Generation
4. ⬜ Distillation for Fast Sampling (progressive distillation · LCM · turbo)

### Control, editing & personalization
5. ⬜ ControlNet & Conditioning Adapters
6. ⬜ Image Editing & Inversion (SDEdit · null-text inversion · instruct-editing)
7. ⬜ Personalization (DreamBooth · textual inversion · diffusion LoRA)

### Beyond still images
8. ⬜ Video Diffusion Models (temporal layers · cascaded and latent video)
9. ⬜ 3D Generation (score distillation · text-to-3D)

### Production
10. ⬜ Diffusion Inference Optimization (samplers · step budgets · caching · batching)
11. ⬜ Safety, Provenance & Watermarking for generative imagery

### Related concepts (covered in another section)
> Foundations stay in their canonical home to avoid repetition.
- **DDPM forward/reverse process** → [GenAI · Diffusion Models — DDPM](diffusion-models-ddpm/diffusion-models-ddpm.md)
- **Score-based & SDE view** → [GenAI · Score-Based & SDE Diffusion](score-based-and-sde-diffusion/score-based-and-sde-diffusion.md)
- **Latent diffusion / Stable Diffusion** → [GenAI · Latent Diffusion](latent-diffusion-stable-diffusion/latent-diffusion-stable-diffusion.md)
- **Samplers & guidance (DDIM · CFG)** → [GenAI · Sampling & Guidance](sampling-and-guidance-techniques/sampling-and-guidance-techniques.md)
- **Text-to-image systems (DALL·E · Imagen)** → [GenAI · Text-to-Image Systems](text-to-image-systems/text-to-image-systems.md)
- **Audio & music diffusion** → [Audio & Speech](../audio-and-speech/README.md)

## 🎓 Courses (free)
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course) — **Hugging Face** — free, code-first; the practical companion to this topic.
- [MIT 6.S184: Generative AI with Stochastic Differential Equations](https://diffusion.csail.mit.edu/) — **MIT** — open course on diffusion & flow matching from first principles.

## 🎥 Videos
- [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** — implement DDPM end to end before going modern.

## 📄 Key Papers
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — **Peebles & Xie (2022)** — the backbone behind modern image/video generators.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — **Lipman et al. (2022)** — the formulation replacing DDPM in current systems.
- [Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)](https://arxiv.org/abs/2302.05543) — **Zhang et al. (2023)** — the standard for structural control.
- [Consistency Models](https://arxiv.org/abs/2303.01469) — **Song et al. (2023)** — few-step generation.

## 📰 Articles
- [Perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html) — **Sander Dieleman (DeepMind)** — how the field's best practitioner thinks about the design space.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face** — runnable code + math, line by line.

## 📚 Books
- [Understanding Deep Learning — Ch. 18 (Diffusion)](https://udlbook.github.io/udlbook/) — **Simon Prince** — free, rigorous, visual.

## 🔗 In this platform
- Foundations: [GenAI](../generative-models/README.md) · Math: [AI-ML-intuition Module 5 (Generation)](../../../AI-ML-intuition/generation/)

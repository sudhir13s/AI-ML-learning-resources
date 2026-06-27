---
id: "10-generative-ai/sampling-guidance"
topic: "Sampling & Guidance Techniques"
parent: "10-generative-ai"
level: advanced
prereqs: ["diffusion-ddpm", "score-sde", "conditional-cfg", "ode-solvers"]
interview_frequency: medium
updated: 2026-06-20
---

# Sampling & Guidance Techniques — DDIM · Solvers · Guidance Scale
> A trained diffusion model is only half the story — *how you sample* decides speed and quality.
> **DDIM** makes the reverse process non-Markovian and deterministic, cutting 1000 steps to ~20–50.
> **ODE solvers** (DPM-Solver, EDM/Heun) push that to ~10 steps via the probability-flow ODE. And
> **guidance** — classifier and classifier-free — is the knob that trades diversity for fidelity, with
> the **guidance scale** as its single most important sampling-time hyperparameter.

**Why it matters:** the practical layer between "I have a model" and "I get good images fast," and a
common follow-up to the diffusion question. Interviews probe: why vanilla **DDPM ancestral sampling**
needs hundreds of steps; how **DDIM** reuses the same network with a deterministic, non-Markovian
update (enabling fewer steps, interpolation, and inversion); the **probability-flow ODE** view that
turns sampling into ODE integration solvable by higher-order solvers in ~10 steps; and the
**guidance-scale trade-off** — higher `s` improves prompt adherence and fidelity but reduces diversity
and can over-saturate.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html) — **Sander Dieleman**. *What the guidance scale does and the fidelity/diversity trade-off, intuitively.*
2. **See why it works** — watch [Diffusion models explained: how does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) — **AI Coffee Break (Letitia)**. *Guidance in action and how it sharpens conditional samples.*
3. **Get the math** — read [What are Diffusion Models? (sampling & DDIM)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** + [Yang Song's score blog (probability-flow ODE)](https://yang-song.net/blog/2021/score/). *DDIM's non-Markovian update and the ODE view behind fast solvers.*
4. **Read the sources** — [DDIM](https://arxiv.org/abs/2010.02502) — **Song et al. (2021)** · [DPM-Solver](https://arxiv.org/abs/2206.00927) — **Lu et al. (2022)** · [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) — **Ho & Salimans (2022)** · [EDM design space](https://arxiv.org/abs/2206.00364) — **Karras et al. (2022)**. *Faster samplers, then the unifying design space.*
5. **Make it concrete** — swap schedulers in the [Diffusers schedulers guide](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) and sweep the guidance scale. *Feeling steps↔quality and scale↔diversity cements it.*

## 🎓 Courses (free)
- [Hugging Face — Diffusion Models Course (sampling & schedulers)](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first; swap DDPM/DDIM/DPM-Solver schedulers and see the effect.
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; score-based sampling and the probability-flow ODE.

## 🎥 Videos
- [Diffusion models explained: how does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) — **AI Coffee Break (Letitia)** — guidance and conditional sampling, clearly visualized.
- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) — **Outlier** — the reverse-process sampling loop the fast samplers accelerate.
- [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** — codes the sampling loop, the natural place to swap in DDIM.
- [Variational Autoencoders and Diffusion Models (M2L summer school)](https://www.youtube.com/watch?v=pea3sH6orMc) — **Tim Salimans (Google)** — a researcher's view of diffusion sampling and distillation for speed.

## 📄 Key Papers
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) — **Song, Meng & Ermon (2021)** — deterministic, non-Markovian sampling; 10–50× fewer steps.
- [DPM-Solver: A Fast ODE Solver for Diffusion Sampling in ~10 Steps](https://arxiv.org/abs/2206.00927) — **Lu et al. (2022)** — high-order ODE solver for the probability-flow ODE.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — **Ho & Salimans (2022)** — the guidance-scale mechanism that trades diversity for fidelity.
- [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — **Karras et al. (2022)** — disentangles noise schedule, sampler, and preconditioning.

## 📰 Articles / Blogs (free, no paywall)
- [Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html) — **Sander Dieleman** — the definitive account of guidance and the guidance-scale trade-off.
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** — the DDIM and faster-sampling sections with the math.
- [Diffusers — Schedulers guide](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) — **Hugging Face** — compares DDPM/DDIM/DPM-Solver/EDM samplers with code, free.

## 📚 Books (free, with chapters)
- [Understanding Deep Learning — **Ch. 18 "Diffusion models"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF; the reverse process, DDIM, and guidance.
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 25 "Diffusion models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; samplers, the probability-flow ODE, and guidance.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md) · [5.03 Diffusion Models](../../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md)
- Prereq: [05 Diffusion Models (DDPM)](../05-Diffusion-Models-DDPM/05-Diffusion-Models-DDPM.md) · [06 Score-Based & SDE Diffusion](../06-Score-Based-and-SDE-Diffusion/06-Score-Based-and-SDE-Diffusion.md) (the probability-flow ODE) · [04 Conditional Generation & CFG](../04-Conditional-Generation-and-Classifier-Free-Guidance/04-Conditional-Generation-and-Classifier-Free-Guidance.md)
- Related: [07 Latent Diffusion & Stable Diffusion](../07-Latent-Diffusion-Stable-Diffusion/07-Latent-Diffusion-Stable-Diffusion.md) · [11 Text-to-Image Systems](../11-Text-to-Image-Systems/11-Text-to-Image-Systems.md) (where guidance scale is the key knob)
- Field overview: [9. Generative AI](../README.md)

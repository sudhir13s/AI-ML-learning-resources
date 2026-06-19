---
id: "09-generative-ai/score-sde"
topic: "Score-Based & SDE Diffusion"
parent: "09-generative-ai"
level: advanced
prereqs: ["diffusion-ddpm", "gradients", "stochastic-processes", "langevin-dynamics"]
interview_frequency: medium
updated: 2026-06-20
---

# Score-Based & SDE Diffusion
> A second, equivalent view of diffusion: instead of predicting noise, learn the **score** — the
> gradient of the log-density `∇_x log p(x)` — at every noise level, then sample by following it with
> **Langevin dynamics**. Yang Song's **SDE framework** unifies score-matching and DDPM: a continuous
> **forward SDE** noises the data and a **reverse-time SDE** generates it, with a deterministic
> **probability-flow ODE** giving fast, exact-likelihood sampling.

**Why it matters:** the theory that explains *why* diffusion works and connects all the variants. It's
the basis for fast samplers (DDIM, DPM-Solver) and exact likelihoods. Interviews probe: what the score
function is and why estimating it avoids the intractable normalizing constant; **denoising score
matching** and its equivalence to DDPM's noise-prediction loss; how **annealed Langevin dynamics**
sample across noise scales; the forward/reverse **SDE** pair (Anderson's theorem) and the **VE vs VP**
SDEs (NCSN vs DDPM as special cases); and the **probability-flow ODE** that turns the SDE into a
deterministic, invertible map.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — **Yang Song**. *The author's own blog: scores, Langevin sampling, and the SDE unification, with animations.*
2. **See why it works** — watch [Diffusion Models From Scratch | Score-Based Generative Models](https://www.youtube.com/watch?v=B4oHJpEJBAA) — **Outlier**. *Connects the score view to the DDPM noise-prediction view you already know.*
3. **Get the math** — watch [Diffusion and Score-Based Generative Models](https://www.youtube.com/watch?v=wMmqCMwuM2Q) — **Yang Song (MIT CBMM lecture)**. *The definitive talk: score matching, the SDE, and the probability-flow ODE.*
4. **Read the sources** — [Generative Modeling by Estimating Gradients (NCSN)](https://arxiv.org/abs/1907.05600) — **Song & Ermon (2019)** → [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — **Song et al. (2021)**. *Annealed Langevin first, then the SDE framework that subsumes DDPM.*
5. **Make it concrete** — read [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face**, then map `ε`-prediction to score `s_θ ≈ −ε/σ`. *Seeing the two parameterizations coincide is the "aha."*

## 🎓 Courses (free)
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free lecture notes; the score-matching and energy/score lectures are the canonical course treatment.
- [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free; the `diffusers` score-SDE pipelines let you sample with SDE/ODE solvers.

## 🎥 Videos
- [Diffusion and Score-Based Generative Models](https://www.youtube.com/watch?v=wMmqCMwuM2Q) — **Yang Song (MIT CBMM)** — the author's lecture: score matching → SDE → probability-flow ODE.
- [Diffusion Models From Scratch | Score-Based Generative Models Explained](https://www.youtube.com/watch?v=B4oHJpEJBAA) — **Outlier** — links the score view to noise-prediction, with clear animations.
- [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) — **Ari Seff** — the diffusion mental model the score view builds on; useful warm-up.
- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) — **Outlier** — the DDPM math whose loss is provably the denoising-score-matching loss.

## 📄 Key Papers
- [Generative Modeling by Estimating Gradients of the Data Distribution (NCSN)](https://arxiv.org/abs/1907.05600) — **Song & Ermon (2019)** — score matching across noise scales + annealed Langevin sampling.
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — **Song et al. (2021)** — the SDE framework: forward/reverse SDEs, VE/VP, and the probability-flow ODE.

## 📰 Articles / Blogs (free, no paywall)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — **Yang Song** — the definitive author blog: scores, Langevin, the SDE unification, with code.
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** — includes the score-based and SDE sections linking to DDPM.
- [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — **Calvin Luo** — derives the score interpretation of the DDPM objective explicitly.

## 📚 Books (free, with chapters)
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 25 "Diffusion models" (score-based & SDE sections)**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; score matching and the SDE view in one place.
- [Understanding Deep Learning — **Ch. 18 "Diffusion models"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF; the score/SDE connection with clean figures.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.03 Diffusion Models](../../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md)
- Prereq: [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md) (the equivalent noise-prediction view)
- Next concepts: [07 Latent Diffusion & Stable Diffusion](07-Latent-Diffusion-Stable-Diffusion.md) · [13 Sampling & Guidance Techniques](13-Sampling-and-Guidance-Techniques.md) (DDIM / ODE samplers)
- Compare with: [09 Energy-Based Models](09-Energy-Based-Models.md) (also score/energy-based, sampled by Langevin)
- Field overview: [9. Generative AI](../README.md)

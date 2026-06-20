---
id: "10-generative-ai/energy-based-models"
topic: "Energy-Based Models"
parent: "10-generative-ai"
level: advanced
prereqs: ["probability", "gradients", "mcmc", "langevin-dynamics", "maximum-likelihood"]
interview_frequency: low
updated: 2026-06-20
---

# Energy-Based Models (EBM)
> Model an unnormalized density `p(x) ∝ exp(−E_θ(x))` with a neural **energy function** `E_θ` — low
> energy for real data, high for everything else. No restrictions on the network (no invertibility, no
> tractable latents), but the **partition function** `Z` is intractable, so training uses
> **contrastive divergence**: lower the energy of real samples, raise the energy of model samples
> drawn by **MCMC / Langevin dynamics**. EBMs unify GANs, score models, and classifiers under one lens.

**Why it matters:** the conceptual glue of generative modeling — score-based diffusion, the
GAN discriminator, and even a softmax classifier are all EBMs in disguise. Interviews probe: why `Z`
makes exact maximum likelihood intractable; the **contrastive-divergence** gradient (a difference of
expectations over data and model samples); how **Langevin dynamics** sample from `E_θ` using only
`∇_x E_θ` (the **score**), linking EBMs directly to score-based diffusion; and the practical
difficulty — MCMC mixing is slow and training is unstable, which is why EBMs are elegant but less used
in production than diffusion.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [NYU DL — Energy-Based Models (Week 7 notes)](https://atcold.github.io/NYU-DLSP20/en/week07/07-1/) — **LeCun / Canziani (NYU)**. *Low-energy-for-real, high-for-fake — the EBM picture and why it generalizes feed-forward nets.*
2. **See why it works** — watch [NYU Deep Learning Week 7 — Energy-based models](https://www.youtube.com/watch?v=PHxKk5Y5ayc) — **LeCun / Canziani (NYU)**. *EBMs as the general framework; contrastive vs architectural methods.*
3. **Get the math** — work through [UvA DL — Deep Energy-Based Generative Models](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html) — **University of Amsterdam**. *Contrastive divergence + Langevin sampling, derived and coded.*
4. **Read the source** — [Implicit Generation and Modeling with Energy-Based Models](https://arxiv.org/abs/1903.08689) — **Du & Mordatch (2019)**. *Modern EBM training with Langevin-dynamics sampling on images.*
5. **Make it concrete** — connect it to diffusion via [Yang Song's score blog](https://yang-song.net/blog/2021/score/). *Seeing `∇_x log p(x) = −∇_x E_θ(x)` is the unifying "aha."*

## 🎓 Courses (free)
- [NYU Deep Learning (DLSP) — Energy-Based Models](https://atcold.github.io/NYU-DLSP21/) — **LeCun & Canziani (NYU)** — free lectures + notes; EBMs as the organizing framework for the course.
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; the energy-based and score-based lectures together.

## 🎥 Videos
- [NYU Deep Learning Week 7 — Energy-based models and self-supervised learning](https://www.youtube.com/watch?v=PHxKk5Y5ayc) — **LeCun / Canziani (NYU)** — the definitive lecture framing of EBMs.
- [Energy-Based Self-Supervised Learning](https://www.youtube.com/watch?v=bDvpuaPq8Vc) — **Alfredo Canziani (NYU)** — a focused, visual talk on the EBM viewpoint and latent-variable EBMs.
- [Concept Learning with Energy-Based Models (Paper Explained)](https://www.youtube.com/watch?v=Cs_j-oNwGgg) — **Yannic Kilcher** — a careful read of an EBM paper; the energy/Langevin machinery in action.
- [L3 Flow Models — CS294-158 Deep Unsupervised Learning](https://www.youtube.com/watch?v=JBb5sSC0JoY) — **Pieter Abbeel (Berkeley)** — sets up likelihood-based vs energy/score models in the same course.

## 📄 Key Papers
- [Implicit Generation and Modeling with Energy-Based Models](https://arxiv.org/abs/1903.08689) — **Du & Mordatch (2019)** — scalable EBM training on images via Langevin-dynamics MCMC sampling.
- [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) — **LeCun et al. (2006)** — the foundational tutorial: energy functions, loss design, and contrastive methods.

## 📰 Articles / Blogs (free, no paywall)
- [Energy-Based Models (NYU DL Week 7 notes)](https://atcold.github.io/NYU-DLSP20/en/week07/07-1/) — **LeCun / Canziani (NYU)** — the clearest high-level account: energy surfaces, inference, and contrastive training.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — **Yang Song** — shows the score `∇log p` equals `−∇E`, linking EBMs to diffusion.
- [UvA DL — Deep Energy-Based Generative Models](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html) — **University of Amsterdam** — runnable contrastive-divergence + Langevin notebook.

## 📚 Books (free, with chapters)
- [Deep Learning — **Ch. 16 "Structured Probabilistic Models" & §18 "Partition Function"**](https://www.deeplearningbook.org/contents/partition.html) — **Goodfellow, Bengio & Courville** — the partition function and contrastive divergence, free online.
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 24 "Energy-based models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; the modern, unified treatment.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Closely related: [06 Score-Based & SDE Diffusion](06-Score-Based-and-SDE-Diffusion.md) (the score is the negative energy gradient; both sample by Langevin)
- Compare with: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) (the discriminator is an implicit energy) · [08 Normalizing Flows](08-Normalizing-Flows.md)
- Field overview: [9. Generative AI](../README.md)

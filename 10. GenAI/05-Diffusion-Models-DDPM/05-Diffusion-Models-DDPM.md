---
id: "10-generative-ai/diffusion-ddpm"
topic: "Diffusion Models (DDPM)"
parent: "10-generative-ai"
level: advanced
prereqs: ["vae", "gaussian", "kl-divergence", "markov-chain", "elbo"]
interview_frequency: very-high
updated: 2026-06-20
---

# Diffusion Models — DDPM
> Destroy an image by adding Gaussian noise over `T` steps (the fixed **forward process**), then train
> a neural network to **reverse** it step by step, denoising pure noise back into a sample. **DDPM**
> shows the reverse step reduces to predicting the noise `ε` that was added, trained with a simple MSE
> loss — a beautifully stable objective that dethroned GANs for image synthesis.

**Why it matters:** the single most-asked generative-AI topic today — diffusion underlies Stable
Diffusion, DALL·E, Imagen, Sora. Interviews probe: the forward process and its **closed-form**
`x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε` (sample any `t` in one shot), why the reverse process is also
Gaussian for small steps, how the **variational bound** simplifies to the `‖ε − ε_θ(x_t, t)‖²`
training loss, and the precise relationship to VAEs (a diffusion model is a deep hierarchical VAE with
a fixed encoder). Be ready to contrast it with GANs: slower sampling, but stable training, mode
coverage, and exact-ish likelihood.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) — **Ari Seff**. *The clearest first picture of forward-noising and learned reverse-denoising.*
2. **See why it works** — watch [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) — **Outlier**. *Walks the forward/reverse process and the simplified loss visually.*
3. **Get the math** — read [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** + [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — **Calvin Luo**. *The full ELBO derivation down to the `ε`-prediction objective.*
4. **Read the source** — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — **Ho, Jain & Abbeel (2020)**. *The paper that made diffusion work; the simplified training objective.*
5. **Make it concrete** — implement it with [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** or [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion). *Coding the noise schedule + U-Net + loss cements it.*

## 🎓 Courses (free)
- [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first; build and sample a DDPM with the `diffusers` library.
- [DeepLearning.AI — How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) — **DeepLearning.AI** — free short course; implement a diffusion model from the ground up.

## 🎥 Videos
- [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) — **Ari Seff** — the best gentle first watch; forward-noising and learned reverse-denoising.
- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) — **Outlier** — the forward/reverse process and the simplified `ε`-prediction loss, beautifully animated.
- [DDPM — Diffusion Models Beat GANs (Paper Explained)](https://www.youtube.com/watch?v=W-O7AZNzbzQ) — **Yannic Kilcher** — a careful read of the DDPM line of work and why it overtook GANs.
- [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** — implements the noise schedule, U-Net, and training loop end to end.
- [Diffusion Models | PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU) — **Outlier** — the code companion to the math video; build DDPM step by step.

## 📄 Key Papers
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — **Ho, Jain & Abbeel (2020)** — the modern formulation; the simplified noise-prediction objective.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — **Nichol & Dhariwal (2021)** — learned variances, cosine schedule, and better likelihoods.

## 📰 Articles / Blogs (free, no paywall)
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng** — the canonical math walkthrough from forward process to the training loss.
- [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — **Calvin Luo** — the cleanest VAE→diffusion ELBO derivation; reads like lecture notes, fully open.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face** — runnable PyTorch + math side by side, free.

## 📚 Books (free, with chapters)
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 25 "Diffusion models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; the rigorous, modern textbook treatment.
- [Understanding Deep Learning — **Ch. 18 "Diffusion models"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF with clear figures and the DDPM derivation.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.03 Diffusion Models](../../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md) · [5.02 ELBO & VAEs](../../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md)
- Prereq: [01 Variational Autoencoders](01-Variational-Autoencoders-VAE-ELBO.md) (diffusion is a deep, fixed-encoder VAE)
- Next concepts: [06 Score-Based & SDE Diffusion](06-Score-Based-and-SDE-Diffusion.md) · [07 Latent Diffusion & Stable Diffusion](07-Latent-Diffusion-Stable-Diffusion.md) · [13 Sampling & Guidance Techniques](13-Sampling-and-Guidance-Techniques.md)
- Compare with: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) (adversarial, faster sampling, less stable)
- Field overview: [9. Generative AI](../README.md)

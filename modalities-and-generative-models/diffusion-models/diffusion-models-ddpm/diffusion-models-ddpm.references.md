---
id: "10-generative-ai/diffusion-ddpm/references"
topic: "Diffusion Models (DDPM) — References"
parent: "10-generative-ai/diffusion-ddpm"
type: references
updated: 2026-07-04
---

# Diffusion Models (DDPM) — references and further reading

> Companion link library for **[Diffusion Models (DDPM)](diffusion-models-ddpm.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on *DDPM specifically* (the forward process and its closed form, the variational bound down to the ε-prediction loss, the sampling update, and the VAE / score connections), not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch ⭐ [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) by **Ari Seff**. *The clearest first picture: forward-noising and learned reverse-denoising, with no equations in the way.*
2. **See why it works** — watch [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) by **Outlier**. *Walks the forward/reverse process, the closed form, and the simplified ε-prediction loss, beautifully animated.*
3. **Get the math** — read [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) by **Lilian Weng** and [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) by **Calvin Luo**. *The full ELBO derivation, VAE→diffusion, down to the ε-prediction objective and the score connection.*
4. **Read the source** — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by **Ho, Jain & Abbeel (2020)**. *The paper that made diffusion work; the simplified training objective this chapter derives.*
5. **Make it concrete** — run this chapter's [notebook](code/diffusion-models-ddpm.ipynb): a from-scratch DDPM on a real 2-D distribution, with the closed-form forward asserted equal to the iterative chain, and the trained model's samples asserted to match the target by energy distance.

**Videos**:
- [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) — **Ari Seff** — the best gentle first watch; forward-noising and learned reverse-denoising, clean and visual.
- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) — **Outlier** — the forward/reverse process and the simplified ε-prediction loss, animated; the ideal companion to this page's math.
- [DDPM — Diffusion Models Beat GANs (Paper Explained)](https://www.youtube.com/watch?v=W-O7AZNzbzQ) — **Yannic Kilcher** — a careful read of the DDPM line of work and why it overtook GANs.
- [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) — **DeepFindr** — implements the noise schedule, U-Net, and training loop end to end — the implementer's view of everything here.
- [Diffusion Models | PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU) — **Outlier** — the code companion to the math video; build a DDPM step by step.

**Courses (free)**:
- [Hugging Face — Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first: build and sample a DDPM, then scale to Stable Diffusion with the `diffusers` library.
- [DeepLearning.AI — How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) — **DeepLearning.AI (Sharon Zhou)** — free short course; implement a diffusion model from the ground up.
- [MIT 6.S191 — Deep Generative Modeling](https://introtodeeplearning.com/) — **MIT (Alexander Amini)** — places diffusion alongside VAEs and GANs in the generative-model landscape; slides + video free.
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Stefano Ermon)** — the score-based / diffusion lectures from the group that co-invented the score view; notes free online.

**Articles / blogs (free, no paywall)**:
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — **Lilian Weng (OpenAI)** — the canonical open math walkthrough from forward process to the training loss and beyond.
- [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — **Calvin Luo** — the cleanest VAE→diffusion ELBO derivation; reads like lecture notes, fully open.
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — **Hugging Face (Rogge & Rasul)** — runnable PyTorch + the DDPM math side by side, free.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — **Yang Song** — the score-based view of diffusion (the ε↔∇log p bridge) from its originator; the natural next step after this page.

**Key papers** (the source and the extensions):
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — **Ho, Jain & Abbeel (2020), NeurIPS** — the DDPM: the closed-form forward, the variational bound, and the simplified noise-prediction objective. The source for everything on this page.
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — **Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015), ICML** — the original diffusion generative model; framed generation as reversing a diffusion process.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — **Nichol & Dhariwal (2021), ICML** — the cosine schedule and learned variances; better likelihoods and samples.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — **Song & Ermon (2019), NeurIPS** — score matching / NCSN; the score view that DDPM's ε-prediction is (up to scale) an instance of.
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — **Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), ICLR** — unifies DDPM and score matching as discretizations of an SDE; the continuous-time picture.
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) — **Song, Meng & Ermon (2021), ICLR** — the deterministic, few-step sampler that fixes DDPM's slow sampling.

**Books (free chapters)**:
- [Understanding Deep Learning — Ch. 18 "Diffusion models"](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF with clear figures and a careful DDPM derivation.
- [Probabilistic Machine Learning: Advanced Topics — Ch. 25 "Diffusion models"](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — the rigorous, modern textbook treatment (free PDF).
- [Deep Learning — generative models chapter](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — the latent-variable and likelihood background under the ELBO diffusion reuses, free online.

**In this platform**:
- Concept page (full explanation): [Diffusion Models (DDPM)](diffusion-models-ddpm.md)
- Runnable code: [the module `ddpm.py`](code/ddpm.py) · [the step-by-step notebook](code/diffusion-models-ddpm.ipynb) — a from-scratch DDPM on a real 2-D distribution, with the closed-form forward asserted equal to the iterative chain (the "nice property"), the trained model's samples asserted to match the target by energy distance, and every figure (forward diffusion, noise schedule, reverse trajectory, generation overlay) from the same measured run; every number computed, none fabricated.
- The one-step seed: [01 Variational Autoencoders (VAE · ELBO)](../../generative-models/variational-autoencoders-vae-elbo/variational-autoencoders-vae-elbo.md) — a diffusion model is a many-step hierarchical VAE with a *fixed* encoder; the ELBO, the reparameterization trick, and the Gaussian KL all come from here.
- The KL in every loss term: [01 Foundations · 23 Cross-Entropy and KL Divergence](../../../foundations/mathematical-foundations/cross-entropy-and-kl-divergence/cross-entropy-and-kl-divergence.md) — where the two-Gaussian KL (each per-step L_{t-1}) is derived; diffusion specializes it to two Gaussians of equal variance.
- The same ELBO, another view: [04 Unsupervised Learning · 04 Gaussian Mixture Models & EM](../../../core-machine-learning/unsupervised-learning/clustering/gaussian-mixture-models-and-em/gaussian-mixture-models-and-em.md) — the variational lower bound and its KL-gap identity that the diffusion ELBO extends over a T-step chain.
- The gradient engine: [05 Deep Learning · 02 Backpropagation and Computational Graphs](../../../deep-learning/neural-network-foundations/backpropagation-and-computational-graphs/backpropagation-and-computational-graphs.md) — the reparameterized forward (x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε) exists so ordinary backprop can train the denoiser.
- Where it goes next: [06 Score-Based & SDE Diffusion](../score-based-and-sde-diffusion/score-based-and-sde-diffusion.md) — predicting noise is learning the score (ε_θ ≈ −√(1−ᾱ_t) ∇log q_t); the continuous-time SDE picture that unifies DDPM and score matching.
- Where it lives in the real world: [07 Latent Diffusion / Stable Diffusion](../latent-diffusion-stable-diffusion/latent-diffusion-stable-diffusion.md) — run this exact DDPM in a pretrained VAE's latent space; the engine of Stable Diffusion.
- Steering it: [04 Conditional Generation & Classifier-Free Guidance](../conditional-generation-and-classifier-free-guidance/conditional-generation-and-classifier-free-guidance.md) — condition the denoiser on text and steer sampling; how text-to-image works on top of the DDPM loop.
- Fixing the slow sampling: [13 Sampling & Guidance Techniques](../sampling-and-guidance-techniques/sampling-and-guidance-techniques.md) — DDIM and fast samplers that cut the T-step chain to tens of steps.
- The family contrast: [02 GANs and DCGAN](../../generative-models/gans-and-dcgan/gans-and-dcgan.md) — adversarial, fast to sample, unstable; diffusion is likelihood-based, stable, sharp, but slow to sample.
- Field overview: [10 Generative AI](../../generative-models/README.md)

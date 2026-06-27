---
id: "10-generative-ai/normalizing-flows"
topic: "Normalizing Flows"
parent: "10-generative-ai"
level: advanced
prereqs: ["change-of-variables", "jacobian", "maximum-likelihood", "neural-networks"]
interview_frequency: medium
updated: 2026-06-20
---

# Normalizing Flows — RealNVP · Glow
> Build a flexible distribution by pushing a simple base (e.g. `N(0,I)`) through a chain of
> **invertible, differentiable** transformations. The **change-of-variables** formula gives the exact
> density: `log p(x) = log p(z) + log|det J|`. The trick is designing layers whose Jacobian
> determinant is cheap — **coupling layers** (RealNVP) give a triangular Jacobian; **Glow** adds
> invertible 1×1 convolutions. Unlike VAEs and GANs, flows give **exact likelihoods** and **exact
> inversion**.

**Why it matters:** the clean "exact-likelihood generative model" and the home of the
change-of-variables question. Interviews probe: why invertibility + a tractable Jacobian determinant
are *both* required, how **affine coupling** splits dimensions so half pass through unchanged (making
the Jacobian triangular and the determinant a product of diagonal terms), why flows preserve
dimensionality (no bottleneck), and the trade-offs vs VAEs/GANs (exact likelihood and invertibility,
but architectural constraints and high parameter cost). Flows also reappear inside diffusion as the
probability-flow ODE.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [What are Normalizing Flows?](https://www.youtube.com/watch?v=i7LjDvsLWCg) — **Ari Seff**. *The clearest first picture: warp a simple density into a complex one, invertibly.*
2. **See why it works** — read [Flow-based Deep Generative Models (Lil'Log)](https://lilianweng.github.io/posts/2018-10-13-flow-models/) — **Lilian Weng**. *Change of variables, coupling layers, RealNVP, and Glow in one place.*
3. **Get the math** — watch [Normalizing Flows — Motivations, The Big Idea & Essential Foundations](https://www.youtube.com/watch?v=IuXU2dBOJyw) — **Kapil Sachdeva** + read [Going with the Flow](https://gebob19.github.io/normalizing-flows/) — **Brennan Gebotys**. *The Jacobian-determinant trick worked through carefully.*
4. **Read the sources** — [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) — **Dinh et al. (2017)** → [Glow](https://arxiv.org/abs/1807.03039) — **Kingma & Dhariwal (2018)**. *Coupling layers, then invertible 1×1 convolutions.*
5. **Make it concrete** — work through the [UvA DL — Normalizing Flows for image modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html). *Coding a coupling-layer flow and computing the exact NLL cements it.*

## 🎓 Courses (free)
- [UC Berkeley CS294-158 — Deep Unsupervised Learning (Flow Models)](https://sites.google.com/view/berkeley-cs294-158-sp20/home) — **Berkeley (Abbeel)** — free lectures + slides; the canonical flow-models lecture.
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; the normalizing-flows lecture with the change-of-variables derivation.

## 🎥 Videos
- [What are Normalizing Flows?](https://www.youtube.com/watch?v=i7LjDvsLWCg) — **Ari Seff** — the best gentle first watch; invertible warping of a simple density.
- [Normalizing Flows — Motivations, The Big Idea & Essential Foundations](https://www.youtube.com/watch?v=IuXU2dBOJyw) — **Kapil Sachdeva** — the change-of-variables and Jacobian-determinant math, carefully.
- [L3 Flow Models — CS294-158 Deep Unsupervised Learning](https://www.youtube.com/watch?v=JBb5sSC0JoY) — **Pieter Abbeel (Berkeley)** — the rigorous lecture: coupling layers, RealNVP, Glow.
- [Introduction to Normalizing Flows (ECCV 2020 Tutorial)](https://www.youtube.com/watch?v=u3vVyFVU_lI) — **Marcus Brubaker** — a thorough survey-style tutorial across flow architectures.

## 📄 Key Papers
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) — **Dinh, Sohl-Dickstein & Bengio (2017)** — affine coupling layers with a tractable triangular Jacobian.
- [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039) — **Kingma & Dhariwal (2018)** — invertible 1×1 convs + actnorm for high-quality image flows.
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) — **Papamakarios et al. (2021)** — the definitive review unifying the flow zoo.

## 📰 Articles / Blogs (free, no paywall)
- [Flow-based Deep Generative Models (Lil'Log)](https://lilianweng.github.io/posts/2018-10-13-flow-models/) — **Lilian Weng** — the canonical math walkthrough: change of variables → RealNVP → Glow.
- [Going with the Flow: An Introduction to Normalizing Flows](https://gebob19.github.io/normalizing-flows/) — **Brennan Gebotys** — derivation plus code; clear and fully open.
- [UvA DL — Normalizing Flows for image modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html) — **University of Amsterdam** — runnable notebook with the exact-NLL objective.

## 📚 Books (free, with chapters)
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 23 "Normalizing flows"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; the rigorous, unified treatment.
- [Understanding Deep Learning — **Ch. 16 "Normalizing flows"**](https://udlbook.github.io/udlbook/) — **Simon Prince** — free PDF; coupling layers and the Jacobian trick with clean figures.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md) · [5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Compare with: [01 Variational Autoencoders](../01-Variational-Autoencoders-VAE-ELBO/01-Variational-Autoencoders-VAE-ELBO.md) (lower-bound likelihood) · [06 Score-Based & SDE Diffusion](../06-Score-Based-and-SDE-Diffusion/06-Score-Based-and-SDE-Diffusion.md) (the probability-flow ODE is a continuous flow)
- Related: [10 Autoregressive Image Generation (PixelCNN)](../10-Autoregressive-Image-Generation-PixelCNN/10-Autoregressive-Image-Generation-PixelCNN.md) (another exact-likelihood model)
- Field overview: [9. Generative AI](../README.md)

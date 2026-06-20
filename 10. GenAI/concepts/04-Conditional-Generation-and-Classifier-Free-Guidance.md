---
id: "10-generative-ai/conditional-cfg"
topic: "Conditional Generation & Classifier-Free Guidance"
parent: "10-generative-ai"
level: advanced
prereqs: ["gans-dcgan", "diffusion-ddpm", "bayes-rule", "cross-entropy"]
interview_frequency: high
updated: 2026-06-20
---

# Conditional Generation & Classifier-Free Guidance
> Make a generator obey a **condition** `y` (a class label, a text prompt, an input image). For GANs
> this is **cGAN** — feed `y` to both generator and discriminator (pix2pix conditions on a whole
> image). For diffusion, the dominant tool is **Classifier-Free Guidance (CFG)**: jointly train a
> conditional and an unconditional model (by randomly dropping `y`), then at sampling time extrapolate
> `ε̃ = ε_uncond + s·(ε_cond − ε_uncond)` to *amplify* the condition — the knob behind every prompt's
> "guidance scale."

**Why it matters:** the bridge from "generate something" to "generate *this*," and a hot interview
topic because CFG powers Stable Diffusion, DALL·E 2, and Imagen. Expect: how a cGAN conditions both
networks; **classifier guidance** vs **classifier-free guidance** (and why the classifier-free version
won — no separate, noise-robust classifier to train); the Bayes-rule derivation
(`∇log p(x|y) = ∇log p(x) + ∇log p(y|x)`) that links the two; and the guidance-scale trade-off —
higher `s` gives better prompt adherence but less diversity and possible artifacts.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Diffusion models explained: how does GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) — **AI Coffee Break (Letitia)**. *Shows conditioning + guidance in the system that introduced CFG to text-to-image.*
2. **See why it works** — read [Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html) — **Sander Dieleman**. *The clearest account of what guidance does and why scaling it trades diversity for fidelity.*
3. **Get the math** — read [Conditional Image Generation with Classifier-Free Guidance](https://www.peterholderrieth.com/blog/2023/Classifier-Free-Guidance-For-Diffusion-Models/) — **Peter Holderrieth (MIT)**. *The training (condition-dropout) and the sampling extrapolation formula, derived from Bayes' rule.*
4. **Read the sources** — [Conditional GANs](https://arxiv.org/abs/1411.1784) — **Mirza & Osindero (2014)** · [Diffusion Models Beat GANs (classifier guidance)](https://arxiv.org/abs/2105.05233) — **Dhariwal & Nichol (2021)** · [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — **Ho & Salimans (2022)**. *cGAN → classifier guidance → the classifier-free version everyone uses.*
5. **Make it concrete** — implement a cGAN/pix2pix with the [TensorFlow pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix). *Conditioning a generator on an image makes "conditioning" concrete before the diffusion version.*

## 🎓 Courses (free)
- [Hugging Face — Diffusion Models Course (conditioning & guidance unit)](https://huggingface.co/learn/diffusion-course/unit0/1) — **Hugging Face** — free, code-first; covers class conditioning and CFG with runnable notebooks.
- [Google — Generative Adversarial Networks](https://developers.google.com/machine-learning/gan) — **Google** — free course; the conditional-GAN section shows label-conditioned generation.

## 🎥 Videos
- [Diffusion models explained: how does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) — **AI Coffee Break (Letitia)** — text conditioning and guidance in the model that popularized CFG.
- [How AI Image Generators Work (Stable Diffusion / DALL·E)](https://www.youtube.com/watch?v=1CIpzeNxIhU) — **Computerphile** — how a text prompt steers generation, in plain language.
- [Pix2Pix Paper Walkthrough](https://www.youtube.com/watch?v=9SGs4Nm0VR4) — **Aladdin Persson** — the canonical conditional GAN: conditioning both networks on an input image.
- [Pix2Pix implementation from scratch](https://www.youtube.com/watch?v=SuddDSqGRzg) — **Aladdin Persson** — codes the conditional generator/discriminator and paired-image loss in PyTorch.

## 📄 Key Papers
- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — **Mirza & Osindero (2014)** — the cGAN: condition both `G` and `D` on a label `y`.
- [Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)](https://arxiv.org/abs/1611.07004) — **Isola et al. (2017)** — conditioning on a full image; the PatchGAN discriminator.
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — **Dhariwal & Nichol (2021)** — **classifier guidance**: steer sampling with a noise-aware classifier's gradient.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — **Ho & Salimans (2022)** — drop the classifier; train one model with condition-dropout and extrapolate at sampling.

## 📰 Articles / Blogs (free, no paywall)
- [Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html) — **Sander Dieleman** — the definitive intuition for what guidance does and the fidelity/diversity trade-off.
- [Conditional Image Generation with Classifier-Free Guidance](https://www.peterholderrieth.com/blog/2023/Classifier-Free-Guidance-For-Diffusion-Models/) — **Peter Holderrieth (MIT)** — the training trick and sampling formula, derived from Bayes' rule, free.
- [pix2pix: Image-to-image translation with a conditional GAN](https://www.tensorflow.org/tutorials/generative/pix2pix) — **TensorFlow** — end-to-end conditional GAN you can run.

## 📚 Books (free, with chapters)
- [Deep Learning — **§20.10.4 (GANs) & §3.9–3.13 (probability / Bayes)**](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — conditional generation and the Bayes background under guidance.
- [Dive into Deep Learning — **Ch. 20 "Generative Adversarial Networks"**](https://d2l.ai/chapter_generative-adversarial-networks/index.html) — **Zhang et al.** — free, with conditional-GAN code to adapt.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.03 Diffusion Models](../../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md) · [5.04 GANs & WGAN](../../../AI-ML-intuition/Module_5_Generation/5.04_GANs_and_WGAN.md)
- Prereq: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) · [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md)
- Next concepts: [07 Latent Diffusion & Stable Diffusion](07-Latent-Diffusion-Stable-Diffusion.md) · [11 Text-to-Image Systems](11-Text-to-Image-Systems.md) · [13 Sampling & Guidance Techniques](13-Sampling-and-Guidance-Techniques.md)
- Field overview: [9. Generative AI](../README.md)

---
id: "09-generative-ai/autoregressive-pixelcnn"
topic: "Autoregressive Image Generation (PixelCNN)"
parent: "09-generative-ai"
level: advanced
prereqs: ["cnns", "softmax", "chain-rule-probability", "maximum-likelihood"]
interview_frequency: low
updated: 2026-06-20
---

# Autoregressive Image Generation — PixelRNN · PixelCNN
> Treat an image as a sequence of pixels and model it the way a language model models text: factor the
> joint as a product of conditionals `p(x) = ∏ p(x_i | x_<i)` and predict each pixel from all the ones
> "before" it (raster order). **PixelRNN** uses recurrence; **PixelCNN** uses **masked convolutions**
> so a pixel never sees future pixels. Exact likelihoods, sharp samples — but generation is slow
> (one pixel at a time).

**Why it matters:** the image analogue of autoregressive language models, and a clean exact-likelihood
baseline. Interviews probe: the **chain-rule factorization** and the raster-scan ordering; how a
**causal mask** on the convolution kernel enforces "no peeking ahead" (and the **blind-spot** problem
that gated PixelCNN fixes with horizontal/vertical stacks); why output is a **softmax over 256
intensities** rather than a regression; and the core trade-off — exact likelihood and stable training,
but `O(H·W)` sequential sampling. It also seeds VQ-VAE / DALL·E-1, which run autoregressive models
over *discrete latent codes* instead of raw pixels.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [PixelCNN for generative modeling explained](https://www.youtube.com/watch?v=tLb9m0m91QU) — **Data Science in your pocket**. *Pixel-by-pixel generation and the masking idea, plainly.*
2. **See why it works** — read [Blog: Autoregressive Models — PixelCNN](https://bjlkeng.io/posts/pixelcnn/) — **Brian Keng**. *Masked convolutions, the blind spot, and the gated fix, with code.*
3. **Get the math** — watch [L2 Autoregressive Models — CS294-158](https://www.youtube.com/watch?v=iyEOk8KCRUw) — **Pieter Abbeel (Berkeley)**. *MADE → WaveNet → PixelCNN(++); the chain-rule factorization and masking, rigorously.*
4. **Read the sources** — [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) — **van den Oord et al. (2016)** → [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328) — **van den Oord et al. (2016)**. *PixelRNN/PixelCNN, then the gated, conditional version.*
5. **Make it concrete** — work through [UvA DL — Autoregressive Image Modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html). *Coding masked convs and sampling pixel-by-pixel cements it.*

## 🎓 Courses (free)
- [UC Berkeley CS294-158 — Deep Unsupervised Learning (Autoregressive Models)](https://sites.google.com/view/berkeley-cs294-158-sp20/home) — **Berkeley (Abbeel)** — free; the canonical autoregressive-models lecture (MADE, WaveNet, PixelCNN).
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; the autoregressive-models lecture with the chain-rule view.

## 🎥 Videos
- [PixelCNN for generative modeling explained](https://www.youtube.com/watch?v=tLb9m0m91QU) — **Data Science in your pocket** — gentle intro to pixel-by-pixel generation and masking.
- [L2 Autoregressive Models — CS294-158 Deep Unsupervised Learning](https://www.youtube.com/watch?v=iyEOk8KCRUw) — **Pieter Abbeel (Berkeley)** — the rigorous lecture: MADE, WaveNet, PixelCNN(++), self-attention.
- [Autoregressive Generative Models with Deep Learning](https://www.youtube.com/watch?v=R8fx2b8Asg0) — **Hugo Larochelle (Google Brain)** — the foundations talk (NADE/MADE) by an originator of neural AR models.
- [Lecture 13 — Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — **Stanford CS231n** — places PixelRNN/PixelCNN alongside VAEs and GANs in the generative-models lecture.

## 📄 Key Papers
- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) — **van den Oord, Kalchbrenner & Kavukcuoglu (2016)** — PixelRNN/PixelCNN; the autoregressive image-modeling formulation.
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328) — **van den Oord et al. (2016)** — gated PixelCNN that fixes the blind spot and adds conditioning.
- [PixelCNN++](https://arxiv.org/abs/1701.05517) — **Salimans et al. (2017)** — a discretized-logistic-mixture likelihood and other improvements.

## 📰 Articles / Blogs (free, no paywall)
- [Autoregressive Models — PixelCNN](https://bjlkeng.io/posts/pixelcnn/) — **Brian Keng** — masked convolutions, the blind-spot problem, and the gated fix, with runnable code.
- [PixelCNN](https://sergeiturukin.com/2017/02/22/pixelcnn.html) — **Sergei Turukin** — a focused walkthrough of the masking scheme and the architecture.
- [UvA DL — Autoregressive Image Modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html) — **University of Amsterdam** — runnable PixelCNN notebook with the exact-NLL objective.

## 📚 Books (free, with chapters)
- [Deep Learning — **§20.10.7 "Other Generation Schemes" & §10 (recurrent/seq models)**](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — autoregressive generation in context, free online.
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 22 "Autoregressive models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; the modern treatment of AR generative models.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)
- Prereq: [Deep Learning — CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md) (masked convolutions are the core trick)
- Related (text analogue): [LLMs](../../08.%20LLMs/concepts/README.md) · [NLP — Decoding Strategies](../../06.%20NLP/concepts/17-Decoding-Strategies.md) (sampling from autoregressive conditionals)
- Compare with: [08 Normalizing Flows](08-Normalizing-Flows.md) (another exact-likelihood model) · [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md)
- Field overview: [9. Generative AI](../README.md)

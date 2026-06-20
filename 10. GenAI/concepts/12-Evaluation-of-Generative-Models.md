---
id: "10-generative-ai/evaluation"
topic: "Evaluation of Generative Models"
parent: "10-generative-ai"
level: advanced
prereqs: ["gans-dcgan", "multivariate-gaussian", "kl-divergence", "cnns"]
interview_frequency: high
updated: 2026-06-20
---

# Evaluation of Generative Models — FID · Inception Score
> Likelihood doesn't exist (GANs) or doesn't correlate with sample quality, so we evaluate generative
> models with **feature-space statistics**. **Inception Score (IS)** rewards confident, diverse
> ImageNet classifications. **Fréchet Inception Distance (FID)** fits Gaussians to Inception features
> of real vs. generated images and measures the Fréchet distance between them — lower is better, and
> it captures *both* fidelity and diversity. **Precision/Recall** disentangles the two.

**Why it matters:** "how do you know your generator is good?" is asked of every GAN/diffusion
candidate. Interviews probe: why we can't just use log-likelihood; how IS works and its flaws (no
comparison to real data, blind to mode collapse within a class, ImageNet-biased); the **FID** formula
`‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)` and why it's the field standard (sensitive to mode
collapse and blur, but biased by sample size); the **precision (fidelity) vs. recall (coverage)** split
that a single number hides; and **CLIP-score** for text-to-image prompt alignment.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Fréchet Inception Distance and Inception Score](https://www.youtube.com/watch?v=mlk3JA2aJ-g) — **AI Bits and Pieces**. *What the two metrics measure, in a few minutes.*
2. **See why it works** — read [How to Implement the Fréchet Inception Distance (FID)](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) — **Jason Brownlee**. *The Inception-feature Gaussians and the Fréchet distance, with code.*
3. **Get the math** — watch [Evaluating GANs with Inception Score and FID](https://www.youtube.com/watch?v=eGbhEDDb8NA) — **Giuseppe Canale** + read [How to Evaluate GANs using FID](https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTc) — **Weights & Biases**. *The IS KL-divergence and the FID formula derived.*
4. **Read the sources** — [Improved Techniques for Training GANs (IS)](https://arxiv.org/abs/1606.03498) → [GANs Trained by a Two Time-Scale Update Rule (FID)](https://arxiv.org/abs/1706.08500) → [Improved Precision and Recall Metric](https://arxiv.org/abs/1904.06991). *IS, then FID, then the precision/recall refinement.*
5. **Make it concrete** — compute FID with [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) on your own samples. *Watching FID drop as samples improve makes it tangible.*

## 🎓 Courses (free)
- [Google — GANs: Evaluation & Common Problems](https://developers.google.com/machine-learning/gan/problems) — **Google** — free; FID/IS and the failure modes they detect (mode collapse, blur).
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) — **Stanford (Ermon)** — free notes; the evaluation lecture covers likelihood-free metrics.

## 🎥 Videos
- [Fréchet Inception Distance and Inception Score](https://www.youtube.com/watch?v=mlk3JA2aJ-g) — **AI Bits and Pieces** — a quick, clear intro to what IS and FID measure.
- [Evaluating GANs with Inception Score and FID](https://www.youtube.com/watch?v=eGbhEDDb8NA) — **Giuseppe Canale** — the metrics worked through with the Inception network in the loop.
- [DDPM — Diffusion Models Beat GANs on Image Synthesis (Paper Explained)](https://www.youtube.com/watch?v=W-O7AZNzbzQ) — **Yannic Kilcher** — uses FID as the head-to-head yardstick; shows how the field compares models.
- [Lecture 13 — Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — **Stanford CS231n** — frames why generative-model evaluation is hard and likelihood-free metrics arise.

## 📄 Key Papers
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) — **Salimans et al. (2016)** — introduces the **Inception Score**.
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — **Heusel et al. (2017)** — introduces **FID** (and TTUR).
- [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) — **Kynkäänniemi et al. (2019)** — separates fidelity (precision) from coverage (recall).

## 📰 Articles / Blogs (free, no paywall)
- [How to Implement the Fréchet Inception Distance (FID) from Scratch](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) — **Jason Brownlee** — the FID computation with runnable code.
- [How to Evaluate GANs using Fréchet Inception Distance (FID)](https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTc) — **Weights & Biases** — IS and FID side by side, with plots, free.
- [FID: Fréchet Inception Distance](https://strikingloo.github.io/wiki/fid) — **Luciano Strika** — a concise, math-first explainer of the FID formula.

## 📚 Books (free, with chapters)
- [Probabilistic Machine Learning: Advanced Topics — **Ch. 26 "Evaluating generative models"**](https://probml.github.io/pml-book/book2.html) — **Kevin Murphy** — free PDF; likelihood-free metrics and their pitfalls.
- [Deep Learning — **§20.14 "Evaluating Generative Models"**](https://www.deeplearningbook.org/contents/generative_models.html) — **Goodfellow, Bengio & Courville** — why evaluation is hard and what the metrics measure, free online.

## 🔗 In this platform
- Prereq: [02 GANs & DCGAN](02-GANs-and-DCGAN.md) · [05 Diffusion Models (DDPM)](05-Diffusion-Models-DDPM.md) (the models you evaluate)
- Related: [03 GAN Training & WGAN](03-GAN-Training-and-WGAN.md) (FID/precision-recall detect mode collapse)
- Compare with: [NLP — Evaluation Metrics (BLEU · perplexity)](../../06.%20NLP/concepts/18-NLP-Evaluation-Metrics.md) (the text-generation analogue)
- Field overview: [9. Generative AI](../README.md)

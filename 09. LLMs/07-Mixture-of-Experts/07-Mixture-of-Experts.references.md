---
id: "09-llms/mixture-of-experts/references"
topic: "Mixture-of-Experts — References"
parent: "09-llms/mixture-of-experts"
type: references
updated: 2026-06-22
---

# Mixture-of-Experts — references and further reading

> Companion link library for **[Mixture-of-Experts](07-Mixture-of-Experts.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build the intuition** — watch [Mixture of Experts (MoE), Visually Explained](https://www.youtube.com/watch?v=0QQlYR1r6pQ) (**Jia-Bin Huang**). *Routers, experts, and sparse activation made visual — the cleanest first picture.*
2. **Read the visual guide** — [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) (**Maarten Grootendorst**). *The clearest illustrated walkthrough of routing and load balancing.*
3. **Read the definitive blog** — [Mixture of Experts Explained](https://huggingface.co/blog/moe) (**Hugging Face**). *Routing, the balance loss, training instabilities, fine-tuning, and serving in one place.*
4. **Read the source** — [Switch Transformers](https://arxiv.org/abs/2101.03961) (**Fedus et al.**). *Top-1 routing, the capacity factor, and scaling to a trillion parameters.*
5. **See the frontier** — [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (**Dai et al.**). *Fine-grained + shared experts — the current state-of-the-art recipe.*

**Videos**:
- [Mixture of Experts (MoE), Visually Explained](https://www.youtube.com/watch?v=0QQlYR1r6pQ) — **Jia-Bin Huang** — the cleanest visual intuition for routers, experts, and sparsity.
- [Understanding Mixture of Experts](https://www.youtube.com/watch?v=0U_65fLoTq0) — **Trelis Research** — architecture walkthrough that connects MoE to real MoE LLMs and the key papers.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the dense FFN block that MoE replaces, visualized.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where the capacity-vs-compute trade-off sits in the big LLM picture.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (architectures & scaling)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — MoE within the modern architecture/scaling toolkit.
- [Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe) — **Hugging Face** — a thorough, course-grade treatment (routing, balance, training, serving).

**Articles / blogs (free, no paywall)**:
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — **Maarten Grootendorst** — the best illustrated explainer of routing + load balancing.
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) — **Hugging Face** — the definitive free deep-dive; covers the balance loss, capacity, instabilities, and serving.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng (OpenAI)** — MoE situated among transformer variants, with the routing math.
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) — **Gale et al.** — block-sparse "dropless" MoE that avoids token dropping and padding.

**Key papers**:
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — **Shazeer et al. (2017)** — the modern MoE layer: noisy top-k gating + the load-balancing loss. The origin.
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) — **Lepikhin et al. (2020)** — top-2 routing, expert capacity, and expert parallelism at 600B.
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — **Fedus et al. (2021)** — top-1 routing, capacity-factor analysis, simplicity at 1.6T params.
- [GLaM: Efficient Scaling of Language Models with MoE](https://arxiv.org/abs/2112.06905) — **Du et al. (2022)** — a 1.2T MoE LLM matching GPT-3 at a fraction of the training energy.
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) — **Zoph et al. (2022)** — the router **z-loss** and the recipe that made large MoE training stable.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — **Jiang et al. (2024)** — open-weights top-2-of-8; the 47B-total / 13B-active flagship.
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) — **Dai et al. (2024)** — fine-grained experts + shared always-on experts.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — **DeepSeek-AI (2024)** — 671B-total / 37B-active MoE with auxiliary-loss-free load balancing.
- [Sparse Upcycling: Training MoE from Dense Checkpoints](https://arxiv.org/abs/2212.05055) — **Komatsuzaki et al. (2022)** — initialize an MoE from a trained dense model.
- [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667) — **Fedus, Dean & Zoph (2022)** — the authoritative survey of MoE methods and history.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — **Kaplan et al. (2020)** — the $\approx 2{\cdot}\text{params}$ forward-pass FLOP rule behind MoE's active-vs-total compute accounting.

**Books (free chapters)**:
- [Dive into Deep Learning — Attention & Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — the transformer block (and its FFN) that the MoE layer modifies.
- [Speech and Language Processing, 3rd ed. — Large Language Models](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the LLM architecture context MoE plugs into.

**In this platform**:
- Concept page (full explanation): [Mixture-of-Experts](07-Mixture-of-Experts.md)
- Foundations (the FFN MoE replaces): [Transformer Architecture](../../05.%20Deep_Learning/16-Transformer-Architecture/16-Transformer-Architecture.md)
- Why capacity matters: [Scaling Laws](../03-Scaling-Laws/03-Scaling-Laws.md) · [Pretraining at Scale](../02-Pretraining-at-Scale/02-Pretraining-at-Scale.md)
- Builds on this: [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md) · [KV Cache](../05-KV-Cache/05-KV-Cache.md) · [Quantization](../10-Quantization/10-Quantization.md)
- Puts it to work: [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)
- Concept depth (the *why*): [Neural Scaling Laws / Chinchilla](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.01_Neural_Scaling_Laws_Chinchilla.md)

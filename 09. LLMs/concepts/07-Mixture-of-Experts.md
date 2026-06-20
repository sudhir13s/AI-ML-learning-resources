---
id: "09-llms/mixture-of-experts"
topic: "Mixture-of-Experts (MoE)"
parent: "09-llms"
level: advanced
prereqs: ["decoder-only-architecture", "softmax", "scaling-laws"]
interview_frequency: high
updated: 2026-06-20
---

# Mixture-of-Experts (MoE)
> Replace the dense feed-forward block with **many** expert FFNs and a router that sends each token to
> just a few (top-k). You get a model with huge total parameter count but small *active* compute per
> token — decoupling capacity from FLOPs. The architecture behind Mixtral, GShard/Switch, and many
> frontier models.

**Why it matters:** an increasingly common interview topic. Expect to explain sparse vs dense compute,
the gating/router and top-k routing, load-balancing (auxiliary) loss, the all-to-all communication
cost, and the memory-vs-FLOPs trade-off (must hold all experts in memory, but only run a few).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Mixture of Experts (MoE), Visually Explained](https://www.youtube.com/watch?v=0QQlYR1r6pQ). *Routers, experts, and sparse activation made visual.*
2. **Read the visual guide** — [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts). *The clearest written walkthrough of routing and load balancing.*
3. **Go deeper** — [Understanding Mixture of Experts](https://www.youtube.com/watch?v=0U_65fLoTq0) (Trelis). *Connects the architecture to real MoE LLMs.*
4. **Read the sources** — [Switch Transformers](https://arxiv.org/abs/2101.03961). *Simplifies routing to top-1 and scales to trillions of params.*
5. **Read the survey** — [HF: Mixture of Experts Explained](https://huggingface.co/blog/moe). *Training instabilities, fine-tuning, and serving considerations.*

## 🎓 Courses (free)
- [Stanford CS336 — Architectures & scaling](https://stanford-cs336.github.io/spring2025/) — **Stanford** — MoE within the modern architecture/scaling toolkit.
- [Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe) — **Hugging Face** — a thorough, course-grade treatment (routing, balance, training, serving).

## 🎥 Videos
- [Mixture of Experts (MoE), Visually Explained](https://www.youtube.com/watch?v=0QQlYR1r6pQ) — **Jia-Bin Huang** — the cleanest visual intuition.
- [Understanding Mixture of Experts](https://www.youtube.com/watch?v=0U_65fLoTq0) — **Trelis Research** — architecture walkthrough with the key papers.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the dense FFN block MoE replaces.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where capacity vs compute trade-offs sit in the big picture.

## 📄 Key Papers
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — **Fedus et al. (2021)** — simplified top-1 routing; capacity at constant FLOPs.
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — **Shazeer et al. (2017)** — the modern MoE layer + load-balancing loss.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — situates MoE among scaling strategies.

## 📰 Articles / Blogs (free, no paywall)
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — **Maarten Grootendorst** — the best illustrated explainer of routing + balance.
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) — **Hugging Face** — definitive free deep-dive.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng** — MoE among transformer variants.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 11 "Attention & Transformers"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — the transformer block MoE modifies.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — book-length free reference covering sparse architectures.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.01 Neural Scaling Laws / Chinchilla](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.01_Neural_Scaling_Laws_Chinchilla.md)
- Foundations (covered elsewhere): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Related concepts: [Scaling Laws](03-Scaling-Laws.md) · [Decoder-only Architecture](04-Decoder-only-Architecture.md) · [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)

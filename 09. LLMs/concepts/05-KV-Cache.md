---
id: "09-llms/kv-cache"
topic: "KV Cache"
parent: "09-llms"
level: advanced
prereqs: ["decoder-only-architecture", "attention"]
interview_frequency: very-high
updated: 2026-06-20
---

# KV Cache
> The single most important LLM inference optimization. During autoregressive decoding, the keys and
> values of past tokens never change — so cache them instead of recomputing. This turns generation
> from O(n²) per step into O(n), at the cost of memory that grows linearly with sequence length.

**Why it matters:** *the* LLM-serving interview question. Be ready to explain why only K and V are
cached (not Q), compute cache size (`2 · layers · heads · head_dim · seq · batch · bytes`), and connect
it to the memory pressure that motivates GQA/MQA, paged attention, and quantized caches.

**⭐ Start here — suggested path:**

1. **Get the core idea** — watch [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4). *Exactly what's cached, why, and the memory cost — the best single explainer.*
2. **See why it's needed** — [How a Transformer works at inference vs training](https://www.youtube.com/watch?v=IGu7ivuy1Ag). *Makes the redundant recomputation obvious.*
3. **Do the math** — [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). *KV-cache memory and bandwidth as the real decode bottleneck.*
4. **See the optimization** — [Efficient Inference (KV cache section)](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/). *MQA/GQA and cache compression that follow directly.*
5. **Connect to serving** — read about [paged attention](09-Inference-Optimization-and-Serving.md). *How real engines manage KV-cache memory without fragmentation.*

## 🎓 Courses (free)
- [Stanford CS336 — Inference & systems](https://stanford-cs336.github.io/spring2025/) — **Stanford** — KV cache within the LLM inference stack.
- [Hugging Face — LLM inference optimization docs](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — KV caching, static cache, and how to use it in practice.

## 🎥 Videos
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) — **Efficient NLP** — the definitive concise explainer with the memory math.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge** — why each decode step would otherwise recompute the past.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — generation loop where a KV cache plugs in.
- [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw) — **AI Makerspace** — how serving systems manage the KV cache at scale.

## 📄 Key Papers
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — **Shazeer (2019)** — multi-query attention to shrink the KV cache.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — grouped-query attention, the KV-cache sweet spot used by LLaMA-2/3.
- [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023)** — paging the KV cache like OS virtual memory.

## 📰 Articles / Blogs (free, no paywall)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **Kipply** — KV-cache memory/bandwidth as the decode bottleneck.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng** — KV cache, MQA/GQA, and cache compression.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — memory accounting that includes the KV cache.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive decoding (the loop the cache accelerates).

## 🔗 In this platform
- Foundations (covered elsewhere): [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Related concepts: [Decoder-only Architecture](04-Decoder-only-Architecture.md) · [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md) · [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)

---
id: "09-llms/kv-cache/references"
topic: "KV Cache — References"
parent: "09-llms/kv-cache"
type: references
updated: 2026-06-21
---

# KV Cache — references and further reading

> Companion link library for **[KV Cache](05-KV-Cache.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Get the core idea** — watch [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) (**Efficient NLP**). *Exactly what's cached, why, and the memory cost — the best concise explainer.*
2. **See why it's needed** — watch [How a Transformer works at inference vs training](https://www.youtube.com/watch?v=IGu7ivuy1Ag) (**Niels Rogge**). *Makes the redundant recomputation obvious.*
3. **Do the math** — read [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (**Kipply**). *KV-cache memory and bandwidth as the real decode bottleneck.*
4. **See the optimizations** — read [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (**Lilian Weng**). *MQA/GQA and cache compression that follow directly.*
5. **Connect to serving** — read the [vLLM / PagedAttention paper](https://arxiv.org/abs/2309.06180) (**Kwon et al.**). *How real engines page the cache to serve at scale.*

**Videos**:
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) — **Efficient NLP** — the definitive concise explainer, with the memory math.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the most visual explanation of Q, K, V anywhere; watch this first if "key" and "value" still feel abstract.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge (Hugging Face)** — why each decode step would otherwise recompute the past.
- [Coding LLaMA 2 from scratch — KV cache, GQA, RoPE](https://www.youtube.com/watch?v=oM4VmoabDAI) — **Umar Jamil** — builds the KV cache and grouped-query attention line by line in PyTorch.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — the generation loop a KV cache plugs into.
- [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw) — **AI Makerspace** — how serving systems manage the cache at scale.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (inference & systems)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — the KV cache within the full LLM inference stack.
- [Hugging Face — LLM inference optimization](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — KV caching, static cache, and how to use it in practice.

**Articles / blogs (free, no paywall)**:
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **Kipply** — KV-cache memory/bandwidth as the decode bottleneck, with the arithmetic.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng (OpenAI)** — KV cache, MQA/GQA, and cache compression in one survey.
- [Optimizing your LLM in production](https://huggingface.co/blog/optimizing-llm-inference) — **Hugging Face** — KV cache, MQA/GQA, and the memory math for serving.
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — **vLLM team (UC Berkeley)** — the cache-paging idea, from the people who built it.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — full memory accounting that includes the KV cache.
- [Accelerating Generative AI II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) — **PyTorch team** — KV cache + quantization in a real fast-decode implementation.
- [How continuous batching enables 23× throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference) — **Anyscale** — why cache management and batching dominate serving throughput.
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — **Databricks** — the KV cache in the memory/latency budget.
- [Optimizing AI Inference at Character.AI](https://research.character.ai/optimizing-inference/) — **Character.AI** — aggressive KV-cache reduction (MQA, cross-layer sharing) in production.

**Key papers**:
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — **Shazeer (2019)** — multi-query attention to shrink the KV cache.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — grouped-query attention, the KV-cache sweet spot used by Llama-2/3.
- [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023)** — paging the KV cache like OS virtual memory.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — **Pope et al. (2022, Google)** — the canonical analysis of inference cost, including the cache.
- [Efficient Streaming LLMs with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — **Xiao et al. (2023)** — bounding the cache with a sliding window + sink tokens.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive decoding, the loop the cache accelerates.

**In this platform**:
- Concept page (full explanation): [KV Cache](05-KV-Cache.md)
- Foundations (the *why* behind K, Q, V): [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Builds on this: [Decoder-only Architecture](04-Decoder-only-Architecture.md) · [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md) · [Long-Context Methods](08-Long-Context-Methods.md) · [Quantization](10-Quantization.md)
- Puts it to work: [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)

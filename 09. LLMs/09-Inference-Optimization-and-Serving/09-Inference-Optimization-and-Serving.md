---
id: "09-llms/inference-optimization-and-serving"
topic: "Inference Optimization & Serving (vLLM · paged attention)"
parent: "09-llms"
level: advanced
prereqs: ["kv-cache", "decoder-only-architecture", "efficient-attention-flashattention"]
interview_frequency: high
updated: 2026-06-20
---

# Inference Optimization & Serving — vLLM · Paged Attention
> Making LLM serving fast and cheap. The decode phase is **memory-bandwidth-bound** and KV-cache
> memory is the scarce resource. The wins: **PagedAttention** (manage the KV cache like OS virtual
> memory, near-zero waste), **continuous batching** (swap requests in/out per token), plus
> speculative decoding, prefix caching, and quantized KV. vLLM packages these into a serving engine.

**Why it matters:** the production-LLM systems interview. Be ready to distinguish prefill vs decode,
explain why naive batching wastes the KV cache (fragmentation + over-allocation), describe paged
attention and continuous batching, and reason about throughput vs latency (TTFT, TPOT) trade-offs.

**⭐ Start here — suggested path:**

1. **Get the serving picture** — watch [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw). *Why KV-cache memory dominates and how paging fixes it.*
2. **Read the vLLM intro** — [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html). *PagedAttention + continuous batching, with numbers.*
3. **Do the bottleneck math** — [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). *Memory bandwidth vs compute; why decode is bandwidth-bound.*
4. **Read the source** — [PagedAttention paper](https://arxiv.org/abs/2309.06180). *The OS-virtual-memory analogy for the KV cache.*
5. **Go broad** — [Lilian Weng: Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/). *Distillation, quantization, sparsity, speculative decoding — the full toolbox.*

## 🎓 Courses (free)
- [Stanford CS336 — Inference & serving systems](https://stanford-cs336.github.io/spring2025/) — **Stanford** — the systems view of efficient LLM inference.
- [Hugging Face — LLM inference optimization](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — KV cache, static cache, quantization, speculative decoding in code.

## 🎥 Videos
- [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw) — **AI Makerspace** — the serving-engine view, paging and batching.
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) — **Efficient NLP** — the memory object serving systems must manage.
- [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — **Stanford MLSys** — kernel-level efficiency under the engine.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge** — prefill vs decode, the basis for batching.

## 📄 Key Papers
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023)** — the vLLM core idea; near-zero KV-cache waste.
- [GQA: Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — fewer KV heads → smaller cache → cheaper serving.
- [FlashAttention: IO-Aware Exact Attention](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — the fast attention kernel serving engines build on.

## 📰 Articles / Blogs (free, no paywall)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — **vLLM team** — the canonical intro with benchmarks.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **Kipply** — the latency/throughput math behind serving decisions.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng** — the full inference-optimization toolbox.

## 📚 Books (free, with chapters)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/) — **vLLM project** — comprehensive, free reference for paged attention, batching, and serving config.
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the autoregressive decode loop being served.

## 🔗 In this platform
- Foundations (covered elsewhere): [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Related concepts: [KV Cache](../05-KV-Cache/05-KV-Cache.md) · [Efficient Attention (FlashAttention)](../06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md) · [Quantization](../10-Quantization/10-Quantization.md) · [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md)

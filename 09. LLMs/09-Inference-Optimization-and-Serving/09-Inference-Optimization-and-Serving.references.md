---
id: "09-llms/inference-optimization-and-serving/references"
topic: "Inference Optimization & Serving — References"
parent: "09-llms/inference-optimization-and-serving"
type: references
updated: 2026-06-26
---

# Inference Optimization & Serving — references and further reading

> Companion link library for **[Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)** (the concept page). Curated, free/open sources only — external references *and* internal cross-links — kept separate so it can be reused as a standalone reading list. Grouped by type, best-first within each group. Every cited formula source from the page (PagedAttention, Orca, speculative decoding, roofline, KV-cache size) appears below as a clickable entry.

**Start here — suggested path**:
1. **See the serving picture** — watch [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw) (**AI Makerspace**). *Why KV-cache memory dominates and how paging + batching fix it.*
2. **Do the bottleneck math** — read [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (**kipply**). *Memory bandwidth vs compute — why decode is bandwidth-bound, the page's core derivation.*
3. **Get the biggest lever** — read [How Continuous Batching Enables 23× Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference) (**Anyscale**). *Static vs continuous batching, with measured numbers.*
4. **Read the source** — read the [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) (**Kwon et al.**). *KV cache as OS virtual memory; 2–4× throughput.*
5. **Go broad** — read [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (**Lilian Weng**). *Quantization, distillation, sparsity, speculative decoding — the full toolbox.*

**Videos**:
- [Inference, Serving, PagedAttention and vLLM](https://www.youtube.com/watch?v=3TBT4WPkDaw) — **AI Makerspace** — the serving-engine view: paging, continuous batching, and the metrics that matter.
- [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) — **Anyscale (Woosuk Kwon)** — the first author of vLLM walks through PagedAttention and continuous batching.
- [Enabling Cost-Efficient LLM Serving with Speculative Decoding](https://www.youtube.com/watch?v=Ttw57Hj1zR8) — **Trelis Research** — draft-and-verify decoding explained and benchmarked end to end.
- [How a Transformer works at inference vs training time](https://www.youtube.com/watch?v=IGu7ivuy1Ag) — **Niels Rogge (Hugging Face)** — prefill vs decode, the split every serving optimization is built on.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (inference & serving systems)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — the systems view of efficient LLM inference: batching, paging, parallelism.
- [Hugging Face — LLM inference optimization](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — KV cache, static cache, quantization, and speculative decoding, in runnable code.

**Articles / blogs (free, no paywall)**:
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **kipply** — the latency/throughput and memory-bandwidth math behind every serving decision (source for the per-token FLOP and KV-cache derivations on the page).
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — **EleutherAI** — the $2\times\text{params}$ inference-FLOP rule and full memory accounting, derived from the layer shapes.
- [How Continuous Batching Enables 23× Throughput in LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) — **Anyscale** — static vs continuous (in-flight) batching, measured on production-shaped traffic (source for the 23× figure).
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — **vLLM team (UC Berkeley)** — the canonical intro to paging the KV cache, with benchmarks.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng (OpenAI)** — the full inference-optimization toolbox: quantization, distillation, sparsity, speculative decoding.
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — **Databricks** — TTFT/TPOT, batching, and the latency/throughput budget in practice.
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/) — **Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov (PyTorch)** — parallelizing a single decode query across the KV cache; the decode-time counterpart to FlashAttention (no paper — this blog is the canonical source).
- [Accelerating Generative AI II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) — **PyTorch team** — KV cache, quantization, and speculative decoding in a real fast-decode implementation.

**Key papers**:
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — **Chen et al. (2023, DeepMind)** — concurrent derivation of speculative sampling with the distribution-preserving acceptance step.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023, SOSP)** — PagedAttention and vLLM; the KV-cache-as-virtual-memory idea and the 2–4× throughput result (source for the paging mechanism and the KV-size formula on the page).
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — **Pope et al. (2022, Google)** — the canonical analysis of inference cost; establishes that autoregressive decode is memory-bound (the roofline applied to transformers).
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — **Leviathan, Kalman & Matias (2023, ICML, Google)** — draft-and-verify decoding; §3.1 derives the expected-speedup formula $\frac{1-\alpha^{k+1}}{1-\alpha}/(1+kc)$ used on the page.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — the IO-aware tiled attention kernel serving engines build on (FlashDecoding follows for decode).
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — **Qin et al. (2024, Moonshot AI)** — disaggregated prefill/decode with a shared KV-cache pool across machines.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — **Yu et al. (2022, OSDI)** — introduces iteration-level (continuous) batching, the source for the static-vs-continuous lever on the page (open-access PDF on the USENIX page).
- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) — **Williams, Waterman & Patterson (CACM 2009)** — the compute-vs-bandwidth model behind "decode is memory-bound"; free Berkeley tech-report version (source for the roofline ridge-point derivation on the page).
- [SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention)](https://arxiv.org/abs/2312.07104) — **Zheng et al. (2023)** — prefix-sharing the KV cache via a radix tree for fast longest-prefix reuse.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's full forward pass and *see* where prefill ends and decode begins.
- [vLLM Documentation](https://docs.vllm.ai/en/latest/) — **vLLM project** — comprehensive, free reference for PagedAttention, continuous batching, chunked prefill, prefix caching, and speculative decoding config.

**In this platform**:
- Concept page (full explanation): [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)
- The memory object being served: [KV Cache](../05-KV-Cache/05-KV-Cache.md) — prefill/decode split, the memory-bound decode arithmetic, and the four cache-shrinking levers this page builds on.
- The decode loop being served: [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md) · [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md)
- The kernels under the engine: [Efficient Attention (FlashAttention)](../06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md)
- Cutting the bytes streamed: [Quantization](../10-Quantization/10-Quantization.md)
- Foundations (the *why* behind K, Q, V): [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/16-Transformer-Architecture/16-Transformer-Architecture.md)

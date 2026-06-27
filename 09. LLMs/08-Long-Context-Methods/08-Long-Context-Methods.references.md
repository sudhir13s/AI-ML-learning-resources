---
id: "09-llms/long-context-methods/references"
topic: "Long-Context Methods — References"
parent: "09-llms/long-context-methods"
type: references
updated: 2026-06-26
---

# Long-Context Methods — references and further reading

> Companion link library for **[Long-Context Methods](08-Long-Context-Methods.md)** (the concept page). This file holds the curated links — external sources *and* internal cross-links — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free/open and chosen for depth on *this* topic — positional extension (RoPE scaling, YaRN, ALiBi), sparse/sliding attention, and bounded-cache streaming — not popularity.

**Start here — suggested path**:
1. **Get RoPE first** — watch [Rotary Positional Embeddings: Combining Absolute and Relative](https://www.youtube.com/watch?v=o29P0Kpobz0) (**Efficient NLP**). *Why rotation gives relative positions and graceful extrapolation — the foundation for everything else here.*
2. **See context extension** — watch [RoPE to 100K context length](https://www.youtube.com/watch?v=DvP8f7eWS7U) (**Discover AI**). *How RoPE scaling pushes far beyond the training length.*
3. **Read the angle problem and its fix** — read [Position Interpolation, interactive](https://mbrenndoerfer.com/writing/position-interpolation-rope-context-extension) (**Michael Brenndoerfer**), then the [YaRN paper](https://arxiv.org/abs/2309.00071) (**Peng et al.**). *Why naive extrapolation breaks and how interpolation/frequency-scaling fixes it.*
4. **Read the alternative philosophy** — [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) (**Press et al.**). *Distance-bias positions that extrapolate without rescaling the geometry.*
5. **Bound the cache** — [StreamingLLM / Attention Sinks](https://arxiv.org/abs/2309.17453) (**Xiao et al.**). *Why a few first tokens are load-bearing, and how that enables endless streaming.*
6. **Connect to compute & memory** — [FlashAttention](../06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md) + [KV Cache](../05-KV-Cache/05-KV-Cache.md). *The other two walls of long context.*

**Videos**:
- [Rotary Positional Embeddings: Combining Absolute and Relative](https://www.youtube.com/watch?v=o29P0Kpobz0) — **Efficient NLP** — the cleanest RoPE explainer; start here if rotation-as-position still feels abstract.
- [RoPE Rotary Position Embedding to 100K context length](https://www.youtube.com/watch?v=DvP8f7eWS7U) — **Discover AI** — RoPE scaling (PI/NTK) walked through for long-context extension.
- [Coding LLaMA 2 from scratch — RoPE, KV cache, GQA](https://www.youtube.com/watch?v=oM4VmoabDAI) — **Umar Jamil** — builds rotary embeddings and the sliding-window-friendly attention line by line in PyTorch.
- [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — **Stanford MLSys** — the kernel that makes the $O(N^2)$ compute wall tractable at long context.

**Interactive & visual**:
- [Position Interpolation: Extending LLM Context with RoPE Scaling](https://mbrenndoerfer.com/writing/position-interpolation-rope-context-extension) — **Michael Brenndoerfer** — interactive walkthrough of squeezing positions back into the trained angle range.
- [YaRN: Selective Interpolation and Temperature Scaling](https://mbrenndoerfer.com/writing/yarn-rope-context-extension-llm) — **Michael Brenndoerfer** — interactive companion showing YaRN's frequency-dependent scaling vs PI.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (long context & efficiency)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — positional schemes and attention cost at length, within the full LLM stack.
- [Hugging Face — GPU inference optimization](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) — **Hugging Face** — long-context attention kernels and RoPE scaling in practice.

**Articles / blogs (free, no paywall)**:
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — **EleutherAI** — the definitive intuition + derivation for RoPE, the basis of every scaling trick here.
- [LLM Context Length Extension](https://aman.ai/primers/ai/context-length-extension/) — **Aman Chadha** — a clear survey of PI, NTK-aware scaling, YaRN, and ALiBi side by side.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng (OpenAI)** — long-range and sparse attention variants surveyed in one place.
- [In the long (context) run](https://www.harmdevries.com/post/context-length/) — **Harm de Vries** — the economics and practical limits of long context.

**Key papers**:
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — **Zaheer et al. (2020)** — sparse attention (window + global + random) proven to be a universal approximator.
- [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — **Xiao et al. (2023)** — the attention-sink phenomenon; bounding the cache with sinks + a recent window for infinite-length streaming.
- [Extending Context Window of LLMs via Positional Interpolation (PI)](https://arxiv.org/abs/2306.15595) — **Chen et al. (2023)** — squeeze the position index by $L_\text{train}/L_\text{target}$ so RoPE angles stay in the trained range.
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — **Beltagy et al. (2020)** — sliding-window + global-token sparse attention, linear in sequence length.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — **Liu et al. (2023)** — the U-shaped retrieval curve; why advertised context length overstates *effective* context.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — **Jiang et al. (2023)** — sliding-window attention with the layered-receptive-field argument (a window $w$ over $L$ layers reaches $\approx L\,w$ tokens).
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — the rotary scheme behind nearly every long-context LLM and the relative-position derivation.
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) — **Press et al. (2021)** — distance-bias positions that extrapolate to longer sequences without learned embeddings.
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — **Dai et al. (2019)** — segment-level recurrence + relative positional encodings, an ancestor of RoPE.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — **Peng et al. (2023)** — NTK-by-parts frequency-dependent interpolation + attention-temperature correction; 128K with minimal fine-tuning.
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — IO-aware tiled attention; the compute-wall solution that makes long sequences trainable.

**In this platform**:
- Concept page (full explanation): [Long-Context Methods](08-Long-Context-Methods.md)
- Foundations (the *why* behind RoPE and attention): [Positional Encoding](../../05.%20Deep_Learning/17-Positional-Encoding/17-Positional-Encoding.md) · [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md)
- The other two walls: [Efficient Attention (FlashAttention)](../06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md) · [KV Cache](../05-KV-Cache/05-KV-Cache.md)
- Builds on this: [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md) (the modern RoPE recipe) · [Quantization](../10-Quantization/10-Quantization.md) (compressing the cache) · [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)

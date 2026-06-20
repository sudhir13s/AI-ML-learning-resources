---
id: "09-llms/long-context-methods"
topic: "Long-Context Methods"
parent: "09-llms"
level: advanced
prereqs: ["positional-encoding", "attention", "kv-cache"]
interview_frequency: high
updated: 2026-06-20
---

# Long-Context Methods
> How LLMs read 32K, 128K, or 1M tokens. Two problems must be solved: **positional generalization**
> (RoPE scaling, ALiBi, NoPE) so the model handles positions it never trained on, and **attention
> cost** (FlashAttention, sliding-window/sparse, ring attention) so the N² compute and KV-cache memory
> stay tractable.

**Why it matters:** a hot interview area. Be ready to explain why vanilla learned positions don't
extrapolate, how RoPE enables length extrapolation (and how NTK/YaRN scaling extends it further),
what ALiBi does, and why the KV cache (not FLOPs) is often the real long-context bottleneck.

**⭐ Start here — suggested path:**

1. **Understand RoPE** — watch [Rotary Positional Embeddings: combining absolute and relative](https://www.youtube.com/watch?v=o29P0Kpobz0). *Why rotation gives relative positions and graceful extrapolation.*
2. **See context extension** — [RoPE to 100K context length](https://www.youtube.com/watch?v=DvP8f7eWS7U). *How RoPE scaling pushes far beyond the training length.*
3. **Read the math** — [RoFormer / RoPE paper](https://arxiv.org/abs/2104.09864) and [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/). *The rotation formulation and why it generalizes.*
4. **Read the alternatives** — [ALiBi (Train Short, Test Long)](https://arxiv.org/abs/2108.12409). *Bias-based positions that extrapolate without learned embeddings.*
5. **Connect to cost** — [FlashAttention](06-Efficient-Attention-FlashAttention.md) + [KV Cache](05-KV-Cache.md). *The compute/memory levers that make long context affordable.*

## 🎓 Courses (free)
- [Stanford CS336 — Long context & efficiency](https://stanford-cs336.github.io/spring2025/) — **Stanford** — positional schemes and attention cost at length.
- [Hugging Face — GPU inference optimization](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) — **Hugging Face** — long-context attention kernels in practice.

## 🎥 Videos
- [Rotary Positional Embeddings: Combining Absolute and Relative](https://www.youtube.com/watch?v=o29P0Kpobz0) — **Efficient NLP** — the cleanest RoPE explainer.
- [RoPE Rotary Position Embedding to 100K context length](https://www.youtube.com/watch?v=DvP8f7eWS7U) — **Discover AI** — RoPE scaling for long context.
- [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — **Stanford MLSys** — the kernel that makes long sequences trainable.
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) — **Efficient NLP** — why memory, not FLOPs, often caps context length.

## 📄 Key Papers
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — **Su et al. (2021)** — the positional scheme behind most long-context LLMs.
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) — **Press et al. (2021)** — extrapolation without learned positions.
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — **Tay et al. (2020)** — sparse/sliding/low-rank attention for long sequences.

## 📰 Articles / Blogs (free, no paywall)
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — **EleutherAI** — the definitive intuition + derivation for RoPE.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng** — long-range attention variants surveyed.
- [In the long (context) run](https://www.harmdevries.com/post/context-length/) — **Harm de Vries** — the economics and limits of long context.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 9 "Transformers"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — positional encodings and self-attention.
- [Dive into Deep Learning — **Ch. 11 "Attention & Transformers"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — positional encoding with code.

## 🔗 In this platform
- Foundations (covered elsewhere): [Positional Encoding](../../05.%20Deep_Learning/concepts/17-Positional-Encoding.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)
- Related concepts: [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md) · [KV Cache](05-KV-Cache.md) · [Decoder-only Architecture](04-Decoder-only-Architecture.md)

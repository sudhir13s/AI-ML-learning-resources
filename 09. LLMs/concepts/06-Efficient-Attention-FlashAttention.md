---
id: "09-llms/efficient-attention-flashattention"
topic: "Efficient Attention (FlashAttention)"
parent: "09-llms"
level: advanced
prereqs: ["attention", "kv-cache", "gpu-memory-hierarchy"]
interview_frequency: high
updated: 2026-06-20
---

# Efficient Attention — FlashAttention
> Standard attention is memory-bound: it materializes the full N×N score matrix in slow HBM.
> **FlashAttention** computes exact attention without ever writing that matrix — it tiles the
> computation, keeps blocks in fast SRAM, and uses the online-softmax trick. Same math, far less
> memory traffic, big speedups. The reason long-context training is even feasible.

**Why it matters:** the canonical "how do you make attention faster?" question. Be ready to explain
that attention is *IO-bound* not compute-bound, what tiling + online softmax do, why it's **exact**
(not an approximation like sparse/linear attention), and the recompute-in-backward trade-off.

**⭐ Start here — suggested path:**

1. **Frame the problem** — watch [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4). *The author explains IO-awareness and tiling — the best overview.*
2. **Go deeper on the algorithm** — [MedAI #54: FlashAttention talk](https://www.youtube.com/watch?v=FThvfkXWqtE). *More detail on the kernel and the softmax recomputation.*
3. **Read the source** — the [FlashAttention paper](https://arxiv.org/abs/2205.14135). *Online softmax + tiling + the IO-complexity analysis.*
4. **See why memory matters** — [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). *Quantifies the memory-bandwidth wall FlashAttention sidesteps.*
5. **Connect to long context** — [Long-Context Methods](08-Long-Context-Methods.md). *FlashAttention is what makes ≥32K-token training practical.*

## 🎓 Courses (free)
- [Stanford CS336 — Systems & efficient attention](https://stanford-cs336.github.io/spring2025/) — **Stanford** — kernels, IO-awareness, and attention efficiency.
- [Hugging Face — GPU inference optimization](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) — **Hugging Face** — using FlashAttention / SDPA in practice.

## 🎥 Videos
- [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — **Stanford MLSys** — the author's clearest full talk.
- [MedAI #54: FlashAttention with IO-Awareness](https://www.youtube.com/watch?v=FThvfkXWqtE) — **Tri Dao / Stanford MedAI** — the algorithm and the memory-hierarchy argument.
- [Flash Attention 2.0 with Tri Dao (author)](https://www.youtube.com/watch?v=IoMSGuiwV3g) — **Aleksa Gordić (AI Epiphany)** — what changed in v2 and why it's faster.
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the exact computation FlashAttention reorganizes.

## 📄 Key Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — the original tiling + online-softmax algorithm.
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — **Tay et al. (2020)** — the landscape of sparse/linear/low-rank attention (the approximate alternatives).
- [GQA: Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — a complementary efficiency lever (fewer KV heads).

## 📰 Articles / Blogs (free, no paywall)
- [FlexAttention: the flexibility of PyTorch with the performance of FlashAttention](https://pytorch.org/blog/flexattention/) — **PyTorch** — how modern frameworks expose fast, fused attention.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng** — efficient-attention techniques in context.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng** — survey of efficient-attention variants.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 11 "Attention Mechanisms & Transformers"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — attention internals, with code.
- [Speech and Language Processing, 3rd ed. — **Ch. 9 "Transformers"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — self-attention foundations FlashAttention optimizes.

## 🔗 In this platform
- Foundations (covered elsewhere): [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md)
- Related concepts: [KV Cache](05-KV-Cache.md) · [Long-Context Methods](08-Long-Context-Methods.md) · [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)

---
id: "09-llms/efficient-attention-flashattention/references"
topic: "Efficient Attention (FlashAttention) — References"
parent: "09-llms/efficient-attention-flashattention"
type: references
updated: 2026-06-26
---

# Efficient Attention (FlashAttention) — references and further reading

> Companion link library for **[Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first within each group. Every entry is free/open and chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame the problem** — watch [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) (**Stanford MLSys**). *The author explains IO-awareness, the memory wall, and tiling — the best single overview.*
2. **See the math** — watch [Coding FlashAttention from scratch in Triton](https://www.youtube.com/watch?v=zy8ChVd_oTM) (**Umar Jamil**). *Online softmax + tiling built line by line; the clearest derivation of the running (m, ℓ) update.*
3. **Read the source** — read [FlashAttention](https://arxiv.org/abs/2205.14135) (**Dao et al. 2022**). *Tiling, online softmax, recomputation, and the IO-complexity theorems.*
4. **Get the one-pass softmax** — read [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (**Milakov & Gimelshein 2018**). *The streaming softmax that is FlashAttention's mathematical heart.*
5. **Connect to long context** — read [Long-Context Methods](../08-Long-Context-Methods/08-Long-Context-Methods.md). *FlashAttention is what makes ≥32K-token training and prefill practical.*

**Videos**:
- [FlashAttention — Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — **Stanford MLSys** — the author's clearest full talk: IO-awareness, the memory hierarchy, tiling, and why it's exact.
- [Coding FlashAttention from scratch in Triton](https://www.youtube.com/watch?v=zy8ChVd_oTM) — **Umar Jamil** — a long, careful build of the kernel and the online-softmax derivation; the best hands-on walkthrough.
- [Flash Attention 2.0 with Tri Dao (author)](https://www.youtube.com/watch?v=IoMSGuiwV3g) — **Aleksa Gordić (The AI Epiphany)** — what changed in v2 (work partitioning, parallelism) and why it's faster.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the exact Q·Kᵀ-softmax-V computation FlashAttention reorganizes; watch first if attention still feels abstract.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's forward pass and *see* the attention scores FlashAttention refuses to materialize.
- [GPU Puzzles](https://github.com/srush/GPU-Puzzles) — **Sasha Rush** — hands-on puzzles that build the SRAM/HBM tiling intuition FlashAttention depends on.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (systems & efficient attention)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — kernels, IO-awareness, and attention efficiency within the full LLM stack.
- [Hugging Face — GPU inference optimization](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) — **Hugging Face** — using FlashAttention / SDPA in practice.

**Articles / blogs (free, no paywall)**:
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) — **Aleksa Gordić** — a careful, intuition-first walkthrough of tiling and online softmax.
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — **Zihao Ye (UW CSE599M)** — a clean lecture-note derivation connecting the streaming softmax to the full algorithm.
- [FlexAttention: the flexibility of PyTorch with the performance of FlashAttention](https://pytorch.org/blog/flexattention/) — **PyTorch** — how modern frameworks expose fast, fused attention over custom masks.
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/) — **Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov (PyTorch)** — the decode-time counterpart: parallelizing a single query across the KV cache.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — **Kipply** — quantifies the memory-bandwidth wall FlashAttention sidesteps.
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — **Lilian Weng (OpenAI)** — survey of efficient-attention variants (sparse, linear, recurrent) that places FlashAttention in context.

**Key papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the transformer; §3.2 defines the scaled-dot-product attention ($QK^\top/\sqrt{d}$, softmax, ×V) FlashAttention reorganizes without changing.
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — **Tay et al. (2020)** — the landscape of sparse / linear / low-rank attention (the *approximate* alternatives FlashAttention is contrasted with).
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — **Dao et al. (2022)** — the original: tiling + online softmax + backward recomputation, with the IO-complexity theorems. The core source for this page.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — **Dao (2023)** — v2's work partitioning and reduced non-matmul FLOPs (~2× over v1).
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — **Shah et al. (2024)** — v3's Hopper asynchrony (warp specialization, TMA) and FP8 (~1.5–2× over v2).
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — **Ainslie et al. (2023)** — a complementary efficiency lever (fewer KV heads); orthogonal to, and composable with, FlashAttention.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — **Jiang et al. (2023)** — sliding-window attention with the layered-receptive-field argument ($w$ over $L$ layers reaches $\approx L\,w$); a sparsity approach that composes with FlashAttention.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — **Milakov & Gimelshein (2018)** — the single-pass streaming softmax (running max + running sum) that is FlashAttention's mathematical heart.
- [Self-attention Does Not Need $O(n^2)$ Memory](https://arxiv.org/abs/2112.05682) — **Rabe & Staats (2021)** — the memory-efficient streaming-attention formulation FlashAttention builds on and makes IO-aware.
- [Roofline: An Insightful Visual Performance Model (CACM 2009)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) — **Williams, Waterman & Patterson** — the compute-vs-bandwidth model behind "attention is memory-bound"; free Berkeley tech-report version.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 11 "Attention Mechanisms & Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — **Zhang et al.** — attention internals with runnable code, the computation FlashAttention optimizes.
- [Speech and Language Processing, 3rd ed. — Ch. 9 "Transformers"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the self-attention foundations FlashAttention accelerates.

**In this platform**:
- Concept page (full explanation): [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md)
- Foundations (the *why* behind Q, K, V and softmax): [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/16-Transformer-Architecture/16-Transformer-Architecture.md)
- Companion (the decode-side IO story): [KV Cache](../05-KV-Cache/05-KV-Cache.md)
- Builds on this: [Long-Context Methods](../08-Long-Context-Methods/08-Long-Context-Methods.md) · [Quantization](../10-Quantization/10-Quantization.md)
- Puts it to work: [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)

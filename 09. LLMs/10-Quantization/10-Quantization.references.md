---
id: "09-llms/quantization/references"
topic: "Quantization — References"
parent: "09-llms/quantization"
type: references
updated: 2026-06-26
---

# Quantization — references and further reading

> Companion link library for **[Quantization](10-Quantization.md)** (the concept page). It holds the curated links — external sources *and* internal cross-links — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first within each group. Every entry is a primary source or a recognized deep explainer, chosen for depth on *this* topic. Every formula cited on the concept page has its source here in **Papers**.

**Start here — suggested path:**
1. **Build the intuition (illustrated)** — read [A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/) (**Maarten Grootendorst**). *The single best illustrated walkthrough of LLM quantization, end to end.*
2. **Map the methods** — watch [Quantization Explained: GPTQ, AWQ, GGUF, bitsandbytes](https://www.youtube.com/watch?v=K75j8MkwgJ0) (**Trelis Research**). *When to reach for which format.*
3. **Do the math from scratch** — read the affine formulation in [Jacob et al. (2017)](https://arxiv.org/abs/1712.05877) §2, then this page's code. *Scale, zero-point, round, clamp.*
4. **Understand the outlier problem** — read [LLM.int8()](https://arxiv.org/abs/2208.07339) (**Dettmers et al.**). *The emergent-outlier diagnosis the whole field is built around.*
5. **Read the 4-bit workhorses** — [GPTQ](https://arxiv.org/abs/2210.17323) then [AWQ](https://arxiv.org/abs/2306.00978). *Hessian error-compensation vs activation-aware salience.*
6. **Connect to fine-tuning** — [QLoRA](https://arxiv.org/abs/2305.14314) and this platform's [LoRA / PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md). *Quantize the base, adapt with LoRA.*

**Papers:**
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) — **Gholami et al. (2021)** — the reference survey; symmetric/asymmetric affine quantization and granularity, the source for the page's general affine + symmetric formulas.
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — **Lin et al. (2023, MLSys 2024)** — protect the ~1% salient weight channels (identified by activation magnitude) via per-channel scaling before 4-bit; the AWQ saliency-scaling formula on the page.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — **Frantar et al. (2022)** — layer-wise 3–4 bit PTQ with Hessian-based error compensation; the GPTQ objective on the page.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — **Dettmers et al. (2022)** — discovers emergent outlier features and the mixed-precision decomposition; the LLM.int8() formula on the page.
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — **Jacob et al. (2017)** — the canonical affine (scale + integer zero-point) quantization formulation the page's core formula derives from.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — **Dettmers et al. (2023, NeurIPS)** — the NF4 quantile-spaced datatype, double quantization, and paged optimizers; fine-tunes a 65B model on one 48 GB GPU. Source for the NF4/QLoRA and double-quantization notes.
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — **Xiao et al. (2022, ICML 2023)** — the mathematically-equivalent migration of activation outliers into weights for INT8 weight+activation; the SmoothQuant migration formula on the page.
- [KIVI: A Tuning-Free 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — **Liu et al. (2024)** — the same outlier asymmetry applied to the KV cache (keys per-channel, values per-token); the bridge to KV-cache quantization.
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) — **Tseng et al. (2024)** — pushing weight quantization to 2 bits with incoherence processing; the frontier below the 3-bit wall.

**Videos:**
- [Quantization Explained: GPTQ, AWQ, GGUF, bitsandbytes](https://www.youtube.com/watch?v=K75j8MkwgJ0) — **Trelis Research** — the practical map of formats and when to use each.
- [LLM Quantization: GPTQ, AWQ, QLoRA, GGUF and More](https://www.youtube.com/watch?v=WmvZwR4rKJg) — **Tales of Tensors** — a clear tour of the method landscape.
- [QLoRA paper walkthrough](https://www.youtube.com/watch?v=y9PHWGOa8HA) — **Umar Jamil** — line-by-line through NF4, double quantization, and paged optimizers from the QLoRA paper.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where precision/quantization sits in the full deployment picture.

**Courses (free):**
- [Hugging Face — Quantization & LLM inference optimization](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — bitsandbytes, GPTQ, AWQ, and FP8 in practice, with code.
- [Stanford CS336 — Language Modeling from Scratch (efficiency & quantization)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — numerical precision and model compression within the full systems stack.
- [MIT 6.5940 — TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) — **Han Lab (MIT)** — the course behind AWQ/SmoothQuant; lectures on quantization, pruning, and efficient inference.

**Articles / blogs (free, no paywall):**
- [A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/) — **Maarten Grootendorst** — the definitive illustrated walkthrough; affine quantization, symmetric/asymmetric, GPTQ/GGUF/bitsandbytes.
- [Introduction to Weight Quantization](https://huggingface.co/blog/maxime-labonne/introduction-to-weight-quantization) — **Maxime Labonne (Hugging Face)** — from-scratch absmax and zero-point int8 quantization in code, the same map this page builds.
- [A Gentle Introduction to 8-bit Matrix Multiplication (LLM.int8())](https://huggingface.co/blog/hf-bitsandbytes-integration) — **Hugging Face / Dettmers et al.** — the outlier problem and mixed-precision decomposition, explained with figures.
- [Making LLMs even more accessible with bitsandbytes, 4-bit & QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — **Hugging Face** — NF4, double quantization, and QLoRA in practice.
- [Overview of natively supported quantization schemes in Transformers](https://huggingface.co/blog/overview-quantization-transformers) — **Hugging Face** — GPTQ vs bitsandbytes vs AWQ side by side, with trade-offs.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng (OpenAI)** — quantization among the full menu of inference optimizations.

**Interactive & visual:**
- [llama.cpp quantization formats (k-quants) README](https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md) — **ggml-org** — the canonical reference for GGUF k-quant naming (`Q4_K_M`, `Q5_K_M`, …) and bit allocations.
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/main/en/index) — **Hugging Face** — runnable LLM.int8() and NF4/QLoRA; the reference implementation of the methods on this page.
- [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) — **Hugging Face Accelerate** — plug in any model and precision to see the weights/optimizer/activation memory; the memory math of this page, interactive.

**In this platform:**
- Concept page (full explanation): [Quantization](10-Quantization.md)
- Builds on this: [KV Cache](../05-KV-Cache/05-KV-Cache.md) (decode is memory-bandwidth-bound; quantizing the *cache* is the sibling lever) · [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)
- Connects to: [LoRA / PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md) (QLoRA = NF4 base + LoRA adapters) · [Knowledge Distillation](../11-Knowledge-Distillation/11-Knowledge-Distillation.md) (the other route to a smaller model) · [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md) (where quantized weights + cache are served)

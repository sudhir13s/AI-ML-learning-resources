---
id: "09-llms/quantization"
topic: "Quantization (GPTQ · AWQ · GGUF)"
parent: "09-llms"
level: advanced
prereqs: ["decoder-only-architecture", "numerical-precision"]
interview_frequency: high
updated: 2026-06-20
---

# Quantization — GPTQ · AWQ · GGUF
> Store and run LLM weights in 8, 4, or even 3 bits instead of 16. Quantization cuts memory and
> bandwidth (the decode bottleneck) with little quality loss. Key families: **GPTQ** (layer-wise
> error-minimizing PTQ), **AWQ** (protect the salient weights identified by activations), **bitsandbytes**
> (LLM.int8 / NF4 for QLoRA), and **GGUF** (the llama.cpp on-device format).

**Why it matters:** a very common practical interview topic. Be ready to contrast PTQ vs QAT,
weight-only vs weight+activation, explain why outlier activations make naive int8 fail (→ LLM.int8),
how GPTQ uses second-order info and AWQ uses activation salience, and the memory/quality trade-offs.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/). *The single best illustrated explainer of LLM quantization.*
2. **Survey the methods** — watch [LLM Quantization Explained: GPTQ, AWQ, QLoRA, GGUF](https://www.youtube.com/watch?v=WmvZwR4rKJg). *When to use which format.*
3. **Go deeper on techniques** — [LLM Quantization Techniques (GPTQ/AWQ/GGUF/BitNet)](https://www.youtube.com/watch?v=0pF6GdbwMo4). *The math behind each method.*
4. **Read the sources** — [GPTQ](https://arxiv.org/abs/2210.17323) then [AWQ](https://arxiv.org/abs/2306.00978). *Error-minimizing PTQ vs activation-aware salience.*
5. **Connect to fine-tuning** — [LoRA / PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md). *QLoRA fine-tunes on top of a 4-bit base — quantization + adapters.*

## 🎓 Courses (free)
- [Hugging Face — Quantization docs](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — bitsandbytes, GPTQ, AWQ usage in practice.
- [Stanford CS336 — Efficiency & quantization](https://stanford-cs336.github.io/spring2025/) — **Stanford** — numerical precision and model compression in the systems stack.

## 🎥 Videos
- [LLM Quantization Explained: GPTQ, AWQ, QLoRA, GGUF and More](https://www.youtube.com/watch?v=WmvZwR4rKJg) — **Tales of Tensors** — the practical map of methods/formats.
- [LLM Quantization Techniques — GPTQ AWQ GGUF HQQ BitNet](https://www.youtube.com/watch?v=0pF6GdbwMo4) — **Joydeep Bhattacharjee** — theory + implementation of each technique.
- [LoRA explained (and a bit about precision and quantization)](https://www.youtube.com/watch?v=t509sv5MT0w) — **DeepFindr** — the precision intuition that underlies QLoRA.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where precision/quantization sits in deployment.

## 📄 Key Papers
- [GPTQ: Accurate Post-Training Quantization for Generative Transformers](https://arxiv.org/abs/2210.17323) — **Frantar et al. (2022)** — layer-wise, second-order PTQ to 3–4 bits.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — **Lin et al. (2023)** — protect the salient 1% of weights using activation statistics.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — **Dettmers et al. (2022)** — the outlier problem and mixed-precision int8; the line that leads to [QLoRA / NF4](https://arxiv.org/abs/2305.14314).

## 📰 Articles / Blogs (free, no paywall)
- [A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/) — **Maarten Grootendorst** — the definitive illustrated walkthrough.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng** — quantization among inference optimizations.
- [Hugging Face — PEFT & QLoRA](https://huggingface.co/blog/peft) — **Hugging Face** — quantized base + LoRA adapters in practice.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 12 "Computational Performance"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — precision/efficiency foundations.
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the models quantization compresses.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.05 Quantization](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.05_Quantization.md)
- Related concepts: [Knowledge Distillation](../11-Knowledge-Distillation/11-Knowledge-Distillation.md) · [LoRA / PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md) · [Inference Optimization & Serving](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)

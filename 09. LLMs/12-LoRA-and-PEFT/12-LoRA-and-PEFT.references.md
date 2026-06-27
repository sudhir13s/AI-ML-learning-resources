---
id: "09-llms/lora-and-peft/references"
topic: "LoRA & PEFT — References"
parent: "09-llms/lora-and-peft"
type: references
updated: 2026-06-26
---

# LoRA & PEFT — references and further reading

> Companion link library for **[LoRA & PEFT](12-LoRA-and-PEFT.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links are free / open-access.

**Start here — suggested path**:
1. **Get the core idea** — watch [LoRA explained (and a bit about precision & quantization)](https://www.youtube.com/watch?v=t509sv5MT0w) (**DeepFindr**). *Low-rank updates and why they're cheap, clearly.*
2. **Nail the concepts** — watch [Low-rank Adaptation: the Key Concepts behind LoRA](https://www.youtube.com/watch?v=dA-NhCtrrVE) (**Chris Alexiuk**). *Rank, alpha, and merging in a few minutes.*
3. **Read the source** — the [LoRA paper](https://arxiv.org/abs/2106.09685) (**Hu et al. 2021**). *The low-rank hypothesis and the $\Delta W = BA$ formulation, with the init and scaling.*
4. **Understand the memory win** — the [QLoRA paper](https://arxiv.org/abs/2305.14314) (**Dettmers et al. 2023**). *NF4 + double quant + paged optimizers → 65B on one GPU.*
5. **Use the library** — [Hugging Face PEFT docs](https://huggingface.co/docs/peft/index). *Apply LoRA/QLoRA to a real model end to end, then `merge_and_unload()`.*

**Videos**:
- [LoRA explained (and a bit about precision & quantization)](https://www.youtube.com/watch?v=t509sv5MT0w) — **DeepFindr** — the cleanest conceptual intro to low-rank updates.
- [Low-rank Adaptation of LLMs: the Key Concepts behind LoRA](https://www.youtube.com/watch?v=dA-NhCtrrVE) — **Chris Alexiuk** — rank, alpha, and merging, concisely.
- [LoRA & QLoRA Fine-tuning Explained In-Depth](https://www.youtube.com/watch?v=t1caDsMzWBk) — **Mark Hennings (Entry Point AI)** — the practical, in-depth treatment including QLoRA.
- [Fine-tuning LLMs with LoRA (hands-on, with code)](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — a runnable end-to-end LoRA fine-tune.

**Interactive & hands-on**:
- [Hugging Face PEFT — quicktour & task guides](https://huggingface.co/docs/peft/quicktour) — **Hugging Face** — `LoraConfig`, `get_peft_model`, training, and `merge_and_unload()` you can run immediately.
- [QLoRA fine-tuning Colab](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k) — **Hugging Face / bitsandbytes** — load a model in 4-bit and attach LoRA adapters in a free notebook.

**Courses (free)**:
- [Hugging Face — PEFT documentation](https://huggingface.co/docs/peft/index) — **Hugging Face** — the canonical library + guide for LoRA, QLoRA, adapters, prompt-tuning, (IA)³.
- [Hugging Face LLM Course — fine-tuning](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — where PEFT fits in the fine-tuning workflow.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — fine-tuning and adaptation within the full LLM stack.

**Articles / blogs (free, no paywall)**:
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/blog/peft) — **Hugging Face** — LoRA/QLoRA in practice with the library.
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — **Sebastian Raschka** — rank, alpha, target-module choices and what actually moves quality, from extensive experiments.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — full-vs-PEFT and where LoRA fits in the adaptation pipeline.
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — **Hugging Face** — NF4 and the 4-bit machinery QLoRA is built on.
- [The Novice's LLM Training Guide — LoRA & QLoRA](https://rentry.org/llm-training) — **Alpin** — a practitioner walkthrough of LoRA hyperparameters and gotchas.

**Key papers**:
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — **Hu et al. (2021)** — the founding method: low-rank weight updates $\Delta W = \tfrac{\alpha}{r}BA$ with $B$ zero-init, mergeable at inference.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — **Dettmers et al. (2023)** — NF4 datatype, double quantization, paged optimizers; 4-bit base + LoRA matches 16-bit full FT and fits 65B on one GPU.
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) — **Aghajanyan et al. (2020)** — measures that fine-tuning lives in a low-dimensional subspace: the empirical premise LoRA rests on.
- [Parameter-Efficient Transfer Learning for NLP (Adapters)](https://arxiv.org/abs/1902.00751) — **Houlsby et al. (2019)** — the original adapter: bottleneck MLPs inserted in each block, base frozen.
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — **Li & Liang (2021)** — trainable continuous K/V prefixes at every layer; steer without touching weights.
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — **Lester, Al-Rfou & Constant (2021)** — soft prompts at the input only; closes the gap to full FT as scale grows.
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning ((IA)³)](https://arxiv.org/abs/2205.05638) — **Liu et al. (2022)** — learned per-feature rescaling vectors; very few parameters, mergeable.
- [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) — **Ben-Zaken, Ravfogel & Goldberg (2021)** — train only the bias terms; a surprisingly strong baseline.
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — **Sheng et al. (2023)** — one base + thousands of adapters, batched across adapters in production.
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — **Liu et al. (2024)** — decomposes weights into magnitude + direction, adapting each; a strong LoRA successor.
- [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354) — **Hayou, Ghosh & Yu (2024)** — different learning rates for $A$ vs $B$ improve LoRA convergence.

**Books (free chapters)**:
- [Dive into Deep Learning — Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html) — **Zhang et al.** — the transfer-learning / fine-tuning foundations PEFT optimizes.
- [Speech and Language Processing, 3rd ed. — Ch. 11 "Fine-Tuning & Masked LMs"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the adaptation context LoRA makes cheap.

**In this platform**:
- Concept page (full explanation): [LoRA & PEFT](12-LoRA-and-PEFT.md)
- Foundations (the *why* behind the weights LoRA adapts): [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/16-Transformer-Architecture/16-Transformer-Architecture.md)
- Builds on this: [Quantization](../10-Quantization/10-Quantization.md) (NF4 / 4-bit, the base of QLoRA) · [Pretraining at Scale](../02-Pretraining-at-Scale/02-Pretraining-at-Scale.md) (the base LoRA adapts) · [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)
- Puts it to work: [Supervised Fine-Tuning](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [Instruction Tuning](../14-Instruction-Tuning/14-Instruction-Tuning.md) · [RLHF and DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md)

---
id: "08-llms/lora-and-peft"
topic: "LoRA / PEFT (parameter-efficient fine-tuning)"
parent: "08-llms"
level: advanced
prereqs: ["fine-tuning", "linear-algebra", "decoder-only-architecture"]
interview_frequency: very-high
updated: 2026-06-20
---

# LoRA / PEFT — Parameter-Efficient Fine-Tuning
> Fine-tune a frozen base model by training tiny add-on parameters. **LoRA** freezes the weights and
> learns a low-rank update ΔW = B·A (rank r ≪ d), so you update <1% of parameters and the base is
> shared across tasks. **QLoRA** adds 4-bit quantization so you can fine-tune a 65B model on one GPU.
> PEFT also covers adapters, prefix/prompt tuning, and (IA)³.

**Why it matters:** *the* fine-tuning interview question today. Be ready to write ΔW = BA, explain the
rank/alpha hyperparameters, why LoRA adds **zero** inference latency (merge BA into W), why it works
(updates are low-rank), and how QLoRA combines NF4 quantization + LoRA to slash memory.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [LoRA explained (and a bit about precision/quantization)](https://www.youtube.com/watch?v=t509sv5MT0w). *Low-rank updates and why they're cheap, clearly.*
2. **See the key concepts** — [LoRA: Explaining the Key Concepts](https://www.youtube.com/watch?v=dA-NhCtrrVE) by Chris Alexiuk. *Rank, alpha, and merging in a few minutes.*
3. **Read the source** — the [LoRA paper](https://arxiv.org/abs/2106.09685). *The low-rank hypothesis and the ΔW = BA formulation.*
4. **Use the library** — [Hugging Face PEFT docs](https://huggingface.co/docs/peft/index). *Apply LoRA/QLoRA to a real model end to end.*
5. **Understand QLoRA** — [HF: PEFT + QLoRA](https://huggingface.co/blog/peft). *4-bit base + LoRA adapters: fine-tune big models on one GPU.*

## 🎓 Courses (free)
- [Hugging Face — PEFT documentation](https://huggingface.co/docs/peft/index) — **Hugging Face** — the canonical library + guide for LoRA, QLoRA, adapters, prompt tuning.
- [Hugging Face LLM Course — fine-tuning](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — where PEFT fits in the fine-tuning workflow.

## 🎥 Videos
- [LoRA explained (and a bit about precision and quantization)](https://www.youtube.com/watch?v=t509sv5MT0w) — **DeepFindr** — the cleanest conceptual intro.
- [Low-rank Adaption of LLMs: the Key Concepts behind LoRA](https://www.youtube.com/watch?v=dA-NhCtrrVE) — **Chris Alexiuk** — rank, alpha, and merging.
- [LoRA & QLoRA Fine-tuning Explained In-Depth](https://www.youtube.com/watch?v=t1caDsMzWBk) — **Mark Hennings** — the practical, in-depth treatment incl. QLoRA.
- [Fine-tuning LLMs with example code](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — LoRA fine-tuning hands-on.

## 📄 Key Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — **Hu et al. (2021)** — the founding method; low-rank weight updates.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — **Dettmers et al. (2023)** — NF4 + paged optimizers + LoRA → big models on one GPU.
- [Parameter-Efficient Transfer Learning for NLP (Adapters)](https://arxiv.org/abs/1902.00751) — **Houlsby et al. (2019)** — the adapter idea PEFT generalizes.

## 📰 Articles / Blogs (free, no paywall)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/blog/peft) — **Hugging Face** — LoRA/QLoRA in practice with the library.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — full-vs-PEFT and where LoRA fits.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — adaptation methods in the training pipeline.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 16 "NLP Applications & Fine-Tuning"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — fine-tuning workflow foundations.
- [Speech and Language Processing, 3rd ed. — **Ch. 11 "Fine-Tuning & Masked LMs"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the adaptation context LoRA optimizes.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.02 LoRA: Low-Rank Adaptation](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.02_LoRA_Low_Rank_Adaptation.md) · [Module 7.03 Transfer Learning & Fine-Tuning](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.03_Transfer_Learning_and_Fine_Tuning.md)
- Related concepts: [Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md) · [Instruction Tuning](14-Instruction-Tuning.md) · [Quantization](10-Quantization.md)

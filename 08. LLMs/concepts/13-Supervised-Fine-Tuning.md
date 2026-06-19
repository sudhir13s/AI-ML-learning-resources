---
id: "08-llms/supervised-fine-tuning"
topic: "Supervised Fine-Tuning (SFT)"
parent: "08-llms"
level: advanced
prereqs: ["language-modeling-objectives", "fine-tuning", "lora-and-peft"]
interview_frequency: high
updated: 2026-06-20
---

# Supervised Fine-Tuning (SFT)
> The first alignment step after pretraining: continue next-token training on curated
> **(prompt → response)** demonstrations so the base model produces helpful, formatted answers.
> SFT turns a raw next-token predictor into an assistant — and is the foundation that RLHF/DPO
> refines. Often done with LoRA to keep it cheap.

**Why it matters:** the "how do you go from a base model to ChatGPT?" pipeline question. Be ready to
explain SFT vs pretraining (same loss, different data), why you mask the loss over the prompt tokens,
how demonstration data is built, the chat template / special tokens, and where SFT sits before RLHF.

**⭐ Start here — suggested path:**

1. **See the full pipeline** — watch [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI). *Pretraining → SFT → RLHF, with SFT's exact role.*
2. **Do it hands-on** — [Fine-tuning LLMs with example code](https://www.youtube.com/watch?v=eC6Hd1hFvos). *A concrete SFT run start to finish.*
3. **Read the recipe** — [InstructGPT paper](https://arxiv.org/abs/2203.02155) §3 (the SFT stage). *How demonstrations are collected and used.*
4. **Read the survey** — [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models). *Full vs PEFT, data quality, and gotchas.*
5. **Make it efficient** — [LoRA / PEFT](12-LoRA-and-PEFT.md). *How SFT is actually run on consumer hardware.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — fine-tuning chapters](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — SFT with the `trl` SFTTrainer, end to end.
- [Stanford CS336 — Alignment & post-training](https://stanford-cs336.github.io/spring2025/) — **Stanford** — SFT within the post-training stack.

## 🎥 Videos
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — SFT's role in the full alignment pipeline.
- [Fine-tuning Large Language Models (with example code)](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — a complete SFT walkthrough.
- [LLM Fine-Tuning Crash Course: 1-Hour End-to-End](https://www.youtube.com/watch?v=mrKuDK9dGlg) — **AI Anytime** — data prep → training → eval.
- [LoRA & QLoRA Fine-tuning Explained In-Depth](https://www.youtube.com/watch?v=t1caDsMzWBk) — **Mark Hennings** — the efficient way SFT is usually done.

## 📄 Key Papers
- [Training LMs to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — §3 details the SFT stage before RLHF.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — **Zhou et al. (2023)** — 1,000 high-quality SFT examples can be enough; data quality > quantity.
- [LLaMA-2](https://arxiv.org/abs/2307.09288) — **Touvron et al. (2023)** — a documented SFT → RLHF chat recipe.

## 📰 Articles / Blogs (free, no paywall)
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — the SFT landscape, full-vs-PEFT, data quality.
- [Hugging Face — PEFT for SFT](https://huggingface.co/blog/peft) — **Hugging Face** — running SFT efficiently with adapters.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — SFT as the foundation that preference methods refine.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — SFT/instruction tuning in the alignment chapter.
- [Dive into Deep Learning — **Ch. 16 "Fine-Tuning"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — supervised fine-tuning foundations with code.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.03 Transfer Learning & Fine-Tuning](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.03_Transfer_Learning_and_Fine_Tuning.md)
- Related concepts: [LoRA / PEFT](12-LoRA-and-PEFT.md) · [Instruction Tuning](14-Instruction-Tuning.md) · [RLHF & DPO](15-RLHF-and-DPO.md)

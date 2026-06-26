---
id: "09-llms/supervised-fine-tuning/references"
topic: "Supervised Fine-Tuning — References"
parent: "09-llms/supervised-fine-tuning"
type: references
updated: 2026-06-27
---

# Supervised Fine-Tuning — references and further reading

> Companion link library for **[Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links are free / no-paywall.

**Start here — suggested path**:
1. **See the whole pipeline** — watch [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (**Andrej Karpathy**). *Pretraining → SFT → RLHF, with SFT's exact role in turning a base model into an assistant.*
2. **Read the canonical recipe** — read [InstructGPT](https://arxiv.org/abs/2203.02155) (**Ouyang et al. 2022**), §3.1 — the SFT stage: how demonstrations are collected and trained with the supervised loss.
3. **Get the "less is more" insight** — read [LIMA](https://arxiv.org/abs/2305.11206) (**Zhou et al. 2023**). *Why 1,000 curated examples beat tens of thousands — the Superficial Alignment Hypothesis.*
4. **Survey the landscape** — read [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) (**Sebastian Raschka**). *Full-vs-PEFT, data quality, and the gotchas.*
5. **Do it hands-on** — work through the [TRL SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer) (**Hugging Face**). *Completion-only masking and chat templates in real code.*

**Videos**:
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — the clearest end-to-end account of where SFT sits between pretraining and RLHF, by one of the field's best teachers.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — builds the exact next-token cross-entropy loop that SFT reuses; watch this if the loss in the math section still feels abstract.
- [Fine-tuning Large Language Models (with example code)](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — a complete, concrete SFT walkthrough from data to trained model.
- [LoRA & QLoRA Fine-tuning Explained In-Depth](https://www.youtube.com/watch?v=t1caDsMzWBk) — **Mark Hennings** — the efficient (LoRA/QLoRA) way SFT is actually run on consumer hardware.

**Courses (free)**:
- [Hugging Face LLM Course — fine-tuning chapters](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — SFT with the `trl` SFTTrainer, chat templates, and completion-only masking, end to end.
- [Stanford CS336 — Language Modeling from Scratch (alignment & post-training)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — SFT within the full post-training stack.

**Articles / blogs (free, no paywall)**:
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — the SFT landscape: full vs PEFT, data quality, and where each fits.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — SFT as the foundation that preference methods (RLHF, DPO) refine.
- [Supervised Fine-tuning Trainer (SFTTrainer) docs](https://huggingface.co/docs/trl/en/sft_trainer) — **Hugging Face TRL** — the reference implementation of SFT, including `completion_only_loss` / response masking and chat-template handling.
- [Chat Templates](https://huggingface.co/docs/transformers/en/chat_templating) — **Hugging Face** — how `apply_chat_template` formats turns; the source of the train/inference-mismatch pitfall.
- [Stanford Alpaca: An Instruction-following LLaMA Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — **Stanford CRFM** — the SFT run that popularized cheap, synthetic-data instruction tuning (52K Self-Instruct examples).
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) — **Hugging Face** — running SFT efficiently with adapters (LoRA), the common production setup.

**Key papers**:
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — §3.1 is the canonical SFT stage: supervised next-token training on labeler demonstrations, before reward modeling and RLHF. The source for the masked-demonstration objective.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — **Zhou et al. (2023)** — 1,000 curated examples, no RLHF, competitive with much larger pipelines; introduces the **Superficial Alignment Hypothesis** (knowledge from pretraining, SFT teaches format). The source for "quality ≫ quantity."
- [Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — **Wang et al. (2022)** — bootstrapping a large instruction-following SFT set from a model's own generations; the method behind Alpaca-style synthetic data.
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — **Bengio et al. (2003)** — the neural next-token (autoregressive cross-entropy) objective that SFT reuses unchanged.
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — **Touvron et al. (2023)** — a documented end-to-end SFT → RLHF chat recipe; emphasizes SFT data *quality* over quantity.
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — **Rafailov et al. (2023)** — the preference-tuning step *after* SFT, optimizing preferences directly from the SFT model without a reward model (the SFT → DPO path in the pipeline diagram).
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — **Wei et al. (2021)** — instruction-tuning at task-scale: SFT applied to a broad multi-task instruction set for generalization (the scaling story continued in the Instruction-Tuning chapter).

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting, and In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — SFT / instruction tuning inside the alignment chapter.
- [Deep Learning — Ch. 5 (maximum likelihood & cross-entropy)](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville (2016)** — the MLE / cross-entropy equivalence underlying the next-token loss SFT minimizes.

**In this platform**:
- Concept page (full explanation): [Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md)
- Foundations (the loss SFT reuses): [Language-Modeling Objectives](../01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md) · [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)
- Run it efficiently: [LoRA & PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md)
- What comes next: [Instruction Tuning](../14-Instruction-Tuning/14-Instruction-Tuning.md) · [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md)
- Alternatives when SFT can't add knowledge: [RAG Fundamentals](../../11.%20RAG_and_LLM_Applications/concepts/01-RAG-Fundamentals.md)

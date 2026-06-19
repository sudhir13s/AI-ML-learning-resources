---
id: "06-nlp/text-summarization"
topic: "Text Summarization (extractive & abstractive)"
parent: "06-nlp"
level: intermediate
prereqs: ["seq2seq-encoder-decoder", "contextual-embeddings"]
interview_frequency: medium
updated: 2026-06-19
---

# Text Summarization — extractive & abstractive
> Condensing a document into a short version that keeps the key information. **Extractive** picks the
> most important existing sentences (graph ranking like TextRank); **abstractive** *generates* new
> sentences with a seq2seq model (pointer-generator, BART, PEGASUS).

**Why it matters:** summarization is a flagship generation task and a clean way to probe seq2seq,
copy mechanisms, and evaluation. Be ready to contrast **extractive vs abstractive**, explain
**TextRank** (PageRank on a sentence graph), why naive seq2seq hallucinates and how
**pointer-generator** copying + coverage help, what **PEGASUS**'s gap-sentence pretraining does, and
why **ROUGE** is the standard (and flawed) metric.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [What is Text Summarization? Extractive & Abstractive](https://www.youtube.com/watch?v=UEikjJ6c63A) (**OnTimeNotes**). *The two paradigms before any model.*
2. **See extractive ranking** — read the [TextRank paper](https://aclanthology.org/W04-3252/) (**Mihalcea & Tarau, 2004**). *PageRank-on-sentences, simple and strong.*
3. **Get abstractive seq2seq** — read the [Pointer-Generator paper](https://arxiv.org/abs/1704.04368) (**See et al., 2017**). *Copying + coverage to stop repetition and OOV errors.*
4. **Read the modern source** — [PEGASUS](https://arxiv.org/abs/1912.08777) + [BART](https://arxiv.org/abs/1910.13461). *Pretraining objectives built for summarization.*
5. **Make it concrete** — code it with the [HF Summarization guide](https://huggingface.co/docs/transformers/tasks/summarization). *Fine-tune and evaluate with ROUGE.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — Ch. 7: Summarization](https://huggingface.co/learn/llm-course/chapter7/5) — **Hugging Face** — fine-tune an abstractive summarizer, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — NLG lectures covering summarization.

## 🎥 Videos
- [What is Text Summarization? Extractive & Abstractive](https://www.youtube.com/watch?v=UEikjJ6c63A) — **OnTimeNotes** — clear framing of the two paradigms.
- [Text Summarization — Extractive vs. Abstractive with HF Transformers](https://www.youtube.com/watch?v=2NQfcS3oIyM) — **SH AI Academy** — both approaches in code.
- [Summarize Text using HuggingFace's Summarization Pipeline](https://www.youtube.com/watch?v=LK9dVN9yMYY) — **Bhavesh Bhatt** — abstractive summarization in a few lines.
- [BERT for Extractive Summarization (paper walkthrough)](https://www.youtube.com/watch?v=JU6eSLsp6vI) — **TechViz** — how encoders rank sentences for extraction.

## 📄 Key Papers
- [TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — **Mihalcea & Tarau (2004)** — graph-based extractive summarization.
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) — **See, Liu & Manning (2017)** — copy mechanism + coverage.
- [PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — **Zhang et al. (2020)** — pretraining tailored to summarization.
- [BART: Denoising Seq2Seq Pre-training](https://arxiv.org/abs/1910.13461) — **Lewis et al. (2020)** — the standard abstractive-summarization backbone.

## 📰 Articles / Blogs (free, no paywall)
- [Summarization (Hugging Face task guide)](https://huggingface.co/docs/transformers/tasks/summarization) — **Hugging Face** — fine-tune + evaluate with ROUGE.
- [Text Summarization using TextRank](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/) — **Analytics Vidhya** — implement extractive TextRank from scratch (free).

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 11 "Information Retrieval and Retrieval-Augmented Generation"**](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — generation + retrieval that underpin summarization.
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Machine Translation" (seq2seq + ROUGE-adjacent eval)**](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — the seq2seq machinery summarization reuses.

## 🔗 In this platform
- Prior step: [08 Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md) — the engine of abstractive summarization.
- Evaluation: [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md) — ROUGE defined precisely.
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)

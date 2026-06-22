---
id: "06-nlp/text-summarization/references"
topic: "Text Summarization — References"
parent: "06-nlp/text-summarization"
type: references
updated: 2026-06-22
---

# Text Summarization — references and further reading

> Companion link library for **[Text Summarization](13-Text-Summarization.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame the two paradigms** — watch [What is Text Summarization? Extractive & Abstractive](https://www.youtube.com/watch?v=UEikjJ6c63A) (**OnTimeNotes**). *The select-vs-generate distinction before any model.*
2. **See extractive ranking** — read the [TextRank paper](https://aclanthology.org/W04-3252/) (**Mihalcea & Tarau, 2004**). *PageRank-on-sentences — simple, unsupervised, still strong.*
3. **Get abstractive copying** — read [Get To The Point (Pointer-Generator)](https://arxiv.org/abs/1704.04368) (**See, Liu & Manning, 2017**). *The $p_{gen}$ copy/generate switch + coverage that fix OOV and repetition.*
4. **Read the modern backbones** — skim [PEGASUS](https://arxiv.org/abs/1912.08777) and [BART](https://arxiv.org/abs/1910.13461). *Pretraining objectives built for (or ideal for) summarization.*
5. **Confront evaluation** — read [On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) (**Maynez et al., 2020**). *Why ROUGE isn't enough and hallucination is the real problem.*
6. **Make it concrete** — code it with the [Hugging Face Summarization guide](https://huggingface.co/docs/transformers/tasks/summarization). *Fine-tune and evaluate with ROUGE.*

**Videos**:
- [What is Text Summarization? Extractive & Abstractive](https://www.youtube.com/watch?v=UEikjJ6c63A) — **OnTimeNotes** — clear framing of the two paradigms.
- [Text Summarization — Extractive vs. Abstractive with HF Transformers](https://www.youtube.com/watch?v=2NQfcS3oIyM) — **SH AI Academy** — both approaches, in code.
- [Summarize Text using Hugging Face's Summarization Pipeline](https://www.youtube.com/watch?v=LK9dVN9yMYY) — **Bhavesh Bhatt** — abstractive summarization in a few lines.
- [BERT for Extractive Summarization (BERTSUM walkthrough)](https://www.youtube.com/watch?v=JU6eSLsp6vI) — **TechViz** — how an encoder ranks sentences for extraction.

**Courses (free)**:
- [Hugging Face LLM Course — Ch. 7: Summarization](https://huggingface.co/learn/llm-course/chapter7/5) — **Hugging Face** — fine-tune an abstractive summarizer, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the seq2seq + attention + NLG lectures summarization builds on.

**Articles / blogs (free, no paywall)**:
- [Summarization (Hugging Face task guide)](https://huggingface.co/docs/transformers/tasks/summarization) — **Hugging Face** — end-to-end fine-tune + ROUGE evaluation.
- [Taming Recurrent Neural Networks for Better Summarization](https://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html) — **Abigail See** — the pointer-generator's first author explaining copy + coverage, with examples.
- [Introduction to Text Summarization with TextRank](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/) — **Analytics Vidhya** — implement extractive TextRank from scratch in Python.
- [PEGASUS: A State-of-the-Art Model for Abstractive Summarization](https://research.google/blog/pegasus-a-state-of-the-art-model-for-abstractive-text-summarization/) — **Google Research** — the gap-sentence-generation idea from the team that built it.

**Key papers**:
- [TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — **Mihalcea & Tarau (2004)** — graph-based extractive summarization (PageRank on a sentence graph).
- [LexRank: Graph-based Lexical Centrality as Salience](https://arxiv.org/abs/1109.2128) — **Erkan & Radev (2004)** — the TF-IDF-cosine variant of sentence-centrality summarization.
- [The Use of MMR for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) — **Carbonell & Goldstein (1998)** — Maximal Marginal Relevance, the relevance−redundancy trade-off.
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) — **See, Liu & Manning (2017)** — copy mechanism ($p_{gen}$) + coverage to fix OOV and repetition.
- [PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — **Zhang et al. (2020)** — pretraining objective tailored to summarization (GSG).
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — **Lewis et al. (2020)** — the standard abstractive-summarization backbone.
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — text-to-text framing ("summarize:" prefix).
- [Text Summarization with Pretrained Encoders (BERTSUM)](https://arxiv.org/abs/1908.08345) — **Liu & Lapata (2019)** — BERT for extractive *and* abstractive summarization.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — **Lin (2004)** — the standard n-gram-overlap metric, defined.
- [On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — **Maynez et al. (2020)** — abstractive models hallucinate; ROUGE doesn't catch it.
- [Evaluating the Factual Consistency of Abstractive Summarization (FactCC)](https://arxiv.org/abs/1910.12840) — **Kryściński et al. (2020)** — a trained consistency classifier.
- [Asking and Answering Questions to Evaluate Factual Consistency (QAGS)](https://arxiv.org/abs/2004.04228) — **Wang et al. (2020)** — QA-based faithfulness evaluation.
- [SummaC: Re-Visiting NLI-based Models for Inconsistency Detection](https://arxiv.org/abs/2111.09525) — **Laban et al. (2022)** — entailment-based faithfulness checking.
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — **Beltagy et al. (2020)** — sparse attention (and LED) for long-input summarization.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Machine Translation" (seq2seq + ROUGE-adjacent eval)](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — the encoder–decoder + attention machinery abstractive summarization reuses.
- [Speech and Language Processing, 3rd ed. — Ch. 11 "Information Retrieval and Retrieval-Augmented Generation"](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — retrieval that underpins query-focused / multi-document summarization.

**In this platform**:
- Concept page (full explanation): [Text Summarization](13-Text-Summarization.md)
- The engine of abstractive summarization: [08 Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md) · [16 Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [15 Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)
- How summaries are decoded: [17 Decoding Strategies](17-Decoding-Strategies.md)
- How summaries are scored (ROUGE in full): [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md)
- Sentence scoring inputs: [03 Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md) · [06 Contextual Embeddings (ELMo, BERT)](06-Contextual-Embeddings-ELMo-BERT.md)
- Query-focused / multi-document retrieval: [16 Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search.md)
- Coherence / dangling pronouns: [14 Coreference Resolution](14-Coreference-Resolution.md)

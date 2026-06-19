---
id: "06-nlp/coreference-resolution"
topic: "Coreference Resolution"
parent: "06-nlp"
level: advanced
prereqs: ["sequence-labeling-pos-ner", "contextual-embeddings"]
interview_frequency: low
updated: 2026-06-19
---

# Coreference Resolution
> Linking all expressions in a text that refer to the same entity — "Barack Obama … he … the
> president" all point to one person. The model clusters **mentions** into coreference chains, which
> is essential for understanding who/what a document is about.

**Why it matters:** coreference is the structured-prediction task behind real document understanding,
information extraction, and summarization. Be ready to explain **mention detection → mention pairing
→ clustering**, the difference between **anaphora** and **coreference**, classic mention-pair vs
**mention-ranking** models, and how the modern **end-to-end span-ranking** model (span
representations + attention, no pipeline) works on top of contextual embeddings.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [SLP3 Ch. 23](https://web.stanford.edu/~jurafsky/slp3/23.pdf) intro on mentions, chains, and anaphora. *Frame the task before the models.*
2. **Watch it explained** — [Anaphora and Coreference Resolution](https://www.youtube.com/watch?v=jaN8kJ7JeY0) (**IIT Madras**). *The linguistic phenomena and why they're hard.*
3. **Get the modern model** — watch [Stanford CS224N Lec 15 — Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk) (**Stanford, Manning**). *Mention-ranking → end-to-end neural coref.*
4. **Read the source** — [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) (**Lee et al., 2017**). *Span-ranking without a hand-built pipeline.*
5. **Make it concrete** — try [NeuralCoref (spaCy)](https://github.com/huggingface/neuralcoref). *Resolve coreference clusters on real text.*

## 🎓 Courses (free)
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the dedicated coreference-resolution lecture.
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — the contextual encoders end-to-end coref builds on.

## 🎥 Videos
- [Anaphora and Coreference Resolution, Discourse Connectives](https://www.youtube.com/watch?v=jaN8kJ7JeY0) — **IIT Madras B.S. Programme** — the linguistic foundations.
- [Stanford CS224N Lec 15 — Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk) — **Stanford (Manning)** — mention-ranking → end-to-end neural model.
- [Reference Resolution — Anaphora & Coreference](https://www.youtube.com/watch?v=v73Xc7GaR60) — **Dhana DataSciEngg Lectures** — worked examples of resolving references.
- [Coreference Resolution in NLP](https://www.youtube.com/watch?v=Zz5UveXquLY) — **TecHno RayZ** — a concise concept-level overview.

## 📄 Key Papers
- [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — **Lee et al. (2017)** — the span-ranking model that removed the pipeline.
- [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392) — **Lee, He & Zettlemoyer (2018)** — refines the end-to-end model with higher-order reasoning.

## 📰 Articles / Blogs (free, no paywall)
- [NeuralCoref (GitHub)](https://github.com/huggingface/neuralcoref) — **Hugging Face** — fast neural coreference in spaCy, with explanation + code.
- [Coreference Resolution chapter (SLP3 draft)](https://web.stanford.edu/~jurafsky/slp3/23.pdf) — **Jurafsky & Martin** — the free, definitive write-up.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 23 "Coreference Resolution and Entity Linking"**](https://web.stanford.edu/~jurafsky/slp3/23.pdf) — **Jurafsky & Martin** — mentions, chains, mention-ranking, neural coref.
- [Speech and Language Processing, 3rd ed. — **Ch. 24 "Discourse Coherence"**](https://web.stanford.edu/~jurafsky/slp3/24.pdf) — **Jurafsky & Martin** — the discourse context coreference lives in.

## 🔗 In this platform
- Prior step: [09 Sequence Labeling — POS & NER](09-Sequence-Labeling-POS-and-NER.md) — mention detection reuses entity tagging.
- Concept depth (the *why*): [AI-ML-intuition 1.06 Scaled Dot-Product (span scoring via attention)](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Used by: [13 Text Summarization](13-Text-Summarization.md) and [11 Question Answering](11-Question-Answering.md) for document-level understanding.

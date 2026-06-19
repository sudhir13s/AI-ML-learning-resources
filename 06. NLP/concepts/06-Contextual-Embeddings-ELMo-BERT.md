---
id: "06-nlp/contextual-embeddings"
topic: "Contextual Embeddings (ELMo · BERT-as-embeddings)"
parent: "06-nlp"
level: intermediate
prereqs: ["word-embeddings", "transformer-architecture"]
interview_frequency: high
updated: 2026-06-19
---

# Contextual Embeddings — ELMo · BERT-as-embeddings
> Word vectors that **change with context**. Static embeddings (word2vec/GloVe) give "bank" one
> vector forever; contextual models (ELMo, BERT) read the whole sentence and emit a *different*
> vector for "bank" in *river bank* vs *savings bank* — capturing polysemy and syntax.

**Why it matters:** this is the leap that reset NLP ("the ImageNet moment"). Be ready to explain the
limitation of static embeddings, how **ELMo** stacks a bidirectional LM, how **BERT** uses masked-LM
pretraining for deep bidirectionality, and how you **extract** contextual vectors (which layers, how
to pool) versus fine-tuning the whole model.

**⭐ Start here — suggested path:**

1. **Build intuition** — [The Illustrated BERT, ELMo & co.](https://jalammar.github.io/illustrated-bert/) (**Jay Alammar**). *The clearest visual story of how contextual representations work.*
2. **Watch it explained** — [BERT Neural Network — EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo) (**CodeEmporium**), then [What is BERT?](https://www.youtube.com/watch?v=7kLi8u2dJz0) (**codebasics**). *Two short, complementary takes.*
3. **Place it in history** — [NLP's ImageNet Moment Has Arrived](https://www.ruder.io/nlp-imagenet/) (**Sebastian Ruder**). *Why pretrained contextual models changed everything.*
4. **Read the sources** — [ELMo](https://arxiv.org/abs/1802.05365) → [BERT](https://arxiv.org/abs/1810.04805). *Deep contextualized representations, then masked-LM bidirectionality.*
5. **Make it concrete** — [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) (**Chris McCormick**). *Extract real contextual vectors and inspect them.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — how pretrained encoders produce and use contextual representations, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the contextual-representations + pretraining lectures (ELMo → BERT).

## 🎥 Videos
- [BERT Neural Network — EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo) — **CodeEmporium** — clear intuition for masked-LM and why bidirectional context matters.
- [What is BERT?](https://www.youtube.com/watch?v=7kLi8u2dJz0) — **codebasics** — gentle, example-driven intro.
- [Transformer Models and BERT Model: Overview](https://www.youtube.com/watch?v=hsp1OAcoLBY) — **Google Cloud** — concise official overview tying transformers to BERT.
- [BERT Explained!](https://www.youtube.com/watch?v=OR0wfP2FD3c) — **Connor Shorten** — walks through the paper's key ideas.

## 📄 Key Papers
- [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/abs/1802.05365) — **Peters et al. (2018)** — contextual vectors from a bidirectional LM.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — masked-LM pretraining; the paradigm shift.

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated BERT, ELMo & co.](https://jalammar.github.io/illustrated-bert/) — **Jay Alammar** — the definitive visual explainer.
- [NLP's ImageNet Moment Has Arrived](https://www.ruder.io/nlp-imagenet/) — **Sebastian Ruder** — the context for why pretrained contextual models took over.
- [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) — **Chris McCormick** — hands-on extraction of contextual vectors (blog + Colab).

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 11 "Fine-Tuning and Masked Language Models"**](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — contextual embeddings + BERT in the standard text.
- [Dive into Deep Learning — **Ch. 15.8–15.10 (BERT: pretraining + fine-tuning)**](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html) — **Zhang et al.** — BERT from scratch with runnable code.

## 🔗 In this platform
- Contrast with static vectors: [05 Word Embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · more in the [concept index](README.md)
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
- Canonical homes: the BERT/Transformer models live in [08. LLMs](../../08.%20LLMs/README.md) and [5. Deep Learning](../../05.%20Deep_Learning/README.md) — this card is about *using* them for representations.

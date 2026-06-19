---
id: "06-nlp/tokenization"
topic: "Tokenization & Subword Algorithms (BPE · WordPiece · SentencePiece · Unigram)"
parent: "06-nlp"
level: intermediate
prereqs: ["text-preprocessing"]
interview_frequency: very-high
updated: 2026-06-19
---

# Tokenization & Subword Algorithms — BPE · WordPiece · SentencePiece · Unigram
> How raw text becomes the integer tokens a model consumes. Modern systems split on **subword**
> units — frequent words stay whole, rare words break into pieces — so a fixed vocabulary can cover
> any input (no out-of-vocabulary holes) while keeping sequences short.

**Why it matters:** tokenization decides vocabulary size, sequence length (hence compute/cost), and
how gracefully a model handles rare words, typos, code, and other languages. Be ready to explain
*why subword beats word- or character-level*, how **BPE** learns merges, **byte-level BPE** (GPT-2),
**WordPiece** (BERT) vs **Unigram** (SentencePiece/T5), and how vocab size trades off against
sequence length.

**⭐ Start here — suggested path:**

1. **Build intuition** — [Tokenization & Byte Pair Encoding](https://www.youtube.com/watch?v=gstdcCDqdlc) (**Luis Serrano**), then skim [Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) (**Hugging Face**). *Get the "why subword" picture before the algorithms.*
2. **See the algorithm** — [BPE](https://huggingface.co/learn/llm-course/en/chapter6/5) and [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6) in the HF course. *Walk through merges and vocab building step by step.*
3. **Build it from scratch** — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (**Karpathy**). *Implementing byte-level BPE makes it stick.*
4. **Read the sources** — [BPE](https://arxiv.org/abs/1508.07909) → [Unigram](https://arxiv.org/abs/1804.10959) → [SentencePiece](https://arxiv.org/abs/1808.06226). *The progression from merge-based to probabilistic subword models.*
5. **Reference** — [SLP3 Ch. 2](https://web.stanford.edu/~jurafsky/slp3/2.pdf) for the textbook treatment.

## 🎓 Courses (free)
- [Hugging Face LLM Course — Ch. 6: The Tokenizers Library](https://huggingface.co/learn/llm-course/chapter6/1) — **Hugging Face** — builds BPE, WordPiece, and Unigram tokenizers in code; the most practical treatment.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the subword-modeling lecture sets the linguistic + modeling context.

## 🎥 Videos
- [Tokenization & Byte Pair Encoding](https://www.youtube.com/watch?v=gstdcCDqdlc) — **Luis Serrano** — gentle, visual intuition for what BPE does and why.
- [Byte Pair Encoding Tokenization](https://www.youtube.com/watch?v=HEikzVL-lZU) — **Hugging Face** — concise, official walkthrough of the BPE training loop.
- [WordPiece Tokenization](https://www.youtube.com/watch?v=qpv6ms_t_1A) — **Hugging Face** — how BERT's tokenizer scores and merges (vs BPE).
- [Unigram Tokenization](https://www.youtube.com/watch?v=TGZfZVuF9Yc) — **Hugging Face** — the probabilistic model behind SentencePiece/T5.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — **Andrej Karpathy** — build a byte-level BPE tokenizer line by line (≈2 hrs, deep).

## 📄 Key Papers
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — **Sennrich et al. (2016)** — brought BPE to NLP; the foundation.
- [Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959) — **Kudo (2018)** — the probabilistic alternative to BPE merges.
- [SentencePiece](https://arxiv.org/abs/1808.06226) — **Kudo & Richardson (2018)** — language-agnostic, whitespace-free tokenization (used by T5, LLaMA).
- [Google's Neural Machine Translation (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — introduces the WordPiece scheme used by BERT.

## 📰 Articles / Blogs (free, no paywall)
- [Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — **Hugging Face** — crisp side-by-side of BPE vs WordPiece vs Unigram, fully open.
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5) — **Hugging Face** — worked numeric example of learning merges.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 2 "Regular Expressions, Text Normalization, and Edit Distance"**](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — tokenization + BPE in the standard text.
- [Dive into Deep Learning — **Ch. 15.6 "Subword Embedding"**](https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html) — **Zhang et al.** — BPE with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.15 Tokenization & BPE](../../../AI-ML-intuition/Module_1_Representation/1.15_Tokenization_and_BPE.md)
- Next concepts: [05 Word Embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · [06 Contextual Embeddings](06-Contextual-Embeddings-ELMo-BERT.md)
- Related domain: [8. LLMs](../../8.%20LLMs/README.md)

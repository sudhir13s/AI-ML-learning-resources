---
id: "06-nlp/tokenization/references"
topic: "Tokenization & Subword Algorithms — References"
parent: "06-nlp/tokenization"
type: references
updated: 2026-06-22
---

# Tokenization & Subword Algorithms — references and further reading

> Companion link library for **[Tokenization & Subword Algorithms](02-Tokenization-and-Subword-Algorithms.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (the paper's authors) or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (**Andrej Karpathy**). *Implements byte-level BPE from scratch; the single best resource for making tokenization stick.*
2. **See the algorithms side by side** — read [Summary of the tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) (**Hugging Face**). *Crisp contrast of BPE vs WordPiece vs Unigram, fully open.*
3. **Walk the merges** — work through [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5) (**Hugging Face course**). *A numeric BPE training example you can redo by hand.*
4. **Read the source** — read [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (**Sennrich et al., 2016**). *The paper that brought BPE to NLP.*
5. **Connect to the probabilistic view** — read [Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959) (**Kudo, 2018**). *The probabilistic alternative to merge-based tokenization.*

**Videos**:
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — **Andrej Karpathy** — builds a byte-level BPE tokenizer line by line (~2 hrs); the definitive deep walkthrough.
- [Tokenization & Byte Pair Encoding](https://www.youtube.com/watch?v=gstdcCDqdlc) — **Luis Serrano** — gentle, visual intuition for what BPE does and why subword wins.
- [Byte Pair Encoding Tokenization](https://www.youtube.com/watch?v=HEikzVL-lZU) — **Hugging Face** — concise official walkthrough of the BPE training loop.
- [WordPiece Tokenization](https://www.youtube.com/watch?v=qpv6ms_t_1A) — **Hugging Face** — how BERT's tokenizer scores merges by likelihood (vs BPE's frequency).
- [Unigram Tokenization](https://www.youtube.com/watch?v=TGZfZVuF9Yc) — **Hugging Face** — the probabilistic prune-down model behind SentencePiece/T5.

**Interactive & visual**:
- [OpenAI Tokenizer playground](https://platform.openai.com/tokenizer) — **OpenAI** — paste any text and watch it split into colored tokens with a live count; the fastest way to *see* rare-word splitting and the multilingual tax.
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — **Diagram (community)** — side-by-side token counts across GPT, Llama, and other tokenizers, with token ids shown.

**Courses (free)**:
- [Hugging Face LLM Course — Ch. 6: The Tokenizers Library](https://huggingface.co/learn/llm-course/en/chapter6/1) — **Hugging Face** — builds BPE, WordPiece, and Unigram tokenizers in code; the most practical hands-on treatment.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the subword-modeling lecture sets the linguistic and modeling context.

**Articles / blogs (free, no paywall)**:
- [Summary of the tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) — **Hugging Face** — crisp side-by-side of BPE vs WordPiece vs Unigram, fully open.
- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5) — **Hugging Face** — worked numeric example of learning BPE merges.
- [WordPiece tokenization](https://huggingface.co/learn/llm-course/en/chapter6/6) — **Hugging Face** — the likelihood-score merge rule and `##` encoding, worked through.
- [Unigram tokenization](https://huggingface.co/learn/llm-course/en/chapter6/7) — **Hugging Face** — the EM training + Viterbi decoding of the Unigram model, step by step.
- [The Technical User's Introduction to LLM Tokenization](https://christophergs.com/blog/understanding-llm-tokenization) — **Christopher Samiullah** — practical tour of tiktoken, byte-level BPE, and the gotchas (digits, whitespace, glitch tokens).

**Papers**:
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — **Sennrich, Haddow & Birch (2016)** — brought BPE to NLP; the foundation of modern subword tokenization.
- [Japanese and Korean Voice Search (WordPiece)](https://research.google/pubs/japanese-and-korean-voice-search/) — **Schuster & Nakajima (2012)** — the original WordPiece, later adopted by BERT.
- [Google's Neural Machine Translation System (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — describes the WordPiece scheme later used by BERT.
- [Subword Regularization: Improving NMT with Multiple Subword Candidates (Unigram LM)](https://arxiv.org/abs/1804.10959) — **Kudo (2018)** — the probabilistic Unigram model and subword-regularization sampling.
- [SentencePiece: A simple and language-independent subword tokenizer](https://arxiv.org/abs/1808.06226) — **Kudo & Richardson (2018)** — whitespace-as-`▁`, fully reversible, language-agnostic (used by T5, LLaMA).
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — **Radford et al. (2019)** — §2.2 introduces byte-level BPE, the no-OOV tokenizer behind GPT.
- [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267) — **Provilkov et al. (2020)** — stochastic BPE segmentation, the BPE analogue of subword regularization.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 2 "Regular Expressions, Text Normalization, and Edit Distance"](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — tokenization and BPE in the standard NLP textbook.
- [Dive into Deep Learning — Ch. 15.6 "Subword Embedding"](https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html) — **Zhang, Lipton, Li & Smola** — BPE explained with runnable code.

**Tools (open source)**:
- [tiktoken](https://github.com/openai/tiktoken) — **OpenAI** — the fast byte-level BPE tokenizer used by GPT-3.5/4; read the code to see byte-level BPE in practice.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) — **Hugging Face** — production BPE/WordPiece/Unigram trainers and encoders in Rust with Python bindings.
- [SentencePiece](https://github.com/google/sentencepiece) — **Google** — the reference implementation of the SentencePiece framework (BPE + Unigram).

**In this platform**:
- Concept page (full explanation): [Tokenization & Subword Algorithms](02-Tokenization-and-Subword-Algorithms.md)
- Concept depth (the *why*): [AI-ML-intuition 1.15 Tokenization & BPE](../../../AI-ML-intuition/Module_1_Representation/1.15_Tokenization_and_BPE.md)
- Comes before this: [01 Text Preprocessing & Normalization](01-Text-Preprocessing-and-Normalization.md)
- Builds on this: [05 Word Embeddings (Word2Vec / GloVe / FastText)](05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · [06 Contextual Embeddings (ELMo / BERT)](06-Contextual-Embeddings-ELMo-BERT.md)
- Puts it to work: [09. LLMs](../../09.%20LLMs/README.md) · [KV Cache (tokens are what fills it)](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md)

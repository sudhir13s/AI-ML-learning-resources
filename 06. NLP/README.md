---
id: "06-nlp"
topic: "Natural Language Processing"
level: intermediate
prereqs: ["deep-learning"]
updated: 2026-06-27
---

# Natural Language Processing
> Teaching machines to understand and generate language — from word vectors to transformers.

**⭐ Start here:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the single best explainer of the architecture behind all modern NLP.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links).
> **✅ ready.** New to NLP? Start with the field overview above, then work top to bottom.

### Representation & classical models
1. ✅ [Text Preprocessing & Normalization (tokenize, stem, lemmatize, stopwords)](01-Text-Preprocessing-and-Normalization/01-Text-Preprocessing-and-Normalization.md)
2. ✅ [Tokenization & Subword Algorithms (BPE · WordPiece · SentencePiece · Unigram)](02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md)
3. ✅ [Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md)
4. ✅ [N-gram Language Models & Smoothing](04-N-gram-Language-Models-and-Smoothing/04-N-gram-Language-Models-and-Smoothing.md)
5. ✅ [Word Embeddings — Word2Vec · GloVe · FastText](05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md)
6. ✅ [Contextual Embeddings (ELMo · BERT-as-embeddings)](06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
7. ✅ [Sentence & Document Embeddings (Sentence-BERT · USE)](07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md)

### Sequence modeling & tasks
8. ✅ [Sequence-to-Sequence & Encoder–Decoder for MT](08-Sequence-to-Sequence-and-Encoder-Decoder/08-Sequence-to-Sequence-and-Encoder-Decoder.md) *(applies the DL attention/transformer to language)*
9. ✅ [Sequence Labeling — POS & NER (HMM/CRF → neural)](09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md)
10. ✅ [Text Classification & Sentiment Analysis](10-Text-Classification-and-Sentiment-Analysis/10-Text-Classification-and-Sentiment-Analysis.md)
11. ✅ [Question Answering (extractive & generative)](11-Question-Answering/11-Question-Answering.md)
12. ✅ [Machine Translation](12-Machine-Translation/12-Machine-Translation.md)
13. ✅ [Text Summarization (extractive & abstractive)](13-Text-Summarization/13-Text-Summarization.md)
14. ✅ [Coreference Resolution](14-Coreference-Resolution/14-Coreference-Resolution.md)
15. ✅ [Topic Modeling (LDA · NMF)](15-Topic-Modeling-LDA-NMF/15-Topic-Modeling-LDA-NMF.md)
16. ✅ [Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md)

### Generation & evaluation
17. ✅ [Decoding Strategies (greedy · beam · top-k · top-p/nucleus · temperature)](17-Decoding-Strategies/17-Decoding-Strategies.md)
18. ✅ [NLP Evaluation Metrics (BLEU · ROUGE · METEOR · perplexity · BERTScore · F1/EM)](18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Architectures & mechanisms** — Attention · Transformers · Positional Encodings · RNN / LSTM / GRU → [Deep Learning](../05.%20Deep_Learning/README.md)
- **Large language models** — BERT · GPT · T5 / BART · Fine-tuning · Prompting · RLHF → [LLMs](../09.%20LLMs/README.md)
- **Retrieval-augmented generation** → [RAG & LLM Applications](../11.%20RAG_and_LLM_Applications/README.md)

## 🎓 Courses (free)
- [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive university NLP course; lectures on YouTube.
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — **Hugging Face** — free, code-first, modern (transformers in practice).

## 🎥 Videos
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — build a transformer line by line.
- [Transformers (chapters 5–7)](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — attention, visualized.

## 📄 Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the transformer; non-negotiable.
- [BERT](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the pretraining paradigm shift.

## 📚 Books (free)
- [Speech and Language Processing (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — free draft; the field's standard reference.

## 🔗 In this platform
- Math: [AI-ML-intuition 1.02 embeddings, 1.15 tokenization, Module 4 attention](../../AI-ML-intuition/) · LLMs: [09. LLMs](../09.%20LLMs/README.md)

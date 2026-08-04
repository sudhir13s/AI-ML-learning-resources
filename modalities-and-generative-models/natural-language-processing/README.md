---
id: "06-nlp"
topic: "Natural Language Processing"
level: intermediate
built_from: ["deep-learning"]
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
1. ✅ [Text Preprocessing & Normalization (tokenize, stem, lemmatize, stopwords)](text-preprocessing-and-normalization/text-preprocessing-and-normalization.md)
2. ✅ [Tokenization & Subword Algorithms (BPE · WordPiece · SentencePiece · Unigram)](tokenization-and-subword-algorithms/tokenization-and-subword-algorithms.md)
3. ✅ [Bag-of-Words & TF-IDF](bag-of-words-and-tf-idf/bag-of-words-and-tf-idf.md)
4. ✅ [N-gram Language Models & Smoothing](n-gram-language-models-and-smoothing/n-gram-language-models-and-smoothing.md)
5. ✅ [Word Embeddings — Word2Vec · GloVe · FastText](word-embeddings-word2vec-glove-fasttext/word-embeddings-word2vec-glove-fasttext.md)
6. ✅ [Contextual Embeddings (ELMo · BERT-as-embeddings)](contextual-embeddings-elmo-bert/contextual-embeddings-elmo-bert.md)
7. ✅ [Sentence & Document Embeddings (Sentence-BERT · USE)](sentence-and-document-embeddings/sentence-and-document-embeddings.md)

### Sequence modeling & tasks
8. ✅ [Sequence-to-Sequence & Encoder–Decoder for MT](sequence-to-sequence-and-encoder-decoder/sequence-to-sequence-and-encoder-decoder.md) *(applies the DL attention/transformer to language)*
9. ✅ [Sequence Labeling — POS & NER (HMM/CRF → neural)](sequence-labeling-pos-and-ner/sequence-labeling-pos-and-ner.md)
10. ✅ [Text Classification & Sentiment Analysis](text-classification-and-sentiment-analysis/text-classification-and-sentiment-analysis.md)
11. ✅ [Question Answering (extractive & generative)](question-answering/question-answering.md)
12. ✅ [Machine Translation](machine-translation/machine-translation.md)
13. ✅ [Text Summarization (extractive & abstractive)](text-summarization/text-summarization.md)
14. ✅ [Coreference Resolution](coreference-resolution/coreference-resolution.md)
15. ✅ [Topic Modeling (LDA · NMF)](topic-modeling-lda-nmf/topic-modeling-lda-nmf.md)
16. ✅ [Information Retrieval & Semantic Search](information-retrieval-and-semantic-search/information-retrieval-and-semantic-search.md)

### Generation & evaluation
17. ✅ [Decoding Strategies (greedy · beam · top-k · top-p/nucleus · temperature)](decoding-strategies/decoding-strategies.md)
18. ✅ [NLP Evaluation Metrics (BLEU · ROUGE · METEOR · perplexity · BERTScore · F1/EM)](nlp-evaluation-metrics/nlp-evaluation-metrics.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Architectures & mechanisms** — Attention · Transformers · Positional Encodings · RNN / LSTM / GRU → [Deep Learning](../../deep-learning/README.md)
- **Large language models** — BERT · GPT · T5 / BART · Fine-tuning · Prompting · RLHF → [LLMs](../../09.%20LLMs/README.md)
- **Retrieval-augmented generation** → [RAG & LLM Applications](../../llms-applications-and-agents/rag-and-knowledge-systems/overview.md)

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
- Math: [AI-ML-intuition 1.02 embeddings, 1.15 tokenization, Module 4 attention](../../../AI-ML-intuition/) · LLMs: [09. LLMs](../../09.%20LLMs/README.md)

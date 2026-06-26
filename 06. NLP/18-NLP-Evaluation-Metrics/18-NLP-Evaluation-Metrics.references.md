---
id: "06-nlp/evaluation-metrics/references"
topic: "NLP Evaluation Metrics — References"
parent: "06-nlp/evaluation-metrics"
type: references
updated: 2026-06-22
---

# NLP Evaluation Metrics — references and further reading

> Companion link library for **[NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (the paper's authors) or a recognized deep explainer, chosen for depth on *this* topic, not popularity. Every link verified.

**Start here — suggested path**:
1. **Build intuition** — read [SLP3 Ch. 3 §"Perplexity"](https://web.stanford.edu/~jurafsky/slp3/3.pdf) and [SLP3 Ch. 13 §"MT Evaluation / BLEU"](https://web.stanford.edu/~jurafsky/slp3/13.pdf) (**Jurafsky & Martin**). *The two anchor metrics, defined precisely.*
2. **See BLEU explained** — watch [BLEU Score (C5W3L06)](https://www.youtube.com/watch?v=DejHQYAGb7Q) (**Andrew Ng**), then [What is the BLEU metric?](https://www.youtube.com/watch?v=M05L1DhFqcw) (**Hugging Face**). *Clipped n-gram precision + brevity penalty, two ways.*
3. **Read the sources** — [BLEU](https://aclanthology.org/P02-1040/) → [ROUGE](https://aclanthology.org/W04-1013/) → [BERTScore](https://arxiv.org/abs/1904.09675). *Surface overlap, then embedding-based scoring.*
4. **See where surface metrics break** — read [How NOT To Evaluate Your Dialogue System](https://aclanthology.org/D16-1230/) (**Liu et al. 2016**). *Why BLEU/ROUGE correlate near-zero on open-ended text.*
5. **Reach the modern paradigm** — read [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (**Zheng et al. 2023**). *LLM-as-judge, its agreement with humans, and its biases.*
6. **Make it concrete** — compute everything with the [Hugging Face `evaluate` library](https://huggingface.co/docs/evaluate/index). *Score real model outputs.*

**Videos**:
- [BLEU Score (C5W3L06)](https://www.youtube.com/watch?v=DejHQYAGb7Q) — **Andrew Ng (DeepLearning.AI)** — the canonical BLEU explanation, n-gram precision and brevity penalty.
- [What is the BLEU metric?](https://www.youtube.com/watch?v=M05L1DhFqcw) — **Hugging Face** — concise official walkthrough with a worked example.
- [What is the ROUGE metric?](https://www.youtube.com/watch?v=TMshhnrEXlg) — **Hugging Face** — recall-oriented summarization scoring, ROUGE-N and ROUGE-L.
- [Understanding BLEU Score in Machine Translation](https://www.youtube.com/watch?v=zZfTFXUMUxc) — **Developers Hutt** — a fully worked BLEU example including the brevity penalty.
- [BERTScore explained](https://www.youtube.com/watch?v=Tkc2vfvBSPg) — **Connor Shorten / Henry AI Labs** — embedding-based scoring and why it beats BLEU on paraphrase.

**Courses (free)**:
- [Hugging Face LLM Course — evaluation & metrics](https://huggingface.co/learn/llm-course/chapter7/4) — **Hugging Face** — compute BLEU/ROUGE/F1 in code alongside training.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — evaluation framed across translation, generation, and QA.

**Articles / blogs (free, no paywall)**:
- [🤗 Evaluate documentation](https://huggingface.co/docs/evaluate/index) — **Hugging Face** — the library plus precise definitions for BLEU, ROUGE, BERTScore, F1, and perplexity.
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) — **Hugging Face** — the exact perplexity computation with a sliding window, free.
- [A Gentle Introduction to Calculating the BLEU Score](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/) — **Jason Brownlee** — BLEU computed step by step in Python.
- [sacreBLEU: standardized, reproducible BLEU](https://github.com/mjpost/sacrebleu) — **Matt Post** — the reference implementation that fixes tokenization so BLEU is comparable across papers.

**Key papers**:
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — **Papineni et al. (2002)** — clipped n-gram precision + brevity penalty; the founding MT metric.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — **Lin (2004)** — recall-oriented summarization metric; ROUGE-N and ROUGE-L (LCS).
- [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909/) — **Banerjee & Lavie (2005)** — unigram alignment with stemming/synonyms + fragmentation penalty.
- [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — **Popović (2015)** — character-level F-score, strong for morphologically rich languages.
- [A Call for Clarity in Reporting BLEU Scores (sacreBLEU)](https://aclanthology.org/W18-6319/) — **Post (2018)** — why tokenization makes BLEU non-reproducible, and the fix.
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — **Zhang et al. (2020)** — greedy cosine matching of contextual embeddings → precision/recall/F1.
- [MoverScore: Text Generation Evaluation with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/abs/1909.02622) — **Zhao et al. (2019)** — embedding matching via optimal transport.
- [BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696) — **Sellam et al. (2020)** — a learned metric fine-tuned on human ratings.
- [COMET: A Neural Framework for MT Evaluation](https://arxiv.org/abs/2009.09025) — **Rei et al. (2020)** — source-aware learned metric, the modern WMT standard.
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — **Rajpurkar et al. (2016)** — the Exact-Match and token-F1 QA metrics.
- [How NOT To Evaluate Your Dialogue System](https://aclanthology.org/D16-1230/) — **Liu et al. (2016)** — surface metrics correlate near-zero with humans on open-ended dialogue.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — **Zheng et al. (2023)** — LLM-as-judge, >80% human agreement, and the position/verbosity/self-preference biases.
- [GLUE: A Multi-Task Benchmark for NLU](https://arxiv.org/abs/1804.07461) — **Wang et al. (2018)** — the multi-task NLU benchmark suite.
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) — **Liang et al. (2022, Stanford)** — many scenarios × many metrics; no single number suffices.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — **Ch. 3 §"Perplexity"**](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — perplexity ↔ cross-entropy, derived.
- [Speech and Language Processing, 3rd ed. — **Ch. 13 "Machine Translation" (BLEU & MT evaluation)**](https://web.stanford.edu/~jurafsky/slp3/13.pdf) — **Jurafsky & Martin** — BLEU and human MT evaluation in context.

**Tools**:
- [sacreBLEU](https://github.com/mjpost/sacrebleu) — **Matt Post** — standardized BLEU/chrF/TER; the reproducible reference implementation.
- [Google Research `rouge-score`](https://github.com/google-research/google-research/tree/master/rouge) — **Google** — the canonical ROUGE-N / ROUGE-L implementation used on the page.
- [`bert-score`](https://github.com/Tiiiger/bert_score) — **Tianyi Zhang et al.** — the official BERTScore library (with baseline rescaling).
- [Hugging Face `evaluate`](https://github.com/huggingface/evaluate) — **Hugging Face** — one API for BLEU, ROUGE, BERTScore, F1, perplexity, and more.

**In this platform**:
- Concept page (full explanation): [NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md)
- Foundations (the metrics these build on): [Classification Metrics — precision/recall/F1](../../03.%20Supervised_Learning/concepts/14-Classification-Metrics.md) · [N-gram Language Models and Smoothing (perplexity)](../04-N-gram-Language-Models-and-Smoothing/04-N-gram-Language-Models-and-Smoothing.md)
- Tasks that use these metrics: [Machine Translation (BLEU/chrF/COMET)](../12-Machine-Translation/12-Machine-Translation.md) · [Text Summarization (ROUGE)](../13-Text-Summarization/13-Text-Summarization.md) · [Question Answering (EM/F1)](../11-Question-Answering/11-Question-Answering.md) · [Sequence Labeling: POS and NER (span F1)](../09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md)
- Related: [Decoding Strategies (how the text being scored is produced)](../17-Decoding-Strategies/17-Decoding-Strategies.md) · [RLHF and Alignment (human preference data & A/B eval)](../../Practitioner-Workflows/RLHF-and-Alignment/RLHF-and-Alignment.md)
- The *why* behind the building blocks: [AI-ML-intuition 3.05 Precision · Recall · F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md) · [5.01 Entropy & KL (perplexity)](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)

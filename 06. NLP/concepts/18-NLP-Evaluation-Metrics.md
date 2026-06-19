---
id: "06-nlp/evaluation-metrics"
topic: "NLP Evaluation Metrics (BLEU · ROUGE · METEOR · perplexity · BERTScore · F1/EM)"
parent: "06-nlp"
level: intermediate
prereqs: ["ngram-language-models", "classification-metrics"]
interview_frequency: high
updated: 2026-06-19
---

# NLP Evaluation Metrics — BLEU · ROUGE · METEOR · perplexity · BERTScore · F1/EM
> How we measure NLP systems. **Perplexity** scores language models intrinsically; **BLEU/METEOR**
> score translation; **ROUGE** scores summarization; **F1/Exact-Match** score QA and labeling;
> **BERTScore** uses embeddings to capture meaning beyond surface n-gram overlap.

**Why it matters:** picking and explaining the right metric is a constant interview and on-the-job
task. Be ready to define **perplexity** (and its link to cross-entropy), how **BLEU** uses
**n-gram precision + brevity penalty**, why **ROUGE** is **recall-oriented** for summaries, what
**METEOR** adds (stemming/synonyms), how **F1/EM** work for QA, and why embedding-based **BERTScore**
fixes BLEU/ROUGE's blindness to paraphrase — plus the limits of all automatic metrics.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [SLP3 Ch. 3 §"Perplexity"](https://web.stanford.edu/~jurafsky/slp3/3.pdf) and [SLP3 Ch. 12 §"BLEU"](https://web.stanford.edu/~jurafsky/slp3/12.pdf). *The two anchor metrics, defined precisely.*
2. **See BLEU explained** — watch [BLEU Score (C5W3L06)](https://www.youtube.com/watch?v=DejHQYAGb7Q) (**Andrew Ng**), then [What is the BLEU metric?](https://www.youtube.com/watch?v=M05L1DhFqcw) (**Hugging Face**). *n-gram precision + brevity penalty, two ways.*
3. **Read the sources** — [BLEU](https://aclanthology.org/P02-1040/) → [ROUGE](https://aclanthology.org/W04-1013/) → [BERTScore](https://arxiv.org/abs/1904.09675). *Surface overlap, then embedding-based scoring.*
4. **Connect to classification** — F1/EM build directly on precision/recall (see the intuition link). *The same metrics, applied to spans and tags.*
5. **Make it concrete** — compute them with the [HF Evaluate metrics guide](https://huggingface.co/learn/llm-course/chapter5/6) and [Hugging Face `evaluate`](https://huggingface.co/docs/evaluate/index). *Score real model outputs.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — evaluation & metrics](https://huggingface.co/learn/llm-course/chapter5/6) — **Hugging Face** — compute BLEU/ROUGE/F1 in code.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — evaluation framed across translation, generation, and QA.

## 🎥 Videos
- [BLEU Score (C5W3L06)](https://www.youtube.com/watch?v=DejHQYAGb7Q) — **Andrew Ng (DeepLearning.AI)** — the canonical BLEU explanation.
- [What is the BLEU metric?](https://www.youtube.com/watch?v=M05L1DhFqcw) — **Hugging Face** — concise official walkthrough.
- [Understanding BLEU Score in Machine Translation](https://www.youtube.com/watch?v=zZfTFXUMUxc) — **Developers Hutt** — worked example with the brevity penalty.
- [Random Sampling for LLMs (generation context for metrics)](https://www.youtube.com/watch?v=mti5XUm22Og) — **Minsuk Heo** — how generated text (that metrics score) is produced.

## 📄 Key Papers
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — **Papineni et al. (2002)** — n-gram precision + brevity penalty.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — **Lin (2004)** — recall-oriented summarization metric.
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — **Zhang et al. (2020)** — embedding-based semantic similarity scoring.

## 📰 Articles / Blogs (free, no paywall)
- [What is the BLEU metric? (HF Course)](https://huggingface.co/learn/llm-course/chapter7/4) — **Hugging Face** — BLEU explained alongside translation training.
- [🤗 Evaluate documentation](https://huggingface.co/docs/evaluate/index) — **Hugging Face** — the library + definitions for BLEU, ROUGE, BERTScore, F1, perplexity.
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) — **Hugging Face** — the precise perplexity computation, free.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 3 §3.7 "Perplexity"**](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — perplexity ↔ cross-entropy.
- [Speech and Language Processing, 3rd ed. — **Ch. 12 §"Automatic Evaluation" (BLEU)**](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — BLEU and MT evaluation.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.05 Precision · Recall · F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md) · [5.01 Entropy & KL (perplexity)](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Used by: [12 Machine Translation](12-Machine-Translation.md) (BLEU), [13 Text Summarization](13-Text-Summarization.md) (ROUGE), [11 Question Answering](11-Question-Answering.md) (F1/EM).

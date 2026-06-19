---
id: "06-nlp/text-classification-sentiment"
topic: "Text Classification & Sentiment Analysis"
parent: "06-nlp"
level: beginner
prereqs: ["bow-tfidf", "logistic-regression"]
interview_frequency: high
updated: 2026-06-19
---

# Text Classification & Sentiment Analysis
> Assigning a label to a whole piece of text — spam/ham, topic, or sentiment (positive/negative/
> neutral). The most common applied NLP task, with a clean ladder of models: **Naive Bayes** →
> **logistic regression on TF-IDF** → **fine-tuned BERT**.

**Why it matters:** classification is the workhorse interview task and the easiest place to reason
about features, baselines, and evaluation. Be ready to derive **Naive Bayes** (and its independence
assumption), explain why **logistic regression on TF-IDF** is a strong baseline, when a **fine-tuned
transformer** is worth it, how to handle **class imbalance**, and why **macro-F1** often beats raw
accuracy for skewed labels.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [SLP3 Ch. 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf). *Naive Bayes + the bag-of-words view of classification.*
2. **See the strong baseline** — read [Text classification](https://lena-voita.github.io/nlp_course/text_classification.html) (**Lena Voita**). *From BoW + logistic regression up to neural classifiers.*
3. **Build a classic pipeline** — watch [Real-World ML with Scikit-Learn (NLP, classifiers)](https://www.youtube.com/watch?v=M9Itm95JzL0) (**Keith Galli**). *TF-IDF → classifier on real text.*
4. **Go neural** — watch [Text Classification Using BERT & TensorFlow](https://www.youtube.com/watch?v=D9yyt6BfgAM) (**codebasics**), and read the [CNN-for-text paper](https://arxiv.org/abs/1408.5882). *When and why deep models help.*
5. **Make it concrete** — fine-tune with the [HF text-classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification). *Train and evaluate a transformer classifier.*

## 🎓 Courses (free)
- [NLP Course for You — Text Classification](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita** — Naive Bayes → logistic regression → neural, fully free.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — classification framed within the modern NLP toolkit.

## 🎥 Videos
- [Real-World ML with Scikit-Learn (NLP, classifiers)](https://www.youtube.com/watch?v=M9Itm95JzL0) — **Keith Galli** — TF-IDF + classic classifiers, end to end.
- [Text Classification Using BERT & TensorFlow](https://www.youtube.com/watch?v=D9yyt6BfgAM) — **codebasics** — fine-tune BERT for spam/sentiment.
- [Neural Network Text Classification with TensorFlow](https://www.youtube.com/watch?v=VtRLrQ3Ev-U) — **freeCodeCamp.org** — long, thorough deep-learning classifier walkthrough.
- [NLP — Text Preprocessing and Text Classification](https://www.youtube.com/watch?v=nxhCyeRR75Q) — **Machine Learning TV** — clean-then-classify in one pipeline.

## 📄 Key Papers
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — **Kim (2014)** — the simple, strong CNN-for-text baseline.
- [Bag of Tricks for Efficient Text Classification (fastText)](https://arxiv.org/abs/1607.01759) — **Joulin et al. (2016)** — linear classifier on n-gram embeddings; fast and competitive.

## 📰 Articles / Blogs (free, no paywall)
- [Text classification (NLP Course for You)](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita** — the clearest free survey from baselines to neural.
- [Text classification (Hugging Face task guide)](https://huggingface.co/docs/transformers/tasks/sequence_classification) — **Hugging Face** — fine-tune and evaluate a transformer classifier.
- [Classification of text documents (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) — **scikit-learn** — runnable BoW/TF-IDF classifier comparison.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 4 "Logistic Regression and Text Classification"**](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — Naive Bayes, logistic regression, sentiment.
- [Natural Language Processing with Python — **Ch. 6 "Learning to Classify Text"**](https://www.nltk.org/book/ch06.html) — **Bird, Klein & Loper** — feature design and classifiers in NLTK.

## 🔗 In this platform
- Prior step: [03 Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md) — the features classic classifiers use.
- Concept depth (the *why*): [AI-ML-intuition 3.05 Precision · Recall · F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Next: [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md) — how classification quality is measured.

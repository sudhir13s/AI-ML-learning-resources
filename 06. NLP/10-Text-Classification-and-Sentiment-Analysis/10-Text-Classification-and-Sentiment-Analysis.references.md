---
id: "06-nlp/text-classification-sentiment/references"
topic: "Text Classification & Sentiment Analysis — References"
parent: "06-nlp/text-classification-sentiment"
type: references
updated: 2026-06-22
---

# Text Classification & Sentiment Analysis — references and further reading

> Companion link library for **[Text Classification & Sentiment Analysis](10-Text-Classification-and-Sentiment-Analysis.md)** (the concept page). External sources *and* internal cross-links, kept separate so this can be reused as a standalone reading list. Grouped by type, best-first; every entry is a primary author or a recognized deep explainer, chosen for depth on *this* topic and verified to load.

**Start here — suggested path**:
1. **Build the intuition** — read [SLP3 Ch. 4 "Naive Bayes, Text Classification, and Sentiment"](https://web.stanford.edu/~jurafsky/slp3/4.pdf) (**Jurafsky & Martin**). *The bag-of-words view of classification, Naive Bayes derived, sentiment.*
2. **See the strong baseline** — read [Text classification](https://lena-voita.github.io/nlp_course/text_classification.html) (**Lena Voita**). *From BoW + logistic regression up to neural classifiers — the clearest free survey.*
3. **Build a classic pipeline** — watch [Real-World ML with Scikit-Learn (NLP, classifiers)](https://www.youtube.com/watch?v=M9Itm95JzL0) (**Keith Galli**). *TF-IDF → classifier on real text, end to end.*
4. **Go neural** — read the [CNN-for-text paper](https://arxiv.org/abs/1408.5882) (**Kim 2014**) and watch [Text Classification Using BERT](https://www.youtube.com/watch?v=D9yyt6BfgAM) (**codebasics**). *When and why deep models help.*
5. **Make it concrete** — follow the [HF text-classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) and try the [zero-shot pipeline](https://huggingface.co/tasks/zero-shot-classification). *Fine-tune a transformer, and classify with no labels at all.*

**Videos**:
- [Real-World Machine Learning with Scikit-Learn (NLP, classifiers)](https://www.youtube.com/watch?v=M9Itm95JzL0) — **Keith Galli** — TF-IDF + classic classifiers, a full runnable pipeline.
- [Text Classification Using BERT & TensorFlow](https://www.youtube.com/watch?v=D9yyt6BfgAM) — **codebasics** — fine-tune BERT for spam/sentiment, clearly walked.
- [Sentiment Analysis with BERT Neural Network and Python](https://www.youtube.com/watch?v=szczpgOEdXs) — **Nicholas Renotte** — a hands-on transformer sentiment build.
- [Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — **StatQuest (Josh Starmer)** — the most visual walkthrough of the multinomial NB text classifier, the bottom rung of the ladder.
- [ROC and AUC, Clearly Explained!](https://www.youtube.com/watch?v=4jRBRDbJemM) — **StatQuest (Josh Starmer)** — the clearest visual intuition for ROC/AUC (and why precision-recall is the better lens under imbalance), the evaluation backbone of this page.

**Courses (free)**:
- [NLP Course for You — Text Classification](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita (Yandex)** — Naive Bayes → logistic regression → neural, fully free, beautifully illustrated.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the canonical modern NLP course; classification within the full pipeline.
- [Hugging Face LLM/NLP Course — Text classification & fine-tuning](https://huggingface.co/learn/nlp-course/chapter3/1) — **Hugging Face** — fine-tune a transformer classifier with the `Trainer`, step by step.

**Articles / blogs (free, no paywall)**:
- [Text classification (NLP Course for You)](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita** — the clearest free survey from baselines to neural classifiers.
- [Text classification (Hugging Face task guide)](https://huggingface.co/docs/transformers/tasks/sequence_classification) — **Hugging Face** — fine-tune and evaluate a transformer classifier, with code.
- [Zero-shot classification task page](https://huggingface.co/tasks/zero-shot-classification) — **Hugging Face** — the NLI-entailment trick, with a live demo and models.
- [Classification of text documents (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) — **scikit-learn** — runnable BoW/TF-IDF classifier comparison (NB vs LogReg vs SVM).
- [TF-IDF + classifier pipeline with grid search (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html) — **scikit-learn** — the canonical TF-IDF → classifier → pipeline tutorial with hyperparameter search.

**Key papers**:
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — **Kim (2014)** — the simple, strong CNN-for-text baseline (filters as n-gram detectors + max-over-time pooling).
- [Bag of Tricks for Efficient Text Classification (fastText)](https://arxiv.org/abs/1607.01759) — **Joulin et al. (2016)** — a linear classifier over averaged word/n-gram embeddings; fast and competitive.
- [Deep Unordered Composition Rivals Syntactic Methods (DAN)](https://aclanthology.org/P15-1162/) — **Iyyer et al. (2015)** — averaging word vectors rivals order-aware models on sentiment.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the `[CLS]`-head fine-tuning recipe that made transformers SOTA for classification.
- [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146) — **Howard & Ruder (2018)** — the transfer-learning recipe (pretrain → fine-tune) for text classification.
- [Benchmarking Zero-shot Text Classification (NLI)](https://arxiv.org/abs/1909.00161) — **Yin et al. (2019)** — classification with no labels by reframing labels as NLI hypotheses.
- [Thumbs up? Sentiment Classification using Machine Learning Techniques](https://arxiv.org/abs/cs/0205070) — **Pang, Lee & Vaithyanathan (2002)** — the paper that launched ML sentiment analysis on movie reviews.
- [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) — **Saito & Rehmsmeier (2015, PLoS ONE)** — *why* PR-AUC, not ROC-AUC, is the honest metric under class imbalance; the provenance for the PR-vs-ROC section.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 4 "Naive Bayes, Text Classification, and Sentiment"](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — Naive Bayes, smoothing, evaluation, sentiment phenomena.
- [Speech and Language Processing, 3rd ed. — Ch. 5 "Logistic Regression"](https://web.stanford.edu/~jurafsky/slp3/5.pdf) — **Jurafsky & Martin** — the discriminative baseline, the sigmoid + cross-entropy derivation, and the $(\sigma(z)-y)\,\mathbf{x}$ gradient, derived for text.
- [Introduction to Information Retrieval — Ch. 8 "Evaluation in information retrieval"](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — **Manning, Raghavan & Schütze (2008)** — precision, recall, F-measure, and the ROC/PR construction from first principles; the provenance for the evaluation section.
- [Opinion Mining and Sentiment Analysis (Foundations & Trends)](https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf) — **Pang & Lee (2008)** — the foundational survey of sentiment's hard problems (negation, aspect, domain, subjectivity).
- [Natural Language Processing with Python — Ch. 6 "Learning to Classify Text"](https://www.nltk.org/book/ch06.html) — **Bird, Klein & Loper** — feature design and classifiers in NLTK.
- [Deep Learning — Ch. 6 "Deep Feedforward Networks" (§6.2.2 Output Units)](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville (2016)** — softmax + cross-entropy for one-of-$K$ and independent sigmoids + BCE for multi-label outputs; the provenance for the output-layer rule.

**In this platform**:
- Concept page (full explanation): [Text Classification & Sentiment Analysis](10-Text-Classification-and-Sentiment-Analysis.md)
- Runnable code (seeded, CPU-only): [teaching notebook](code/10-Text-Classification-and-Sentiment-Analysis.ipynb) · [source-of-truth module](code/text_classification.py) · [figure generator](code/make_figures_10.py)
- Builds on these features: [Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) · [Word Embeddings (word2vec/GloVe/fastText)](../05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · [Contextual Embeddings (ELMo · BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
- The classifiers, in depth: [Naive Bayes](../../03.%20Supervised_Learning/05-Naive-Bayes/05-Naive-Bayes.md) · [Logistic Regression](../../03.%20Supervised_Learning/02-Logistic-Regression/02-Logistic-Regression.md) · [Support Vector Machines](../../03.%20Supervised_Learning/06-Support-Vector-Machines/06-Support-Vector-Machines.md)
- Evaluation: [Classification Metrics (precision · recall · F1 · ROC-AUC)](../../03.%20Supervised_Learning/14-Classification-Metrics/14-Classification-Metrics.md) · [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
- The token-level cousin: [Sequence Labeling (POS & NER)](../09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md)

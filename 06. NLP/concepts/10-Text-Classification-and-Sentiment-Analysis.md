---
id: "06-nlp/text-classification-sentiment"
topic: "Text Classification & Sentiment Analysis"
parent: "06-nlp"
level: intermediate
prereqs: ["bow-tfidf", "word-embeddings", "naive-bayes", "logistic-regression"]
interview_frequency: very-high
template: concept-deep
updated: 2026-06-22
---

# Text Classification & Sentiment Analysis: assigning a label to a span of text

Show a model the sentence *"this movie was not good"* and ask it one question — **is this a happy review or an unhappy one?** That single question is **text classification**: map a whole piece of text to a discrete label. It is, by a wide margin, the most *applied* task in all of NLP. The spam filter on your inbox, the "is this comment toxic?" gate on a forum, the routing that sends *"my card was charged twice"* to billing instead of returns, the dashboard that tells a brand whether today's tweets about them lean positive or negative — every one of those is a text classifier. **Sentiment analysis** — deciding whether text is positive, negative, or neutral — is the most famous instance, and the running example we'll use, because it exposes everything that makes the task subtle: the four-character word *not* can flip a label, two clauses can disagree, and *"yeah, brilliant"* can mean the opposite of what it says.

What makes this topic such fertile interview ground is that it is the *cleanest place in ML to reason about the whole pipeline at once* — features, baselines, model choice, class imbalance, evaluation, and the modern fork between **fine-tuning a small model** and **prompting a big one**. There is a beautiful, well-trodden **ladder** of approaches, and the senior-engineer skill is not knowing the top rung; it's knowing *which rung a given problem actually needs*. A linear model on TF-IDF features, trained in two seconds, is still the baseline that every fancier model must beat — and surprisingly often, it isn't beaten by enough to justify the cost.

This page is the definitive treatment. We will build the task from first principles, climb the **representation ladder** rung by rung (deriving *how classification rides on each representation*), confront what makes **sentiment** specifically hard, separate **multi-class** from **multi-label**, handle **class imbalance**, weigh **zero-/few-shot LLM prompting against fine-tuning a small model**, and ground all of it in **four worked, measured examples** — including a multinomial Naive Bayes classification done **by hand** and proven against scikit-learn. By the end you'll be able to:

- frame any labeling problem as **binary / multi-class / multi-label** and pick the right output layer + loss;
- climb the ladder — **BoW/TF-IDF + classical ML → pooled embeddings → CNN-for-text → RNN/biLSTM → fine-tuned BERT** — and explain *what each rung buys you*;
- **derive the multinomial Naive Bayes sentiment decision** end to end (log-posterior, smoothing) and reproduce it in code;
- explain why **linear models on sparse TF-IDF** are a famously strong baseline, and what a transformer adds;
- handle **negation, aspects, sarcasm, and domain shift** — the things that break naive sentiment;
- choose between **fine-tuning a small model and prompting an LLM** from a cost/latency/accuracy argument, with **zero-shot NLI** as the no-data option;
- evaluate with **macro-F1 and the confusion matrix**, and **calibrate a decision threshold**.

We'll go intuition and pictures first, then the math (every step shown), then runnable, **verified** code — the numbers in every figure below were measured in Python 3.12, not invented.

> **Note:** "classification" here means assigning **one of a fixed set of labels** to a span of text, where the span is usually a whole document/sentence. The cousin task where you label **each token** (part-of-speech, named-entity) is **[sequence labeling](09-Sequence-Labeling-POS-and-NER.md)** — same intuition, different output shape (one label *per token* instead of one *per document*). Keep them distinct in an interview.

---

## The problem: from a span of text to a discrete label

Formally, a text classifier is a function

$$f : \text{text} \;\longrightarrow\; y \in \mathcal{Y},$$

where $\mathcal{Y}$ is a finite **label set**. The flavor of the task is set entirely by the shape of $\mathcal{Y}$ and how many labels a document may carry:

- **Binary** — $\mathcal{Y} = \{0, 1\}$. *Spam / ham. Positive / negative. Toxic / clean.* One decision, one threshold.
- **Multi-class** — $\mathcal{Y} = \{1, \dots, K\}$, and **exactly one** label is correct. *Topic = {sports, politics, tech, …}. Intent = {billing, returns, shipping, …}. Star rating ∈ {1,2,3,4,5}.* The classes are **mutually exclusive**; the model must pick one.
- **Multi-label** — $\mathcal{Y}$ is a set of $K$ tags and a document may carry **any subset** (zero, one, or several). *A news article tagged both `politics` and `economy`. A support ticket that is both `billing` and `urgent`. A movie review that is `positive-about-acting` and `negative-about-pacing`.* The labels are **not** mutually exclusive.

That distinction — *exactly one* vs *any subset* — is not a footnote; it changes the **output layer and the loss function** (softmax + cross-entropy vs per-label sigmoid + binary cross-entropy), and we derive exactly why below. The most common mistake juniors make is reaching for softmax on a multi-label problem, which forces the probabilities to sum to one and therefore makes the labels compete when they shouldn't.

Sentiment analysis is usually framed as binary (pos/neg) or 3-way (pos/neg/neutral), sometimes as **ordinal** (1–5 stars, where the classes have an order), and at its most ambitious as **aspect-based** (positive about the *food*, negative about the *service* — really a structured, per-aspect classification). We'll touch all of these.

> **Note:** **ordinal** labels (1–5 stars) are a real third category between multi-class and regression: the classes have an *order* (4 is closer to 5 than to 1), so a misprediction of 4-vs-5 should cost less than 1-vs-5. Treating stars as plain multi-class throws that ordering away; treating them as regression and rounding ignores that they're discrete. Ordinal regression (e.g. cumulative-link models) sits in between — worth naming if asked, though plain multi-class is the common pragmatic choice.

### The general pipeline

Every classifier, from a 1990s Naive Bayes spam filter to a fine-tuned transformer, is the same three-stage pipeline. The only thing that changes up the ladder is **stage 2** — how text becomes numbers.

```mermaid
graph LR
    T(["Raw text<br/>'this movie was not good'"]):::data --> P["Preprocess<br/>tokenize · normalize"]:::prep
    P --> R{"Choose a<br/>representation"}:::choice
    R -->|"sparse"| BOW["BoW / TF-IDF<br/>(word identity)"]:::blue
    R -->|"dense static"| EMB["Embeddings + pooling<br/>(word similarity)"]:::navy
    R -->|"contextual"| BERT["BERT encoder<br/>([CLS] vector)"]:::purple
    BOW --> CLF["Classifier head<br/>NB / LogReg / SVM / MLP"]:::proc
    EMB --> CLF
    BERT --> CLF
    CLF --> O{"Label space"}:::choice
    O -->|"one of K"| MC["softmax<br/>multi-class"]:::green
    O -->|"many at once"| ML["per-label sigmoid<br/>multi-label (BCE)"]:::green
    MC --> Y(["predicted label(s)<br/>+ calibrated probability"]):::amber
    ML --> Y

    classDef data fill:#3A6B96,stroke:#2A5B86,color:#fff
    classDef prep fill:#4A5B6E,stroke:#3A4B5E,color:#fff
    classDef choice fill:#7A6528,stroke:#6A5518,color:#fff
    classDef blue fill:#3A6B96,stroke:#2A5B86,color:#fff
    classDef navy fill:#2A5B80,stroke:#1A4B70,color:#fff
    classDef purple fill:#5D4A8A,stroke:#4D3A7A,color:#fff
    classDef proc fill:#5D4A8A,stroke:#4D3A7A,color:#fff
    classDef green fill:#2E7A5A,stroke:#1E6A4A,color:#fff
    classDef amber fill:#7A6528,stroke:#6A5518,color:#fff
```

1. **Text → features / representation.** Turn the variable-length string into a fixed (or sequential) numeric form. This is the rung of the ladder you're on.
2. **Representation → classifier.** A decision rule over that representation — counts of words, a learned hyperplane, a neural head.
3. **Classifier → label.** A softmax over classes (multi-class) or independent sigmoids (multi-label), then a threshold.

Hold that three-stage skeleton in mind; the entire rest of this page is just *filling in stage 1 with better and better representations and watching the accuracy climb.*

---

## Intuition: the ladder of representations

Here is the whole story in one picture. As you climb, the representation captures **more of how language actually works** — from "which words appeared" (a bag, order-blind) all the way to "what each word means *in this exact sentence*" (deep, bidirectional context). Accuracy generally climbs with you. So do cost and latency.

![The text-classification representation ladder: BoW/TF-IDF + classical ML at the bottom, then pooled word embeddings, then CNN-for-text, then RNN/biLSTM, then fine-tuned BERT at the top. Each rung models more context at more compute; a TF-IDF linear model is the baseline everything must beat.](images/textcls_ladder.png)

The pedagogical spine of this page is to walk **up** that ladder, and for each rung answer the same two questions: *what does the representation capture?* and *how does the classification decision ride on it?* The punchline — visible in the measured comparison later — is that the **bottom rung is shockingly competitive**, and most of the accuracy gap to the top is closed by the *jump to contextual transformers*, with the middle rungs (DAN, CNN, RNN) being historically important way-stations.

> **Tip:** the senior move is to **always build the bottom rung first** — TF-IDF + logistic regression takes seconds, needs no GPU, and gives you (a) a real accuracy number to beat, (b) a sanity check on your labels and splits, and (c) often a shippable product. Only climb when that baseline genuinely isn't good enough, and *measure* the gain before paying for it.

---

## Rung 1: Bag-of-words / TF-IDF + classical ML

The oldest and still the strongest baseline. Represent a document as a **vector over the vocabulary** — one dimension per word — and feed that to a linear classifier. The representation is built in **[Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md)** (read it for the construction of the counts and the IDF weighting); here we focus on the **classifiers that ride on it**.

A document like *"great fun great"* over a vocabulary `[boring, dull, film, fun, great, movie]` becomes a count vector $\mathbf{x} = [0, 0, 0, 1, 2, 0]$ (or its TF-IDF-weighted cousin). It is **sparse** (mostly zeros — a 200-word review touches a handful of a 40,000-word vocabulary) and **high-dimensional**. Crucially, it is **order-blind**: *"good not bad"* and *"bad not good"* produce the identical vector. That blindness is the rung's central weakness — and exactly what every rung above tries to fix.

Three classical classifiers dominate this rung, and an interviewer may ask you to compare them.

### Multinomial Naive Bayes: the canonical text classifier

[Naive Bayes](../../03.%20Supervised_Learning/concepts/05-Naive-Bayes.md) is the textbook sentiment/spam classifier, and you should be able to derive its decision on the spot. (That page is the full treatment — generative-vs-discriminative, smoothing as a Dirichlet prior, the proof that NB is a linear classifier. Read it. Here we derive *just the text-classification decision rule* and then work it numerically.)

The **multinomial** event model treats a document as a bag of word **counts** drawn from a per-class word distribution. By Bayes' rule, the posterior probability of class $c$ given document $\mathbf{x}$ (word counts $x_w$) is

$$P(c \mid \mathbf{x}) \;=\; \frac{P(c)\,P(\mathbf{x}\mid c)}{P(\mathbf{x})} \;\propto\; P(c)\prod_{w \in V} P(w \mid c)^{\,x_w}.$$

The "naive" assumption — words are conditionally independent given the class — is what turns the joint likelihood into that product. We classify with the **MAP** rule (pick the class with the highest posterior), and because the evidence $P(\mathbf{x})$ doesn't depend on $c$, we can drop it. Taking $\log$ (to turn the product into a sum and dodge floating-point underflow) gives the **log-posterior score** we actually compute:

$$\boxed{\;\hat{c} \;=\; \arg\max_{c}\;\Big[\;\underbrace{\log P(c)}_{\text{prior}} \;+\; \sum_{w \in V} x_w \,\underbrace{\log P(w\mid c)}_{\text{log-likelihood}}\;\Big]\;}$$

The per-word likelihoods are estimated from training counts with **Laplace (add-$\alpha$) smoothing**, so that a word never seen in class $c$ doesn't zero out the whole product:

$$P(w \mid c) \;=\; \frac{\text{count}(w, c) + \alpha}{\Big(\sum_{w' \in V}\text{count}(w', c)\Big) + \alpha\,|V|}.$$

Notice the *shape* of that boxed rule: a sum of `count × log-likelihood`, which is a **linear function of the word-count vector**. That's the secret the Naive Bayes page proves in full — multinomial NB is a **linear classifier** in disguise, a sibling of logistic regression. Same decision *geometry*, different way of estimating the weights (NB from class-conditional counts, generatively; logistic regression by directly optimizing the conditional likelihood, discriminatively).

> **Note:** people dismiss Naive Bayes as a toy, but for text it is *genuinely strong* — fast, needs little data, and hard to beat as a one-line baseline. The independence assumption is wildly false (words travel in packs) yet the **ranking** of classes usually comes out right even when the probabilities are nonsense — which is all classification needs. See the measured 0.83 accuracy below: not far behind a logistic-regression baseline.

> **Gotcha:** the multinomial event model uses word **counts** (term frequency); the **Bernoulli** model uses binary present/absent flags and explicitly models *absent* words too. For longer documents the **multinomial** model is the standard for sentiment/topic; Bernoulli can win on very short texts. And the **Complement NB** variant (CNB) is the one to reach for under **class imbalance** — it estimates each class's parameters from the *complement* of that class, which is more stable when one class is rare.

We work this rule fully by hand in **Worked example 1** below — and prove it matches scikit-learn to the last decimal.

### Logistic regression on TF-IDF: the discriminative baseline

[Logistic regression](../../03.%20Supervised_Learning/concepts/02-Logistic-Regression.md) learns a weight vector $\mathbf{w}$ and bias $b$ and predicts $P(y{=}1\mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b)$, fitting the weights by maximizing the conditional log-likelihood (equivalently minimizing cross-entropy) with L2 regularization. On **sparse TF-IDF features** it is the workhorse text baseline: it learns a signed weight *per word* (large positive for `excellent`, large negative for `terrible`), so the model is **directly interpretable** — sort the weights and you can read off the most positive and most negative words. It typically edges out Naive Bayes because it doesn't assume independence; instead it can *down-weight* correlated, redundant features. In the measured comparison it lands at **0.858** accuracy — a couple of points above NB.

### Linear SVM: the margin-maximizer for sparse high-dim text

A **linear support vector machine** ([SVM](../../03.%20Supervised_Learning/concepts/06-Support-Vector-Machines.md)) finds the **maximum-margin** separating hyperplane. On high-dimensional **sparse** TF-IDF data it is, historically, *the* text-classification champion (Joachims showed in 1998 that text is "linearly separable in high-dimensions" and SVMs exploit exactly that). The intuition for *why linear models win here*: with tens of thousands of sparse features and relatively few examples, the data is **almost always linearly separable**, so a linear decision boundary has plenty of capacity — and a kernel would only overfit. The margin objective also makes the SVM robust to the many irrelevant features. In the comparison `LinearSVC` ties LogReg at **0.860**.

> **Tip:** *why are linear models + sparse TF-IDF such a strong baseline?* Three reasons worth saying out loud: (1) in tens-of-thousands of dimensions with sparse features, classes are usually **linearly separable**, so a linear boundary suffices; (2) the model is **fast and data-efficient** — it trains in seconds and works with a few thousand labels; (3) it is **interpretable** — every word has a signed weight. The non-linearity that deep models add only pays off when *interactions between words* matter, which for topic/spam is rarely, and for sentiment is sometimes (negation, sarcasm).

---

## Rung 2: word embeddings + pooling (the Deep Averaging Network)

The bag-of-words representation has no notion that `great` and `excellent` are related — they're two orthogonal dimensions. **[Word embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md)** fix that: each word becomes a dense vector where similar words sit close together. To classify a whole document, we need to collapse its *sequence* of word vectors into one fixed vector — and the simplest, surprisingly effective way is **pooling**:

$$\mathbf{h} \;=\; \text{pool}\big(\mathbf{e}_{w_1}, \mathbf{e}_{w_2}, \dots, \mathbf{e}_{w_n}\big), \qquad \text{pool} \in \{\text{mean},\ \text{max}\}.$$

**Mean pooling** averages the word vectors (every word votes equally); **max pooling** takes the element-wise max (each dimension fires if *any* word activates it strongly). Feed the pooled vector $\mathbf{h}$ to a small MLP + softmax, and you have a neural classifier. When the embeddings are averaged and passed through a couple of feed-forward layers, this is the **Deep Averaging Network (DAN)** of Iyyer et al. (2015) — and the startling result of that paper is that *averaging* word vectors, with **no word order at all**, rivals far more complex order-aware models on sentiment, at a fraction of the cost. **fastText** ([Joulin et al. 2016](https://arxiv.org/abs/1607.01759)) is the industrial version: a linear classifier over averaged word- *and n-gram*-embeddings, trained so fast it classifies a billion words in minutes while matching deep models on many benchmarks.

> **Note:** mean-pooled embeddings are still **mostly order-blind** — *"not good"* and *"good not"* average to nearly the same vector. The DAN's surprise is precisely that *this barely matters* for coarse sentiment, because the **presence** of strong-polarity words carries most of the signal. The cases where order *does* matter (negation, contrast) are exactly where the next rungs earn their keep. Add **word n-grams** to the bag (as fastText does) to recover some local order cheaply — *"not_good"* as its own feature.

---

## Rung 3: CNN-for-text — convolutions as n-gram detectors

To capture **local word order** without the cost of a full recurrent model, Kim (2014) had an elegant idea: slide **1-D convolutional filters** over the *sequence of embeddings*. Lay the sentence out as a matrix of shape $(T \text{ tokens} \times d \text{ dims})$, and a filter of width $h$ is a small weight patch that covers $h$ consecutive word-vectors at a time. As it slides, at each position it computes a single number — a **feature** — that fires when that particular $h$-gram pattern is present.

![CNN-for-text: an embedding matrix (tokens by embedding dim) with a width-3 filter sliding over 'was not good', producing a feature map, then max-over-time pooling collapsing it to one feature ('this n-gram is present'). Many filters of widths 2/3/4/5 feed a softmax classifier (Kim 2014).](images/textcls_cnn.png)

Formally, with embeddings $\mathbf{e}_1,\dots,\mathbf{e}_T \in \mathbb{R}^d$ stacked, a filter $\mathbf{W}\in\mathbb{R}^{h\times d}$ produces, at position $i$, the feature

$$c_i \;=\; g\big(\mathbf{W}\odot \mathbf{e}_{i:i+h-1} + b\big),$$

a non-linearity $g$ (ReLU) over the element-wise product summed across the $h\times d$ window. Sweeping $i$ from $1$ to $T-h+1$ gives a **feature map** $\mathbf{c} = [c_1,\dots,c_{T-h+1}]$ — the filter's response *at every position*. Then comes the key trick: **max-over-time pooling**, $\hat{c} = \max_i c_i$, which keeps only the *strongest* activation across the whole sentence. This says, in effect, *"did this n-gram pattern appear anywhere?"* — and it conveniently produces a fixed-size output regardless of sentence length.

What do the filters **learn**? They become **n-gram detectors**: a width-3 filter might learn to fire on *"was not good"*, another on *"highly recommend it"*, another on *"a complete waste"*. By using **many** filters at **several widths** (2, 3, 4, 5) you build a bank of learned phrase detectors; max-pooling each one and concatenating gives a feature vector that feeds a softmax. This is how the CNN *fixes the order-blindness of rungs 1–2 for local patterns* — it can represent *"not good"* as a single firing feature, which bag-of-words never could.

> **Tip:** the mental model that makes CNN-for-text click: **it's a learned, soft version of n-gram features.** Where bag-of-n-grams hard-codes *"not_good"* as a vocabulary entry (and explodes the feature count), the CNN *learns* which 3-grams matter and shares parameters via the embedding, so it generalizes — a filter that fires on *"not good"* also partly fires on *"not great"* because their embeddings are close.

> **Gotcha:** max-over-time pooling throws away **where** and **how many times** a pattern fired — only *that it fired strongest somewhere*. For most sentence classification that's a feature (position-invariance), but for tasks where *count* or *position* matters you'd lose information. It also makes CNNs weak at **long-range** dependencies (a negation 30 words from the word it negates), which is the gap RNNs and especially transformers close.

---

## Rung 4: RNN / LSTM / biLSTM — sequence-aware classification

A recurrent network reads the sentence **one token at a time**, maintaining a hidden state $\mathbf{h}_t$ that is a running summary of everything seen so far: $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{e}_t)$. For classification you typically take the **final hidden state** (a summary of the whole sentence) — or pool all hidden states — and feed it to a softmax. Unlike pooling or a width-$h$ CNN, an RNN can in principle carry information **across the whole sentence**, so a *"not"* early on can color the interpretation of an adjective much later. **LSTMs** (and GRUs) add gates that let gradients and information persist over long spans, fixing the vanishing-gradient problem of vanilla RNNs.

The standard sentiment architecture is the **bidirectional LSTM (biLSTM)**: run one LSTM left-to-right and another right-to-left, and concatenate their final states. *Why bidirectional?* Because the meaning of a word can depend on what comes **after** it as much as before — *"not"* needs to see the word it negates, which may be downstream. A biLSTM gives every position both its left and right context. This rung was the sentiment state-of-the-art for several years (circa 2015–2018) and remains a reasonable choice when a transformer is too heavy.

> **Note:** the line from this rung to the next is direct. RNNs read **sequentially** (slow, and long-range info still degrades over many steps); the **transformer** replaced recurrence with **[attention](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)**, letting every token attend to every other token in **one parallel step** — better long-range modeling *and* far faster training. That architectural leap is what makes Rung 5 possible.

---

## Rung 5: fine-tuning BERT — the modern default

The top of the ladder, and the SOTA for most text-classification tasks since 2018, is to **fine-tune a pretrained transformer**. The idea is **transfer learning**: a model like BERT is first **pretrained** on a huge unlabeled corpus with masked-language-modeling — learning deep, **bidirectional, contextual** representations of language (the full story is in **[Contextual Embeddings — ELMo · BERT](06-Contextual-Embeddings-ELMo-BERT.md)**). Then you **fine-tune** it on your (much smaller) labeled classification set.

The mechanics for classification are clean. BERT prepends a special **`[CLS]`** token to every input. After the encoder runs, the final-layer hidden vector at the `[CLS]` position is treated as a **pooled representation of the whole sentence**; you attach a single linear **classification head** on top of it:

$$\mathbf{h}_{\texttt{[CLS]}} \in \mathbb{R}^{768} \;\xrightarrow{\;\mathbf{W}\in\mathbb{R}^{K\times 768},\,\mathbf{b}\;}\; \text{softmax}\big(\mathbf{W}\,\mathbf{h}_{\texttt{[CLS]}} + \mathbf{b}\big) \in \Delta^{K}.$$

You then train **the whole model end to end** (head *and* the pretrained weights, with a small learning rate) on your labels. Because the encoder already "knows language," fine-tuning needs comparatively few labeled examples to reach high accuracy — this is the transfer that **ULMFiT** ([Howard & Ruder 2018](https://arxiv.org/abs/1801.06146)) introduced for text and **BERT** ([Devlin et al. 2018](https://arxiv.org/abs/1810.04805)) perfected. It works because the `[CLS]` vector is a *context-aware* summary: it can represent that *"not good"* is negative in a way no bag-of-words ever could, because self-attention lets *"not"* and *"good"* interact.

> **Tip:** for production you rarely fine-tune full BERT-large. **DistilBERT** (40% smaller, 60% faster, ~97% of BERT's accuracy) and **RoBERTa** (better-pretrained BERT) are the pragmatic choices. For sentiment specifically, off-the-shelf fine-tuned checkpoints (e.g. `distilbert-base-uncased-finetuned-sst-2-english`) exist — sometimes you don't need to train at all.

> **Gotcha:** fine-tuning a transformer is **not free** — it needs a GPU, the inference is 10–100× slower and more expensive than a TF-IDF linear model, and on small/clean datasets the accuracy gain over the baseline can be **a couple of points** (see the measured comparison: 0.86 → 0.88 on this subset). Always ask whether those points are worth the cost. The honest answer is often *yes* for hard, nuanced tasks (sarcasm, nuanced toxicity) and *no* for clean topic/spam.

---

## The measured ladder: NB vs linear models vs transformer

Enough theory — here is the ladder **measured**. I trained all four on the same **IMDb** sentiment subset (4,000 reviews train, 2,000 test), TF-IDF with 1–2-grams for the linear models, and a DistilBERT fine-tuned for **one epoch on a 2,000-review subset** (deliberately small, to be honest about a quick fine-tune). Every number below came out of the verification script in Python 3.12.

![Measured accuracy and macro-F1 on the IMDb sentiment subset: Multinomial Naive Bayes 0.832, TF-IDF+LogReg 0.858, Linear SVM 0.860, fine-tuned DistilBERT 0.877. The transformer leads, but the linear baselines are within a few points.](images/textcls_model_compare.png)

| Model | Representation | Accuracy | Macro-F1 | Train cost |
|---|---|---|---|---|
| Multinomial Naive Bayes | bag-of-counts (1–2 gram) | **0.832** | 0.831 | ~1 s, CPU |
| TF-IDF + Logistic Regression | TF-IDF (1–2 gram) | **0.858** | 0.858 | ~2 s, CPU |
| Linear SVM | TF-IDF (1–2 gram) | **0.860** | 0.860 | ~2 s, CPU |
| DistilBERT (fine-tuned, 1 epoch, 2k) | contextual `[CLS]` | **0.877** | 0.877 | ~85 s, GPU |

Read this table like a senior engineer. The **bottom rung gets you to 0.86** in two seconds on a CPU. The transformer — even a quick, under-trained one — leads at **0.88**, and a *fully* fine-tuned DistilBERT on all 25k IMDb reviews reaches **~0.91–0.93**. So the realistic picture is: *the linear baseline captures the bulk of the achievable accuracy, and the transformer buys you the last ~5–7 points at 40× the training cost and ~50× the inference cost.* Whether that trade is worth it is the entire engineering decision — and it depends on how much each point of accuracy is worth in your product.

> **Note:** notice **accuracy ≈ macro-F1** for every model here. That's because IMDb is **perfectly balanced** (50/50 pos/neg). On a **skewed** dataset they diverge sharply, and that gap is the whole reason macro-F1 exists — more on this under evaluation and imbalance.

---

## What makes *sentiment* specifically hard

Topic and spam classification are comparatively easy — the presence of a few telltale words usually settles it. **Sentiment is harder**, because polarity is a property of *meaning in context*, and language has many ways to invert, qualify, or hide it. These are the issues an interviewer will probe, and the cases where the higher rungs of the ladder earn their cost.

![Left: the measured confusion matrix for TF-IDF + LogReg on IMDb (867 true-neg, 849 true-pos, ~140 errors each way). Right: the hard cases — 'not good' (negation, BoW votes positive, wrong), 'not bad at all' (double negation, actually positive), 'great cast, dull plot' (two aspects, needs aspect-level), 'yeah, brilliant /s' (sarcasm, surface says positive), vs 'the movie was great' (clear polarity, easy).](images/textcls_confusion.png)

- **Polarity vs intensity.** *"good"* and *"phenomenal"* are both positive but not equally so. Binary pos/neg throws intensity away; star ratings and regression-style sentiment keep it.
- **Negation.** *"not good"*, *"hardly a masterpiece"*, *"I can't say I enjoyed it"* — a single negator flips polarity. **Bag-of-words is structurally blind to this**: it sees the positive word `good` and votes positive (the top-left of the hard-cases panel). Classic fixes prepend a `NOT_` tag to every token between a negator and the next punctuation (so `not good` → `not NOT_good`); CNNs and transformers learn it from data.
- **Double negation & litotes.** *"not bad at all"* is *positive* — two negations and an understatement. These defeat simple negation rules and need a model that reads the whole phrase.
- **Aspect-based sentiment (ABSA).** *"great food, terrible service"* is **positive about one aspect and negative about another**. A single document label is the wrong shape; ABSA predicts a sentiment *per aspect/entity*, turning sentiment into a structured extraction-plus-classification task. This is where real product analytics live ("what do customers like vs dislike?").
- **Sarcasm & irony.** *"Oh, fantastic, another crash."* The surface words are positive; the meaning is negative. This is genuinely hard even for large models, because it often requires world knowledge or tone the text doesn't carry.
- **Domain dependence.** A model trained on movie reviews transfers poorly to financial news or product reviews — *"unpredictable"* is **praise** for a thriller and a **complaint** for a car. Sentiment lexicons and models are **domain-specific**; always validate on your target domain.
- **Subjectivity.** Much text is purely factual ("the phone has 128 GB of storage") and carries **no** sentiment. A robust pipeline often first classifies *subjective vs objective*, then scores polarity only on the subjective spans.

> **Note:** these difficulties are precisely *why the ladder exists.* Negation and double-negation reward **local order** (CNN) and **long-range context** (biLSTM, transformer); aspects reward **structured** outputs; sarcasm rewards the **world knowledge** baked into large pretrained models. When someone asks "why not just use Naive Bayes for sentiment?", the answer is this list.

> **Gotcha:** the canonical reference for *why sentiment is its own field* is **Pang & Lee's 2008 survey, "Opinion Mining and Sentiment Analysis"** — it catalogs exactly these phenomena (negation, domain transfer, subjectivity, aspect) and is still the best map of the territory. Cite it if asked for the foundational work.

---

## Multi-class (softmax) vs multi-label (per-label sigmoid): derive the difference

This is the single most testable structural detail, and a place juniors slip. The output layer and loss differ because the **label spaces have different shapes**.

**Multi-class (exactly one of $K$).** The model outputs $K$ logits $\mathbf{z}=[z_1,\dots,z_K]$, and a **softmax** turns them into a probability distribution that **sums to one**:

$$P(y{=}k \mid \mathbf{x}) \;=\; \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_{k=1}^{K} P(y{=}k\mid\mathbf{x}) = 1.$$

Train with **categorical cross-entropy**: $\mathcal{L} = -\log P(y{=}y^\star\mid\mathbf{x})$ for the true class $y^\star$. The crucial property: softmax makes the classes **compete** — raising one class's probability *lowers* the others. That is exactly what you want when only one label can be right.

**Multi-label (any subset of $K$).** Now the labels are **independent** — a document can be both `politics` and `economy`. You must **not** use softmax (it would force them to compete and sum to one). Instead use $K$ **independent sigmoids**, one per label, each its own binary classifier:

$$P(\text{label}_k \mid \mathbf{x}) \;=\; \sigma(z_k) \;=\; \frac{1}{1+e^{-z_k}}, \qquad \text{each in } (0,1) \text{ independently.}$$

Train with **binary cross-entropy summed over labels**: $\mathcal{L} = -\sum_{k=1}^{K}\big[y_k\log\sigma(z_k) + (1-y_k)\log(1-\sigma(z_k))\big]$. At inference you **threshold each sigmoid independently** (typically at 0.5, but tune per label) to decide which tags to emit — so a document can fire zero, one, or many labels.

$$\boxed{\;\text{one-of-}K \Rightarrow \text{softmax + cross-entropy}\qquad\text{any-subset} \Rightarrow K\ \text{sigmoids + BCE}\;}$$

> **Gotcha:** the most common bug in applied multi-label NLP is using **softmax + cross-entropy** on a problem where documents legitimately carry multiple labels. The symptom: the model can never confidently predict two labels at once (their probabilities are forced to trade off), so multi-tag recall collapses. The fix is the sigmoid+BCE head above. Conversely, using independent sigmoids on a genuinely mutually-exclusive problem wastes the helpful inductive bias that the classes compete.

---

## Class imbalance: when one label dominates

Real classification data is often **skewed** — 1% spam, 0.3% fraud, a rare toxic class among mostly-clean comments. Three things go wrong, and all are covered in depth in **[Classification Metrics](../../03.%20Supervised_Learning/concepts/14-Classification-Metrics.md)** (read it for the full treatment of the confusion matrix, precision/recall, and the metric choices) — here is the text-specific summary:

1. **Accuracy lies.** A classifier that predicts "not-spam" for everything scores 99% accuracy on a 1%-spam stream while catching zero spam. Report **macro-F1** (averages the per-class F1, so the rare class counts equally) or **precision/recall/PR-AUC for the minority class**, never bare accuracy.
2. **The model under-learns the rare class.** Standard cross-entropy is dominated by the majority class's gradient. Fixes:
   - **Class weighting** — scale each class's loss by the inverse of its frequency (`class_weight="balanced"` in scikit-learn; `pos_weight` in a BCE head), so a rare-class mistake costs more.
   - **Resampling** — oversample the minority (or SMOTE-style synthesis, though it's awkward for text) or undersample the majority.
   - **Focal loss** — down-weights easy, well-classified examples so training focuses on the hard, often-minority ones (originally from object detection, widely used for imbalanced text too).
   - **Complement Naive Bayes** — the NB variant designed for imbalance.
3. **The default threshold is wrong.** A 0.5 sigmoid threshold is rarely optimal under imbalance — tune it on a validation set to hit your precision/recall target (next section).

> **Tip:** for the rare-but-critical class (fraud, toxicity, medical), the product question is usually *"of the bad cases, how many did we catch?"* — that's **recall** — traded against the false-alarm rate. Optimize the metric your product actually cares about, not accuracy. Macro-F1 is the safe default summary; PR-AUC is best when the positive class is rare.

---

## Zero-shot and few-shot classification with LLMs

The newest fork in the road: you may not need to *train a classifier at all*. Two LLM-era techniques classify text with **no task-specific training data**.

**Zero-shot via NLI entailment** ([Yin et al. 2019](https://arxiv.org/abs/1909.00161)). The clever reframe: turn classification into **natural language inference**. Take a model fine-tuned on NLI (does sentence A *entail* sentence B?), feed the document as the premise and a templated hypothesis per candidate label — *"This text is about {politics}."*, *"This text is {positive}."* — and read off the **entailment probability** for each label. The label whose hypothesis is most entailed wins. No labeled training data, any label set you like at runtime. This is exactly what Hugging Face's `zero-shot-classification` pipeline does (with `bart-large-mnli`), and we run it live in **Worked example 4** — it scores a clearly-negative review as `negative` 0.86.

**Few-shot / in-context prompting.** Hand a general LLM the task as a prompt: a short instruction, optionally a few labeled examples ("in-context learning"), and the text to classify, then parse the label out of the completion. No training, no gradient updates — just a well-crafted prompt. It handles arbitrary, fuzzy label sets and nuanced cases (sarcasm) that a small model struggles with, and it shines when you have *no* labels.

**When to prompt an LLM vs fine-tune a small model.** This is the decision an interviewer wants you to reason through — it's a **cost / latency / accuracy** trade.

```mermaid
graph TD
    S(["New text-classification task"]):::data --> L{"Have labeled<br/>training data?"}:::choice
    L -->|"none / a handful"| Z["Zero-/few-shot:<br/>prompt an LLM or NLI"]:::purple
    L -->|"hundreds–thousands"| B{"Latency &<br/>cost budget?"}:::choice
    B -->|"tight (high QPS,<br/>cheap, on-prem)"| C["TF-IDF + LogReg/SVM<br/>strong baseline"]:::blue
    B -->|"can afford a GPU"| F["Fine-tune a small<br/>transformer (DistilBERT)"]:::green
    Z --> E{"Enough accuracy?"}:::choice
    E -->|"yes"| SHIP(["ship it"]):::amber
    E -->|"no, and labels exist"| F
    C --> SHIP
    F --> SHIP

    classDef data fill:#3A6B96,stroke:#2A5B86,color:#fff
    classDef choice fill:#7A6528,stroke:#6A5518,color:#fff
    classDef purple fill:#5D4A8A,stroke:#4D3A7A,color:#fff
    classDef blue fill:#2A5B80,stroke:#1A4B70,color:#fff
    classDef green fill:#2E7A5A,stroke:#1E6A4A,color:#fff
    classDef amber fill:#7A6528,stroke:#6A5518,color:#fff
```

- **Prompt an LLM when:** you have **no/few labels**, the label set changes often, volume is low, latency is forgiving, and the task is nuanced. It's the fastest path to a working v0.
- **Fine-tune (or train a linear model) when:** you have **labeled data** and need **high throughput, low latency, low per-call cost, or on-prem/offline** deployment. A 66M-parameter DistilBERT (or a TF-IDF linear model) serving thousands of requests per second per CPU is **orders of magnitude cheaper** than calling a 100B+ LLM per document — and on a well-defined task it is usually **as accurate or more**, because it was trained on *your* distribution.

> **Tip:** a powerful hybrid: use an **LLM to label** a few thousand examples (cheap, one-time), then **fine-tune a small model** on those labels (cheap, fast at serve time). You get the LLM's quality distilled into a model you can run at scale for pennies. This "distill the labels, serve the small model" pattern is now standard practice for high-volume classification.

---

## Evaluation and calibration

**Metrics.** Use the **confusion matrix** as the source of truth, then the metric matched to your problem (the full derivations are in **[Classification Metrics](../../03.%20Supervised_Learning/concepts/14-Classification-Metrics.md)** and the NLP-specific angle in **[NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md)**):

- **Accuracy** only when classes are balanced *and* errors cost the same.
- **Macro-F1** as the default for multi-class/imbalanced (per-class F1 averaged, every class equal); **micro-F1** weights by support; **weighted-F1** is in between.
- **Per-class precision/recall** when one class is the one that matters; **PR-AUC** for a rare positive class; **ROC-AUC** for balanced ranking quality.

**Threshold tuning & calibration.** A model outputs a *probability*; turning it into a *decision* needs a **threshold**. The default 0.5 is rarely optimal, especially under imbalance or asymmetric costs — sweep the threshold on a validation set to hit your target precision/recall (or maximize F1). Separately, **calibration** asks whether a predicted "0.8" really means "right 80% of the time." Naive Bayes is notoriously **over-confident** (its probabilities are unreliable even when its decisions are good — see the NB page); logistic regression is better calibrated by construction; transformers often need **temperature scaling** to be trustworthy. If you *act* on the probability (e.g. only auto-action above 0.95 confidence), calibrate it first.

> **Note:** *accuracy is a decision metric (needs a threshold); AUC is a ranking metric (threshold-free).* A model can have great AUC (it ranks positives above negatives) but poor accuracy at the default threshold simply because the threshold is mis-set. Separating "is the ranking good?" from "is the threshold right?" is a senior-level distinction interviewers look for.

---

## Worked example 1: multinomial Naive Bayes by hand

Let's classify a tiny sentiment corpus completely by hand, then prove it against scikit-learn. Training set — four documents, two classes:

| Document | Class |
|---|---|
| `great fun great` | pos |
| `great great movie` | pos |
| `boring boring film` | neg |
| `boring dull movie` | neg |

**Vocabulary** $V = \{$`boring, dull, film, fun, great, movie`$\}$, so $|V| = 6$. **Priors:** 2 pos, 2 neg docs → $P(\text{pos}) = P(\text{neg}) = 0.5$.

**Word counts per class** (count each word across that class's docs):

- **pos** (`great fun great` + `great great movie`): `great`=4, `fun`=1, `movie`=1 → **total = 6** words.
- **neg** (`boring boring film` + `boring dull movie`): `boring`=3, `film`=1, `dull`=1, `movie`=1 → **total = 6** words.

**Classify the test document** $d^\star =$ `great fun`. With Laplace smoothing $\alpha = 1$ and $|V| = 6$, the smoothed likelihoods we need are:

$$P(\text{great}\mid\text{pos}) = \frac{4+1}{6+6} = \frac{5}{12},\qquad P(\text{fun}\mid\text{pos}) = \frac{1+1}{6+6} = \frac{2}{12},$$
$$P(\text{great}\mid\text{neg}) = \frac{0+1}{6+6} = \frac{1}{12},\qquad P(\text{fun}\mid\text{neg}) = \frac{0+1}{6+6} = \frac{1}{12}.$$

Now the **log-posterior scores** (counts of `great` and `fun` in $d^\star$ are both 1):

$$\log P(\text{pos}\mid d^\star) = \log 0.5 + \log\tfrac{5}{12} + \log\tfrac{2}{12} = -0.693 - 0.875 - 1.792 = \mathbf{-3.360},$$
$$\log P(\text{neg}\mid d^\star) = \log 0.5 + \log\tfrac{1}{12} + \log\tfrac{1}{12} = -0.693 - 2.485 - 2.485 = \mathbf{-5.663}.$$

Since $-3.360 > -5.663$, the model predicts **pos** — correctly, since `great` and `fun` are both positive words and never appeared in negative reviews. The gap (2.3 in log-space, i.e. $e^{2.3} \approx 10\times$ more probable) reflects that confidence.

**Proof against scikit-learn.** The verification script fits `MultinomialNB(alpha=1.0)` on the same corpus and reads its *joint log-likelihood* for the two classes:

```
by hand:  logp_pos = -3.3604   logp_neg = -5.6630   -> pos
sklearn:  jll_pos  = -3.3604   jll_neg  = -5.6630   -> pos
match: True
```

Bit-for-bit identical — the by-hand derivation *is* exactly what scikit-learn computes.

---

## Worked example 2: TF-IDF + logistic regression on the same corpus

Now the discriminative baseline on the identical tiny corpus, classifying the same `great fun`. The verification script fits a `TfidfVectorizer` + `LogisticRegression` and predicts:

```
TF-IDF + LogReg:  P(pos | "great fun") = 0.620  -> pos
```

It agrees with Naive Bayes (**pos**), but notice the probability is a calmer **0.62** versus NB's effective ~0.9. Logistic regression's probabilities tend to be **better calibrated**; Naive Bayes, because of its false independence assumption, **double-counts** correlated evidence and pushes probabilities toward 0/1. Same decision, more trustworthy number — a concrete instance of the calibration point above. (On this two-word toy the margin is small; on real data the LogReg/SVM models pull a couple of accuracy points ahead, as the measured comparison showed.)

---

## Worked example 3: the measured ladder comparison

This is the table and bar chart from earlier, produced end to end by the verification script on **IMDb** (4k train / 2k test for the linear models; 2k-review, 1-epoch DistilBERT fine-tune):

```
Multinomial Naive Bayes   acc 0.832   macro-F1 0.831
TF-IDF + LogReg           acc 0.858   macro-F1 0.858
Linear SVM (TF-IDF)       acc 0.860   macro-F1 0.860
DistilBERT (fine-tuned)   acc 0.877   macro-F1 0.877   [measured]
confusion (LogReg): [[867, 148],
                     [136, 849]]
```

The confusion matrix (also in the figure above) shows the TF-IDF+LogReg model getting **867** true-negatives and **849** true-positives, with a near-symmetric ~140 errors each way — exactly what you expect on a balanced set with no systematic bias toward either class. The ~5-point climb from NB to DistilBERT is the cost-vs-accuracy trade made concrete: each rung up buys a little more, the linear baseline already captures most of it, and a *fully* trained DistilBERT (all 25k reviews) would push to ~0.91–0.93.

---

## Worked example 4: zero-shot classification via NLI

Finally, classification with **zero training data**, using the NLI-entailment trick. The verification script runs Hugging Face's `zero-shot-classification` pipeline (`facebook/bart-large-mnli`) on a clearly negative review:

```
text:   "The plot was predictable and I nearly fell asleep."
labels: ["negative", "neutral", "positive"]
scores: [ 0.860,      0.103,     0.037 ]   -> negative
```

The model never saw a single labeled sentiment example, yet it confidently (**0.86**) labels the review **negative** — by checking which of *"This text is negative/neutral/positive"* the NLI model thinks the review **entails**. That's the whole zero-shot story: reframe your labels as hypotheses and let a pretrained entailment model vote. Great for a v0 with no data; you'd still fine-tune a small model once you have labels and need scale.

---

## Code: the full pipeline, end to end

Here is a single runnable script (verified in Python 3.12, scikit-learn 1.9) that builds the bottom three rungs, classifies, and reports — the playbook you'd actually run on day one of a new classification task. The by-hand Naive Bayes block matches Worked example 1.

```python
"""Text classification: by-hand NB check + the strong TF-IDF baselines.
Verified in Python 3.12, scikit-learn 1.9 (CPU, ~2 s)."""
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# --- 1. By-hand multinomial NB on the tiny corpus (matches sklearn) ---------
train = [("great fun great", 1), ("great great movie", 1),
         ("boring boring film", 0), ("boring dull movie", 0)]
docs, y = [d for d, _ in train], [c for _, c in train]

cv = CountVectorizer()
Xc = cv.fit_transform(docs)
nb = MultinomialNB(alpha=1.0).fit(Xc, y)
jll = nb._joint_log_likelihood(cv.transform(["great fun"]))[0]   # [neg, pos]
print(f"NB joint-log-lik  neg={jll[0]:.4f}  pos={jll[1]:.4f}  -> "
      f"{'pos' if jll[1] > jll[0] else 'neg'}")

# the same numbers, derived by hand (Laplace alpha=1, |V|=6, totals 6 each):
V, alpha = 6, 1.0
lp_pos = math.log(.5) + math.log((4+alpha)/(6+alpha*V)) + math.log((1+alpha)/(6+alpha*V))
lp_neg = math.log(.5) + math.log((0+alpha)/(6+alpha*V)) + math.log((0+alpha)/(6+alpha*V))
print(f"by-hand           neg={lp_neg:.4f}  pos={lp_pos:.4f}")

# --- 2. The strong linear baselines on TF-IDF features ----------------------
# (swap in your own train/test text + labels here)
X_train = ["loved every minute, brilliant film", "a complete waste of time",
           "wonderful, heartwarming and funny", "boring, predictable and dull",
           "the best movie i have seen", "terrible acting, awful script"]
y_train = [1, 0, 1, 0, 1, 0]
X_test  = ["a brilliant, wonderful film", "an awful waste of time",
           "this was not good at all"]   # the 3rd needs negation — watch it fail

tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

for name, clf in [("LogReg", LogisticRegression(max_iter=1000)),
                  ("LinearSVM", LinearSVC())]:
    clf.fit(Xtr, y_train)
    print(name, "->", clf.predict(Xte).tolist())   # 1 = positive, 0 = negative
```

```
NB joint-log-lik  neg=-5.6630  pos=-3.3604  -> pos
by-hand           neg=-5.6630  pos=-3.3604
LogReg -> [1, 0, 1]
LinearSVM -> [1, 0, 1]
```

Both baselines correctly call *"a brilliant, wonderful film"* positive (`1`) and *"an awful waste of time"* negative (`0`) — they learned `brilliant`/`wonderful` and `awful`/`waste` from the six training docs. But both **miss** the third, *"this was not good at all"*, predicting positive (`1`): this six-document toy never saw the words `not` or `good`, so the model has **no signal** for that example and falls back toward the majority direction. That failure is the whole lesson in miniature — a linear model only knows the patterns its **training data** contains; give it thousands of real reviews (as in the IMDb comparison) and the 1–2-gram features *do* learn *"not good"* as a negative cue. To climb to the transformer rung — which carries negation knowledge from pretraining — the pattern is just as short:

```python
"""Zero-shot sentiment with no training data, via NLI entailment.
Verified in Python 3.12, transformers 5.10 (downloads bart-large-mnli once)."""
from transformers import pipeline
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
out = clf("The plot was predictable and I nearly fell asleep.",
          candidate_labels=["positive", "negative", "neutral"])
print(out["labels"][0], round(out["scores"][0], 3))   # -> negative 0.86
```

For a fine-tuned model, swap in `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)` with the Hugging Face `Trainer` — the full, verified training loop lives in `tools/_verify_textcls.py` in this repo, and the diagrams in `tools/gen_text_classification_diagrams.py`.

---

## Application: a step-by-step playbook

The reasoning I'd actually run when handed a new text-classification task:

1. **Frame the label space.** Binary, multi-class (one-of-$K$, softmax), or multi-label (any-subset, sigmoids+BCE)? Ordinal? This fixes your output layer and loss before anything else.
2. **Check the balance.** Class frequencies. If skewed, plan for **macro-F1 / PR-AUC** (not accuracy) and **class weighting / threshold tuning** from the start.
3. **Build the bottom rung.** TF-IDF (1–2 grams) + logistic regression (or LinearSVM). Two seconds, no GPU. This is your **baseline to beat** and often your **shipping candidate**.
4. **Read the errors.** Look at the confusion matrix and the misclassified examples. Is the model losing on **negation**? **Aspects**? **Sarcasm**? **Domain words**? The failure mode tells you which rung to climb to (or whether better features/labels would help more than a bigger model).
5. **Decide: prompt vs train.** No labels or shifting label set → **zero-/few-shot LLM** for a v0. Labels + scale/latency needs → **fine-tune DistilBERT** (or stay on the linear baseline if it's close enough). Consider the **LLM-labels → small-model** distillation hybrid.
6. **Tune the threshold, calibrate the probability.** Sweep the decision threshold for your target precision/recall; temperature-scale if you act on the confidence.
7. **Evaluate honestly.** Report macro-F1 + the confusion matrix on a held-out set from the **target domain**, and re-measure when the domain drifts.

---

## Recap and rapid-fire

**If you remember nothing else:** text classification is a three-stage pipeline — *text → representation → classifier → label* — and the only thing that changes up the **ladder** (BoW/TF-IDF → pooled embeddings → CNN → biLSTM → fine-tuned BERT) is how text becomes numbers. A **linear model on sparse TF-IDF** is the strong baseline everything must beat; a **fine-tuned transformer** is the modern SOTA but buys its last few points at a large cost. **Sentiment** is the hard instance because polarity is contextual — negation, aspects, sarcasm, and domain all flip it. Choose your **output layer by the label shape** (softmax for one-of-$K$, sigmoids+BCE for multi-label), handle **imbalance** with macro-F1 + weighting, and decide **prompt-vs-fine-tune** on a cost/latency/accuracy argument.

**Quick-fire — say these out loud:**

- *Strongest cheap baseline for text?* TF-IDF (1–2 gram) + logistic regression / linear SVM — seconds to train, no GPU, often within a few points of a transformer.
- *Derive the NB sentiment decision?* $\arg\max_c \big[\log P(c) + \sum_w x_w \log P(w\mid c)\big]$, likelihoods Laplace-smoothed; it's a **linear** classifier.
- *Why is multinomial NB still good for text?* The independence assumption is false but the class **ranking** comes out right; fast and data-efficient.
- *What does a CNN-for-text filter learn?* An **n-gram detector**; max-over-time pooling answers "did this pattern fire anywhere?"
- *Multi-class vs multi-label output?* Softmax + cross-entropy (classes compete, sum to 1) vs **per-label sigmoid + BCE** (independent, threshold each).
- *Why does bag-of-words fail on "not good"?* It's **order-blind** — sees `good`, votes positive. CNNs/transformers model the local order.
- *Metric for imbalanced sentiment?* **Macro-F1** (or PR-AUC for a rare class), never bare accuracy; weight the loss and tune the threshold.
- *Zero-shot with no labels?* Reframe labels as **NLI hypotheses** and read the entailment score (Yin et al. 2019) — or prompt an LLM in-context.
- *Prompt an LLM or fine-tune a small model?* No/few labels & low volume → **prompt**; labels + high QPS / low cost / on-prem → **fine-tune** (or distill LLM labels into a small model).
- *Does the transformer always win?* It usually leads, but on clean topic/spam the linear baseline is within a couple of points at a fraction of the cost — measure before you pay.

---

## References and further reading

The curated link library for this topic — start-here path, videos, courses, articles, papers, books, and internal cross-links — lives in a companion file so it can be reused as a standalone reference list:

**→ [Text Classification & Sentiment Analysis — references and further reading](10-Text-Classification-and-Sentiment-Analysis.references.md)**

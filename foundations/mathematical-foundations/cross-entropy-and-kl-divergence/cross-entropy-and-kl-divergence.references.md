---
id: "01-foundations/cross-entropy-and-kl-divergence/references"
topic: "Cross-Entropy & KL Divergence — References"
parent: "01-foundations/cross-entropy-and-kl-divergence"
type: references
updated: 2026-07-02
---

# Cross-Entropy & KL Divergence — references and further reading

> Companion link library for **[Cross-Entropy & KL Divergence](cross-entropy-and-kl-divergence.md)**
> (the concept page). Kept separate so it can be reused as a standalone reference list. Grouped by
> type, best-first. Everything here is **free / open** — no paywall. Every source cited under a
> "Source / derivation" line on the concept page appears here, so each formula is traceable to a
> primary source. Chosen for depth on *this* topic, not popularity.

**⭐ Start here — suggested path**:
1. **Get the picture** — read [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (**Christopher Olah**). *The definitive free visual treatment of entropy → cross-entropy → KL as code lengths — the coding intuition this chapter is built on.*
2. **The bridge, fast** — watch [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) (**Aurélien Géron**). *All three quantities and how they relate, in ten minutes.*
3. **Cross-entropy as a loss** — watch [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) (**StatQuest**). *Exactly how cross-entropy is used to train a classifier.*
4. **KL intuition & asymmetry** — watch [Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM) (**Adian Liusie**). *Why KL measures distributional distance and why it's asymmetric.*
5. **The math + MLE link** — read [MML Ch. 8 (MLE ↔ cross-entropy)](https://mml-book.github.io/book/mml-book.pdf) and [MacKay Ch. 2 (relative entropy, Gibbs)](https://www.inference.org.uk/itprnn/book.pdf). *Minimising cross-entropy = maximising likelihood = minimising KL.*

**🎥 Videos** (trusted creators only):
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — all three concepts and their relationships, the cleanest single bridge.
- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — **StatQuest (Josh Starmer)** — cross-entropy as a training loss, step by step.
- [Entropy (for data science) Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw) — **StatQuest (Josh Starmer)** — the entropy / surprise baseline that KL is measured against.
- [Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM) — **Adian Liusie** — the meaning and asymmetry of KL, forward vs reverse.
- [The KL Divergence: Information Theory meets Machine Learning](https://www.youtube.com/watch?v=LJ5oCK4mQZM) — **Serrano.Academy (Luis Serrano)** — KL divergence built from first principles with worked examples.
- [Solving Wordle using information theory](https://www.youtube.com/watch?v=v68zYyaEmEA) — **3Blue1Brown** — builds Shannon entropy and information-as-bits from first principles (expected surprise = code length) — the exact intuition this page opens with.

**🎓 Courses (free)**:
- [Stanford CS231n — Linear Classification (Softmax & cross-entropy loss)](https://cs231n.github.io/linear-classify/) — **Stanford** — cross-entropy derived as *the* classification loss, with the softmax gradient (the Step 4 derivation).
- [Khan Academy — Journey into Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory) — **Khan Academy** — the entropy / bits / surprise foundations cross-entropy and KL build on.
- [MIT 6.050J — Information & Entropy](https://ocw.mit.edu/courses/6-050j-information-and-entropy-spring-2008/) — **MIT OCW** — a full free course on entropy, coding, and information from first principles.

**📰 Articles / blogs (free, no paywall)**:
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — **Christopher Olah** — the definitive free visual treatment of cross-entropy and KL as code lengths; the intuition this chapter rests on.
- [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) — **Will Kurt (Count Bayesie)** — an intuitive, free walkthrough from information loss to VAEs.
- [KL Divergence for Machine Learning](https://dibyaghosh.com/blog/probability/kldivergence.html) — **Dibya Ghosh** — forward vs reverse KL (mode-covering vs mode-seeking) made precise, with the fitting pictures this chapter's Demo 2 mirrors.
- [Cross-entropy and its relation to log loss](https://sebastianraschka.com/faq/docs/cross-entropy.html) — **Sebastian Raschka** — the classifier-loss view, connecting cross-entropy, NLL, and sklearn's `log_loss`.

**📄 Key papers**:
- [On Information and Sufficiency](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full) — **Kullback & Leibler (1951), *Annals of Mathematical Statistics***. — the original definition of relative entropy (KL divergence); open access on Project Euclid.
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — **Shannon (1948), *Bell System Technical Journal***. — the source-coding theorem: entropy is the optimal average code length, the foundation of the "bits / surprise" reading.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — **Hinton, Vinyals & Dean (2015)**. — knowledge distillation as minimising KL to a teacher's soft labels (a "where it's used" reference).
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)**. — the RLHF objective with its KL penalty $\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})$; where the KL "leash" of the crux section lives.

**📚 Books (free chapters / full PDFs)**:
- [Elements of Information Theory — Ch. 2 "Entropy, Relative Entropy, and Mutual Information"](http://staff.ustc.edu.cn/~cgong821/Wiley.Interscience.Elements.of.Information.Theory.Jul.2006.eBook-DDU.pdf) — **Cover & Thomas** — the standard reference; the definitions of entropy, cross-entropy, and KL used in the derivation (free chapter PDF).
- [Information Theory, Inference, and Learning Algorithms — Ch. 2 (Relative Entropy) & Ch. 4](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — KL, cross-entropy, Gibbs' inequality, and the source-coding theorem, in the free classic (full PDF on the author's site).
- [Mathematics for Machine Learning — Ch. 8 (esp. 8.3 MLE / cross-entropy)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — derives minimising cross-entropy / KL to the empirical distribution = maximum likelihood. Free full PDF.
- [Deep Learning — Ch. 3.13 (Information Theory) & Ch. 6.2 (softmax + cross-entropy)](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — the cross-entropy classification loss and the softmax gradient $\hat q - y$; free online.

**🔗 In this platform**:
- Concept page (full explanation): [Cross-Entropy & KL Divergence](cross-entropy-and-kl-divergence.md)
- Prerequisite (the *why* behind bits and surprise): [22 Entropy](../entropy/entropy.md)
- The unifying theorem it rests on: [19 Maximum Likelihood Estimation](../maximum-likelihood-estimation/maximum-likelihood-estimation.md) (minimising cross-entropy = MLE)
- Where it goes next: [24 Mutual Information](../mutual-information/mutual-information.md) — $I(X;Y) = D_{KL}(\text{joint}\,\|\,\text{product of marginals})$, KL applied to dependence
- Applied downstream: the classification/LM loss deep dives → [AI-ML-intuition 3.03 Categorical Cross-Entropy / NLL](../../../../AI-ML-intuition/objectives-and-evaluation/training-objectives/categorical-cross-entropy-intuition.md) · [5.01 Entropy & KL](../../../../AI-ML-intuition/foundational-mental-models/information-and-dependence/entropy-and-kl-divergence-intuition.md); the RLHF KL penalty → [LLMs](../../../09.%20LLMs/README.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Information Theory)](../maths-for-ai-ml/README.md)

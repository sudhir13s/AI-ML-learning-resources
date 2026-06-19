---
id: "01-foundations/entropy"
topic: "Entropy"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/random-variables-and-distributions"]
interview_frequency: high
updated: 2026-06-20
---

# Entropy
> Entropy `H(p) = −Σ p log p` measures the average uncertainty (or information content) of a
> distribution — the expected number of bits to encode samples from it. It's the bedrock of
> information theory and the reason cross-entropy is the default classification loss, why decision
> trees split on information gain, and how we quantify a model's surprise.

**Why it matters:** entropy is the parent concept behind cross-entropy/KL (the losses) and
information gain (tree splits). Interviewers ask what entropy measures, why a uniform distribution
maximizes it, the units (bits vs nats), and how it connects to coding/compression and to the loss
functions you train with.

**⭐ Start here — suggested path:**

1. **Information intuition** — watch [Khan: Information entropy](https://www.youtube.com/watch?v=2s3aJfRr9gE) (and [Measuring information](https://www.youtube.com/watch?v=PtmzfpV6CDE)). *Why "surprise" and "bits" are the right units.*
2. **For data science** — watch [StatQuest: Entropy (for data science), Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw). *Entropy as used in trees and ML losses.*
3. **Entropy → cross-entropy → KL** — watch [Aurélien Géron: A Short Introduction to Entropy, Cross-Entropy and KL](https://www.youtube.com/watch?v=ErfnhcEV1O8). *The cleanest single bridge to the loss functions.*
4. **Formalize** — read [MacKay, Information Theory, Inference & Learning Algorithms — Ch. 2 & 4](https://www.inference.org.uk/itprnn/book.pdf). *Entropy, the source-coding theorem, and the units.*
5. **Connect to ML** — read [AI-ML-intuition 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md). *Where entropy powers ML objectives.*

## 🎓 Courses (free)
- [Khan Academy — Journey into Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory) — **Khan Academy** — entropy and information from first principles, free.
- [Stanford EE376A — Information Theory (course materials)](https://web.stanford.edu/class/ee376a/) — **Stanford** — the rigorous treatment of entropy and coding; free notes.

## 🎥 Videos
- [Information entropy | Journey into information theory](https://www.youtube.com/watch?v=2s3aJfRr9gE) — **Khan Academy** — entropy as expected surprise.
- [Entropy (for data science) Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw) — **StatQuest (Josh Starmer)** — entropy for ML and decision trees.
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — entropy → cross-entropy → KL in 10 minutes.
- [Measuring information | Journey into information theory](https://www.youtube.com/watch?v=PtmzfpV6CDE) — **Khan Academy** — bits, symbols, and information content.

## 📄 Key Papers
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — **Claude Shannon (1948)** — the paper that defined entropy and founded information theory.
- [MacKay — Information Theory, Inference & Learning Algorithms (Ch. 2, 4)](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — entropy and source coding; the free canonical textbook.

## 📰 Articles / Blogs (free, no paywall)
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — **Christopher Olah** — the best free visual essay on entropy, cross-entropy, and KL.
- [Entropy (MML book, Ch. 6.5 / Information theory notes)](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth et al.** — entropy in the probability chapter.

## 📚 Books (free, with chapters)
- [Information Theory, Inference, and Learning Algorithms — **Ch. 2, 4 (Entropy, Source Coding)**](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — the free classic.
- [Elements of Information Theory — **Ch. 2 (Entropy, Relative Entropy & Mutual Information)**](http://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover&Thomas-Ch2.pdf) — **Cover & Thomas** — the standard reference's entropy chapter (free PDF).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Information Theory, row 3.5)](../Maths%20for%20AI-ML/README.md)
- Prereq: [16 Random Variables & Distributions](16-Random-Variables-and-Distributions.md) · Next: [23 Cross-Entropy & KL Divergence](23-Cross-Entropy-and-KL-Divergence.md) · [24 Mutual Information](24-Mutual-Information.md)
</content>

---
id: "04-unsupervised-learning/association-rule-learning/references"
topic: "Association Rule Learning — References"
parent: "04-unsupervised-learning/association-rule-learning"
type: references
updated: 2026-06-22
---

# Association Rule Learning — references and further reading

> Companion link library for **[Association Rule Learning](11-Association-Rule-Learning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (Agrawal, Han, Zaki), the canonical free textbook (MMDS, ESL), or a recognized deep maintainer (Raschka's mlxtend, Hahsler's arules) — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Apriori Algorithm Explained | Finding Frequent Itemsets](https://www.youtube.com/watch?v=guVvtZ7ZClw) (**edureka!**). *Support / confidence / lift and the candidate-generation loop on a small basket example — the clearest first pass.*
2. **See the pruning** — watch [Association Rule Mining — Apriori, step by step](https://www.youtube.com/watch?v=rXBc25pXhpk) (**Sinzsma**). *Why "every subset of a frequent set is frequent" lets you discard most candidates, worked itemset by itemset.*
3. **Get the math** — read [MMDS Ch. 6 "Frequent Itemsets"](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf) (**Leskovec, Rajaraman & Ullman**). *The formal support / confidence / lift definitions and the A-priori principle at scale.*
4. **Read the sources** — [Apriori (Agrawal & Srikant, 1994)](https://www.macs.hw.ac.uk/~dwcorne/Teaching/agrawal94fast.pdf) → [FP-Growth (Han, Pei & Yin, 2000)](https://link.springer.com/article/10.1023/B:DAMI.0000005258.31418.83). *The candidate-generation algorithm, then the prefix-tree method that avoids it.*
5. **Make it concrete** — code it with [mlxtend's Apriori & association rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) (**Sebastian Raschka**). *Mining real baskets and ranking rules by lift cements it.*

**Videos**:
- [Apriori Algorithm Explained | Association Rule Mining | Finding Frequent Itemset](https://www.youtube.com/watch?v=guVvtZ7ZClw) — **edureka!** — support / confidence / lift and candidate generation on a clear example; the best concise first watch.
- [Association Rule Mining — Apriori Algorithm Explained Step by Step](https://www.youtube.com/watch?v=rXBc25pXhpk) — **Sinzsma** — the downward-closure pruning worked through itemset by itemset.
- [Apriori Algorithm — Association Rule Learning](https://www.youtube.com/watch?v=T3Pd_3QP9J4) — **Super Data Science** — a second walkthrough that emphasizes the rule-quality metrics and thresholds.

**Courses (free)**:
- [Mining of Massive Datasets — Frequent Itemsets](http://www.mmds.org/) — **Leskovec, Rajaraman & Ullman (Stanford)** — free course + book; the definitive treatment of Apriori, support / confidence / lift, and scaling to huge baskets.
- [mlxtend — Frequent patterns user guide](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) — **Sebastian Raschka** — the practical Python reference with runnable Apriori, FP-Growth, and rule generation.

**Articles / blogs (free, no paywall)**:
- [Frequent Itemsets via Apriori (mlxtend docs)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) — **Sebastian Raschka** — copy-paste recipe with support / confidence / lift on real baskets.
- [Association rules (mlxtend docs)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) — **Sebastian Raschka** — generating and ranking rules, with every metric (lift, leverage, conviction) defined.
- [FP-Growth (mlxtend docs)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/) — **Sebastian Raschka** — the drop-in faster alternative to Apriori, same API, identical itemsets.
- [A Probabilistic Comparison of Interestingness Measures](https://mhahsler.github.io/arules/docs/measures) — **Michael Hahsler** — the canonical catalog of support, confidence, lift, leverage, conviction, and dozens more, each derived.
- [arules: Mining Association Rules in R](https://mhahsler.github.io/arules/) — **Michael Hahsler** — the reference implementation and documentation maintained by a domain authority.
- [Frequent Itemsets — MMDS Ch. 6 (PDF)](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf) — **Leskovec, Rajaraman & Ullman** — the clearest free written treatment of the algorithms and metrics.

**Papers**:
- [Mining Association Rules between Sets of Items in Large Databases](https://courses.cs.washington.edu/courses/cse544/11wi/papers/agrawal-sigmod-1993.pdf) — **Agrawal, Imieliński & Swami (SIGMOD 1993)** — the paper that introduced association rules, support, and confidence.
- [Fast Algorithms for Mining Association Rules (Apriori)](https://www.macs.hw.ac.uk/~dwcorne/Teaching/agrawal94fast.pdf) — **Agrawal & Srikant (VLDB 1994)** — the A-priori principle and the candidate-generation algorithm.
- [Mining Frequent Patterns without Candidate Generation (FP-Growth)](https://link.springer.com/article/10.1023/B:DAMI.0000005258.31418.83) — **Han, Pei & Yin (DMKD 2000/2004)** — the FP-tree and pattern-growth method that avoids Apriori's expensive scans. ([SIGMOD 2000 conference PDF](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf).)
- [Scalable Algorithms for Association Mining (Eclat)](https://www.cs.rpi.edu/~zaki/PaperDir/TKDE00.pdf) — **Zaki (IEEE TKDE 2000)** — the vertical tid-list / tidset-intersection approach.

**Books (free chapters)**:
- [Mining of Massive Datasets — **Ch. 6 "Frequent Itemsets"**](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; Apriori, the A-priori principle, and scaling to huge baskets.
- [The Elements of Statistical Learning — **§14.2 "Association Rules"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the statistical view of support, confidence, and the Apriori algorithm.

**In this platform**:
- Concept page (full explanation): [Association Rule Learning](11-Association-Rule-Learning.md)
- Concept depth (probability foundations): [AI-ML-intuition 0.01 Probability & Bayes' Theorem](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) — confidence and lift are conditional-probability quantities.
- Compare with: [01 K-Means Clustering](../01-K-Means-Clustering/01-K-Means-Clustering.md) · [02 Hierarchical Clustering](../02-Hierarchical-Clustering/02-Hierarchical-Clustering.md) — the other classic "unsupervised structure discovery" techniques.
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

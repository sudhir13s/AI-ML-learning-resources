---
id: "04-unsupervised-learning/association-rule-learning"
topic: "Association Rule Learning (Apriori · FP-Growth)"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["sets", "probability", "conditional-probability"]
interview_frequency: medium
updated: 2026-06-19
---

# Association Rule Learning
> Find "if-then" patterns in transactional data — *people who buy {bread, butter} also buy {jam}*.
> **Apriori** prunes the search using the rule that any subset of a frequent itemset is also frequent;
> **FP-Growth** does the same far faster by compressing the data into a prefix tree. Classic market-basket
> analysis.

**Why it matters:** the canonical "market-basket / recommendation without ML models" topic. Interviews
test the three metrics — **support** (how often an itemset appears), **confidence** (conditional
probability of the consequent), and **lift** (confidence over baseline, the one that filters out
coincidences) — plus the combinatorial blowup of itemsets and how each algorithm tames it: Apriori's
**downward-closure** pruning (and its many database scans) vs **FP-Growth**'s single-tree, scan-light
approach. Knowing *why lift matters* (high-confidence rules can still be useless if the consequent is
just popular) is the differentiator.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Apriori Algorithm Explained | Finding Frequent Itemsets](https://www.youtube.com/watch?v=guVvtZ7ZClw). *Support/confidence/lift and the candidate-generation loop on a small basket example.*
2. **See the pruning** — [Association Rule Mining — Apriori, step by step](https://www.youtube.com/watch?v=rXBc25pXhpk). *Why "every subset of a frequent set is frequent" lets you discard most candidates.*
3. **Get the math** — read [MMDS Ch. 6 "Frequent Itemsets"](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf). *The formal support/confidence/lift definitions and the A-priori principle at scale.*
4. **Read the sources** — [Apriori (Agrawal & Srikant, 1994)](https://www.vldb.org/conf/1994/P487.PDF) → [FP-Growth (Han, Pei & Yin, 2000)](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf). *The candidate-generation algorithm, then the prefix-tree method that avoids it.*
5. **Make it concrete** — code it with [mlxtend's Apriori & association rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/). *Mining real baskets and ranking rules by lift cements it.*

## 🎓 Courses (free)
- [Mining of Massive Datasets — Frequent Itemsets](http://www.mmds.org/) — **Leskovec, Rajaraman & Ullman (Stanford)** — free course + book; the definitive treatment of Apriori, support/confidence/lift, and scaling.
- [mlxtend — Frequent patterns & association rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) — **Sebastian Raschka** — the practical Python reference with runnable Apriori/FP-Growth and rule generation.

## 🎥 Videos
- [Apriori Algorithm Explained | Association Rule Mining | Finding Frequent Itemset](https://www.youtube.com/watch?v=guVvtZ7ZClw) — **edureka!** — support/confidence/lift and candidate generation on a clear example.
- [Association Rule Mining — Apriori Algorithm Explained Step by Step](https://www.youtube.com/watch?v=rXBc25pXhpk) — **Augmented Startups** — the pruning principle worked through itemset by itemset.
- [Apriori Algorithm — Association Rule Learning](https://www.youtube.com/watch?v=T3Pd_3QP9J4) — **Anuj Shah** — a second walkthrough emphasizing the rule-quality metrics.
- [Unsupervised Anomaly Detection Explained: Clustering, Density & Isolation Forest](https://www.youtube.com/watch?v=QZNEJHbophM) — **Data Science Garage** — context for where pattern-mining sits among unsupervised methods.

## 📄 Key Papers
- [Fast Algorithms for Mining Association Rules (Apriori)](https://www.vldb.org/conf/1994/P487.PDF) — **Agrawal & Srikant (1994)** — the original; the A-priori principle and candidate generation.
- [Mining Frequent Patterns without Candidate Generation (FP-Growth)](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf) — **Han, Pei & Yin (2000)** — the prefix-tree method that avoids Apriori's expensive scans.

## 📰 Articles / Blogs (free, no paywall)
- [Frequent Itemsets via Apriori (mlxtend docs)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) — **Sebastian Raschka** — copy-paste recipe with support/confidence/lift on real baskets.
- [Association rules (mlxtend docs)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) — **Sebastian Raschka** — generating and ranking rules, with the metric definitions.
- [Frequent Itemsets — MMDS Ch. 6 (PDF)](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf) — **Leskovec, Rajaraman & Ullman** — the clearest free written treatment of the algorithms and metrics.

## 📚 Books (free, with chapters)
- [Mining of Massive Datasets — **Ch. 6 "Frequent Itemsets"**](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; Apriori, the A-priori principle, and scaling to huge baskets.
- [The Elements of Statistical Learning — **§14.2 "Association Rules"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the statistical view of support, confidence, and the Apriori algorithm.

## 🔗 In this platform
- Concept depth (probability foundations): [AI-ML-intuition 0.01 Probability & Bayes' Theorem](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md) — confidence and lift are conditional-probability quantities
- Related: [01 K-Means Clustering](01-K-Means-Clustering.md) — the other classic "unsupervised structure discovery" technique
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)

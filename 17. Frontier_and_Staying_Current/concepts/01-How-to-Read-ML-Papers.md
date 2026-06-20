---
id: "17-frontier/how-to-read-papers"
topic: "How to Read ML Papers"
parent: "17-frontier"
level: intermediate
prereqs: ["basic-deep-learning", "linear-algebra"]
interview_frequency: medium
updated: 2026-06-20
---

# How to Read ML Papers
> A repeatable, multi-pass system for extracting the idea, the evidence, and the limitations from an
> ML paper — without reading every word. The single highest-leverage skill for staying current:
> the field moves in papers, and the people who keep up are the ones who can triage 20 and deep-read 2.

**Why it matters:** in interviews and on the job you'll be handed a paper ("walk me through how
this works / would you trust this result?") and expected to find the contribution, the baseline,
the ablations, and the threats to validity fast. Reading skill is what separates "I saw the
headline" from "I understand the method and its limits."

**⭐ Start here — suggested path:**

1. **Adopt the three-pass method** — read ⭐ ["How to Read a Paper" (Keshav)](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf), one page, then watch [Andrew Ng on reading papers (CS230 Lec 8)](https://www.youtube.com/watch?v=733m6qBH-jI). *Pass 1 = title/abstract/figures/conclusion; pass 2 = method + results skim; pass 3 = reconstruct it yourself. This is the backbone of everything below.*
2. **Practice triage** — read titles → abstracts → figures of 10 papers in your area before deep-reading any. *Andrew Ng's rule: 5–20 papers gives basic grasp of a field, 50–100 gives mastery. Breadth first, depth on demand.*
3. **Read for the contribution and the evidence** — for each paper answer: what's new? what's the baseline? what do the ablations remove? *A method "works" only relative to a baseline; the ablation table tells you which part actually matters.*
4. **Hunt the limitations** — read the limitations/related-work sections and ask what the authors did *not* test. *This is the "evaluate hype vs substance" muscle — and exactly what interviewers probe.*
5. **Reconstruct, then move on** — write 3 sentences (problem / method / result) in your own words; only re-open the paper if you'd build on it. *Active recall beats re-reading; the note becomes your searchable memory.*

## 🎓 Courses (free)
- [Stanford CS230 — Career Advice / Reading Research Papers (Lec 8)](https://cs230.stanford.edu/lecture) — **Andrew Ng / Stanford** — the canonical lecture on a paper-reading workflow and how many to read.
- [Harvard CS197: AI Research Experiences](https://www.cs197.seas.harvard.edu/course-content) — **Pranav Rajpurkar / Harvard** — a full free course (lecture notes online) on reading, critiquing, and producing AI research.

## 🎥 Videos
- [Stanford CS230 Lec 8 — Reading Research Papers](https://www.youtube.com/watch?v=733m6qBH-jI) — **Andrew Ng / Stanford** — the multi-pass method and a concrete reading plan, straight from the source.
- [How To Read AI Research Papers Effectively](https://www.youtube.com/watch?v=K6Wui3mn-uI) — **DeepLearning.AI** — a practical, ML-specific reading workflow.
- [How to actually learn AI/ML: Reading Research Papers](https://www.youtube.com/watch?v=x6slke5niqw) — **Jean Lee** — a beginner-friendly walkthrough of approaching your first papers.
- [Read a research paper effectively — tools and tricks](https://www.youtube.com/watch?v=g8qatelVS7c) — **Andy Stapleton** — triage + tooling to move through papers faster.
- [How To Read A Paper Quickly & Effectively](https://www.youtube.com/watch?v=0w61Ou-F5vo) — **Dr Amina Yonis** — a quick demonstration of the skim-first approach.

## 📄 Key Papers
- [How to Read a Paper](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) — **S. Keshav (2007)** — the original "three-pass" method; the canonical reference for this skill.
- [How to Write a Great Research Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) — **Simon Peyton Jones (Microsoft Research)** — reading the structure authors are taught to write makes papers easier to read.
- [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341) — **Lipton & Steinhardt (2018)** — what to be skeptical of when reading ML papers (a critical-reading checklist).

## 📰 Articles / Blogs (free, no paywall)
- [How to Read a Paper (Keshav) — HTML mirror](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) — **Stanford** — the three-pass method as a printable handout.
- [How to read research papers — a pragmatic guide](https://www.semanticscholar.org/product/tutorials) — **Semantic Scholar** — search/triage tooling that supports the reading workflow.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — a model of how a hard paper (Attention Is All You Need) becomes readable; use it as a "what good understanding looks like" target.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed.](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — each chapter's "Bibliographical and Historical Notes" models how to place a paper in its lineage.
- [Dive into Deep Learning](https://d2l.ai/) — **Zhang et al.** — pairs prose with runnable code; the gold standard for "reconstruct the method to understand it."

## 🔗 In this platform
- Per-concept index: [Frontier & Staying Current — concepts](README.md)
- Where to find papers to read: [02 arXiv & Papers with Code](02-arXiv-and-Papers-with-Code.md) · [04 Newsletters & Blogs to Follow](04-Newsletters-and-Blogs-to-Follow.md)
- Apply it to a frontier area: [08. LLMs](../../08.%20LLMs/concepts/README.md) · [09. GenAI](../../09.%20GenAI/concepts/README.md)

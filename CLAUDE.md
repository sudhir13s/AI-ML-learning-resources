# Project: AI-ML-learning-resources

> Reference notes for Claude. The **curated resource library** of the platform (renamed from
> `AI-study-notes-and-links`, 2026-06-11).

## Purpose
A curated library of **the best free resources on the internet** per AI/ML topic — courses,
videos, papers, articles, books — chosen for authority + clarity (top institutions/researchers/
best explainers), free/open preferred, ~2 per type, each with a one-line "why it's the best."
Doubles as a **dataset** for the interview-prep project, so the format must stay consistent and
parseable.

## Structure (ONE pattern — do not deviate)
- **One `README.md` per topic folder.** No `Links.md` / `Notes.md` / `Resources/` (deleted).
- Each topic README = **YAML frontmatter** (`id`, `topic`, `level`, `prereqs`, `updated`) +
  curated sections in this order: **⭐ Start here · 🎓 Courses · 🎥 Videos · 📄 Papers ·
  📰 Articles · 📚 Books · 🔗 In this platform**. ~2 entries per section, format
  `[Title](url) — **Author/Institution** — why it's the best.`
- Canonical example: [05. Deep_Learning/README.md](05.%20Deep_Learning/README.md).

## Topics (numbered folders; **zero-padded** `NN. Name` so they sort correctly everywhere, incl. GitHub)
00 Basics · 01 Foundations (+ `Maths for AI-ML/` deep math curriculum) · 02 Data-Preprocessing ·
03 Supervised · 04 Unsupervised · 05 Deep-Learning · 06 NLP · 07 Computer-Vision · 08 LLMs ·
09 GenAI · 10 RL · 11 Tools-and-Frameworks · 12 MLOps-and-Deployment · 13 Neuroscience ·
14 Advanced-Research-Math · **15 Agentic-AI · 16 RAG-and-LLM-Apps · 17 Frontier-Staying-Current** (new).

Specialization folders (07/13/14) hold deeper *curricula* (what/why/resources by sub-module);
`llm_systems_curriculum.md` is the 14-chapter LLM-systems syllabus. The root `README.md` is the
master index.

## How to help here
- Add/curate resources, keep the bar high (best explainer, free, ~2 per type, with a "why").
- Keep the one-README-per-topic pattern + frontmatter (the dataset contract).
- Cross-link into [`AI-ML-intuition`](../AI-ML-intuition/) (the *why*) and
  [`AI-ML-problemsets`](../AI-ML-problemsets/) (the *practice*); don't duplicate their depth here.

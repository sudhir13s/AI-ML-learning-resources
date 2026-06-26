# Project: AI-ML-learning-resources

> Reference notes for Claude. The **curated resource library** of the platform (renamed from
> `AI-study-notes-and-links`, 2026-06-11).

## Purpose
Two complementary layers per topic:
1. **Curated resource library** — the best free resources on the internet per AI/ML topic —
   courses, videos, papers, articles, books — chosen for authority + clarity (top institutions/
   researchers/best explainers), free/open preferred, each with a one-line "why it's the best."
2. **Deep concept pages** (in each topic's `concepts/` folder) — blog-quality, intuition-first
   teaching pages that are the *canonical explanation* of a concept on this platform. These carry
   real depth (math, diagrams, runnable code) and use the **two-file standard** below.

Doubles as a **dataset** for the interview-prep app, so formats must stay consistent and parseable.

## Structure (ONE pattern — do not deviate)
- **One `README.md` per topic folder.** No `Links.md` / `Notes.md` / `Resources/` (deleted).
- Each topic README = **YAML frontmatter** (`id`, `topic`, `level`, `prereqs`, `updated`) +
  curated sections in this order: **⭐ Start here · 🎓 Courses · 🎥 Videos · 📄 Papers ·
  📰 Articles · 📚 Books · 🔗 In this platform**. ~2 entries per section, format
  `[Title](url) — **Author/Institution** — why it's the best.`
- Canonical example: [05. Deep_Learning/README.md](05.%20Deep_Learning/README.md).

## Concept pages (deep teaching content) — the two-file standard

> **GOLD STANDARD — MUST (no exceptions).** The KV-Cache pages are the ratified gold standard for
> every concept page. **Before creating or modifying ANY `concepts/` file** — yourself or via any
> specialist skill / subagent you invoke — **first read both gold-standard files**
> ([05-KV-Cache.md](09.%20LLMs/05-KV-Cache/05-KV-Cache.md) + [05-KV-Cache.references.md](09.%20LLMs/05-KV-Cache/05-KV-Cache.references.md))
> and match them. **Before marking the work done, verify the result against them**: section flow
> present & in order, visuals generated and rendering (PNGs viewed, mermaid validates), code runs in
> `~/.uv/envs/ml-py312`, every reference link verified, bold-not-highlight, no emoji in headings,
> two-file split correct. Mark done only after that verification passes. A subagent's prompt MUST
> include this read-first + verify-before-done instruction.
>
> **Reference-grade depth (raised).** Pages are **exhaustive deep-dives, not summaries** — the
> definitive page on the topic. The repeated rework failure is "comprehensive but not exhaustive."
> Build to this from the start: cover every sub-concept; **multiple worked numeric examples at
> increasing complexity** (minimal scalar → realistic vector/matrix → full end-to-end trace), not
> one token example; **derive key results step by step** (show the algebra), don't just state them;
> **over-explain** so a learner has no gaps; **several diagrams per major concept**. Length is a
> feature — these typically run **~500–800+ lines**; never cap it. Enumerate the sub-concepts first.
>
> **Specialist-judge review (mandatory).** After building and self-verifying, review the page as an
> **AI/ML specialist-teacher judge** (score /100 + concrete comments on gaps and weak spots), then
> **address every comment** before the PR — **target 95–98**; if below, the fix is almost always to
> *expand* (add an example, a derivation, a diagram) and re-judge. (Inline by default; subagent only if asked.)

Each topic's `concepts/` folder holds the **deep, blog-quality teaching pages**. As of 2026-06-21
each concept is **two files** (canonical example: [09. LLMs/05-KV-Cache/05-KV-Cache.md](09.%20LLMs/05-KV-Cache/05-KV-Cache.md)
+ [05-KV-Cache.references.md](09.%20LLMs/05-KV-Cache/05-KV-Cache.references.md)):

- **`NN-Concept.md` — the content.** A progressive, intuition-first page written in the voice of a
  researcher-teacher writing a tech blog (style bar = Practitioner-Workflows `RLHF-and-Alignment.md`).
  Section flow: **Problem / why it was introduced → What it is → Intuition (analogy) → Why it matters
  → How it works → The math (derived, every symbol defined) → Where/when used → Application
  (step-by-step playbook) → Code (runnable, verified in `~/.uv/envs/ml-py312`) → Recap & rapid-fire**.
  Plus: **many woven visuals** (matplotlib PNGs via `tools/` generators + palette mermaid — author is
  a visual learner, multiple images encouraged, placed at the moment each idea needs one); inline
  **Note / Tip / Gotcha** callouts wherever a point earns one (book-margin style, not bucketed);
  **bold** for emphasis (no highlighter); **no emoji in headings**; keep contextual links inline in
  the body. Generators live in `tools/`; every code block and PNG must actually run/render and be
  visually verified.
- **`NN-Concept.references.md` — the links.** The curated link library for that concept, kept
  **separate on purpose**: later it doubles as a standalone references list, and it holds **internal**
  links (to our own pages, incl. the content page itself) alongside **external** ones. Flat (one
  level), best-first, in order: **Start-here path · Videos · Courses · Articles · Papers · Books · In
  this platform**. Bar: **15+ entries, authority sources only** (primary authors / recognized deep
  explainers — Raschka, Olah/Distill, 3Blue1Brown, Karpathy, Lilian Weng, the paper's authors — not
  generic popular tutorials); every link verified. The content page ends with a one-line pointer to
  this companion.

This **intentionally inverts** the old "curated links only — don't duplicate depth" rule for
`concepts/` pages: when a concept has no `AI-ML-intuition` page, its concept page is the canonical
deep home. (Topic-level `README.md`s stay link-only as described above.)

## Topics (numbered folders; **zero-padded** `NN. Name` so they sort correctly everywhere, incl. GitHub)
Ordered by learning progression: 00 Basics · 01 Foundations (+ `Maths for AI-ML/` deep math curriculum) ·
02 Data-Preprocessing · 03 Supervised · 04 Unsupervised · 05 Deep-Learning · 06 NLP · 07 Computer-Vision ·
08 Reinforcement-Learning · 09 LLMs · 10 GenAI · 11 RAG-and-LLM-Apps · 12 Agentic-AI ·
13 Tools-and-Frameworks · 14 MLOps-and-Deployment · 15 Advanced-Research-Math · 16 Neuroscience ·
17 Frontier-Staying-Current.

Specialization folders (07/13/14) hold deeper *curricula* (what/why/resources by sub-module);
`llm_systems_curriculum.md` is the 14-chapter LLM-systems syllabus. The root `README.md` is the
master index.

## How to help here
- **Topic `README.md`s**: add/curate resources, keep the bar high (best explainer, free, with a
  "why"); keep the one-README-per-topic pattern + frontmatter (the dataset contract). These stay
  link-only — don't put concept depth here.
- **`concepts/` pages**: author/raise them to the two-file standard above (deep content + separate
  references). These *are* allowed to carry full depth; they're the canonical deep home for a
  concept when no `AI-ML-intuition` page exists.
- Cross-link into [`AI-ML-intuition`](../AI-ML-intuition/) (the *why*) and
  [`AI-ML-problemsets`](../AI-ML-problemsets/) (the *practice*) where those pages exist.

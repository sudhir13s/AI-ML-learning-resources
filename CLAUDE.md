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
- **The tree is `section/sub-area/topic-package/`, all kebab-case** — the chartered shape
  landed by T-86 (`docs/plans/structure-rework/ai-ml-blueprints/ai-ml-learning-resources-structure.md`).
  No ordinal prefixes, no spaces, no underscores, nothing about order or difficulty in a name.
- **`course.yaml` at the repo root declares every section**; a `metadata.yaml` carrying
  `page-order:` (the ONE ordering key estate-wide) fixes reading order inside every sub-area.
- **A topic package is a folder**: `<topic>/<topic>.md` plus its `<topic>.references.md`
  companion and its own `images/`, `code/` and `notebooks/`.
- **One `README.md` per sub-area** — the curated resource index. No `Links.md` / `Notes.md` /
  `Resources/` (deleted). It is repo-facing and is deliberately never a declared chapter.
- Each sub-area README = **YAML frontmatter** (`id`, `topic`, `level`, `prereqs`, `updated`) +
  curated sections in this order: **⭐ Start here · 🎓 Courses · 🎥 Videos · 📄 Papers ·
  📰 Articles · 📚 Books · 🔗 In this platform**. ~2 entries per section, format
  `[Title](url) — **Author/Institution** — why it's the best.`
- Canonical example: [deep-learning/README.md](deep-learning/README.md).

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
  **Tiered beat flow (v2, decision-10 in the parent store — the page's `tier` decides which beats it
  owes; `##` section headings stay natural sentence case, e.g. `## The problem: decoding repeats
  itself` — ALL-CAPS headings were piloted 2026-07-14 and REVERSED by the owner 2026-07-15
  ("doesn't look good"); do not reintroduce them), in this order:**
  - **CORE (every page):** The problem / why it was introduced → What it is → Intuition (analogy) →
    Why it works / why it matters → How it works → The math (derived, every symbol defined) →
    Numerical example(s) (minimal scalar → realistic vector → end-to-end trace) → Code (**PyTorch
    ONLY** — no NumPy/TF variants; runnable, verified in `~/.uv/envs/ml-py312`) → Where it is used —
    **and where it is NOT** (trade-offs, when-not-to-use) → Production failure modes → Recap &
    rapid-fire → Related concepts → References.
  - **STANDARD adds:** **Common misconceptions** (after intuition — consolidates the traps
    interviewers probe; woven Gotcha callouts stay) and **What-if analysis** (after code — X↑ / X↓ /
    remove-the-component ablations, each with expected behavior, failure case, and trade-off).
  - **FLAGSHIP adds:** **Interactive explorer** and **Implementation visualization** (step-player) —
    each authored as a widget marker (`<!-- EXPLORER: <id> -->` / `<!-- STEP-PLAYER: <id> -->`) plus
    2–4 lines of spec-fallback prose so the page reads complete with zero renderer support. Never an
    empty section; a non-applicable beat is omitted entirely.
  **Related concepts is RENDERED from frontmatter `built_from`/`leads_to` — never authored prose.
  References is ALWAYS the last section** (the one-line pointer to the companion file).
  Plus: **many woven visuals** (matplotlib PNGs via `tools/` generators + palette mermaid — author is
  a visual learner, multiple images encouraged, placed at the moment each idea needs one); **no GIFs,
  ever** — animation stack is animated SVG/CSS → React-driven SVG → Manim muted looping MP4; inline
  **Note / Tip / Gotcha** callouts wherever a point earns one (book-margin style, not bucketed);
  **bold** for emphasis (no highlighter); **no emoji in headings**; keep contextual links inline in
  the body. **Lead-then-bullets (owner directive 2026-07-14):** every section/block opens with a
  1–3 line lead sentence, then bullets / sub-sections carry the detail — never a dense multi-line
  paragraph. Generators live in `tools/`; every code block and PNG must actually run/render and be
  visually verified.
- **Chaptered concepts (pilot: KV-Cache, T-85, 2026-07-14).** When one page can't hold the depth
  (main page pushing past ~550 lines with production/kernel/variant material), split into the
  **main page + numbered chapter files in the SAME topic folder**: `NN-Concept.md` (the tiered
  core/standard/flagship page — complete on its own) plus `NN-Concept.chK-<slug>.md` depth
  chapters (each: own frontmatter with `chapter: K` + `chapter_of: <main id>`, natural-case
  headings, lead-then-bullets, recap + next-chapter link, references pointer last). The main page
  lists chapters in a `chapters:` frontmatter array AND a `## Going deeper: the chapters` section
  (1-line-per-chapter index). The reader surfaces every chapter `.md` as an ordered lesson under
  the topic card (main page first). Canonical example: `09. LLMs/05-KV-Cache/` (main + ch1
  variants · ch2 optimization-ladder · ch3 flashattention-flashdecoding · ch4 production).
- **`NN-Concept.references.md` — the links.** The curated link library for that concept, kept
  **separate on purpose**: later it doubles as a standalone references list, and it holds **internal**
  links (to our own pages, incl. the content page itself) alongside **external** ones. Flat (one
  level), best-first, in order: **Start-here path · Videos · Courses · Articles · Papers · Books · In
  this platform**. Bar: **15+ entries, authority sources only** (primary authors / recognized deep
  explainers — Raschka, Olah/Distill, 3Blue1Brown, Karpathy, Lilian Weng, the paper's authors — not
  generic popular tutorials); every link verified. The content page ends with a one-line pointer to
  this companion.

**Frontmatter contract (v2 — field names harmonized with decision-9 / AI-ML-intuition):**
`id`, `topic`, `parent`, `level`, `tier` (core | standard | flagship), `est_minutes`,
`built_from` (upstream prereq concepts — **renames the old `prereqs` field**), `leads_to`
(downstream concepts), `interview_frequency`, `template`, `updated`. The hub hero renders
difficulty, tier badge, est-minutes, and prereq chips from these fields — never author hero
content in the body. **Migration is additive:** existing v1 pages stay valid; the tier sweep
(T-73) and Standard upgrade (T-74) land the new fields/beats without big-bang rewrites. The 20
gold LLM chapters default to `tier: standard`; the owner-approved Flagship shortlist gets
`flagship` (pilot: KV-Cache, T-75, which also fixes the explorer/step-player spec format).

This **intentionally inverts** the old "curated links only — don't duplicate depth" rule for
`concepts/` pages: when a concept has no `AI-ML-intuition` page, its concept page is the canonical
deep home. (Topic-level `README.md`s stay link-only as described above.)

## Sections (chartered, kebab-case — the ordinal `NN. Name` folders are gone)
Ordered by learning progression, each declared as a section in `course.yaml`:

- **`foundations/`** — ai-ml-orientation · programming-and-data-foundations ·
  mathematical-foundations (+ the `maths-for-ai-ml/` deep math curriculum) · data-preparation ·
  tools-and-frameworks · research-literacy.
- **`core-machine-learning/`** — supervised-learning (regression · classification ·
  trees-and-ensembles) · unsupervised-learning (clustering · dimensionality-reduction ·
  density-estimation · anomaly-detection · association-rules) · reinforcement-learning
  (foundations · value-based-learning · policy-learning · model-based · offline · multi-agent) ·
  model-selection-and-evaluation.
- **`deep-learning/`** — neural-network-foundations · optimization-and-training ·
  stabilization-and-architectural-blocks · neural-architectures · attention-and-transformers ·
  self-supervised-learning.
- **`modalities-and-generative-models/`** — natural-language-processing · computer-vision ·
  generative-models · diffusion-models · multimodal-learning · video-understanding ·
  audio-and-speech.
- **`llms-applications-and-agents/`** — rag-and-knowledge-systems · agentic-ai ·
  inference-and-runtime · reasoning-evaluation-and-alignment.
- **`deployment-and-mlops/`** — lifecycle-and-reproducibility · data-and-training-platforms ·
  packaging-and-serving · release-and-deployment · monitoring-and-reliability ·
  governance-and-economics.
- **`specialized-studies/`** — advanced-mathematics-for-ai-research ·
  neuroscience-and-brain-inspired-ai.

`09. LLMs/` is the **one folder still on its legacy name**: its twenty topics belong to
`llms-applications-and-agents/`, and the LLM wave re-homes them together with its
`course.yaml` section. `world-models-and-embodied-intelligence/` is chartered but has no
content yet, so it is deliberately absent rather than stubbed.

`llm_systems_curriculum.md` is the 14-chapter LLM-systems syllabus. The root `README.md` is
the master index.

## How to help here
- **Sub-area `README.md`s**: add/curate resources, keep the bar high (best explainer, free, with
  a "why"); keep the one-README-per-sub-area pattern + frontmatter (the dataset contract). These
  stay link-only — don't put concept depth here.
- **Topic pages** (`<topic>/<topic>.md`): author/raise them to the two-file standard above (deep
  content + separate `<topic>.references.md`). These *are* allowed to carry full depth; they're
  the canonical deep home for a concept when no `AI-ML-intuition` page exists.
- **Adding a topic**: create `<section>/<sub-area>/<topic>/<topic>.md`, give it `title`,
  `minutes` and a `category` equal to its sub-area folder, add the path to `course.yaml`, and
  add it to that sub-area's `metadata.yaml` `page-order:` at the right position.
- Cross-link into [`AI-ML-intuition`](../AI-ML-intuition/) (the *why*) and
  [`AI-ML-problemsets`](../AI-ML-problemsets/) (the *practice*) where those pages exist.

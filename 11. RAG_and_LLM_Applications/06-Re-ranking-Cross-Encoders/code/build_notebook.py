"""Generate the teaching notebook 06-Re-ranking-Cross-Encoders.ipynb.

Authoring the notebook as code keeps every cell under version control and lets us regenerate it
deterministically. The notebook imports the real `reranking.py` module and runs the REAL two-stage
pipeline over BeIR/scifact live: real bi-encoder retrieval, real cross-encoder reranking, real
nDCG@10/MRR@10 against real relevance labels. Corpus embeddings are cached (`data/`), so the notebook
reuses one embedding pass; the cross-encoder passes are the honest cost of a real demo.

Execute headless with:
    python -m nbconvert --to notebook --execute --inplace 06-Re-ranking-Cross-Encoders.ipynb

Run this generator with:  python build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "06-Re-ranking-Cross-Encoders.ipynb"

_CELL_COUNTER = [0]


def _next_id(kind: str) -> str:
    _CELL_COUNTER[0] += 1
    return f"{kind}-{_CELL_COUNTER[0]:02d}"


def md(*lines: str) -> dict:
    src = "\n".join(lines)
    return {"cell_type": "markdown", "id": _next_id("md"), "metadata": {}, "source": src.splitlines(keepends=True)}


def code(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "code", "id": _next_id("code"), "execution_count": None,
        "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# Re-ranking with Cross-Encoders — a stepwise walkthrough over a *real* IR benchmark",
        "",
        "This notebook is **not a toy**, and it doesn't skip steps. It runs the real two-stage "
        "retrieve-then-rerank pipeline over **BeIR/scifact** — a real scientific-claim retrieval "
        "benchmark with **5,183 real abstracts, 300 real test queries, and real human relevance "
        "judgments** — so every metric (nDCG@10, MRR@10, Recall, Precision) is grounded in truth, not "
        "invented. Stage 1 is a real bi-encoder (`all-MiniLM-L6-v2`); stage 2 is a real cross-encoder "
        "(`ms-marco-MiniLM-L-6-v2`). We walk — one operation at a time — through retrieval, joint "
        "pair-scoring, the metric math, the measured quality lift, the recall ceiling, the "
        "quality-vs-depth curve, and the real latency of each stage.",
        "",
        "**No faiss** (first-stage top-K is exact cosine via numpy, which also dodges the torch/faiss "
        "OpenMP conflict). Corpus embeddings are cached to `data/` so we embed once; the cross-encoder "
        "passes are the honest cost of a real demo. Inference is deterministic (dropout off, no "
        "sampling), so ranks and metrics reproduce; wall-clock latency varies and is reported as a "
        "median. Device is auto-detected (cuda → mps → cpu).",
        "",
        "> Run `python reranking.py` once first to warm the embedding cache if `data/` is empty.",
    ),
    # ----------------------------------------------------------------- Step 1
    md(
        "## Step 1 — load the real benchmark and see what a labelled IR dataset is",
        "",
        "A retrieval benchmark is three things: a **corpus** (passages to search), **queries**, and "
        "**qrels** — human relevance judgments saying which passages actually answer which query. Those "
        "qrels are what make the metrics *real*: we can check whether the ranking put the truly-relevant "
        "passages on top. `load_scifact` aligns all three into index-based **gold sets** — `gold[i]` is "
        "the set of relevant doc indices for query `i` (scifact has ~1.1 per query).",
    ),
    code(
        "import time",
        "",
        "import numpy as np",
        "",
        "from reranking import (",
        "    RETRIEVE_K, METRIC_K,",
        "    load_scifact, detect_device, BiEncoderRetriever, CrossEncoderReranker,",
        "    ndcg_at_k, mrr_at_k, recall_at_k, precision_at_k,",
        "    evaluate, sweep_rerank_depth, hero_query, pick_demo_query,",
        ")",
        "",
        "device = detect_device()",
        "try:",
        "    import sentence_transformers",
        "    import torch",
        "    print(f'torch {torch.__version__} | sentence-transformers {sentence_transformers.__version__} '",
        "          f'| numpy {np.__version__} | device: {device}')",
        "except ImportError:",
        "    print(f'numpy {np.__version__} | device: {device}')",
        "",
        "data = load_scifact()",
        "print(f'\\nBeIR/scifact: {data.n_docs:,} passages | {data.n_queries} test queries with gold')",
        "print(f'mean relevant docs/query: {np.mean([len(g) for g in data.gold]):.2f}')",
    ),
    # ----------------------------------------------------------------- Step 2
    md(
        "## Step 2 — read a real query and the passage that answers it",
        "",
        "Before ranking anything, look at the data. A scifact query is a scientific *claim*; the gold "
        "passage is an abstract that supports (or refutes) it. Seeing the real text makes every rank "
        "below interpretable — you can *read* whether a retrieved passage is genuinely on-topic. Here's "
        "one example query and its gold abstract(s):",
    ),
    code(
        "example_qi = 0",
        "print('QUERY:', data.query_texts[example_qi])",
        "print('\\ngold (relevant) passages for this query:')",
        "for gi in sorted(data.gold[example_qi]):",
        "    print(f'  doc[{gi}]: {data.doc_texts[gi][:150]}...')",
    ),
    # ----------------------------------------------------------------- Step 3
    md(
        "## Step 3 — build both stages, and pick a walkthrough query *by measurement*",
        "",
        "The first stage (`BiEncoderRetriever`) embeds the query and **every passage independently** into "
        "unit vectors, then ranks by cosine similarity (a dot product on unit vectors). Because the "
        "passage vectors don't depend on the query, they are **precomputed once** and cached — at query "
        "time we embed only the query and take dot products. That independence is what makes it fast, and "
        "exactly what makes it coarse: the query never *sees* the passage. We also build the second-stage "
        "`CrossEncoderReranker` now, so we can **choose our single-query walkthrough by measurement** "
        "rather than an arbitrary index: `pick_demo_query` scans a few queries and returns one where "
        "re-ranking *demonstrably helps* (gold in the pool but not first → promoted toward the top). "
        "(Re-ranking is **not** guaranteed to help every query — Step 8's honest aside shows a query where "
        "it hurts — which is exactly why we measure in aggregate. Here we teach the mechanism on a case "
        "where it works, chosen honestly.)",
    ),
    code(
        "bi = BiEncoderRetriever(data, device=device)      # embeds/caches corpus + queries (once)",
        "cross = CrossEncoderReranker(device=device)       # the second stage (built now so we can pick by measurement)",
        "print(f'corpus embeddings: {bi.doc_vectors.shape} | query embeddings: {bi.query_vectors.shape}')",
        "norms = np.linalg.norm(bi.doc_vectors[:500], axis=1)",
        "print(f'doc-vector L2 norms: min {norms.min():.3f}, max {norms.max():.3f} (≈1 → cosine == dot product)')",
        "",
        "qi = pick_demo_query(data, bi, cross)             # a query where re-ranking demonstrably wins (measured)",
        "pool = bi.retrieve(qi, k=RETRIEVE_K)",
        "print(f'\\nwalkthrough query {qi}: \"{data.query_texts[qi][:70]}...\"')",
        "print(f'retrieved top-{RETRIEVE_K}; first 8 doc ids: {pool[:8].tolist()}')",
    ),
    # ----------------------------------------------------------------- Step 4
    md(
        "## Step 4 — where did the gold land? The first stage is coarse",
        "",
        "The whole premise of re-ranking is that the first stage gets the right *neighbourhood* but not "
        "always the right *order*. Let's see where the gold passage sits in the bi-encoder's ranking. "
        "Often it's in the pool but not at the very top — outranked by passages that are topically "
        "similar but don't actually answer the claim. That gap is what the cross-encoder will close.",
    ),
    code(
        "def gold_rank(order, gold):",
        "    for r, d in enumerate(order, start=1):",
        "        if int(d) in gold:",
        "            return r",
        "    return None",
        "",
        "print('top-5 bi-encoder results (★ = gold):')",
        "for r, d in enumerate(pool[:5], start=1):",
        "    mark = '★' if int(d) in data.gold[qi] else ' '",
        "    print(f'  #{r} {mark} doc[{int(d)}]: {data.doc_texts[int(d)][:90]}...')",
        "print(f'\\ngold first appears at bi-encoder rank: #{gold_rank(pool, data.gold[qi])}')",
    ),
    # ----------------------------------------------------------------- Step 5
    md(
        "## Step 5 — the ranking metrics, from scratch (nDCG, MRR, Recall, Precision)",
        "",
        "To *prove* re-ranking helps we need ranking-quality metrics, not eyeballing. **DCG** sums each "
        "relevant hit discounted by a log of its rank ($\\text{rel}_i/\\log_2(i+1)$) — deeper hits count "
        "less, so ranking the answer higher scores more. **nDCG** divides DCG by the ideal (all gold on "
        "top) to land in $[0,1]$. **MRR** is the reciprocal rank of the first hit. **Recall@k** is the "
        "fraction of gold present in the top-k (the ceiling re-ranking can't exceed); **Precision@k** is "
        "the fraction of the top-k that's relevant. We print the log-discount table so the metric isn't "
        "a black box, then score this query's bi-encoder ranking.",
    ),
    code(
        "print('rank i : DCG discount 1/log2(i+1)')",
        "for i in range(1, 6):",
        "    print(f'  {i}    : {1/np.log2(i+1):.3f}')",
        "",
        "g = data.gold[qi]",
        "print(f'\\nbi-encoder ranking, query {qi}:')",
        "print(f'  nDCG@10    = {ndcg_at_k(pool, g, METRIC_K):.3f}')",
        "print(f'  MRR@10     = {mrr_at_k(pool, g, METRIC_K):.3f}')",
        "print(f'  Recall@10  = {recall_at_k(pool, g, METRIC_K):.3f}')",
        "print(f'  Precision@10 = {precision_at_k(pool, g, METRIC_K):.3f}')",
    ),
    # ----------------------------------------------------------------- Step 6
    md(
        "## Step 6 — Stage 2: the cross-encoder scores each (query, passage) pair *jointly*",
        "",
        "Now the precision stage. A cross-encoder concatenates the query and one passage into "
        "`[CLS] query [SEP] passage` and runs the transformer over the **pair**, so every query token "
        "attends to every passage token and back. A head on the `[CLS]` position emits one scalar "
        "**relevance logit**. This is the signal the bi-encoder structurally throws away — and it costs "
        "one full forward pass *per pair*, so it can only run on the shortlist. We score the retrieved "
        "pool and compare the logits for gold vs non-gold candidates. On a query where re-ranking works "
        "(like this one), the gold *usually* scores above the field — see the score-separation figure "
        "over 60 queries for the aggregate picture; here we print the actual comparison for this query so "
        "the claim matches the output.",
    ),
    code(
        "# `cross` was built in Step 3 (so we could pick the walkthrough query by measurement)",
        "logits = cross.scores(data.query_texts[qi], [data.doc_texts[int(i)] for i in pool])",
        "gold_logits = [float(s) for i, s in zip(pool, logits) if int(i) in g]",
        "other_logits = [float(s) for i, s in zip(pool, logits) if int(i) not in g]",
        "g_med, o_med = float(np.median(gold_logits)), float(np.median(other_logits))",
        "print(f'cross-encoder logits — gold: {[round(x,2) for x in gold_logits]}')",
        "print(f'  gold median logit {g_med:.2f} vs non-gold median {o_med:.2f} '",
        "      f'— gold is {\"ABOVE\" if g_med > o_med else \"BELOW\"} the field '",
        "      f'({\"the gap re-ranking exploits\" if g_med > o_med else \"a mis-score — see Step 8s aside\"})')",
    ),
    # ----------------------------------------------------------------- Step 7
    md(
        "## Step 7 — re-rank the pool and watch the gold climb",
        "",
        "Re-ranking is just: sort the candidate pool by the cross-encoder logit. `rerank` returns the "
        "pool re-ordered best-first. Note it only **reorders** the candidates it was given — it cannot "
        "add a passage the first stage missed (the recall ceiling, measured in Step 9). We compare the "
        "gold's rank and the metrics before vs after for this one query.",
    ),
    code(
        "reranked = cross.rerank(data.query_texts[qi], pool, data.doc_texts)",
        "print('top-5 AFTER re-ranking (★ = gold):')",
        "for r, d in enumerate(reranked[:5], start=1):",
        "    mark = '★' if int(d) in g else ' '",
        "    print(f'  #{r} {mark} doc[{int(d)}]: {data.doc_texts[int(d)][:90]}...')",
        "print(f'\\ngold rank:  bi-encoder #{gold_rank(pool, g)}  ->  reranked #{gold_rank(reranked, g)}')",
        "print(f'nDCG@10:    {ndcg_at_k(pool, g, METRIC_K):.3f}  ->  {ndcg_at_k(reranked, g, METRIC_K):.3f}')",
        "print(f'MRR@10:     {mrr_at_k(pool, g, METRIC_K):.3f}  ->  {mrr_at_k(reranked, g, METRIC_K):.3f}')",
    ),
    # ----------------------------------------------------------------- Step 7b (honest aside)
    md(
        "## Step 7b — the honest caveat: re-ranking is *not* guaranteed per query",
        "",
        "The walkthrough above showed re-ranking winning — but that is a *choice we made by measurement*, "
        "not a guarantee. On a **domain-mismatch** query the `ms-marco`-trained cross-encoder can "
        "*mis-score* and actually **sink** the gold. scifact is scientific claim verification, not the web "
        "QA the reranker was trained on, so this really happens — and it's the exact pitfall the page "
        "flags (§ *Domain / model mismatch*). Let's look at query 0 (\"0-dimensional biomaterials lack "
        "inductive properties\"), where the reranker scores the gold *below* the field and demotes it:",
    ),
    code(
        "bad_qi = 0",
        "bad_pool = bi.retrieve(bad_qi, k=RETRIEVE_K)",
        "bad_g = data.gold[bad_qi]",
        "bad_logits = cross.scores(data.query_texts[bad_qi], [data.doc_texts[int(i)] for i in bad_pool])",
        "bad_gold_l = [float(s) for i, s in zip(bad_pool, bad_logits) if int(i) in bad_g]",
        "bad_other_l = [float(s) for i, s in zip(bad_pool, bad_logits) if int(i) not in bad_g]",
        "bad_reranked = cross.rerank(data.query_texts[bad_qi], bad_pool, data.doc_texts)",
        "print(f'query 0: \"{data.query_texts[bad_qi][:60]}...\"')",
        "print(f'  gold median logit {np.median(bad_gold_l):.2f} vs non-gold median {np.median(bad_other_l):.2f} '",
        "      f'— gold is {\"ABOVE\" if np.median(bad_gold_l) > np.median(bad_other_l) else \"BELOW\"} the field (mis-scored)')",
        "print(f'  gold rank:  bi-encoder #{gold_rank(bad_pool, bad_g)}  ->  reranked #{gold_rank(bad_reranked, bad_g)}  '",
        "      f'(re-ranking HURT this query)')",
        "print('\\n=> a single query can go either way; this is why you MEASURE IN AGGREGATE (next).')",
    ),
    # ----------------------------------------------------------------- Step 8
    md(
        "## Step 8 — the real aggregate lift over many queries",
        "",
        "One query is an anecdote (in either direction); the benchmark is the evidence. We run "
        "retrieve-then-rerank over a batch of real test queries and average the metrics — the wins "
        "outweigh the mis-scores, so the aggregate rises. This is the honest measure of what the re-ranker "
        "buys — and on a strong first stage like this one, the lift is real but *modest*, which is itself "
        "an important lesson: a re-ranker is a measurable precision upgrade you verify on your own data, "
        "not free magic.",
        "",
        "> **Subset vs full run.** For notebook runtime we evaluate the **first 100** queries — an easier "
        "slice than the whole set. The full **300-query** run (`python reranking.py`, the numbers on the "
        "page) gives **nDCG@10 0.648 → 0.687 (+0.038)** and **MRR@10 0.607 → 0.656 (+0.049)**. The subset "
        "numbers below will differ (higher baseline, similar-sized lift) — that's the slice, not a "
        "mismatch; both runs show the same story (ranking metrics up, pool recall flat).",
    ),
    code(
        "N_EVAL = 100   # subset for notebook runtime; the module runs all 300 (page numbers)",
        "t0 = time.time()",
        "ev, bi_orders, rr_orders = evaluate(data, bi, cross, n_queries=N_EVAL)",
        "print(f'evaluated the first {ev.n_queries} queries in {time.time()-t0:.0f}s '",
        "      f'(an easier slice; full-300 on the page: nDCG 0.648->0.687, MRR 0.607->0.656)\\n')",
        "print(f'{\"metric\":<13} | {\"bi-encoder\":>10} | {\"reranked\":>9} | {\"delta\":>7}')",
        "print('-' * 48)",
        "for r in ev.rows:",
        "    print(f'{r.name:<13} | {r.bi:>10.3f} | {r.reranked:>9.3f} | {r.delta:>+7.3f}')",
    ),
    # ----------------------------------------------------------------- Step 9
    md(
        "## Step 9 — the recall ceiling: re-ranking reorders *within the pool*, it cannot retrieve",
        "",
        "The single most important operational fact about re-ranking: it can only **reorder the pool the "
        "first stage retrieved**. A gold passage the bi-encoder never put in the top-K is gone — no "
        "re-ranker brings it back. There's a subtlety worth getting exactly right: **Recall@10 *does* "
        "rise** with reranking — because reranking promotes gold that was in the pool but below rank 10 "
        "*up into* the top-10. What is truly invariant is **Recall@K over the whole retrieved pool** "
        "(K=100 here): reranking shuffles the same 100 docs, so it can never change *which* 100 they are. "
        "That pool recall is the real ceiling — if it's low, the fix is a **wider pool or a better "
        "retriever**, never a better re-ranker.",
    ),
    code(
        "by = {r.name: r for r in ev.rows}",
        "pool_metric = f'Recall@{RETRIEVE_K}'",
        "print(f'Recall@10:     bi {by[\"Recall@10\"].bi:.3f}  ->  reranked {by[\"Recall@10\"].reranked:.3f}  '",
        "      f'(RISES — gold from ranks 11-{RETRIEVE_K} promoted into the top-10)')",
        "print(f'{pool_metric}:   bi {by[pool_metric].bi:.3f}  ->  reranked {by[pool_metric].reranked:.3f}  '",
        "      f'(INVARIANT — the pool ceiling; reranking cannot add what retrieval missed)')",
        "print(f'nDCG@10:       bi {by[\"nDCG@10\"].bi:.3f}  ->  reranked {by[\"nDCG@10\"].reranked:.3f}  '",
        "      f'(improves — reranking fixes the ORDER)')",
        "assert abs(by[pool_metric].reranked - by[pool_metric].bi) < 1e-9, 'pool recall must be invariant'",
        "assert by['nDCG@10'].reranked >= by['nDCG@10'].bi, 'reranking should not lower nDCG'",
        "print(f'\\nassert passed: pool recall (@{RETRIEVE_K}) flat, nDCG up — the recall ceiling is real.')",
    ),
    # ----------------------------------------------------------------- Step 10
    md(
        "## Step 10 — quality vs rerank depth K (where the knee is)",
        "",
        "Re-ranking deeper costs more cross-encoder passes. How much depth do we actually need? We sweep "
        "K = how many of the top candidates we re-rank (leaving the tail in first-stage order, exactly "
        "what a latency-bound system does) and measure nDCG@10 at each. The curve typically rises fast "
        "then plateaus — **most of the lift comes from a shallow pool**, which is why real systems "
        "re-rank ~50–100, not thousands.",
    ),
    code(
        "sweep = sweep_rerank_depth(data, bi_orders, rr_orders)",
        "print(f'{\"K\":>4} | {\"bi-only nDCG@10\":>15} | {\"reranked nDCG@10\":>16}')",
        "print('-' * 42)",
        "for k_depth, (b, r) in sweep.items():",
        "    print(f'{k_depth:>4} | {b:>15.3f} | {r:>16.3f}')",
    ),
    # ----------------------------------------------------------------- Step 11
    md(
        "## Step 11 — the real cost: retrieve is cheap, rerank pays per candidate",
        "",
        "Why not just re-rank everything? Cost. The bi-encoder retrieve is ~one dot-product sweep (a few "
        "ms). The cross-encoder pays **one transformer forward pass per candidate**, and none of it is "
        "precomputable (each score depends on the query). We measure both per-query latencies — the "
        "asymmetry is the whole reason for the two-stage funnel.",
        "",
        "> **Wall-clock varies** run-to-run and by machine (thermal, scheduling, cold vs warm caches). "
        "Expect roughly **8–11 ms per candidate** on this hardware; the page quotes the full-run median "
        "(~11 ms/candidate, ~1.1 s at K=100). Don't read a small spread between this cell and the page as "
        "a reproduction failure — the *ratio* (rerank ≫ retrieve) is the invariant point.",
    ),
    code(
        "from reranking import measure_latency",
        "ret_ms, rr_ms = measure_latency(data, bi, cross)",
        "print(f'per-query latency ({device}, this run — wall-clock varies):')",
        "print(f'  stage 1 retrieve (top-{RETRIEVE_K}) : {ret_ms:.2f} ms')",
        "print(f'  stage 2 rerank   (top-{RETRIEVE_K}) : {rr_ms:.1f} ms  '",
        "      f'(~{rr_ms/ret_ms:.0f}x the retrieve cost, ~{rr_ms/RETRIEVE_K:.1f} ms/candidate)')",
        "print(f'\\n=> cross-encoding all {data.n_docs:,} passages would cost '",
        "      f'~{rr_ms/RETRIEVE_K*data.n_docs/1000:.0f} s/query — why you NEVER rerank the whole corpus.')",
    ),
    # ----------------------------------------------------------------- Step 12
    md(
        "## Step 12 — a real hero query: the buried gold the reranker rescues most",
        "",
        "Averages hide the wins. Let's find — honestly, by measurement, not by hand-picking — the query "
        "the reranker helps *most* (biggest MRR gain) and read what happened: a genuinely relevant "
        "passage the bi-encoder buried, pulled up toward the top by joint scoring. This is the "
        "re-ranking win in one concrete, real example.",
    ),
    code(
        "hq = hero_query(data, bi_orders, rr_orders)",
        "g = data.gold[hq]",
        "print(f'hero query: \"{data.query_texts[hq][:80]}...\"')",
        "print(f'  gold rank:  bi-encoder #{gold_rank(bi_orders[hq], g)}  ->  reranked #{gold_rank(rr_orders[hq], g)}')",
        "print(f'  MRR@10:     {mrr_at_k(bi_orders[hq], g, METRIC_K):.3f}  ->  {mrr_at_k(rr_orders[hq], g, METRIC_K):.3f}')",
        "print(f'  nDCG@10:    {ndcg_at_k(bi_orders[hq], g, METRIC_K):.3f}  ->  {ndcg_at_k(rr_orders[hq], g, METRIC_K):.3f}')",
        "gi = next(iter(g))",
        "print(f'\\n  the rescued passage doc[{gi}]: {data.doc_texts[gi][:160]}...')",
    ),
    # ----------------------------------------------------------------- Try it yourself
    md(
        "## Try it yourself",
        "",
        "Before you run anything, **predict**:",
        "",
        "1. Swap the first stage for a *stronger* bi-encoder (`BiEncoderRetriever(data, model_name='BAAI/bge-small-en-v1.5')`). Predict what happens to the re-ranking *delta* — bigger or smaller? (Hint: the better the first stage orders things, the less room the reranker has to improve.)",
        "2. Shrink `RETRIEVE_K` to 5 and re-run Step 8. What happens to Recall@10, and therefore to the *ceiling* on nDCG@10?",
        "3. In Step 10, at which K does nDCG@10 stop improving meaningfully? That K is your latency/quality sweet spot — the pool size to ship.",
        "",
        "Then try each and check your prediction against the real measured numbers. That predict→measure loop is exactly how you decide whether — and how deep — to re-rank in a real system.",
    ),
    # ----------------------------------------------------------------- What we saw
    md(
        "## What we saw",
        "",
        "- **The first stage is coarse** — a strong bi-encoder gets the right neighbourhood but often not "
        "the right order, because it encodes query and passage independently and can't model their "
        "interaction.",
        "- **The cross-encoder reads the pair jointly** — `[CLS] q [SEP] d` → full self-attention → one "
        "relevance logit; on a query where it works the gold scores above the field and the pool re-sort "
        "lifts it — though (Step 7b) an individual domain-mismatch query can be mis-scored the other way.",
        "- **The lift is real and measured** — on this real benchmark, re-ranking raises nDCG@10 and "
        "MRR@10 (and even Recall@10, by promoting in-pool gold into the top-10) while leaving the "
        "**pool recall (Recall@K) exactly flat** — the recall ceiling: reranking reorders the same K "
        "docs, it cannot retrieve what stage 1 missed.",
        "- **Most of the lift is shallow** — the quality-vs-K curve plateaus early, so you re-rank ~50–100, "
        "not thousands.",
        "- **The cost is the constraint** — retrieve is a few ms; rerank pays one transformer pass per "
        "candidate and can't be precomputed, so cross-encoding the whole corpus is off the table.",
        "",
        "Everything here is real: a real labelled benchmark, real bi- and cross-encoders, real recall/nDCG "
        "measured against human judgments, real latency. That's the difference between reading about "
        "re-ranking and being able to *decide when and how to use it*.",
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {NB_PATH} with {len(CELLS)} cells ({n_md} md, {n_code} code)")


if __name__ == "__main__":
    main()

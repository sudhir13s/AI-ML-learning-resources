---
id: "11-rag-and-llm-apps/graphrag/references"
topic: "GraphRAG — References"
parent: "11-rag-and-llm-apps/graphrag"
type: references
updated: 2026-07-02
---

# GraphRAG — references and further reading

> Companion link library for **[GraphRAG](09-GraphRAG.md)** (the concept page). External sources *and* internal cross-links, kept separate so it can be reused as a standalone list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this* topic — knowledge-graph construction, multi-hop traversal (local search), community detection (modularity / Louvain / Leiden), and global query-focused summarization.

**Start here — suggested path**:
1. **Get the core idea** — read [the GraphRAG paper](https://arxiv.org/abs/2404.16130) (**Edge et al. 2024**), abstract + §1–2. *Why flat RAG fails on global questions, and the local→global (entity graph → community summaries → map-reduce) design.*
2. **See the two search modes** — read [Microsoft GraphRAG — Local Search](https://microsoft.github.io/graphrag/query/local_search/) and [Global Search](https://microsoft.github.io/graphrag/query/global_search/). *Entity-linked traversal vs map-reduce over community reports — exactly the two paths the page builds.*
3. **Understand community detection** — read [Newman, "Modularity and community structure" (PNAS 2006)](https://doi.org/10.1073/pnas.0601602103). *The modularity Q the page derives — edges within communities vs expected at random.*
4. **See the fast algorithms** — skim [Louvain (Blondel et al. 2008)](https://doi.org/10.1088/1742-5468/2008/10/P10008) and [Leiden (Traag et al. 2019)](https://arxiv.org/abs/1810.08473). *How Q is maximized in practice, and why Leiden fixed Louvain's connectivity defect.*
5. **Build one** — read [LlamaIndex — Property Graph Index](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/) or run [Microsoft GraphRAG get-started](https://microsoft.github.io/graphrag/get_started/). *The extract-triples → graph → query pipeline, runnable.*

**Videos**:
- [GraphRAG: The Marriage of Knowledge Graphs and RAG](https://www.youtube.com/watch?v=knDDGYHnnSI) — **Emil Eifrem (Neo4j)** — why combine knowledge graphs with RAG: local vs global questions, entity graphs, and community-level reasoning.
- [GraphRAG: LLM-Derived Knowledge Graphs for RAG](https://www.youtube.com/watch?v=r09tJfON6kE) — **Prompt Engineering** — a clear walkthrough of the entity-graph → community-summary → map-reduce pipeline.
- [Road to NODES: Mastering RAG with the GraphRAG Python Package](https://www.youtube.com/watch?v=OALrsghrP_I) — **Neo4j** — a hands-on workshop building a GraphRAG system: LLM entity extraction, knowledge-graph construction, and relationship-aware retrieval.
- [NetSci 06-2: Modularity and the Louvain Method](https://www.youtube.com/watch?v=QfTxqAxJp0U) — **Andrew Beveridge** — the math behind modularity and how the Louvain algorithm maximizes it; the community-detection engine under GraphRAG.
- [LlamaIndex Webinar: Advanced RAG with Knowledge Graphs (with Tomaz from Neo4j)](https://www.youtube.com/watch?v=LDh5MdR-CPQ) — **LlamaIndex** — the `PropertyGraphIndex` abstractions end to end, from extraction to graph retrieval.

**Interactive & visual**:
- [Microsoft GraphRAG — visualization & get-started](https://microsoft.github.io/graphrag/get_started/) — **Microsoft** — run the indexer on a corpus and explore the resulting graph and community reports.
- [Neo4j Graph Data Science — community detection](https://neo4j.com/docs/graph-data-science/current/algorithms/community/) — **Neo4j** — Louvain/Leiden/modularity as graph-database operations, with playgrounds.

**Courses (free)**:
- [DeepLearning.AI — Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) — **DeepLearning.AI + Neo4j (free short course)** — building and querying a knowledge graph for retrieval, hands-on.
- [Stanford CS224W — Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/) — **Stanford** — community detection, modularity, and graph algorithms from first principles (lectures + slides free).

**Articles / blogs (free, no paywall)**:
- [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) — **Microsoft Research** — the announcement blog: the motivation (global sensemaking) and the local→global design, with examples.
- [GraphRAG Costs Explained: What You Need to Know](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978) — **Microsoft (Azure AI Foundry)** — the indexing-cost breakdown the page's cost anchor cites: LLM extraction calls per chunk drive the majority of GraphRAG's cost (vs ~$0.006 to embed a whole book for flat RAG).
- [Microsoft GraphRAG — Local Search](https://microsoft.github.io/graphrag/query/local_search/) & [Global Search](https://microsoft.github.io/graphrag/query/global_search/) — **Microsoft docs** — the exact mechanics of the two query modes the page cites (entity access points; map-reduce where each point carries a numerical importance rating).
- [Constructing knowledge graphs from text with LLMs](https://neo4j.com/blog/developer/construct-knowledge-graphs-unstructured-text/) — **Neo4j** — LLM entity/relation extraction and entity resolution, the indexing steps this page's pitfalls cover.
- [Using a Property Graph Index](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/) — **LlamaIndex docs** — the `PropertyGraphIndex` API (kg_extractors, synonym + vector-context retrievers) the page cites.
- [NetworkX — community detection](https://networkx.org/documentation/stable/reference/algorithms/community.html) — **NetworkX docs** — `greedy_modularity_communities`, `louvain_communities`, and `modularity` — the exact functions the from-scratch code runs.

**Key papers**:
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) — **Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Metropolitansky, Ness & Larson (2024)** — the primary GraphRAG source: entity graph + community summaries + local/global search; substantial gains in comprehensiveness and diversity on global sensemaking over ~1 M-token corpora.
- [Modularity and community structure in networks](https://doi.org/10.1073/pnas.0601602103) — **Newman (PNAS 2006)** — the modularity $Q$ this page derives (edges within groups minus the expected number at random) and the spectral/modularity-matrix view.
- [Fast unfolding of communities in large networks (Louvain)](https://doi.org/10.1088/1742-5468/2008/10/P10008) — **Blondel, Guillaume, Lambiotte & Lefebvre (2008)** — the greedy modularity-optimization heuristic, fastest of its era, scaling to 100 M-node graphs.
- [From Louvain to Leiden: guaranteeing well-connected communities](https://arxiv.org/abs/1810.08473) — **Traag, Waltman & van Eck (2019)** — fixes Louvain's badly-/dis-connected-community defect; guaranteed-connected communities, faster; GraphRAG's default detector.
- [Corrective / Self-reflective RAG context](https://arxiv.org/abs/2404.16130) — see the GraphRAG paper's related-work discussion of query-focused summarization (QFS) vs retrieval, the framing that motivates global search.

**Books (free chapters)**:
- [Networks, Crowds, and Markets (Easley & Kleinberg) — Ch. 3 "Strong and Weak Ties" / community structure](https://www.cs.cornell.edu/home/kleinber/networks-book/) — **Easley & Kleinberg (Cambridge, free online)** — graph community structure and clustering, the network-science foundation under modularity.

**In this platform**:
- Concept page (full explanation): [GraphRAG](09-GraphRAG.md)
- Foundations this builds on: [RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [Embedding Models for Retrieval (the flat-RAG baseline)](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [Hybrid Search + RRF](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [Advanced RAG (Parent-Doc · Self-RAG)](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Composes with: [Agentic RAG (multi-hop reasoning as an agent loop)](../10-Agentic-RAG/10-Agentic-RAG.md)
- Measure it: [RAG Evaluation (multi-hop and global-answer quality)](../11-RAG-Evaluation/11-RAG-Evaluation.md)

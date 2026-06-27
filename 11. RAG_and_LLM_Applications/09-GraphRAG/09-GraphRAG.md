---
id: "11-rag-and-llm-apps/graphrag"
topic: "GraphRAG"
parent: "11-rag-and-llm-apps"
level: advanced
prereqs: ["advanced-rag", "rag-fundamentals", "embedding-models-for-retrieval"]
interview_frequency: medium
updated: 2026-06-20
---

# GraphRAG
> Plain RAG retrieves isolated chunks and struggles with questions that span many documents
> ("summarize the main themes," "how are X and Y connected?"). **GraphRAG** first uses an LLM to
> extract entities and relationships into a **knowledge graph**, detects communities, and summarizes
> them — then retrieves over graph structure (local entity neighborhoods + global community summaries)
> instead of flat vectors. Better multi-hop reasoning and whole-corpus ("global") questions.

**Why it matters:** the "when does vector RAG fail and what replaces it?" question. You'll explain the
index-build pipeline (entity/relation extraction → graph → community detection → community summaries),
local vs global search, why it shines on connected/multi-hop queries, and its costs (LLM-heavy
indexing, graph maintenance) versus standard RAG.

**⭐ Start here — suggested path:**

1. **Get the intuition** — watch [GraphRAG Explained: AI Retrieval with Knowledge Graphs](https://www.youtube.com/watch?v=Za7aG-ooGLQ). *Why graph structure beats flat chunks for connected questions.*
2. **Read the motivation** — read [Neo4j: What is GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/). *Grounding LLMs in entities + relationships to cut hallucination.*
3. **See the full system** — watch [GraphRAG: Building a Smarter AI System](https://www.youtube.com/watch?v=JTVx6i6MzVw). *End-to-end build over Microsoft's GraphRAG: extract → graph → communities → query.*
4. **Read the source** — skim [From Local to Global (Microsoft GraphRAG)](https://arxiv.org/abs/2404.16130). *Community summaries + map-reduce for global "sensemaking" questions.*
5. **Build it with a graph DB** — watch [Practical GraphRAG with Knowledge Graphs](https://www.youtube.com/watch?v=XNneh6-eyPg), and explore the [Microsoft GraphRAG docs](https://microsoft.github.io/graphrag/). *Hands-on patterns for indexing and querying a real knowledge graph.*

## 🎓 Courses (free)
- [Knowledge Graphs & GraphRAG (GraphAcademy)](https://graphacademy.neo4j.com/knowledge-graph-rag/) — **Neo4j** — free, hands-on course on building knowledge graphs and graph-grounded RAG.
- [Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) — **DeepLearning.AI × Neo4j** — the canonical free short course on querying graphs for retrieval.

## 🎥 Videos
- [GraphRAG Explained: AI Retrieval with Knowledge Graphs & Cypher](https://www.youtube.com/watch?v=Za7aG-ooGLQ) — **IBM Technology** — concise conceptual intro to GraphRAG vs vector RAG.
- [GraphRAG: Building a Smarter AI System (full walkthrough)](https://www.youtube.com/watch?v=JTVx6i6MzVw) — **Thu Vu** — end-to-end build on Microsoft GraphRAG with clear visuals.
- [Practical GraphRAG: Making LLMs smarter with Knowledge Graphs](https://www.youtube.com/watch?v=XNneh6-eyPg) — **Neo4j (AI Engineer)** — local vs global retrieval and production patterns.
- [Advanced RAG with Knowledge Graphs (with Tomaz, Neo4j)](https://www.youtube.com/watch?v=LDh5MdR-CPQ) — **LlamaIndex** — combining property graphs with RAG using LlamaIndex + Neo4j.

## 📄 Key Papers
- [From Local to Global: A GraphRAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) — **Edge et al. (2024)** — the Microsoft GraphRAG method (communities + summaries for global queries).
- [Retrieval-Augmented Generation with Graphs (GraphRAG survey)](https://arxiv.org/abs/2501.00309) — **Han et al. (2025)** — broad survey of graph-based RAG approaches and design choices.
- [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302) — **Pan et al. (2023)** — the conceptual landscape of LLM + KG integration.

## 📰 Articles / Blogs (free, no paywall)
- [What is GraphRAG?](https://neo4j.com/blog/genai/what-is-graphrag/) — **Neo4j** — clear motivation, architecture, and when graphs beat vectors.
- [GraphRAG documentation](https://microsoft.github.io/graphrag/) — **Microsoft** — the reference for the indexing pipeline, local/global search, and config.
- [microsoft/graphrag (GitHub)](https://github.com/microsoft/graphrag) — **Microsoft** — the open-source implementation to read and run.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 15 "Relation and Event Extraction"**](https://web.stanford.edu/~jurafsky/slp3/15.pdf) — **Jurafsky & Martin** — the entity/relation-extraction foundations behind building a knowledge graph, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [1.04 Graph Representations](../../../AI-ML-intuition/Module_1_Representation/1.04_Graph_Representations.md)
- Prereqs: [08 Advanced RAG](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md) · Next: [10 Agentic RAG](../10-Agentic-RAG/10-Agentic-RAG.md)
- Related domain: [06. NLP — Sequence Labeling (POS & NER)](../../06.%20NLP/09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md) (entity extraction feeds the graph)

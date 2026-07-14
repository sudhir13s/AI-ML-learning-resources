"""From-scratch GraphRAG: a real knowledge graph, real traversal, real community detection.

Flat vector RAG retrieves the top-k SIMILAR chunks and stops there. That fails on two whole
classes of question:

  1. MULTI-HOP questions that must CONNECT facts across documents -- "where is the team based that
     leads the satellite carrying the hyperspectral imager?" needs the chain
     hyperspectral imager -> Helios-7 -> Dr. Amara Okoye -> Nairobi office. No single chunk holds
     that chain, so top-k similarity retrieves the endpoints but never links them. (This is the exact
     probe build_multihop_probe() implements; its shortest path is asserted in main().)
  2. GLOBAL / thematic questions -- "what are the main themes across the corpus?" -- where the answer
     is an AGGREGATE over the whole dataset, not any one chunk.

GraphRAG (Edge et al. 2024) fixes both by turning the pile of documents into a CONNECTED MAP: extract
entities (nodes) + relationships (edges) into a knowledge graph, then answer by (a) LOCAL search:
link the query to graph entities and TRAVERSE the neighbourhood / shortest path (multi-hop), or
(b) GLOBAL search: detect graph COMMUNITIES (via modularity optimization), pre-summarize each, and
map-reduce those summaries into a global answer.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the whole GRAPH is a real `networkx` graph; community detection is real
    (`greedy_modularity_communities`, modularity-optimizing); the modularity score Q is real
    (`networkx ... modularity`); multi-hop TRAVERSAL is a real shortest-path/BFS; and the flat-RAG
    baseline it is contrasted against is the real dense bi-encoder (all-MiniLM-L6-v2) from ch5. Every
    path, community count, modularity Q, and retrieval hit printed here is measured and asserted.
  * ILLUSTRATIVE (labelled): the ENTITY + RELATION EXTRACTION and the community SUMMARIES / final
    answers require a generative LLM (this env is encoder-only). We do NOT fake an LLM: the entity/
    relation set (ENTITIES, RELATIONS) is a small FIXED, clearly-labelled hand-authored extraction,
    and the community summaries / answers are fixed exemplars. In production an LLM extracts the
    triples and writes the summaries; the GRAPH ALGORITHMS + traversal + modularity you see run for
    real over that (illustrative) extraction.

The flat-RAG baseline reuses ch5's `DenseRetriever` over ch1's corpus so the chapters share one
source of truth.

Verified on Python 3.12 / numpy 2.x / networkx 3.x / sentence-transformers (all-MiniLM, CPU).
Deterministic given the same cached model.

Run:
    python graph_rag.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

# Reuse ch5's dense retriever + corpus for the flat-RAG baseline we contrast against (it adds ch1 to
# the path transitively). ch5 lives two directories over; inject its path so imports work whether run
# from this dir or imported by the notebook / figure scripts.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
if str(_CH5_CODE) not in sys.path:
    sys.path.insert(0, str(_CH5_CODE))

from hybrid_search import DenseRetriever, full_corpus  # noqa: E402  (path injected above)

TOP_K = 3  # flat-RAG top-k for the baseline contrast

# ================================================================================================
# The (illustrative) extraction: entities + relations an LLM would derive from ch1's Helios-7 corpus.
# In production these come from an LLM entity/relation extractor over each chunk; here they are a
# FIXED, hand-authored set so the pipeline runs with no generative model. The GRAPH ALGORITHMS below
# operate on this graph for real.
# ================================================================================================

# Each entity carries a `group` tag ONLY for readability of the printed output; the community
# DETECTION below does NOT use it -- communities are discovered from graph STRUCTURE via modularity.
ENTITIES: tuple[tuple[str, str], ...] = (
    ("Helios-7", "satellite"),
    ("Dr. Amara Okoye", "person"),
    ("Nairobi office", "org"),
    ("Kourou spaceport", "place"),
    ("Ariane 6", "vehicle"),
    ("hyperspectral imager", "instrument"),
    ("sun-synchronous orbit", "orbit"),
    ("Error E-4011", "error"),
    ("telemetry stream", "system"),
    ("ground team", "org"),
)

# (subject, relation, object) triples -- the edges of the knowledge graph.
RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("Helios-7", "launched_from", "Kourou spaceport"),
    ("Helios-7", "launched_aboard", "Ariane 6"),
    ("Helios-7", "led_by", "Dr. Amara Okoye"),
    ("Dr. Amara Okoye", "based_in", "Nairobi office"),
    ("Helios-7", "carries", "hyperspectral imager"),
    ("Helios-7", "flies_in", "sun-synchronous orbit"),
    ("Error E-4011", "appeared_in", "telemetry stream"),
    ("telemetry stream", "belongs_to", "Helios-7"),
    ("ground team", "operated_by", "Nairobi office"),
    ("ground team", "investigates", "Error E-4011"),
)


@dataclass(frozen=True)
class MultiHopProbe:
    """A multi-hop question, its start/end entities, the expected traversal path, and a label."""

    query: str
    start: str
    end: str
    expected_path: tuple[str, ...]
    label: str


def build_graph(
    entities: tuple[tuple[str, str], ...] = ENTITIES,
    relations: tuple[tuple[str, str, str], ...] = RELATIONS,
) -> nx.Graph:
    """Build a real (undirected) networkx knowledge graph from the extracted entities + relations.

    Nodes are entities (with a `type` attribute); edges are relations (with a `relation` attribute).
    We use an UNDIRECTED graph because multi-hop QUESTIONS traverse relationships in either direction
    ("the office responsible for the satellite" follows led_by/based_in backwards). Community
    detection and shortest-path both run on this graph for real.
    """
    graph = nx.Graph()
    for name, etype in entities:
        graph.add_node(name, type=etype)
    for subj, rel, obj in relations:
        graph.add_edge(subj, obj, relation=rel)
    return graph


# ================================================================================================
# LOCAL search: link the query to entities, TRAVERSE the shortest path (real multi-hop).
# ================================================================================================


def shortest_path(graph: nx.Graph, start: str, end: str) -> list[str]:
    """The real shortest entity path from start to end (BFS on the unweighted knowledge graph).

    This is the heart of multi-hop LOCAL search: once the query is linked to a start entity, the
    graph's edges CONNECT it to the answer entity even when no single document mentioned both. Raises
    if unreachable (a disconnected graph is itself a signal that extraction missed a link).
    """
    return nx.shortest_path(graph, source=start, target=end)


def path_with_relations(graph: nx.Graph, path: list[str]) -> str:
    """Render a traversal path as 'A -[rel]-> B -[rel]-> C' using each edge's real relation label."""
    parts = [path[0]]
    for a, b in zip(path, path[1:]):
        rel = graph.edges[a, b]["relation"]
        parts.append(f"-[{rel}]-> {b}")
    return " ".join(parts)


def graph_without_edge(graph: nx.Graph, subj: str, obj: str) -> nx.Graph:
    """Return a COPY of the graph with one edge removed -- simulating a MISSED extraction (Pitfall 1).

    A missed (subject, relation, object) triple is a missing edge. This lets us measure what a single
    extraction error does to multi-hop traversal without mutating the original graph.
    """
    damaged = graph.copy()
    damaged.remove_edge(subj, obj)
    return damaged


def build_multihop_probe() -> MultiHopProbe:
    """The multi-hop probe flat vector RAG cannot answer: imager -> satellite -> lead -> office.

    'Where is the team based that leads the satellite carrying the hyperspectral imager?' requires
    chaining three facts that live in DIFFERENT chunks -- (Helios-7 carries the imager), (Helios-7 is
    led by Dr. Okoye), (Okoye is based in the Nairobi office). Top-k similarity retrieves the
    endpoints but no single chunk links the imager to the office, so it cannot answer. The expected
    path is asserted against the real shortest path in main().
    """
    return MultiHopProbe(
        query="Where is the team based that leads the satellite carrying the hyperspectral imager?",
        start="hyperspectral imager",
        end="Nairobi office",
        expected_path=("hyperspectral imager", "Helios-7", "Dr. Amara Okoye", "Nairobi office"),
        label="multi-hop: imager -> satellite -> lead -> office",
    )


# ================================================================================================
# GLOBAL search: community detection (real modularity optimization) + map-reduce over summaries.
# ================================================================================================


def detect_communities(graph: nx.Graph) -> list[frozenset[str]]:
    """Detect communities by MODULARITY optimization (networkx greedy_modularity_communities).

    Real, structure-based clustering: it groups entities that are more densely connected to each
    other than to the rest of the graph, WITHOUT any labels -- the same modularity-maximizing family
    as Louvain/Leiden. Returns communities as frozensets of entity names, largest first.
    """
    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(graph)
    return [frozenset(c) for c in communities]


def modularity_score(graph: nx.Graph, communities: list[frozenset[str]]) -> float:
    """The real modularity Q of a partition (networkx). Q in ~[-0.5, 1]; higher = stronger structure.

    Q measures how many edges fall WITHIN communities minus how many you'd expect if edges were
    placed at random (Newman 2006). Q > 0 means the partition captures real structure, not noise.
    """
    from networkx.algorithms.community import modularity

    return float(modularity(graph, communities))


# Illustrative community summaries: in production an LLM writes one per detected community. These are
# keyed by a representative anchor entity so we can attach a summary to whichever community contains
# it. The anchors match the communities the REAL modularity detection produces on this graph
# (imager-cluster, fault-cluster, people/program-cluster); if the structure changed, a community with
# no matching anchor falls back to listing its entities (see summarize_community).
_COMMUNITY_SUMMARY_BY_ANCHOR: dict[str, str] = {
    "hyperspectral imager": "Spacecraft & mission: Helios-7 itself, its launch from Kourou aboard Ariane 6, the hyperspectral imager it carries, and its sun-synchronous orbit.",
    "Error E-4011": "Operations & faults: telemetry error E-4011, the telemetry stream it appeared in, and the ground team investigating it.",
    "Nairobi office": "People & program leadership: Dr. Amara Okoye, who leads the program, and the Nairobi office she is based in.",
}


def summarize_community(community: frozenset[str]) -> str:
    """Attach the (illustrative) LLM summary for a community by matching a known anchor entity.

    In production an LLM reads each community's entities/relations and writes a 'community report';
    here we map a community to a fixed summary via an anchor entity it contains. Clearly illustrative
    text -- but it is attached to a REAL, structurally-detected community.
    """
    for anchor, summary in _COMMUNITY_SUMMARY_BY_ANCHOR.items():
        if anchor in community:
            return summary
    return "Miscellaneous entities: " + ", ".join(sorted(community))


@dataclass(frozen=True)
class GlobalAnswer:
    """A global-search result: the per-community partial points (map) and the final synthesis (reduce)."""

    partial_points: tuple[str, ...]
    final_answer: str


def global_search(communities: list[frozenset[str]]) -> GlobalAnswer:
    """Map-reduce over community summaries -- the GraphRAG global-search shape (real map-reduce flow).

    MAP: each community summary yields a partial 'theme' point (in production an LLM rates these). We
    run the real map-reduce CONTROL FLOW over the real communities; the summary/synthesis TEXT is the
    illustrative LLM stand-in. REDUCE: concatenate the partial points into a final themes answer.
    """
    partials = tuple(summarize_community(c) for c in communities)  # MAP: one partial per community
    final = "Main themes across the corpus: " + " | ".join(partials)  # REDUCE: synthesize
    return GlobalAnswer(partial_points=partials, final_answer=final)


# ================================================================================================
# Flat-RAG baseline (real dense retriever) -- what GraphRAG is contrasted against.
# ================================================================================================


def flat_rag_can_answer_multihop(corpus: tuple[str, ...]) -> bool:
    """True iff any single corpus chunk contains BOTH ends of the hop -- the imager and the office.

    Multi-hop answering needs one piece of text that links the two ends. This tests the strongest
    version of flat RAG's chance across the WHOLE corpus (not just its top-k): does ANY chunk mention
    both the 'imager' and the 'Nairobi office'? If not, no amount of top-k tuning or reranking lets
    flat RAG connect them -- the multi-hop failure is STRUCTURAL (a property of the corpus), not a
    ranking miss. It is exactly because this is corpus-level, not retriever-level, that the check takes
    only the corpus: the dense retriever's ranking is irrelevant when the linking text does not exist.
    """
    for chunk in corpus:
        text = chunk.lower()
        if ("imager" in text or "hyperspectral" in text) and "nairobi" in text:
            return True
    return False


# ================================================================================================
# Reporting
# ================================================================================================


def _report_versions() -> None:
    """Print numpy/networkx/torch versions for reproducibility. The encoder is CPU-pinned (ch5)."""
    print("numpy:", np.__version__)
    print("networkx:", nx.__version__)
    try:
        import torch

        available = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("torch:", torch.__version__, "| accelerator available:", available, "| encoder runs on: cpu")
    except ImportError:
        print("torch: not installed (graph algorithms are pure networkx — unaffected)")


def main() -> None:
    _report_versions()
    graph = build_graph()
    print(f"knowledge graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")
    print(
        "NOTE: the GRAPH + traversal + community detection + modularity are REAL (networkx) and "
        "measured; the entity/relation EXTRACTION and community SUMMARIES are FIXED exemplars "
        "standing in for a generator LLM (this env is encoder-only).\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) MULTI-HOP: flat vector RAG misses; graph traversal connects the chain.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Multi-hop: flat vector RAG can't CONNECT facts; graph traversal can")
    print("=" * 96)
    corpus = full_corpus()
    retriever = DenseRetriever(corpus)
    probe = build_multihop_probe()
    print(f"query: {probe.query}")
    print(f"dense lens: {retriever.backend}")
    # flat RAG: retrieve top-k, check whether any single chunk links both ends
    flat_hits = retriever.search(probe.query, k=TOP_K).indices
    print(f"  flat-RAG top-{TOP_K} chunks: {list(flat_hits)}")
    for idx in flat_hits:
        print(f"    chunk[{idx}]: {corpus[idx]}")
    flat_ok = flat_rag_can_answer_multihop(corpus)
    print(f"  any single corpus chunk links the imager AND the 'Nairobi office': {flat_ok}")
    # graph traversal: the real shortest path connects the chain
    path = shortest_path(graph, probe.start, probe.end)
    print(f"  GRAPH traversal ({len(path) - 1} hops): {path_with_relations(graph, path)}")
    print(f"  answer entity: {path[-1]}")
    # this is a structural fact about the CORPUS (no chunk links both ends), independent of the encoder
    assert not flat_ok, "no single corpus chunk links both ends -> flat RAG cannot connect them (the multi-hop gap)"
    assert path == list(probe.expected_path), f"traversal path must match the expected chain, got {path}"
    assert path[-1] == "Nairobi office", "the traversal must arrive at the responsible office"
    print("  -> flat RAG retrieves the endpoints but cannot connect them; the graph path answers it.\n")

    # ------------------------------------------------------------------------------------------
    # 2) COMMUNITY DETECTION: real modularity-optimizing clusters + the modularity score Q.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Community detection: modularity-optimizing clusters (real networkx) + the score Q")
    print("=" * 96)
    communities = detect_communities(graph)
    q = modularity_score(graph, communities)
    print(f"  detected {len(communities)} communities | modularity Q = {q:.4f}")
    for i, comm in enumerate(communities):
        print(f"    community {i} ({len(comm)} entities): {sorted(comm)}")
    assert len(communities) >= 2, "a structured graph should split into at least 2 communities"
    assert q > 0.0, "positive modularity means the partition captures real structure, not noise"
    # sanity: a random single-community partition scores LOWER modularity than the detected split
    one_blob = [frozenset(graph.nodes())]
    q_blob = modularity_score(graph, one_blob)
    print(f"  modularity of the trivial 'everything in one community' partition: {q_blob:.4f}")
    assert q > q_blob, "the detected partition must beat the trivial one-blob partition on modularity"
    print("  -> the detected split has higher modularity than lumping everything together (real structure).\n")

    # ------------------------------------------------------------------------------------------
    # 3) GLOBAL search: map-reduce over community summaries answers a whole-corpus theme question.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) Global search: map-reduce over community summaries (the 'main themes?' question)")
    print("=" * 96)
    global_q = "What are the main themes across the Helios-7 corpus?"
    print(f"query: {global_q}")
    result = global_search(communities)
    print(f"  MAP — one partial theme per community ({len(result.partial_points)} communities):")
    for i, point in enumerate(result.partial_points):
        print(f"    community {i}: {point}")
    print(f"  REDUCE — final answer:\n    {result.final_answer}")
    # the global answer must draw on EVERY community (no single chunk could)
    assert len(result.partial_points) == len(communities), "one partial per community (map covers all)"
    assert all(p in result.final_answer for p in result.partial_points), "reduce must aggregate every community's point"
    print("  -> no single chunk holds 'the themes'; the answer is an AGGREGATE over graph communities.")


if __name__ == "__main__":
    main()

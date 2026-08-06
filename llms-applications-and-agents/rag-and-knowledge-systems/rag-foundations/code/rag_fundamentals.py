"""A real, production-shaped mini-RAG over a real Hugging Face corpus.

This is the retrieve-then-generate pipeline built out of the *actual* components a production
RAG uses -- no toy corpus, no hand-rolled embedder, no stubbed generation:

  * corpus     -- a real Hugging Face dataset (`rag-datasets/rag-mini-wikipedia`): 3,200 real
                  Wikipedia passages + 918 real question/answer pairs.
  * embeddings -- a real sentence-transformers bi-encoder (`all-MiniLM-L6-v2`, 384-d).
  * index      -- a real FAISS approximate/exact nearest-neighbour index (`IndexFlatIP`).
  * rerank     -- a real cross-encoder (`ms-marco-MiniLM-L-6-v2`) that re-scores candidates.
  * generation -- a real LLM through the Hugging Face Inference API
                  (`meta-llama/Llama-3.1-8B-Instruct` by default), grounded on the retrieved text.

The module is import-safe and side-effect-light: constructing the objects loads models and calls
the network, so nothing runs at import time. Build a `RagPipeline` and call it.

Reproducibility / honesty
--------------------------
Retrieval is deterministic given fixed model weights (the embedder + FAISS are pure math).
Generation goes through a hosted LLM: we send `temperature=0` for as-deterministic-as-possible
decoding, but a hosted endpoint can still vary run-to-run (provider routing, model updates,
sampling floors). We never fabricate an answer -- if the API is unreachable, the call raises.

The libomp / OpenMP guard
-------------------------
FAISS and PyTorch (via sentence-transformers) can each load their own OpenMP runtime; on macOS
this collides and segfaults. Setting `KMP_DUPLICATE_LIB_OK` and pinning `OMP_NUM_THREADS`
*before* faiss/torch import defuses it. We do that at the very top of this module, so any file
that imports `rag_fundamentals` first is protected -- including the notebook and figure scripts.

Run the end-to-end demo:
    python rag_fundamentals.py
"""

from __future__ import annotations

# --- OpenMP guard: MUST run before faiss / torch import (see module docstring) --------------
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # allow the two libomp copies to coexist
os.environ.setdefault("OMP_NUM_THREADS", "1")  # pin threads so faiss+torch don't fight over them
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence the HF tokenizers fork warning

import time
from dataclasses import dataclass, field

import faiss
import numpy as np
from datasets import load_dataset
from huggingface_hub import InferenceClient
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- Real component identifiers (all verified to run in the target env) --------------------
DATASET_ID = "rag-datasets/rag-mini-wikipedia"  # real Wikipedia passages + QA pairs
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d bi-encoder, CPU-fast
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder re-scorer
# Generation model: Llama-3.1-8B is an open instruct model reliably served by the HF router.
# The fallbacks are tried in order if the primary is momentarily unavailable (the router's
# per-model availability genuinely fluctuates -- we saw Qwen 7B flip between runs).
GEN_MODEL_IDS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
)

# ---- Retrieval / generation hyperparameters (hoisted; no magic numbers inline) -------------
TOP_K = 3  # passages retrieved into the prompt by default
RERANK_CANDIDATES = 20  # how many dense hits to hand the cross-encoder before it re-ranks
MAX_NEW_TOKENS = 200  # cap on generated answer length
GEN_TEMPERATURE = 0.0  # greedy-ish decoding for maximum reproducibility

# The augmented-prompt template: the instruction that makes the model answer *from context*,
# plus a hard "say you don't know" clause so a corpus miss surfaces as an honest refusal.
PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. If the context does not contain the "
    "answer, say you don't know. Cite the passage number(s) in square brackets that you used.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass(frozen=True)
class RetrievedPassage:
    """One retrieved passage: its corpus index, text, and the score that surfaced it."""

    doc_id: int
    text: str
    score: float  # cosine similarity (dense) or cross-encoder logit (after rerank)


@dataclass
class RagResult:
    """The full trace of one RAG query -- everything needed to inspect *why* an answer was given."""

    question: str
    retrieved: list[RetrievedPassage]  # passages fed to the generator, best-first
    prompt: str  # the exact augmented prompt sent to the LLM
    answer: str  # the LLM's grounded answer
    model: str  # which generation model actually produced it
    latency_s: dict[str, float] = field(default_factory=dict)  # per-stage wall time


def load_corpus(*, max_passages: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Load the real Wikipedia corpus and QA pairs from the Hugging Face Hub.

    Returns `(passages, qa)` where `passages` is a list of raw passage strings (each already a
    reasonable chunk -- this dataset ships pre-split) and `qa` is a list of `{question, answer}`
    dicts. `max_passages` truncates the corpus for a faster demo; `None` uses all 3,200.
    """
    corpus_split = load_dataset(DATASET_ID, "text-corpus")["passages"]
    qa_split = load_dataset(DATASET_ID, "question-answer")["test"]
    passages = [row["passage"] for row in corpus_split]
    if max_passages is not None:
        passages = passages[:max_passages]
    qa = [{"question": row["question"], "answer": str(row["answer"])} for row in qa_split]
    return passages, qa


class RagPipeline:
    """A real retrieve-then-generate pipeline: embed -> FAISS index -> retrieve -> (rerank) -> generate.

    Construction is where the work happens: it downloads/loads the embedding model, embeds the
    whole corpus once, and builds the FAISS index. The reranker and the LLM client are created
    lazily on first use so a retrieval-only workflow pays for neither.
    """

    def __init__(
        self,
        passages: list[str],
        *,
        embed_model_id: str = EMBED_MODEL_ID,
        gen_model_ids: tuple[str, ...] = GEN_MODEL_IDS,
        hf_token: str | None = None,
    ) -> None:
        self.passages = passages
        self.embed_model_id = embed_model_id
        self.gen_model_ids = gen_model_ids
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")

        # 1) Embed the corpus with the real bi-encoder. `normalize_embeddings=True` gives unit
        #    vectors so that inner product == cosine similarity -- the geometry FAISS-IP expects.
        self.embedder = SentenceTransformer(embed_model_id)
        t = time.perf_counter()
        embeddings = self.embedder.encode(
            passages, normalize_embeddings=True, batch_size=128, show_progress_bar=False
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.embed_seconds = time.perf_counter() - t
        self.dim = int(self.embeddings.shape[1])

        # 2) Build the real FAISS index. IndexFlatIP does exact inner-product search -- at this
        #    scale (thousands of vectors) exact is ~2 ms/query, so we get ground-truth neighbours
        #    with no approximation error. (Vector-DB chapter swaps in HNSW/IVF for millions.)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

        # Lazily initialised heavy components.
        self._reranker: CrossEncoder | None = None
        self._client: InferenceClient | None = None
        self._active_gen_model: str | None = None

    # -- retrieval -----------------------------------------------------------------------------

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a query with the SAME model as the corpus (mismatched embedders => garbage)."""
        vec = self.embedder.encode([question], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def retrieve(self, question: str, *, k: int = TOP_K) -> list[RetrievedPassage]:
        """Dense retrieval: embed the query, ask FAISS for the k nearest passages by cosine."""
        query_vec = self.embed_query(question)
        scores, ids = self.index.search(query_vec, k)  # (1, k) each; scores are cosine sims
        return [
            RetrievedPassage(doc_id=int(i), text=self.passages[int(i)], score=float(s))
            for i, s in zip(ids[0], scores[0])
        ]

    def rerank(
        self, question: str, candidates: list[RetrievedPassage], *, k: int = TOP_K
    ) -> list[RetrievedPassage]:
        """Re-score dense candidates with the cross-encoder and keep the top-k.

        A bi-encoder embeds query and passage independently; a cross-encoder reads the
        (query, passage) pair jointly and is far more accurate -- but too slow to run over the
        whole corpus, so it only re-orders a shortlist the fast dense stage already narrowed.
        """
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL_ID)
        pairs = [(question, c.text) for c in candidates]
        ce_scores = self._reranker.predict(pairs)
        reranked = [
            RetrievedPassage(doc_id=c.doc_id, text=c.text, score=float(s))
            for c, s in zip(candidates, ce_scores)
        ]
        reranked.sort(key=lambda p: p.score, reverse=True)
        return reranked[:k]

    def retrieve_and_rerank(
        self, question: str, *, k: int = TOP_K, candidates: int = RERANK_CANDIDATES
    ) -> list[RetrievedPassage]:
        """Full retrieval: cast a wide dense net (`candidates`), then cross-encoder rerank to `k`."""
        dense = self.retrieve(question, k=candidates)
        return self.rerank(question, dense, k=k)

    # -- augmentation --------------------------------------------------------------------------

    @staticmethod
    def build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
        """Splice retrieved passages + the question into the augmented prompt (the 'open book')."""
        context = "\n".join(f"[{rank}] {p.text}" for rank, p in enumerate(passages, start=1))
        return PROMPT_TEMPLATE.format(context=context, question=question)

    # -- generation ----------------------------------------------------------------------------

    def _generate(self, prompt: str, *, max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[str, str]:
        """Call the real HF Inference API, trying each generation model until one responds.

        Returns `(answer_text, model_id)`. Raises `RuntimeError` if every candidate model fails --
        we never silently fabricate an answer.
        """
        if self._client is None:
            self._client = InferenceClient(token=self._hf_token)
        last_error: Exception | None = None
        for model_id in self.gen_model_ids:
            try:
                response = self._client.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=GEN_TEMPERATURE,
                )
                self._active_gen_model = model_id
                return response.choices[0].message.content.strip(), model_id
            except Exception as error:  # noqa: BLE001 -- provider errors vary; we try the next model
                last_error = error
                continue
        raise RuntimeError(
            f"all generation models failed ({self.gen_model_ids}); last error: {last_error}"
        )

    def generate_ungrounded(self, question: str, *, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Ask the LLM the bare question -- no retrieval, answer from parametric memory alone."""
        answer, _ = self._generate(question, max_new_tokens=max_new_tokens)
        return answer

    # -- end to end ----------------------------------------------------------------------------

    def answer(
        self, question: str, *, k: int = TOP_K, rerank: bool = True, max_new_tokens: int = MAX_NEW_TOKENS
    ) -> RagResult:
        """Full RAG: retrieve (+rerank) -> augment -> generate, returning the whole trace."""
        latency: dict[str, float] = {}

        t = time.perf_counter()
        retrieved = (
            self.retrieve_and_rerank(question, k=k) if rerank else self.retrieve(question, k=k)
        )
        latency["retrieve_s"] = time.perf_counter() - t

        prompt = self.build_prompt(question, retrieved)

        t = time.perf_counter()
        answer_text, model_id = self._generate(prompt, max_new_tokens=max_new_tokens)
        latency["generate_s"] = time.perf_counter() - t

        return RagResult(
            question=question,
            retrieved=retrieved,
            prompt=prompt,
            answer=answer_text,
            model=model_id,
            latency_s=latency,
        )


def _print_banner(pipeline: RagPipeline) -> None:
    """Print the reproducibility banner: versions + the real components in play."""
    import datasets as _ds
    import huggingface_hub as _hf
    import sentence_transformers as _st

    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        torch_v = torch.__version__
    except ImportError:  # pragma: no cover -- torch ships with sentence-transformers
        device, torch_v = "cpu", "n/a"

    print("=" * 78)
    print("REAL mini-RAG -- reproducibility banner")
    print(f"  numpy {np.__version__} | faiss {faiss.__version__} | torch {torch_v} | device {device}")
    print(f"  datasets {_ds.__version__} | sentence-transformers {_st.__version__} | hub {_hf.__version__}")
    print(f"  dataset : {DATASET_ID}  ({len(pipeline.passages)} passages)")
    print(f"  embed   : {pipeline.embed_model_id}  ({pipeline.dim}-d, {pipeline.embed_seconds:.1f}s to embed corpus)")
    print(f"  rerank  : {RERANK_MODEL_ID}")
    print(f"  generate: {' -> '.join(pipeline.gen_model_ids)} (first that responds)")
    print("=" * 78)


def main() -> None:
    """Run the headline demonstration: a real hallucination fixed by real retrieval."""
    passages, qa = load_corpus()
    pipeline = RagPipeline(passages)
    _print_banner(pipeline)

    # The headline question. The answering passage exists in the corpus, but the fact is obscure
    # enough that the bare LLM misattributes it -- exactly the closed-book failure RAG fixes.
    question = "What was reversed about the temperature scale in 1745?"

    print(f"\nQUESTION: {question}\n")

    # 1) Dense retrieval -- FAISS nearest neighbours, with real cosine scores.
    dense = pipeline.retrieve(question, k=TOP_K)
    print(f"DENSE top-{TOP_K} (cosine):")
    for rank, p in enumerate(dense, 1):
        print(f"  {rank}. doc[{p.doc_id}] cos={p.score:.3f} | {p.text[:96]}")
    assert dense[0].score > dense[-1].score, "dense results must be sorted best-first"

    # 2) The headline contrast: ungrounded (parametric) vs grounded (retrieved) on the SAME question.
    ungrounded = pipeline.generate_ungrounded(question)
    result = pipeline.answer(question, rerank=False)
    print("\nUNGROUNDED (no retrieval, parametric memory only):")
    print(f"  -> {ungrounded}")
    print("\nGROUNDED (retrieve-then-generate):")
    print(f"  -> {result.answer}")
    print(f"     [model: {result.model} | retrieve {result.latency_s['retrieve_s']*1000:.1f} ms "
          f"| generate {result.latency_s['generate_s']:.2f} s]")

    # 3) An out-of-corpus question: retrieval returns junk, so the honest answer is 'I don't know'.
    oo = "What was the closing share price of Nvidia stock yesterday?"
    oo_dense = pipeline.retrieve(oo, k=TOP_K)
    oo_result = pipeline.answer(oo, rerank=False)
    print(f"\nOUT-OF-CORPUS: {oo}")
    print(f"  best cosine only {oo_dense[0].score:.3f} (junk) -> grounded answer: {oo_result.answer}")


if __name__ == "__main__":
    main()

"""Question Answering from scratch: the span decode, the SQuAD scorer, and a retriever+reader.

One seeded, deterministic source of truth shared by the concept page, the teaching notebook
(`11-Question-Answering.ipynb`), and the figure generator (`make_figures_11.py`). Everything the
page claims numerically is computed here, so prose, notebook, and figures cannot drift apart.

Three demos, each asserting its qualitative point before printing:

  1. SPAN DECODE -- pick the best valid answer span (argmax over i<=j, length-capped) from
     independent start/end logits. Works on a real fine-tuned SQuAD model when one is cached,
     and on a DETERMINISTIC SYNTHETIC fallback otherwise, so the notebook ALWAYS runs offline.
  2. SQuAD SCORER -- Exact Match and token-level F1 from the official definitions (normalize ->
     bag-of-tokens overlap), reproducing the worked-example numbers exactly.
  3. RETRIEVER + READER -- a tiny open-domain sketch: embed question + passages, retrieve the
     top passage by cosine, then read a span out of it -- the DPR -> reader pipeline in miniature.

Determinism: numpy is seeded once at import, torch is seeded in `load_qa_model`, and any synthetic
vectors come from a STABLE md5 hash (NEVER Python's per-process-salted `hash()`), so a fresh
process reproduces bit-identical numbers. The math here is pure-numpy / torch-on-CPU; the device is
honestly CPU (the tiny tensors gain nothing from an accelerator), but DEVICE is detected and
reported so the banner is truthful about what ran.

Verified on Python 3.12 / numpy 2.x / torch 2.x / transformers 5.x, CPU.

Run:
    python question_answering.py
"""

from __future__ import annotations

import hashlib
import re
import string
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")  # silence HF deprecation chatter so the demo output stays clean

SEED = 0
np.random.seed(SEED)  # seed once at import so every downstream synthetic draw is deterministic

# A small fine-tuned extractive-QA model. Loaded if cached locally; otherwise we fall back to a
# deterministic synthetic span model so the notebook runs fully offline with no network.
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# A SQuAD-2.0 model that was trained WITH unanswerable questions, so it has a meaningful null
# (no-answer) score -- used by abstain_demo() to reproduce Worked Example 3's measured logits.
ABSTAIN_MODEL_NAME = "deepset/roberta-base-squad2"

# ---- Reproducibility helpers ------------------------------------------------------------------


def stable_hash(text: str) -> int:
    """Deterministic across processes -- md5, NEVER Python's salted hash()."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _detect_device() -> str:
    """Best available accelerator for reporting; the QA math itself runs on CPU for reproducibility."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DETECTED_DEVICE = _detect_device()
DEVICE = "cpu"  # pinned: the tiny QA tensors are CPU-bound and must be bit-reproducible


def runtime_banner() -> str:
    """One honest line about device + versions, so a reader knows exactly what produced the numbers."""
    import torch

    return (
        f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU -- these tensors are tiny, "
        f"reproducibility beats throughput)\n"
        f"numpy {np.__version__} | torch {torch.__version__} | seed {SEED}"
    )


# =================================================================================================
# 1. THE SPAN DECODE -- the heart of extractive QA
# =================================================================================================


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (subtract the max before exp)."""
    shifted = logits - np.max(logits)  # subtract max so the largest exp is 1.0, never overflows
    exp = np.exp(shifted)
    return exp / exp.sum()


def best_valid_span(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    *,
    l_max: int = 30,
    valid: np.ndarray | None = None,
) -> tuple[int, int, float]:
    """Return (i, j, score) maximizing start_i + end_j over VALID spans with j>=i and j-i<l_max.

    This is the derived decode: argmax_{j>=i, j-i<l_max} (S.h_i + E.h_j). The j>=i constraint is
    NOT optional -- the two heads are independent softmaxes, so a naive (argmax start, argmax end)
    can produce a backwards span. `valid[k]` marks positions eligible to be span endpoints (e.g.
    passage tokens only, never the question); None means every position is eligible.
    """
    n = len(start_logits)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    best_i, best_j, best_score = -1, -1, -np.inf
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i, min(i + l_max, n)):  # j>=i (no backwards span) and the length cap
            if not valid[j]:
                continue
            score = float(start_logits[i] + end_logits[j])  # sum of logits == log P_start+P_end+const
            if score > best_score:
                best_i, best_j, best_score = i, j, score
    return best_i, best_j, best_score


def span_confidence(start_logits: np.ndarray, end_logits: np.ndarray, i: int, j: int) -> float:
    """Joint span probability P_start(i) * P_end(j) -- the confidence behind a predicted span."""
    return float(softmax(start_logits)[i] * softmax(end_logits)[j])


def null_score(start_logits: np.ndarray, end_logits: np.ndarray) -> float:
    """SQuAD-2.0 no-answer score: start_[CLS] + end_[CLS] (the [CLS] token is position 0)."""
    return float(start_logits[0] + end_logits[0])


# The hand-worked logit table from the page (passage [The, capital, of, France, is, Paris, .]).
# Kept as data so the notebook and figures use the EXACT numbers the prose derives by hand.
HAND_TOKENS = ("The", "capital", "of", "France", "is", "Paris", ".")
HAND_START = np.array([-2.0, -1.0, -3.0, -0.5, -2.0, 4.0, -1.0])
HAND_END = np.array([-3.0, -2.5, -3.0, -1.0, -2.0, 4.5, -0.5])


# =================================================================================================
# 2. THE SQuAD SCORER -- Exact Match and token-level F1, from the definitions
# =================================================================================================

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation, drop a/an/the, collapse whitespace.

    This is why "the Champ de Mars" and "Champ de Mars" score a perfect match -- the article and
    the difference in spacing are normalized away before either metric looks at the strings.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalized strings are identical, else 0.0 -- all-or-nothing."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 with partial credit: bag-of-tokens precision/recall, harmonic mean."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)  # both empty -> 1.0; one empty -> 0.0
    # shared tokens counting multiplicity: min(count in pred, count in gold) per token type
    shared = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens) & set(gold_tokens))
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)  # of the words I said, how many were right
    recall = shared / len(gold_tokens)  # of the right words, how many did I say
    return 2 * precision * recall / (precision + recall)


def max_over_golds(prediction: str, golds: list[str], metric) -> float:
    """SQuAD scores against MULTIPLE gold answers by taking the max -- credit for matching any one."""
    return max(metric(prediction, g) for g in golds)


# The worked-example prediction/gold pairs from the page (reproduced exactly by the scorer).
EM_F1_PAIRS = (
    ("Paris, France", "Paris"),
    ("the Champ de Mars", "Champ de Mars"),
    ("Leonardo da Vinci", "Leonardo da Vinci"),
    ("1889", "in 1889"),
    ("London", "Paris"),
)


# =================================================================================================
# 3. RETRIEVER + READER -- a tiny open-domain QA sketch (DPR -> reader, in miniature)
# =================================================================================================

# A miniature corpus for the retrieve-then-read demo: only passage (a) actually answers the question.
RETRIEVER_QUESTION = "Who painted the Mona Lisa?"
RETRIEVER_CORPUS = (
    "The Mona Lisa is a half-length portrait painting by the Italian artist Leonardo da Vinci.",
    "The Louvre in Paris is the world's most-visited museum and home to the Mona Lisa.",
    "Renaissance portraiture emphasized realistic depiction of the human face and posture.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
)
RETRIEVER_GOLD = "Leonardo da Vinci"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")  # mixed-case: keep 'Mona' whole for readable answers
_STOPWORDS = frozenset(
    "the a an of in on at to is are was were by and or for with as from "
    "who what when where which whose s most into".split()
)


def _stem(token: str) -> str:
    """Crude suffix stripping so 'painted'/'painting'/'paints' share one token -- a transparent
    stand-in for the morphology a learned encoder captures, so the cosine sees a real overlap."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> list[str]:
    """Lowercase, stemmed content tokens (stopwords dropped) -- the vocabulary basis for the bag."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def embed_bow(text: str, vocabulary: list[str]) -> np.ndarray:
    """A transparent bag-of-words embedding over a fixed vocabulary -- a stand-in for a DPR encoder.

    Real DPR uses a learned bi-encoder; this keeps the SAME retrieve-by-cosine mechanics with math a
    reader can verify by hand. Meaning-matching is approximated by lexical overlap, which is exactly
    DPR's weakness (paraphrases) that dense retrieval was built to fix -- a fair, honest stand-in.
    """
    counts = np.zeros(len(vocabulary), dtype=float)
    index = {word: k for k, word in enumerate(vocabulary)}
    for token in _content_tokens(text):
        if token in index:
            counts[index[token]] += 1.0
    return counts


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity, guarding the zero-vector case (returns 0.0 rather than dividing by 0)."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if denom == 0 else float(u @ v / denom)


def retrieve_top_passage(
    question: str, corpus: tuple[str, ...]
) -> tuple[int, list[float]]:
    """Return (index of the best passage, cosine score per passage) -- the retriever stage.

    Vocabulary is the union of content words across question + corpus, so the question's words and
    each passage live in the same space; the passage whose bag points most like the question's wins.
    """
    vocabulary = sorted({w for text in (question, *corpus) for w in _content_tokens(text)})
    q_vec = embed_bow(question, vocabulary)
    scores = [cosine(q_vec, embed_bow(passage, vocabulary)) for passage in corpus]
    return int(np.argmax(scores)), scores


def read_span(question: str, passage: str, *, l_max: int = 6) -> str:
    """Reader stage: run the SAME `best_valid_span` decode over synthetic per-token logits.

    This deliberately reuses the extractive span decode -- the reader's job in retrieve-then-read is
    exactly the SQuAD span head, just over a retrieved passage. We build question-aware start/end
    logits (a named-entity / number heuristic, identical to the synthetic-fallback model), then
    decode the best valid (i, j) span. The point: the reader is not a separate trick, it is the span
    head from Family 1 applied to whatever the retriever surfaced.
    """
    tokens, start, end, valid = _synthetic_logits(question, passage)
    i, j, _ = best_valid_span(start, end, l_max=l_max, valid=valid)
    return " ".join(tokens[i : j + 1])


# =================================================================================================
# Live extractive-QA model: real if cached, deterministic synthetic fallback otherwise
# =================================================================================================


@dataclass
class QABackend:
    """A loaded extractive-QA backend. `is_real` distinguishes the fine-tuned model from the fallback."""

    name: str
    is_real: bool
    tokenizer: object = None
    model: object = None


# The end-to-end passage + questions the page measures (kept here so the notebook reuses them).
APOLLO_PASSAGE = (
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. "
    "Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above."
)
APOLLO_QUESTIONS = (
    "Who were the first humans to walk on the Moon?",
    "When did Apollo 11 land on the Moon?",
    "Who stayed in orbit during Apollo 11?",
)


def load_qa_model(*, force_synthetic: bool = False) -> QABackend:
    """Load the fine-tuned SQuAD model if it is cached locally; else a synthetic fallback.

    Offline-first: we pass `local_files_only=True` so a fresh, network-free run never hangs on a
    download -- it simply falls back to the synthetic backend and the notebook still completes.
    """
    import torch

    torch.manual_seed(SEED)
    if force_synthetic:
        return QABackend(name="synthetic", is_real=False)
    try:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL_NAME, local_files_only=True
        ).eval()
        return QABackend(name=QA_MODEL_NAME, is_real=True, tokenizer=tokenizer, model=model)
    except Exception:  # not cached / offline / transformers missing -> deterministic fallback
        return QABackend(name="synthetic", is_real=False)


def _synthetic_logits(question: str, passage: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic start/end logits that peak on the passage's answer-bearing tokens.

    With no model available we still need a span the decode can find. We score each passage token by
    a stable, question-aware heuristic: tokens that look like the answer (capitalized entities,
    numbers, the words after the question's focus) get a high start/end logit. This is NOT a model
    -- it is a fixture that makes the DECODE observable offline, and it is fully reproducible via
    the md5-stable hash. Returns (tokens, start_logits, end_logits, valid-mask over passage tokens).
    """
    tokens = _TOKEN_RE.findall(passage) + ["."]
    n = len(tokens)
    start = np.full(n, -3.0)
    end = np.full(n, -3.0)
    q_focus = set(_content_tokens(question))
    q_lower = question.lower()
    wants_number = any(w in q_lower for w in ("when", "what year", "how many", "how much"))
    connectors = {"da", "de", "of", "van", "von", "del"}  # internal lowercase parts of an entity name

    def is_entity(tok: str) -> bool:
        return bool(tok) and tok[:1].isupper()

    for k, tok in enumerate(tokens):
        base = 0.0
        if tok.isdigit():  # a number -> likely a date/quantity answer
            base += 8.0 if wants_number else 4.0  # a 'when'/'how many' question wants the number
        if tok.lower() in q_focus or _stem(tok.lower()) in q_focus:  # echoed question words aren't the answer
            base -= 3.0
        jitter = (stable_hash(tok) % 1000) / 1e6  # stable tie-break, deterministic every run
        start[k] += base + jitter
        end[k] += base + jitter

    # Entity-RUN aware: a maximal run of capitalized tokens (with internal connectors like "da")
    # is one answer span. Boost the run's FIRST token's start-logit and its LAST token's end-logit,
    # so the decode prefers "Leonardo da Vinci" whole over any single token inside it.
    k = 0
    while k < n:
        if is_entity(tokens[k]):
            j = k
            while j + 1 < n and (is_entity(tokens[j + 1]) or tokens[j + 1].lower() in connectors):
                j += 1
            # trim a trailing connector that isn't followed by another capitalized token
            while j > k and tokens[j].lower() in connectors:
                j -= 1
            start[k] += 6.0  # this token is a strong span START
            end[j] += 6.0  # this token is a strong span END
            k = j + 1
        else:
            k += 1
    valid = np.ones(n, dtype=bool)
    return tokens, start, end, valid


def answer_question(backend: QABackend, question: str, passage: str, *, l_max: int = 30) -> tuple[str, float, float]:
    """Answer with the derived span decode. Returns (answer_text, span_score, null_score).

    Identical decode whether the logits come from the real model or the synthetic fallback -- the
    point of the page is the DECODE, and it does not care where the logits came from.
    """
    if backend.is_real:
        import torch

        inp = backend.tokenizer(question, passage, return_tensors="pt")
        with torch.no_grad():
            out = backend.model(**inp)
        start = out.start_logits[0].numpy()
        end = out.end_logits[0].numpy()
        seq = inp.sequence_ids(0)
        ids = inp["input_ids"][0]
        valid = np.array([s == 1 for s in seq])  # only passage tokens are eligible span endpoints
        i, j, score = best_valid_span(start, end, l_max=l_max, valid=valid)
        text = backend.tokenizer.decode(ids[i : j + 1]).strip()
        return text, score, null_score(start, end)
    # synthetic fallback
    tokens, start, end, valid = _synthetic_logits(question, passage)
    i, j, score = best_valid_span(start, end, l_max=l_max, valid=valid)
    return " ".join(tokens[i : j + 1]), score, null_score(start, end)


# The two questions Worked Example 3 measures against the SQuAD-2.0 model (one unanswerable, one
# answerable), plus a synthetic abstain fixture so the *rule* is observable even with no model.
ABSTAIN_QUESTIONS = (
    "What is the diameter of the Moon?",  # unanswerable from the passage -> should abstain
    "Who were the first humans to walk on the Moon?",  # answerable -> should answer
)
# The exact (best-span text, span score, null score) the real roberta-base-squad2 produces, to two
# decimals -- asserted in abstain_demo() so the page's table is genuinely reproduced when cached.
ABSTAIN_EXPECTED = (
    ("lunar surface", -6.59, 1.04),
    ("Neil Armstrong and Buzz Aldrin", 16.64, 4.81),
)


def abstain_demo(*, force_synthetic: bool = False, tol: float = 0.05) -> list[tuple[str, float, float, str]]:
    """Reproduce Worked Example 3's abstain table on the SQuAD-2.0 model `deepset/roberta-base-squad2`.

    Returns one (answer_text, span_score, null_score, decision) row per ABSTAIN_QUESTIONS. When the
    real model is cached, the measured logits are ASSERTED against ABSTAIN_EXPECTED (the page's
    numbers), so the table cannot drift from the code. Offline, it falls back to a synthetic fixture
    -- the same offline-first, synthetic-fallback pattern as load_qa_model -- so the abstain RULE
    (compare s_best vs s_null) is still observable, just on illustrative logits rather than measured.

    The decision rule is the SQuAD-2.0 rule: answer iff s_best - s_null > tau (here tau = 0).
    """
    import torch

    torch.manual_seed(SEED)
    backend: QABackend
    if force_synthetic:
        backend = QABackend(name="synthetic", is_real=False)
    else:
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(ABSTAIN_MODEL_NAME, local_files_only=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                ABSTAIN_MODEL_NAME, local_files_only=True
            ).eval()
            backend = QABackend(name=ABSTAIN_MODEL_NAME, is_real=True, tokenizer=tokenizer, model=model)
        except Exception:  # not cached / offline -> synthetic fixture (rule still observable)
            backend = QABackend(name="synthetic", is_real=False)

    rows: list[tuple[str, float, float, str]] = []
    if backend.is_real:
        for idx, question in enumerate(ABSTAIN_QUESTIONS):
            text, span_s, null_s = answer_question(backend, question, APOLLO_PASSAGE)
            decision = "answer" if span_s - null_s > 0.0 else "abstain (no answer)"
            # assert the measured numbers match the page's table, to two decimals
            exp_text, exp_span, exp_null = ABSTAIN_EXPECTED[idx]
            assert text == exp_text, f"abstain[{idx}] text {text!r} != expected {exp_text!r}"
            assert abs(span_s - exp_span) < tol, f"abstain[{idx}] span {span_s:.2f} != {exp_span}"
            assert abs(null_s - exp_null) < tol, f"abstain[{idx}] null {null_s:.2f} != {exp_null}"
            rows.append((text, span_s, null_s, decision))
    else:
        # The synthetic span model has NO null-trained head, so it cannot honestly "abstain" -- it
        # always finds some entity. To keep the abstain RULE observable offline, we substitute an
        # explicit fixture: hand-set (span, null) scores that make the comparison fire both ways.
        # These are the same numbers the notebook's standalone abstain cell uses.
        fixtures = (
            ("lunar surface", -2.0, 4.0),  # unanswerable: null beats span -> abstain
            ("Neil Armstrong and Buzz Aldrin", 6.0, 0.0),  # answerable: span beats null -> answer
        )
        for text, span_s, null_s in fixtures:
            decision = "answer" if span_s - null_s > 0.0 else "abstain (no answer)"
            rows.append((text, span_s, null_s, decision))
    # whichever backend: the unanswerable question must abstain, the answerable one must answer
    assert rows[0][3].startswith("abstain"), "unanswerable question should abstain"
    assert rows[1][3] == "answer", "answerable question should answer"
    return rows


# =================================================================================================
# main: run all three demos, each asserting its qualitative point before printing
# =================================================================================================


def main() -> None:
    print(runtime_banner())
    print()

    # ---- Demo 1: span decode on the hand-worked logits (always offline, no model needed) --------
    i, j, score = best_valid_span(HAND_START, HAND_END, l_max=30)
    conf = span_confidence(HAND_START, HAND_END, i, j)
    # the prose derives (5,5) "Paris" with confidence ~0.957 -- assert it before printing
    assert (i, j) == (5, 5), f"expected span (5,5)='Paris', got ({i},{j})"
    assert abs(score - 8.5) < 1e-9, f"expected score 8.5, got {score}"
    assert 0.95 < conf < 0.96, f"expected confidence ~0.957, got {conf:.4f}"
    print("1. SPAN DECODE (hand-worked logits)")
    print(f"   best valid span = ({i},{j}) -> {HAND_TOKENS[i]!r}   score={score:.1f}   confidence={conf:.3f}")
    print()

    # ---- Demo 2: SQuAD EM / F1 scorer (always offline) -----------------------------------------
    # assert the two headline rows before printing the table
    assert exact_match("Paris, France", "Paris") == 0.0 and abs(token_f1("Paris, France", "Paris") - 2 / 3) < 1e-9
    assert exact_match("the Champ de Mars", "Champ de Mars") == 1.0  # the article normalizes away
    print("2. SQuAD SCORER (EM = exact, F1 = token overlap)")
    for pred, gold in EM_F1_PAIRS:
        print(f"   EM={exact_match(pred, gold):.0f}  F1={token_f1(pred, gold):.3f}   {pred!r:20s} vs {gold!r}")
    print()

    # ---- Demo 3: retriever + reader sketch (always offline) -------------------------------------
    top_idx, scores = retrieve_top_passage(RETRIEVER_QUESTION, RETRIEVER_CORPUS)
    # passage 0 is the only one that answers the question -- assert the retriever found it
    assert top_idx == 0, f"retriever should pick passage 0, picked {top_idx}"
    read = read_span(RETRIEVER_QUESTION, RETRIEVER_CORPUS[top_idx])
    assert RETRIEVER_GOLD.lower() in read.lower(), f"reader missed the answer in {read!r}"
    print("3. RETRIEVER + READER (open-domain sketch)")
    print(f"   question: {RETRIEVER_QUESTION!r}")
    for k, (passage, s) in enumerate(zip(RETRIEVER_CORPUS, scores)):
        mark = "  <-- retrieved (top cosine)" if k == top_idx else ""
        print(f"   cos={s:.3f}  p{k}: {passage[:54]}...{mark}")
    print(f"   reader span -> {read!r}")
    print()

    # ---- Demo 4: live extractive QA -- real model if cached, synthetic fallback otherwise -------
    backend = load_qa_model()
    print(f"4. LIVE EXTRACTIVE QA  (backend: {'real ' + backend.name if backend.is_real else 'synthetic fallback'})")
    for q in APOLLO_QUESTIONS:
        text, span_s, null_s = answer_question(backend, q, APOLLO_PASSAGE)
        print(f"   {text:34s}  span={span_s:6.2f}  null={null_s:6.2f}  <- {q}")
    print()

    # ---- Demo 5: abstaining (SQuAD 2.0) -- reproduces Worked Example 3's measured table ----------
    rows = abstain_demo()
    print("5. ABSTAIN (SQuAD 2.0: compare best span vs the [CLS] null span)")
    for (text, span_s, null_s, decision), question in zip(rows, ABSTAIN_QUESTIONS):
        print(f"   span={span_s:6.2f}  null={null_s:6.2f}  -> {decision:20s}  <- {question}")


if __name__ == "__main__":
    main()

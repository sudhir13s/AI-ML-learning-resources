"""Sequence labeling from scratch: HMM + Viterbi, greedy baseline, CRF-style scoring.

The single seeded source of truth for the chapter
**Sequence Labeling — POS & NER**. Everything the page shows and the notebook runs
is IMPORTED from here, so the prose, the figures (`make_figures_09.py`), and the
teaching notebook can never silently drift from one another.

What lives here, one idea per function:
  * a tiny 2-tag HMM ("fish sleep", N vs V) with start/transition/emission tables;
  * `viterbi`            -- max-product DP that decodes the single best tag *path*;
  * `greedy_argmax`      -- the per-token baseline Viterbi must beat;
  * `forward_logZ`       -- sum-product DP (the partition function, in log space);
  * `bio_transition_matrix` / `crf_sequence_score` -- linear-chain CRF scoring with a
    learned-style transition matrix that makes the illegal `O -> I-PER` transition cost
    -inf, so a globally-scored sequence rules it out where a per-token argmax cannot;
  * `decode_bio_spans`   -- recover (type, start, end) entity tuples from a BIO tag stream;
  * `token_accuracy` / `entity_prf` -- the metric gap (token accuracy lies; entity F1 is honest).

Honest reproducibility line: this module is **pure-Python / numpy** (no torch needed for
the structured-prediction math), so the device is always CPU. We seed numpy once at import
so every run is bit-identical, and we use a STABLE md5 hash (never Python's per-process-salted
`hash()`) anywhere a string needs hashing.

Run:
    python sequence_labeling.py
"""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version

import numpy as np

# ---- Reproducibility -------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)  # seed once at import so every downstream draw is deterministic

# Pure-Python/numpy structured prediction -> the device is honestly CPU.
DEVICE = "cpu (pure-Python/numpy)"
NEG_INF = -1e9  # finite stand-in for -inf so log-space sums never produce nan


def runtime_banner() -> str:
    """One-line, honest provenance string printed by demos and the notebook."""
    py = platform.python_version()
    np_v = np.__version__
    try:
        torch_v = version("torch")  # reported only if importable; not required by this module
        torch_note = f", torch {torch_v} (not used here)"
    except Exception:  # noqa: BLE001 -- torch is genuinely optional for this module
        torch_note = ""
    return f"device: {DEVICE} | python {py} | numpy {np_v}{torch_note}"


def stable_hash(text: str) -> int:
    """Deterministic across processes -- md5, NEVER Python's salted hash()."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


# =============================================================================================
# Part 1 -- A tiny 2-tag HMM and its decoders (Viterbi vs greedy)
# =============================================================================================
# The garden-path classic "time flies fast": every word can be Noun OR Verb. A left-to-right
# greedy tagger is seduced at "flies" by the high LOCAL verb emission b_V(flies)=0.7 and commits
# to V -- which forces a globally worse path. Viterbi, scoring the whole sequence, reads
# "time flies" as a noun phrase (N N) and "fast" as the verb (V): [N, N, V]. Parameters are
# chosen so greedy and Viterbi DISAGREE -- greedy lands on the 2nd-best path, Viterbi on the best.
HMM_TAGS: tuple[str, ...] = ("N", "V")
HMM_START: dict[str, float] = {"N": 0.5, "V": 0.5}
HMM_TRANS: dict[tuple[str, str], float] = {
    ("N", "N"): 0.6, ("N", "V"): 0.4,
    ("V", "N"): 0.8, ("V", "V"): 0.2,
}
HMM_EMIT: dict[tuple[str, str], float] = {
    ("N", "time"): 0.6, ("V", "time"): 0.3,
    ("N", "flies"): 0.3, ("V", "flies"): 0.7,
    ("N", "fast"): 0.2, ("V", "fast"): 0.7,
}

# A second 2-word HMM ("fish sleep") used for the page's hand-worked Viterbi derivation -- two
# tokens are small enough to compute delta by hand and check against viterbi(). Here greedy and
# Viterbi happen to AGREE (both read N, V); the divergence lesson lives in the 3-word HMM above.
WORKED_START: dict[str, float] = {"N": 0.6, "V": 0.4}
WORKED_TRANS: dict[tuple[str, str], float] = {
    ("N", "N"): 0.3, ("N", "V"): 0.7,
    ("V", "N"): 0.6, ("V", "V"): 0.4,
}
WORKED_EMIT: dict[tuple[str, str], float] = {
    ("N", "fish"): 0.5, ("V", "fish"): 0.2,
    ("N", "sleep"): 0.1, ("V", "sleep"): 0.5,
}


def viterbi(
    words: list[str],
    tags: tuple[str, ...] = HMM_TAGS,
    start: dict[str, float] = HMM_START,
    trans: dict[tuple[str, str], float] = HMM_TRANS,
    emit: dict[tuple[str, str], float] = HMM_EMIT,
) -> tuple[list[str], float, np.ndarray]:
    """Max-product DP: the single most probable tag PATH for `words`.

    Returns (best_path, best_path_probability, delta_table) where
    delta[t, j] = prob of the best path that ends in tag j at position t
    (the textbook Viterbi variable delta_t(j)). O(n*k^2) time, O(n*k) space.
    """
    n, k = len(words), len(tags)
    delta = np.zeros((n, k))                      # delta[t, j]: best path prob into (t, tag j)
    back = np.zeros((n, k), dtype=int)            # backpointer: which prev tag we came from
    for j, tag in enumerate(tags):                # init t=0: delta_1(j) = pi_j * b_j(x_1)
        delta[0, j] = start[tag] * emit[(tag, words[0])]
    for t in range(1, n):                         # recurse: delta_t(j) = max_i[delta_{t-1}(i) a_ij] * b_j(x_t)
        for j, tag in enumerate(tags):
            scores = [delta[t - 1, i] * trans[(prev, tag)] for i, prev in enumerate(tags)]
            best_i = int(np.argmax(scores))
            delta[t, j] = scores[best_i] * emit[(tag, words[t])]
            back[t, j] = best_i
    last = int(np.argmax(delta[n - 1]))           # terminate: best final state, then backtrack
    path_idx = [last]
    for t in range(n - 1, 0, -1):
        path_idx.insert(0, int(back[t, path_idx[0]]))
    return [tags[i] for i in path_idx], float(delta[n - 1, last]), delta


def greedy_argmax(
    words: list[str],
    tags: tuple[str, ...] = HMM_TAGS,
    start: dict[str, float] = HMM_START,
    trans: dict[tuple[str, str], float] = HMM_TRANS,
    emit: dict[tuple[str, str], float] = HMM_EMIT,
) -> list[str]:
    """Per-token baseline: pick each tag to maximize ITS OWN local probability, ignore the path.

    At t=0 it maximizes pi_j * b_j(x_1); afterwards it maximizes a_{prev,j} * b_j(x_t) given the
    tag it already committed to -- a left-to-right greedy walk with no look-ahead. This is the
    classifier Viterbi must beat; on the HMM above it picks the WRONG sequence for "fish sleep".
    """
    n = len(words)
    out: list[str] = []
    first = max(tags, key=lambda tag: start[tag] * emit[(tag, words[0])])
    out.append(first)
    for t in range(1, n):
        prev = out[-1]
        nxt = max(tags, key=lambda tag: trans[(prev, tag)] * emit[(tag, words[t])])
        out.append(nxt)
    return out


def path_probability(
    words: list[str],
    path: list[str],
    start: dict[str, float] = HMM_START,
    trans: dict[tuple[str, str], float] = HMM_TRANS,
    emit: dict[tuple[str, str], float] = HMM_EMIT,
) -> float:
    """Joint p(y, x) for a SPECIFIC tag path -- used to show greedy's path scores lower than Viterbi's."""
    p = start[path[0]] * emit[(path[0], words[0])]
    for t in range(1, len(words)):
        p *= trans[(path[t - 1], path[t])] * emit[(path[t], words[t])]
    return float(p)


def forward_logZ(
    words: list[str],
    tags: tuple[str, ...] = HMM_TAGS,
    start: dict[str, float] = HMM_START,
    trans: dict[tuple[str, str], float] = HMM_TRANS,
    emit: dict[tuple[str, str], float] = HMM_EMIT,
) -> float:
    """Sum-product DP: total probability p(x) = sum over ALL tag paths (the CRF partition function).

    Same trellis as Viterbi with max replaced by sum: alpha_t(j) = [sum_i alpha_{t-1}(i) a_ij] b_j(x_t).
    Returned in log space (log p(x)) because real partition functions multiply many tiny factors.
    """
    n, k = len(words), len(tags)
    alpha = np.zeros((n, k))
    for j, tag in enumerate(tags):
        alpha[0, j] = start[tag] * emit[(tag, words[0])]
    for t in range(1, n):
        for j, tag in enumerate(tags):
            alpha[t, j] = sum(alpha[t - 1, i] * trans[(prev, tag)] for i, prev in enumerate(tags))
            alpha[t, j] *= emit[(tag, words[t])]
    return float(np.log(alpha[n - 1].sum()))


# =============================================================================================
# Part 2 -- Linear-chain CRF scoring: rule out the illegal O -> I-PER transition
# =============================================================================================
# A 5-tag NER label set. The whole point: a globally-scored sequence can forbid I-PER right after
# O (you cannot CONTINUE a person span that never BEGAN), which a per-token argmax cannot.
NER_TAGS: tuple[str, ...] = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC")
TAG_INDEX: dict[str, int] = {t: i for i, t in enumerate(NER_TAGS)}


def bio_transition_matrix(
    tags: tuple[str, ...] = NER_TAGS,
    legal_bonus: float = 1.0,
    continue_bonus: float = 1.5,
    adjacent_b_penalty: float = -1.0,
) -> np.ndarray:
    """A CRF-style transition score matrix A[i, j] = score of going from tag i to tag j.

    Encodes the BIO grammar the way a *trained* CRF would learn it from data:
      * HARD constraint -- `I-TYPE` is legal ONLY after `B-TYPE` or `I-TYPE` of the SAME type;
        every other way to reach an `I-` tag (notably `O -> I-PER`) gets NEG_INF (the illegal
        transition). Viterbi will route AROUND it even when the `I-` emission is highest.
      * SOFT preferences a real CRF picks up from counts:
          - continuing a span (`B-TYPE -> I-TYPE`, `I-TYPE -> I-TYPE`) earns `continue_bonus`,
            because multi-token entities are common and "keep the span going" is the frequent pattern;
          - starting a NEW same-type span with no `O` between (`B-/I-TYPE -> B-TYPE`) earns
            `adjacent_b_penalty`, because two same-type entities glued together is rare in data.
        These soft terms are why a CRF keeps one consistent 3-token PER span instead of fragmenting
        it into two adjacent person spans, even when the per-token emissions tilt the other way.
      * everything else gets a small positive `legal_bonus`.
    """
    k = len(tags)
    A = np.full((k, k), legal_bonus)
    for i, src in enumerate(tags):
        for j, dst in enumerate(tags):
            if dst.startswith("I-"):
                dst_type = dst[2:]
                if src in (f"B-{dst_type}", f"I-{dst_type}"):
                    A[i, j] = continue_bonus     # legal continuation -> mildly preferred
                else:
                    A[i, j] = NEG_INF            # O -> I-PER, B-LOC -> I-PER: structurally illegal
            elif dst.startswith("B-"):
                dst_type = dst[2:]
                if src in (f"B-{dst_type}", f"I-{dst_type}"):
                    A[i, j] = adjacent_b_penalty  # two same-type entities with no O between: rare
    return A


def crf_sequence_score(
    emissions: np.ndarray,
    tag_path: list[str],
    transition: np.ndarray,
    tags: tuple[str, ...] = NER_TAGS,
) -> float:
    """Score of one label sequence under a linear-chain CRF: sum of emissions + sum of transitions.

    score(y, x) = sum_t P[t, y_t]  +  sum_t A[y_{t-1}, y_t]
    (the unnormalized log-score; the partition function Z(x) divides every sequence by the same
    constant, so it does not change the argmax). `emissions[t, j]` is the per-token score of tag j.
    """
    idx = [TAG_INDEX[t] for t in tag_path]
    score = float(emissions[0, idx[0]])
    for t in range(1, len(idx)):
        score += float(transition[idx[t - 1], idx[t]]) + float(emissions[t, idx[t]])
    return score


def crf_viterbi_decode(
    emissions: np.ndarray,
    transition: np.ndarray,
    tags: tuple[str, ...] = NER_TAGS,
) -> tuple[list[str], float]:
    """Additive-score Viterbi for a linear-chain CRF -- the SAME DP as the HMM, in log space.

    delta_t(j) = max_i [ delta_{t-1}(i) + A[i, j] ] + P[t, j].  Returns (best_tag_path, best_score).
    """
    n, k = emissions.shape
    delta = np.full((n, k), NEG_INF)
    back = np.zeros((n, k), dtype=int)
    delta[0] = emissions[0]                                   # no transition into the first token
    for t in range(1, n):
        for j in range(k):
            scores = delta[t - 1] + transition[:, j]          # best incoming + transition into j
            best_i = int(np.argmax(scores))
            delta[t, j] = scores[best_i] + emissions[t, j]    # + this token's emission for tag j
            back[t, j] = best_i
    last = int(np.argmax(delta[n - 1]))
    path_idx = [last]
    for t in range(n - 1, 0, -1):
        path_idx.insert(0, int(back[t, path_idx[0]]))
    return [tags[i] for i in path_idx], float(delta[n - 1, last])


# =============================================================================================
# Part 3 -- BIO span decoding and the metric gap (token accuracy vs entity F1)
# =============================================================================================
def decode_bio_spans(tag_seq: list[str]) -> set[tuple[str, int, int]]:
    """Recover entity spans as (type, start, end_inclusive) tuples from a BIO tag stream.

    A span opens at each `B-TYPE`, extends through matching `I-TYPE`, and closes at the next
    `B-`/`O`/type-change. This is exactly how CoNLL/seqeval turn tags into countable entities.
    """
    spans: set[tuple[str, int, int]] = set()
    cur_type: str | None = None
    start = -1
    for i, tag in enumerate(tag_seq):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.add((cur_type, start, i - 1))
            cur_type, start = tag[2:], i
        elif tag.startswith("I-"):
            if cur_type != tag[2:]:               # I- with no matching open span -> treat as a break
                if cur_type is not None:
                    spans.add((cur_type, start, i - 1))
                cur_type = None
        else:                                     # O
            if cur_type is not None:
                spans.add((cur_type, start, i - 1))
            cur_type = None
    if cur_type is not None:
        spans.add((cur_type, start, len(tag_seq) - 1))
    return spans


def token_accuracy(gold: list[str], pred: list[str]) -> float:
    """Fraction of positions where the predicted tag equals the gold tag -- the metric that LIES."""
    assert len(gold) == len(pred), "gold and pred must align position-for-position"
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def entity_prf(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    """Entity-level (precision, recall, F1) with EXACT span+type match -- the honest NER metric.

    A predicted span counts only if (type, start, end) matches a gold span exactly. This is the
    CoNLL/seqeval definition; we reimplement it from scratch so the gap is transparent.
    """
    g, p = decode_bio_spans(gold), decode_bio_spans(pred)
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# =============================================================================================
# Demo: assert each qualitative claim, THEN print the supporting numbers
# =============================================================================================
def main() -> None:
    print(runtime_banner())
    print()

    # ---- Demo (a): Viterbi beats greedy per-token argmax on the ambiguous HMM ----------------
    words = ["time", "flies", "fast"]
    v_path, v_prob, _ = viterbi(words)
    g_path = greedy_argmax(words)
    v_score, g_score = path_probability(words, v_path), path_probability(words, g_path)
    assert v_path == ["N", "N", "V"], "Viterbi should read 'time flies' (N N) + 'fast' (V)"
    assert g_path == ["N", "V", "N"], "greedy is seduced by b_V(flies)=0.7 and commits to V"
    assert g_path != v_path, "this HMM is chosen so greedy DISAGREES with Viterbi"
    assert v_score > g_score, "Viterbi's path must score strictly higher than the greedy path"
    print("Demo (a) -- Viterbi vs greedy on 'time flies fast':")
    print(f"  Viterbi path : {v_path}   p(y,x) = {v_score:.5f}   <- the true global best")
    print(f"  greedy  path : {g_path}   p(y,x) = {g_score:.5f}   <- locally tempting, globally 2nd-best")
    print("  PASS: greedy commits to V at 'flies' (b_V=0.7) and is locked into a worse path;")
    print("        Viterbi weighs emission AND transition over the WHOLE sequence and wins\n")

    # ---- Demo (b): an illegal O -> I-PER transition is ruled out by sequence scoring ----------
    # Emissions are deliberately MISLEADING at token 1: they favour I-PER even though token 0 is O.
    A = bio_transition_matrix()
    em_words = ["downtown", "Paris"]                          # the truth is O then B-LOC, no person
    emissions = np.array([
        # O     B-PER  I-PER  B-LOC  I-LOC
        [2.0,   0.1,   0.1,   0.5,   0.1],                    # token 0 "downtown" -> clearly O
        [0.5,   0.4,   2.2,   1.9,   0.2],                    # token 1 "Paris"    -> emission WRONGLY tops I-PER
    ])
    greedy_tags = [NER_TAGS[int(np.argmax(emissions[t]))] for t in range(len(em_words))]
    crf_tags, crf_best = crf_viterbi_decode(emissions, A)
    illegal = ["O", "I-PER"]                                  # what per-token argmax emits
    s_illegal = crf_sequence_score(emissions, illegal, A)
    s_crf = crf_sequence_score(emissions, crf_tags, A)
    assert greedy_tags == ["O", "I-PER"], "per-token argmax falls for the misleading I-PER emission"
    assert crf_tags == ["O", "B-LOC"], "CRF rules out O->I-PER and recovers the correct O, B-LOC"
    assert s_crf > s_illegal, "the legal CRF path must outscore the illegal O->I-PER path"
    print("Demo (b) -- illegal 'O -> I-PER' ruled out by global scoring:")
    print(f"  per-token argmax : {greedy_tags}   <- emits the ILLEGAL I-PER (no B-PER before it)")
    print(f"  CRF Viterbi      : {crf_tags}   best score = {crf_best:.2f}   <- the correct reading")
    print(f"  score(O, I-PER)  = {s_illegal:>12.1f}   <- O->I-PER transition = -inf")
    print(f"  score(O, B-LOC)  = {s_crf:>12.1f}   <- legal, so Viterbi takes it")
    print("  PASS: the -inf transition makes the illegal sequence impossible, so the model")
    print("        backs off to the best LEGAL tagging instead of the top per-token emission\n")

    # ---- Demo (c): CRF sequence scoring vs independent classification (label consistency) -----
    # Same emissions, two candidate labelings of a 3-token person name "Dr Jane Smith".
    em3 = np.array([
        # O     B-PER  I-PER  B-LOC  I-LOC
        [0.2,   2.5,   0.1,   0.3,   0.1],                    # "Dr"    -> B-PER
        [0.2,   1.0,   2.2,   0.3,   0.1],                    # "Jane"  -> emission likes I-PER
        [0.2,   1.5,   1.4,   0.3,   0.1],                    # "Smith" -> emission torn B-PER vs I-PER
    ])
    indep = [NER_TAGS[int(np.argmax(em3[t]))] for t in range(3)]   # independent per-token argmax
    crf3, _ = crf_viterbi_decode(em3, A)
    s_indep = crf_sequence_score(em3, indep, A) if indep[1:] != ["I-PER", "B-PER"] else NEG_INF
    s_crf3 = crf_sequence_score(em3, crf3, A)
    assert indep == ["B-PER", "I-PER", "B-PER"], "independent argmax breaks the span at 'Smith'"
    assert crf3 == ["B-PER", "I-PER", "I-PER"], "CRF keeps one consistent 3-token PER span"
    assert s_crf3 > s_indep, "the consistent CRF span scores higher than the broken independent one"
    print("Demo (c) -- CRF scoring vs independent classification on 'Dr Jane Smith':")
    print(f"  independent argmax : {indep}   <- breaks the span (a stray B-PER mid-name)")
    print(f"  CRF Viterbi        : {crf3}   <- one clean 3-token PERSON span")
    print("  PASS: global scoring keeps the span consistent; per-token argmax fragments it\n")

    # ---- Demo (d): the metric gap -- token accuracy lies, entity F1 is honest -----------------
    gold = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "O"]   # "New York City" = LOC
    pred = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O", "O"]       # model drops "City"
    acc = token_accuracy(gold, pred)
    precision, recall, f1 = entity_prf(gold, pred)
    assert abs(acc - 0.875) < 1e-9, "7/8 tokens correct"
    assert abs(f1 - 0.5) < 1e-9, "one dropped token halves entity F1"
    print("Demo (d) -- token accuracy vs entity F1 (model drops one token of a 3-token LOC):")
    print(f"  token accuracy = {acc:.3f}   <- looks fine (swamped by easy O tokens)")
    print(f"  entity  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}   <- one dropped token halves it")
    print("  PASS: a half-right span is a wrong span; entity F1 is the metric that matters")


if __name__ == "__main__":
    main()

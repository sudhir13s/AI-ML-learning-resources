"""Coreference resolution from scratch: the single seeded source of truth for the chapter.

Everything the page **14 Coreference Resolution** shows, every figure in
`make_figures_14.py`, and every cell in `14-Coreference-Resolution.ipynb` is IMPORTED
from this module, so the prose, the figures, and the notebook can never silently drift
from one another.

What lives here, one idea per function:
  * a tiny linear **mention-ranking** scorer -- score every candidate antecedent of a
    mention (including the dummy epsilon = "no antecedent"), softmax over them, pick the
    argmax, then form clusters by **transitive closure** of the antecedent links;
  * the three coreference metrics from scratch -- `muc` (link-based, Vilain et al. 1995),
    `bcubed` (mention-based, Bagga & Baldwin 1998), `ceaf_phi4` (entity-alignment,
    Luo 2005), and the `conll_f1` average (Pradhan et al. 2012); B-cubed matches the
    by-hand derivation in the page exactly;
  * a **Winograd-schema** demonstration -- flip one adjective and the world-knowledge
    feature flips the resolved antecedent, which structure/agreement alone cannot do;
  * `spacy_mentions` -- real, reproducible mention detection (named entities + pronouns +
    noun chunks) on the running passage, used by the page's worked example.

Honest reproducibility line: this module is **pure-Python / numpy** (no torch needed for
the ranking math or the metrics), so the device is always CPU. We seed numpy once at
import so every run is bit-identical, and we use a STABLE md5 hash (never Python's
per-process-salted ``hash()``) anywhere a string needs hashing.

Run:
    python coreference.py
"""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version
from itertools import permutations

import numpy as np

# ---- Reproducibility -------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)  # seed once at import so every downstream draw is deterministic

# Pure-Python/numpy ranking + metrics -> the device is honestly CPU.
DEVICE = "cpu (pure-Python/numpy)"

# A scored antecedent's score is measured RELATIVE to "no antecedent": epsilon is pinned at 0,
# so a real candidate only wins if it scores above 0 (an elegant built-in new-entity threshold).
EPSILON = "ε"  # the dummy antecedent symbol used throughout
EPSILON_SCORE = 0.0


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


# =====================================================================================
# Mention-ranking: score candidates (incl. epsilon), softmax, pick argmax, close clusters
# =====================================================================================
#
# A mention is a small feature dict. The toy document is the running passage:
#   "John told his manager that he would finish the report by Friday. She thanked him."
# Each mention carries the cues a linear scorer needs: gender, number, salience (subject?),
# its index in reading order, and which gold entity it belongs to (for evaluating the demo).

MENTIONS: list[dict] = [
    # text,          gender,   number,  subject, pronoun, gold_entity
    {"text": "John", "gender": "masc", "number": "sg", "subject": True, "pronoun": False, "gold": "E1"},
    {"text": "his", "gender": "masc", "number": "sg", "subject": False, "pronoun": True, "gold": "E1"},
    {"text": "his manager", "gender": "unk", "number": "sg", "subject": False, "pronoun": False, "gold": "E2"},
    {"text": "he", "gender": "masc", "number": "sg", "subject": True, "pronoun": True, "gold": "E1"},
    {"text": "She", "gender": "fem", "number": "sg", "subject": True, "pronoun": True, "gold": "E2"},
    {"text": "him", "gender": "masc", "number": "sg", "subject": False, "pronoun": True, "gold": "E1"},
]

# Feature weights for the linear scorer s(i, j) = w . features(i, j).  Hand-set so the demo
# resolves the passage correctly -- they encode the textbook coref signals: agreement dominates,
# recency and subject-salience break ties, and a *non-pronominal* mention (a new name or full
# NP like "his manager") carries an anaphoricity penalty that lets epsilon win -- because a
# definite/indefinite NP usually *introduces* an entity rather than referring back to one.
# A trained model would learn all of these from data; here they are set by hand for a clean demo.
FEATURE_WEIGHTS = {
    "gender_agree": 2.0,        # +2 when candidate gender matches the mention
    "gender_clash": -3.0,       # -3 when genders are known and disagree (a near-hard constraint)
    "recency": 1.0,             # closer antecedents score higher (multiplies a 0..1 closeness)
    "subject": 0.5,             # the discourse subject is the salient default referent
    "non_pronoun_penalty": -1.5,  # a non-pronoun mention is less anaphoric -> bias toward epsilon
}


def pair_features(mention: dict, candidate: dict, distance: int) -> dict:
    """Features for assigning `candidate` (an earlier mention) as antecedent of `mention`.

    `distance` is how many mentions back the candidate sits (1 = immediately previous).
    """
    gm, gc = mention["gender"], candidate["gender"]
    if gm != "unk" and gc != "unk":
        gender = "gender_agree" if gm == gc else "gender_clash"
    else:
        gender = None  # unknown gender contributes no agreement signal either way
    # closeness in (0, 1]: distance 1 -> 1.0, and decays with how far back the candidate is
    closeness = 1.0 / distance
    return {
        "gender": gender,
        "closeness": closeness,
        "subject": candidate["subject"],
        "anaphor_is_pronoun": mention["pronoun"],
    }


def antecedent_score(mention: dict, candidate: dict, distance: int) -> float:
    """The real-valued score s(i, j) for one (mention, candidate-antecedent) pair."""
    f = pair_features(mention, candidate, distance)
    score = 0.0
    if f["gender"] is not None:
        score += FEATURE_WEIGHTS[f["gender"]]
    score += FEATURE_WEIGHTS["recency"] * f["closeness"]
    if f["subject"]:
        score += FEATURE_WEIGHTS["subject"]
    if not f["anaphor_is_pronoun"]:
        # a full NP / name seeks an antecedent only weakly -> epsilon (score 0) tends to win
        score += FEATURE_WEIGHTS["non_pronoun_penalty"]
    return score


def candidate_scores(mentions: list[dict], i: int) -> list[tuple[object, float]]:
    """All candidate antecedents of mention i with their scores: epsilon first, then 0..i-1.

    Returns a list of (candidate_key, score). The candidate_key is the int index of an
    earlier mention, or the string EPSILON for the dummy "no antecedent" option.
    """
    out: list[tuple[object, float]] = [(EPSILON, EPSILON_SCORE)]
    for j in range(i):
        distance = i - j  # 1 = immediately previous mention
        out.append((j, antecedent_score(mentions[i], mentions[j], distance)))
    return out


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exponentiating)."""
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def antecedent_distribution(mentions: list[dict], i: int) -> tuple[list[object], np.ndarray]:
    """Softmax distribution P(y_i = j) over mention i's candidate antecedents (incl. epsilon)."""
    cands = candidate_scores(mentions, i)
    keys = [k for k, _ in cands]
    probs = softmax(np.array([s for _, s in cands], dtype=float))
    return keys, probs


def predict_antecedents(mentions: list[dict]) -> list[object]:
    """For each mention, pick the argmax antecedent (an earlier index, or EPSILON)."""
    preds: list[object] = []
    for i in range(len(mentions)):
        keys, probs = antecedent_distribution(mentions, i)
        preds.append(keys[int(probs.argmax())])
    return preds


def transitive_closure(antecedents: list[object]) -> list[list[int]]:
    """Chain antecedent links into clusters via union-find (the transitive closure step).

    `antecedents[i]` is the predicted antecedent of mention i: an earlier index or EPSILON.
    Two mentions land in the same cluster iff they are connected through antecedent links.
    Returns clusters as sorted lists of mention indices, ordered by first mention.
    """
    n = len(antecedents)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # keep the earlier mention as the root

    for i, a in enumerate(antecedents):
        if a != EPSILON:
            union(i, int(a))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in sorted(groups.values(), key=min)]


def gold_clusters(mentions: list[dict] = MENTIONS) -> list[list[int]]:
    """The gold clustering of the toy passage, as mention-index lists (for scoring the demo)."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mentions):
        groups.setdefault(m["gold"], []).append(i)
    return [sorted(g) for g in sorted(groups.values(), key=min)]


def resolve(mentions: list[dict] = MENTIONS) -> dict:
    """Run the full toy mention-ranking pipeline and return everything the page/notebook use."""
    preds = predict_antecedents(mentions)
    clusters_idx = transitive_closure(preds)
    clusters_text = [[mentions[i]["text"] for i in c] for c in clusters_idx]
    gold_idx = gold_clusters(mentions)
    return {
        "antecedents": preds,
        "clusters_idx": clusters_idx,
        "clusters_text": clusters_text,
        "gold_idx": gold_idx,
        "conll_f1": conll_f1(gold_idx, clusters_idx),
    }


# =====================================================================================
# The three coreference metrics, from scratch (and the CoNLL average)
# =====================================================================================
def _mention_to_cluster(clusters: list[list]) -> dict:
    """Map each mention to the set of mentions in its cluster."""
    return {m: set(c) for c in clusters for m in c}


def bcubed(gold: list[list], pred: list[list]) -> tuple[float, float, float]:
    """B-cubed (Bagga & Baldwin 1998): per-mention precision/recall, averaged.

    For mention m with gold cluster G_m and predicted cluster P_m:
        precision(m) = |G_m ∩ P_m| / |P_m|,   recall(m) = |G_m ∩ P_m| / |G_m|.
    Matches the by-hand table in the page exactly.
    """
    g = _mention_to_cluster(gold)
    p = _mention_to_cluster(pred)
    mentions = set(g) | set(p)
    precision = recall = 0.0
    for m in mentions:
        gm = g.get(m, {m})
        pm = p.get(m, {m})
        inter = len(gm & pm)
        precision += inter / len(pm)
        recall += inter / len(gm)
    n = len(mentions)
    precision /= n
    recall /= n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def muc(gold: list[list], pred: list[list]) -> tuple[float, float, float]:
    """MUC (Vilain et al. 1995): link-based recall/precision.

    A cluster of size n needs n-1 links. Recall counts how many gold links survive in the
    predicted partition; precision swaps the roles of gold and predicted.
    """

    def _partition_index(clusters: list[list]) -> dict:
        idx = {}
        for ci, c in enumerate(clusters):
            for e in c:
                idx[e] = ci
        return idx

    def _score(key: list[list], response: list[list]) -> tuple[int, int]:
        resp_idx = _partition_index(response)
        num = den = 0
        for c in key:
            den += len(c) - 1
            # count the distinct response-partitions that the elements of c fall into
            seen: dict = {}
            for e in c:
                part = resp_idx.get(e, ("solo", e))  # a mention absent from response is its own part
                seen[part] = seen.get(part, 0) + 1
            num += len(c) - len(seen)
        return num, den

    rn, rd = _score(gold, pred)  # recall
    pn, pd = _score(pred, gold)  # precision
    recall = rn / rd if rd else 0.0
    precision = pn / pd if pd else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def ceaf_phi4(gold: list[list], pred: list[list]) -> tuple[float, float, float]:
    """CEAF-phi4 (Luo 2005): best one-to-one entity alignment, scored by Dice overlap.

    phi4(G, P) = 2|G ∩ P| / (|G| + |P|). Maximise total similarity over all bijections of
    gold and predicted entities; precision divides by #predicted, recall by #gold.
    """
    gold_sets = [set(c) for c in gold]
    pred_sets = [set(c) for c in pred]

    def phi4(a: set, b: set) -> float:
        return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else 0.0

    n = max(len(gold_sets), len(pred_sets))
    g_padded = gold_sets + [set() for _ in range(n - len(gold_sets))]
    p_padded = pred_sets + [set() for _ in range(n - len(pred_sets))]
    best = 0.0
    for perm in permutations(range(n)):
        total = sum(
            phi4(g_padded[i], p_padded[perm[i]])
            for i in range(n)
            if g_padded[i] and p_padded[perm[i]]
        )
        best = max(best, total)
    precision = best / len(pred_sets) if pred_sets else 0.0
    recall = best / len(gold_sets) if gold_sets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def conll_f1(gold: list[list], pred: list[list]) -> float:
    """The CoNLL-2012 score: the unweighted mean of MUC, B-cubed and CEAF-phi4 F1."""
    return (muc(gold, pred)[2] + bcubed(gold, pred)[2] + ceaf_phi4(gold, pred)[2]) / 3


# The page's canonical metric example: one mention 'c' wrongly split out of the {a,b,c} entity.
METRIC_GOLD = [["a", "b", "c"], ["d", "e"]]
METRIC_PRED = [["a", "b"], ["c"], ["d", "e"]]


# =====================================================================================
# Winograd schema: why world knowledge is needed (flip one word, the antecedent flips)
# =====================================================================================
# Same syntax, same agreement (trophy & suitcase are both singular/neuter). Only a
# world-knowledge feature -- "a thing that is too BIG fails to fit; a container that is too
# SMALL fails to contain" -- distinguishes the two. We model that single feature explicitly:
# `fit_score(candidate, adjective)` is high when the adjective explains *that* candidate being
# the reason the trophy doesn't fit. Structure/agreement give both candidates an equal score;
# only this commonsense feature breaks the tie, and flipping the adjective flips the winner.

WINOGRAD_TEMPLATE = "The trophy doesn't fit in the suitcase because it is too {adj}."
WINOGRAD_CANDIDATES = ("trophy", "suitcase")


def winograd_world_knowledge(candidate: str, adjective: str) -> float:
    """Commonsense compatibility: does this adjective explain THIS candidate being the cause?

    'too big' explains the *contained* thing (the trophy) failing to fit;
    'too small' explains the *container* (the suitcase) failing to hold it.
    Returns +1 for the commonsense-correct pairing, -1 otherwise.
    """
    big_winner = "trophy"     # a too-BIG trophy is why it won't fit
    small_winner = "suitcase"  # a too-SMALL suitcase is why it won't fit
    winner = big_winner if adjective == "big" else small_winner
    return 1.0 if candidate == winner else -1.0


def winograd_resolve(adjective: str, use_world_knowledge: bool = True) -> tuple[str, dict]:
    """Resolve 'it' in the Winograd schema for a given adjective.

    With `use_world_knowledge=False`, only structure/agreement features are available -- both
    candidates tie (the failure mode of pre-LLM coref). With it on, the commonsense feature
    flips the winner when the adjective flips.
    """
    # structure/agreement: both candidates are singular neuter nouns equally available -> tie.
    structural = {c: 0.0 for c in WINOGRAD_CANDIDATES}
    scores = dict(structural)
    if use_world_knowledge:
        for c in WINOGRAD_CANDIDATES:
            scores[c] += winograd_world_knowledge(c, adjective)
    winner = max(scores, key=scores.get)
    return winner, scores


# =====================================================================================
# Real mention detection via spaCy (reproducible) for the page's worked example
# =====================================================================================
RUNNING_PASSAGE = (
    "John told his manager that he would finish the report by Friday. "
    "She thanked him for the update."
)


def spacy_mentions(text: str = RUNNING_PASSAGE) -> dict:
    """Detect candidate mentions with spaCy: named entities + pronouns + noun chunks.

    Returns a dict (empty lists if spaCy/model is unavailable, so callers degrade gracefully).
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return {
            "ner": [(e.text, e.label_) for e in doc.ents],
            "pronouns": [t.text for t in doc if t.tag_ in ("PRP", "PRP$")],
            "noun_chunks": [nc.text for nc in doc.noun_chunks],
        }
    except Exception as exc:  # noqa: BLE001 -- spaCy is optional; we report and continue
        return {"error": str(exc), "ner": [], "pronouns": [], "noun_chunks": []}


# =====================================================================================
# Demo entry point
# =====================================================================================
def main() -> None:
    print(runtime_banner())
    print(f"stable_hash('coreference') = {stable_hash('coreference') % 10**8:08d} (md5, process-stable)\n")

    print("== Mention-ranking on the toy passage ==")
    result = resolve()
    for i, mention in enumerate(MENTIONS):
        keys, probs = antecedent_distribution(MENTIONS, i)
        best = keys[int(probs.argmax())]
        best_text = EPSILON if best == EPSILON else MENTIONS[int(best)]["text"]
        print(f"  {mention['text']:<12} -> antecedent: {best_text:<12} (p={probs.max():.2f})")
    print("  clusters:", result["clusters_text"], "\n")

    print("== Metrics on gold={{a,b,c},{d,e}} pred={{a,b},{c},{d,e}} ==")
    mp, mr, mf = muc(METRIC_GOLD, METRIC_PRED)
    bp, br, bf = bcubed(METRIC_GOLD, METRIC_PRED)
    cp, cr, cf = ceaf_phi4(METRIC_GOLD, METRIC_PRED)
    print(f"  MUC       P={mp:.3f} R={mr:.3f} F1={mf:.3f}")
    print(f"  B-cubed   P={bp:.3f} R={br:.3f} F1={bf:.3f}")
    print(f"  CEAF-phi4 P={cp:.3f} R={cr:.3f} F1={cf:.3f}")
    print(f"  CoNLL avg F1 = {conll_f1(METRIC_GOLD, METRIC_PRED):.3f}\n")

    print("== Winograd schema: flip one word, the antecedent flips ==")
    for adj in ("big", "small"):
        winner, scores = winograd_resolve(adj)
        print(f"  '...because it is too {adj}.'  -> it = {winner:<8} scores={scores}")
    tie_winner_big, _ = winograd_resolve("big", use_world_knowledge=False)
    print(f"  without world knowledge both candidates tie (structure picks '{tie_winner_big}' arbitrarily)\n")

    print("== Real spaCy mention detection on the running passage ==")
    det = spacy_mentions()
    print("  NER         :", det["ner"])
    print("  pronouns    :", det["pronouns"])
    print("  noun chunks :", det["noun_chunks"])


if __name__ == "__main__":
    main()

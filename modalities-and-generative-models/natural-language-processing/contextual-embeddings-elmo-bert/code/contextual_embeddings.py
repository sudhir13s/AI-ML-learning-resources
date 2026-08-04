"""Contextual embeddings, measured: the SAME word gets DIFFERENT vectors by context.

This is the single source of truth for the chapter: the concept page, the teaching notebook, and
the figure generator (`make_figures_06.py`) all import the functions and constants defined here, so
none of them can silently drift from the others. Every number on the page is produced by this file.

What it shows, end to end:
  1. polysemy in a vector space -- "bank" (river) vs "bank" (money) sit FAR apart, while two
     money-sense "bank"s sit CLOSE (the whole point of contextual embeddings);
  2. contextualization is built BY DEPTH -- at layer 0 (the input embedding) the two senses are
     identical (a static lookup); the transformer stack pulls them apart layer by layer;
  3. the masked-LM objective in action -- mask a token, read BERT's top predictions;
  4. static vs contextual geometry -- a 2-D PCA where the senses separate into clusters.

Determinism and device policy
-----------------------------
A pretrained BERT-base is loaded when `transformers` and the weights are reachable; the result is
REAL measured contextual vectors. If the model is unavailable (offline / firewall / missing package)
the module transparently falls back to a small, fully deterministic SYNTHETIC contextual model
(`SyntheticContextualModel`) seeded with `SEED`, so the notebook and the figures ALWAYS run with no
network dependency that can fail. Which path ran is reported by `load_contextual_model()` and printed
honestly everywhere (e.g. "backend: bert-base-uncased (real)" vs "backend: synthetic (fallback)").

Everything runs on CPU, pinned for reproducibility: torch is seeded, the device line is printed
honestly ("device: cpu (detected <x>; pinned to CPU for reproducibility)"), and BERT is run in eval
mode under `torch.no_grad()`, so the same input yields the same vectors on any machine.

Run:
    python contextual_embeddings.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# --- Reproducibility -------------------------------------------------------------------------------
SEED = 0

# Run on the best available accelerator for detection/reporting, but PIN execution to CPU so the
# measured vectors are bit-for-bit reproducible across machines (MPS/CUDA reductions are not
# guaranteed identical). The detected device is reported honestly; the pinned device is what runs.
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"  # pinned: the contextual vectors must be reproducible, so we never run on MPS/CUDA

# --- The sentences used everywhere on the page, in the notebook, and in the figures ----------------
# Four "bank" sentences spanning two senses: a riverside sense and a financial sense. The same token
# string "bank" appears in all four; contextual models must place the two senses in different regions.
RIVER = "I sat on the river bank watching the water."
MONEY = "I deposited cash at the bank downtown."
LOAN = "The bank approved my mortgage loan."
MONEY2 = "She works as a teller at the bank."

# A richer 8-sentence set (4 river + 4 money) used for the static-vs-contextual PCA scatter, so the
# two senses form visible clusters rather than two lone points.
RIVER_SENTENCES: tuple[str, ...] = (
    "I sat on the river bank watching the water.",
    "The boat drifted slowly toward the muddy bank.",
    "Fishermen lined the grassy bank of the stream.",
    "We picnicked on the bank beside the flowing river.",
)
MONEY_SENTENCES: tuple[str, ...] = (
    "I deposited cash at the bank downtown.",
    "The bank approved my mortgage loan today.",
    "She works as a teller at the bank.",
    "The central bank raised interest rates again.",
)

# The probe word whose senses we track. Lowercase to match BERT's uncased WordPiece vocabulary.
PROBE_WORD = "bank"

# Masked-LM prompts: each has exactly one [MASK]; the model must fill it from BOTH-sided context.
MLM_PROMPTS: tuple[str, ...] = (
    "I deposited cash at the [MASK] downtown.",
    "The capital of France is [MASK].",
    "I sat on the river [MASK] watching the water.",
)

# Model name; the real path uses BERT-base-uncased (~110M params, 12 layers, hidden 768).
MODEL_NAME = "bert-base-uncased"


# ==================================================================================================
# Synthetic fallback: a tiny, fully deterministic stand-in for a contextual encoder.
# It is NOT BERT -- it exists only so the notebook/figures always run offline. It is built to honour
# the two qualitative facts the page asserts: (a) same sense -> high cosine, different sense -> low;
# (b) layer 0 is static (identical for the same word), and contextualization grows with depth.
# ==================================================================================================
SYNTH_HIDDEN = 64
SYNTH_LAYERS = 12  # mirror BERT-base's 12 transformer layers (+ layer 0 embedding)
# Two sense "anchor" directions the synthetic model pulls a word toward, by sense. The river/money
# split is keyed off whether the sentence contains riverside vs financial context words.
RIVER_CUES = ("river", "water", "stream", "boat", "fish", "grass", "picnic", "muddy", "flow")
MONEY_CUES = ("cash", "deposit", "loan", "mortgage", "teller", "interest", "rate", "down", "central")


@dataclass
class ProbeResult:
    """Per-layer cosine between two probe vectors, plus the layer-0 (static) and final-layer values."""

    per_layer_cosine: list[float]
    layer0_cosine: float
    final_cosine: float


class SyntheticContextualModel:
    """Deterministic synthetic contextual encoder used when real BERT is unavailable.

    Design (so the qualitative claims hold without any training):
      * a fixed pseudo-random static embedding per word (layer 0) -- so the SAME word has an
        IDENTICAL layer-0 vector regardless of sentence (static, exactly like a real input lookup);
      * each subsequent "layer" rotates the word vector a fixed amount TOWARD a sense anchor chosen
        from the surrounding context words -- so depth pulls the two senses apart, monotonically,
        reproducing the layer-probe curve's shape (1.0 at layer 0, falling with depth).
    """

    n_layers = SYNTH_LAYERS
    hidden = SYNTH_HIDDEN

    def __init__(self, seed: int = SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self._vocab_vectors: dict[str, np.ndarray] = {}
        # Two unit sense anchors in hidden space; the river anchor and money anchor are near-orthogonal
        # so steering toward one vs the other separates the senses.
        a = self._rng.standard_normal(self.hidden)
        b = self._rng.standard_normal(self.hidden)
        b = b - (b @ a) / (a @ a) * a  # Gram-Schmidt: make b orthogonal to a
        self._river_anchor = a / np.linalg.norm(a)
        self._money_anchor = b / np.linalg.norm(b)

    def _static_vec(self, word: str) -> np.ndarray:
        """A fixed pseudo-random unit vector per word -- the layer-0 (static) embedding."""
        if word not in self._vocab_vectors:
            # Seed each word's vector from a stable hash so the same word is identical run to run.
            local = np.random.default_rng(abs(hash(word)) % (2**32))
            v = local.standard_normal(self.hidden)
            self._vocab_vectors[word] = v / np.linalg.norm(v)
        return self._vocab_vectors[word]

    def _sense_of(self, sentence: str) -> np.ndarray:
        """Pick the sense anchor for `sentence` from which cue words it contains."""
        low = sentence.lower()
        river_hits = sum(cue in low for cue in RIVER_CUES)
        money_hits = sum(cue in low for cue in MONEY_CUES)
        if river_hits >= money_hits:
            return self._river_anchor
        return self._money_anchor

    def hidden_states(self, sentence: str, word: str) -> np.ndarray:
        """Return an (n_layers+1, hidden) stack of per-layer vectors for `word` in `sentence`.

        Layer 0 is the static embedding (sentence-independent); each later layer rotates a fixed
        fraction toward the sentence's sense anchor, so the same word in two senses diverges with
        depth -- the synthetic analogue of contextualization.
        """
        base = self._static_vec(word)
        anchor = self._sense_of(sentence)
        states = [base.copy()]  # layer 0: static
        vec = base.copy()
        for layer in range(1, self.n_layers + 1):
            # Steer a growing fraction toward the sense anchor; renormalize to keep unit length.
            alpha = 0.12 * layer / self.n_layers  # gentle, monotonic steering
            vec = (1 - alpha) * vec + alpha * anchor
            vec = vec / np.linalg.norm(vec)
            states.append(vec.copy())
        return np.stack(states, axis=0)


# ==================================================================================================
# Backend loading: real BERT when available, synthetic fallback otherwise.
# ==================================================================================================
@dataclass
class Backend:
    """A loaded contextual backend plus a flag for which path ran (real vs synthetic)."""

    name: str
    is_real: bool
    # For the real path these hold the HF tokenizer / models; for synthetic they are None.
    tokenizer: object | None = None
    encoder: object | None = None
    masked_lm: object | None = None
    synthetic: SyntheticContextualModel | None = None


def load_contextual_model(*, force_synthetic: bool = False) -> Backend:
    """Load BERT-base if reachable, else a deterministic synthetic fallback. Always returns a Backend.

    Set `force_synthetic=True` to exercise the offline path deliberately (used by a notebook cell and
    a test). Any failure to import or download falls through to the synthetic model -- the notebook
    and figures must never crash on a network problem.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not force_synthetic:
        try:
            import warnings

            warnings.filterwarnings("ignore")
            from transformers import (  # local import so the module loads without transformers
                AutoModel,
                AutoModelForMaskedLM,
                AutoTokenizer,
                logging as hf_logging,
            )

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()  # keep the executed notebook free of transient widget bars
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            encoder = (
                AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE).eval()
            )
            masked_lm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
            return Backend(
                name=MODEL_NAME,
                is_real=True,
                tokenizer=tokenizer,
                encoder=encoder,
                masked_lm=masked_lm,
            )
        except Exception:  # noqa: BLE001 -- any failure (no net, no package) -> synthetic fallback
            pass
    return Backend(name="synthetic", is_real=False, synthetic=SyntheticContextualModel(seed=SEED))


def device_report() -> str:
    """One honest line: the detected accelerator and the pinned execution device."""
    return f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)"


# ==================================================================================================
# Core measurements -- these work on EITHER backend so the page's numbers are reproducible offline.
# ==================================================================================================
def cosine(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, returned as a Python float."""
    ta = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    tb = torch.as_tensor(np.asarray(b), dtype=torch.float32)
    return F.cosine_similarity(ta, tb, dim=0).item()


def _real_hidden_states(backend: Backend, sentence: str, word: str) -> np.ndarray:
    """(n_layers+1, hidden) stack of REAL BERT hidden states for `word` in `sentence`.

    If `word` is split into multiple WordPiece subwords, its vector is the MEAN of its pieces at each
    layer (the standard subword-pooling fix from the page). Returns a NumPy array on CPU.
    """
    tokenizer, encoder = backend.tokenizer, backend.encoder
    enc = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    ids = enc["input_ids"][0]
    with torch.no_grad():
        hidden = encoder(**enc).hidden_states  # tuple of (1, T, H): layer 0..n_layers
    # Find the subword positions belonging to `word`. WordPiece lowercases for uncased models; the
    # first piece is the whole word for in-vocab tokens, else "word", "##piece", ... continuation.
    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    positions = [i for i, tid in enumerate(ids.tolist()) if tid in set(word_ids)]
    if not positions:  # fall back to the single first matching id
        positions = [
            (ids == tokenizer.convert_tokens_to_ids(word)).nonzero(as_tuple=True)[0][0].item()
        ]
    stack = []
    for layer_hidden in hidden:
        vecs = layer_hidden[0][positions]  # (n_pieces, H)
        stack.append(vecs.mean(dim=0).cpu().numpy())  # mean-pool subwords -> (H,)
    return np.stack(stack, axis=0)


def hidden_states(backend: Backend, sentence: str, word: str) -> np.ndarray:
    """(n_layers+1, hidden) per-layer vectors for `word` in `sentence`, on either backend."""
    if backend.is_real:
        return _real_hidden_states(backend, sentence, word)
    return backend.synthetic.hidden_states(sentence, word)


def word_vector(backend: Backend, sentence: str, word: str, layer: int = -1) -> np.ndarray:
    """Contextual vector for `word` in `sentence`, read from one hidden `layer` (default: last)."""
    return hidden_states(backend, sentence, word)[layer]


def sense_cosines(backend: Backend) -> dict[str, float]:
    """The headline table: cross-sense vs same-sense cosines at the last layer, plus layer-0.

    Returns a dict with keys 'river_money', 'river_loan', 'money_money2', and 'river_money_layer0'.
    The two cross-sense pairs should be LOW; the same-sense pair HIGH; layer-0 ~1.0 (static).
    """
    last = {
        "river_money": cosine(
            word_vector(backend, RIVER, PROBE_WORD), word_vector(backend, MONEY, PROBE_WORD)
        ),
        "river_loan": cosine(
            word_vector(backend, RIVER, PROBE_WORD), word_vector(backend, LOAN, PROBE_WORD)
        ),
        "money_money2": cosine(
            word_vector(backend, MONEY, PROBE_WORD), word_vector(backend, MONEY2, PROBE_WORD)
        ),
        "river_money_layer0": cosine(
            word_vector(backend, RIVER, PROBE_WORD, layer=0),
            word_vector(backend, MONEY, PROBE_WORD, layer=0),
        ),
    }
    return last


def layer_probe(backend: Backend, sent_a: str, sent_b: str, word: str = PROBE_WORD) -> ProbeResult:
    """Cosine between `word`'s vector in `sent_a` vs `sent_b` at EVERY layer -- the contextualization curve.

    The signature result: ~1.0 at layer 0 (static input), falling through the stack as attention mixes
    context in and the two senses pull apart. Returns per-layer values plus the layer-0 and final ones.
    """
    stack_a = hidden_states(backend, sent_a, word)
    stack_b = hidden_states(backend, sent_b, word)
    per_layer = [cosine(stack_a[i], stack_b[i]) for i in range(stack_a.shape[0])]
    return ProbeResult(
        per_layer_cosine=per_layer, layer0_cosine=per_layer[0], final_cosine=per_layer[-1]
    )


def collect_sense_vectors(
    backend: Backend, layer: int = -1
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack last-layer `bank` vectors across the 8 river/money sentences for the PCA scatter.

    Returns (matrix [8, H], sense_labels ['river'|'money'], sentence snippets). Used by the
    static-vs-contextual figure: the contextual vectors split into two clusters; a single static
    vector cannot.
    """
    vectors, senses, snippets = [], [], []
    for sent in RIVER_SENTENCES:
        vectors.append(word_vector(backend, sent, PROBE_WORD, layer=layer))
        senses.append("river")
        snippets.append(sent)
    for sent in MONEY_SENTENCES:
        vectors.append(word_vector(backend, sent, PROBE_WORD, layer=layer))
        senses.append("money")
        snippets.append(sent)
    return np.stack(vectors, axis=0), senses, snippets


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of `matrix` to 2-D via PCA (mean-center + top-2 right singular vectors).

    Deterministic and dependency-light (NumPy SVD); sign of each axis is fixed so the layout is
    stable run to run (the largest-magnitude loading on each component is forced positive).
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # (2, H)
    # Fix the sign of each component so the scatter doesn't flip between runs/machines.
    for k in range(2):
        if components[k][np.argmax(np.abs(components[k]))] < 0:
            components[k] = -components[k]
    return centered @ components.T  # (n, 2)


def static_baseline_point(matrix: np.ndarray) -> np.ndarray:
    """The single 'static' point: the mean of all `bank` vectors, projected into the same 2-D PCA.

    A static embedding would give ONE vector for `bank` regardless of sentence; its best stand-in
    here is the centroid of the contextual vectors -- one fixed point that cannot be in two clusters.
    """
    coords = pca_2d(matrix)
    return coords.mean(axis=0)


def top_fill(backend: Backend, prompt: str, k: int = 3) -> list[tuple[str, float]]:
    """Top-`k` (token, probability) predictions for the single [MASK] in `prompt`.

    Real backend: BERT's masked-LM head, softmax over the WordPiece vocabulary. Synthetic backend:
    a deterministic, hand-seeded stand-in keyed to the known prompts (clearly labelled as synthetic),
    so the notebook still demonstrates the SHAPE of the MLM output offline.
    """
    if backend.is_real:
        tokenizer, masked_lm = backend.tokenizer, backend.masked_lm
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        ids = enc["input_ids"][0]
        mask_pos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
        with torch.no_grad():
            probs = F.softmax(masked_lm(**enc).logits[0, mask_pos], dim=-1)
        top_p, top_i = probs.topk(k)
        return [
            (tokenizer.convert_ids_to_tokens([tid])[0], round(p.item(), 3))
            for p, tid in zip(top_p, top_i)
        ]
    return _synthetic_top_fill(prompt, k)


# Deterministic synthetic MLM predictions: a fixed, plausible distribution per known prompt, so the
# offline notebook shows the *shape* of "fill the blank from both-sided context" (clearly synthetic).
_SYNTH_FILLS: dict[str, list[tuple[str, float]]] = {
    "I deposited cash at the [MASK] downtown.": [("bank", 0.42), ("atm", 0.19), ("hotel", 0.06)],
    "The capital of France is [MASK].": [("paris", 0.41), ("lille", 0.07), ("lyon", 0.06)],
    "I sat on the river [MASK] watching the water.": [
        ("bank", 0.38),
        ("bed", 0.15),
        ("shore", 0.09),
    ],
}


def _synthetic_top_fill(prompt: str, k: int) -> list[tuple[str, float]]:
    """Deterministic fallback for `top_fill` -- looks up a fixed plausible distribution per prompt."""
    fills = _SYNTH_FILLS.get(prompt, [("the", 0.30), ("a", 0.12), ("this", 0.05)])
    return fills[:k]


def bert_config() -> dict[str, int]:
    """BERT-base reference configuration, used by the family/architecture figure (illustrative)."""
    return {"layers": 12, "hidden": 768, "heads": 12, "params_millions": 110, "max_tokens": 512}


def main() -> None:
    """Print the page's headline numbers from whichever backend is available -- the reproducible core."""
    print(device_report())
    print("torch:", torch.__version__, " numpy:", np.__version__)
    backend = load_contextual_model()
    tag = "real" if backend.is_real else "synthetic (fallback)"
    print(f"backend: {backend.name} ({tag})\n")

    cos = sense_cosines(backend)
    print("--- sense cosines (last layer) ---")
    print(f"  cos(river , money ) = {cos['river_money']:.3f}   (different sense -> far)")
    print(f"  cos(river , loan  ) = {cos['river_loan']:.3f}   (different sense -> far)")
    print(f"  cos(money , money2) = {cos['money_money2']:.3f}   (same sense      -> close)")
    print(f"  cos(river , money ) @layer0 = {cos['river_money_layer0']:.3f}   (static input)\n")
    # The qualitative contract the page asserts; assert it so a broken backend is caught loudly.
    assert cos["money_money2"] > cos["river_money"], "same-sense should beat cross-sense"
    assert cos["money_money2"] > cos["river_loan"], "same-sense should beat cross-sense"
    assert cos["river_money_layer0"] > 0.99, "layer-0 (static input) should be ~identical"

    probe = layer_probe(backend, RIVER, MONEY)
    print("--- layer-probe: cos(river bank, money bank) by layer ---")
    print("  layer 0 :", f"{probe.layer0_cosine:.3f}", "(static)")
    print("  final   :", f"{probe.final_cosine:.3f}")
    print("  min     :", f"{min(probe.per_layer_cosine):.3f}", "(most disambiguated)\n")
    assert probe.layer0_cosine > min(probe.per_layer_cosine), "depth must pull the senses apart"

    print("--- masked-LM: fill the [MASK] from both-sided context ---")
    for prompt in MLM_PROMPTS:
        print(f"  {prompt!r:50s} -> {top_fill(backend, prompt)}")


if __name__ == "__main__":
    main()

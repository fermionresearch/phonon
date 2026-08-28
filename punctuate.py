#!/usr/bin/env python3
"""Opt-in punctuation + truecasing restoration for transcribed text.

This is a legacy pass from an earlier build whose decoder emitted normalized
text. The current model punctuates and cases natively, so it is not recommended.
It is a *display-only* polish pass: it never touches the raw hypothesis that
feeds accuracy measurements or benchmarks.

Two restorers live here:

``RuleRestorer``
    Zero-dependency, zero-download deterministic baseline: sentence-initial
    capitalization, ``i`` -> ``I`` (and its contractions), and one terminal mark
    chosen by a question heuristic.  Always word-identity safe.

``ModelRestorer``
    ONNX punctuation/capitalization/segmentation models from 1-800-BAD-CODE
    (Apache-2.0) driven through the ``punctuators`` package (Apache-2.0).
    Runs on CPU via onnxruntime; no network access after the one-time model
    download; no cloud inference.

Both are wrapped by a hard word-identity guard (:func:`words_match`).  If a
restorer ever changes, drops, merges or splits a word, the guard discards its
output and returns the model's original text unchanged.  A polish pass is not
allowed to alter what was recognized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Word-identity gate
# ---------------------------------------------------------------------------

# Everything the restorers are allowed to add or change.  Apostrophes are kept
# on purpose: V18 already emits them ("beggar's"), so silently dropping one
# would be a word change the founder can see.
_STRIP_CHARS = ".,?!;:—–-…\"“”‘’()[]{}<>/\\*_"
_STRIP_KEEP_APOS = str.maketrans({c: " " for c in _STRIP_CHARS})
_STRIP_WITH_APOS = str.maketrans({c: " " for c in _STRIP_CHARS + "'"})


def identity_words(text: str, *, keep_apostrophes: bool = True) -> List[str]:
    """Lowercased word sequence used for the hard identity comparison."""
    table = _STRIP_KEEP_APOS if keep_apostrophes else _STRIP_WITH_APOS
    return text.lower().translate(table).split()


def prepare(text: str) -> str:
    """Put V18 output into the form the restoration models were trained on.

    Measured on the 400-utterance gate set, V18's raw text is *inconsistently*
    cased and punctuated (231/400 contain a capital, 265/400 contain some mark).
    The 1-800-BAD-CODE models expect lowercase, unpunctuated input - the English
    and 47-language checkpoints even use a lowercase SentencePiece model, so an
    uppercase character becomes ``<unk>`` and the word is destroyed.  Feeding
    them already-punctuated text also produces doubled marks (``perfectly..``).

    So the pass deliberately discards V18's own sporadic punctuation/casing and
    re-predicts all of it consistently.  Apostrophes are preserved, because V18
    gets those right and they are part of the word.
    """
    return " ".join(text.lower().translate(_STRIP_KEEP_APOS).split())


# The English and XLM-R checkpoints have an <ACRONYM> post-label that writes a
# period after every character of a word: "un" -> "U.N.", "mlb" -> "M.L.B.".
# That is a *word-identity* change under our gate (the word splits into letters),
# so it is undone before the guard runs.  Genuine spaced initials ("J. R. R.")
# are unaffected because this only matches an unspaced run inside one token.
_ACRONYM_RUN = re.compile(r"(?<![A-Za-z.])((?:[A-Za-z]\.){2,})(?![A-Za-z])")


def repair(text: str) -> str:
    """Undo restoration artifacts that would otherwise break word identity."""
    return _ACRONYM_RUN.sub(lambda m: m.group(1).replace(".", ""), text)


def words_match(before: str, after: str, *, keep_apostrophes: bool = True) -> bool:
    return identity_words(before, keep_apostrophes=keep_apostrophes) == identity_words(
        after, keep_apostrophes=keep_apostrophes
    )


# ---------------------------------------------------------------------------
# Rule baseline
# ---------------------------------------------------------------------------

_QUESTION_OPENERS = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "am", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might", "have", "has", "had",
}
_I_FORMS = {
    "i", "i'm", "i've", "i'll", "i'd", "im", "ive", "ill", "id",
}
# Only the unambiguous ones; "ill" and "id" are real words, so they are not
# rewritten by the baseline.
_I_REWRITE = {"i": "I", "i'm": "I'm", "i've": "I've", "i'll": "I'll", "i'd": "I'd"}


def rule_restore(text: str) -> str:
    """Deterministic, dependency-free casing/terminal-punctuation baseline."""
    stripped = prepare(text)
    if not stripped:
        return stripped
    tokens = stripped.split()
    out: List[str] = []
    capitalize_next = True
    for token in tokens:
        low = token.lower()
        if low in _I_REWRITE:
            word = _I_REWRITE[low]
        elif capitalize_next:
            word = token[:1].upper() + token[1:]
        else:
            word = token
        capitalize_next = word.endswith((".", "?", "!"))
        out.append(word)
    result = " ".join(out)
    if not result.endswith((".", "?", "!")):
        result += "?" if tokens[0].lower() in _QUESTION_OPENERS else "."
    return result


# ---------------------------------------------------------------------------
# ONNX model restorer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    spe_filename: str
    model_filename: str
    license: str
    note: str


MODEL_SPECS = {
    # English-only, 210 MB ONNX.  Punctuation (. , ?), truecasing, sentence
    # boundary detection.  Default for this English dictation product.
    "english": ModelSpec(
        repo_id="1-800-BAD-CODE/punctuation_fullstop_truecase_english",
        spe_filename="spe_32k_lc_en.model",
        model_filename="punct_cap_seg_en.onnx",
        license="apache-2.0",
        note="English-only PCS, 210 MB",
    ),
    # 47-language, 233 MB ONNX.
    "multi": ModelSpec(
        repo_id="1-800-BAD-CODE/punct_cap_seg_47_language",
        spe_filename="spe_unigram_64k_lowercase_47lang.model",
        model_filename="punct_cap_seg_47lang.onnx",
        license="apache-2.0",
        note="47-language PCS, 233 MB",
    ),
    # XLM-RoBERTa based, 1.11 GB ONNX.  Best quality, worst size/latency.
    "xlmr": ModelSpec(
        repo_id="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
        spe_filename="sp.model",
        model_filename="model.onnx",
        license="apache-2.0",
        note="XLM-R PCS, 1112 MB",
    ),
}

DEFAULT_MODEL = "english"


class ModelRestorer:
    """ONNX punctuation + truecasing, guarded by the word-identity gate."""

    def __init__(self, name: str = DEFAULT_MODEL, *, providers: Sequence[str] | None = None):
        if name not in MODEL_SPECS:
            raise ValueError(
                f"unknown punctuation model {name!r}; choose from {sorted(MODEL_SPECS)}"
            )
        from punctuators.models import PunctCapSegModelONNX
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX

        spec = MODEL_SPECS[name]
        self.name = name
        self.spec = spec
        cfg = PunctCapSegConfigONNX(
            hf_repo_id=spec.repo_id,
            spe_filename=spec.spe_filename,
            model_filename=spec.model_filename,
        )
        self._model = PunctCapSegModelONNX(cfg, ort_providers=list(providers or ["CPUExecutionProvider"]))
        self.guard_fallbacks = 0

    def raw_batch(self, texts: Sequence[str]) -> List[str]:
        """Unguarded model output.  Used for benchmark scoring only."""
        payload = [prepare(t) for t in texts]
        keep = [i for i, t in enumerate(payload) if t]
        results = list(payload)
        if keep:
            raw = self._model.infer(texts=[payload[i] for i in keep], apply_sbd=True)
            for i, sentences in zip(keep, raw):
                results[i] = " ".join(s.strip() for s in sentences if s.strip())
        return results

    def restore_batch(self, texts: Sequence[str]) -> List[str]:
        """Guarded output.  On any word-identity violation the original V18 text
        is returned untouched - a display polish is never allowed to change what
        was recognized."""
        originals = [t.strip() for t in texts]
        candidates = [repair(c) for c in self.raw_batch(originals)]
        results = []
        for original, candidate in zip(originals, candidates):
            if not original:
                results.append(original)
            elif candidate and words_match(original, candidate):
                results.append(candidate)
            else:
                self.guard_fallbacks += 1
                results.append(original)
        return results

    def restore(self, text: str) -> str:
        return self.restore_batch([text])[0]


# ---------------------------------------------------------------------------
# Public entry point used by stt.py
# ---------------------------------------------------------------------------

class Punctuator:
    """Thin facade so the CLI can say ``--punctuate`` and forget the details."""

    def __init__(self, name: str = DEFAULT_MODEL):
        self.name = name
        self.rule_only = name == "rules"
        self._model = None if self.rule_only else ModelRestorer(name)

    @property
    def guard_fallbacks(self) -> int:
        return 0 if self._model is None else self._model.guard_fallbacks

    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if self._model is None:
            return rule_restore(text)
        polished = self._model.restore(text)
        # The models never insert "I" for a bare lowercase "i" that opens no
        # sentence; the cheap rule fixes that without risking word identity.
        polished = _fix_bare_i(polished)
        return polished


_BARE_I = re.compile(r"(?<![\w'])i(?=(?:'(?:m|ve|ll|d))?(?![\w']))")


def _fix_bare_i(text: str) -> str:
    return _BARE_I.sub("I", text)


def load(name: str = DEFAULT_MODEL) -> Punctuator:
    return Punctuator(name)


CHOICES = ("rules",) + tuple(MODEL_SPECS)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()
    punct = load(args.model)
    lines: Iterable[str] = [" ".join(args.text)] if args.text else sys.stdin
    for line in lines:
        line = line.rstrip("\n")
        if line.strip():
            print(punct(line))

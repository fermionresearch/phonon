"""Hotwords for Phonon-2's greedy TDT decoder: a vocabulary list the decoder favours when the audio is close.

Input: up to MAX_WORDS names or terms (strings, or ``{"word": ..., "spoken": [...]}`` dicts) and a strength
``lam``.  Output: per-step score bonuses that make the decoder prefer spelling a listed word when the audio
supports it.  Nothing is ever penalised, the blank symbol is never biased, and each word is also matched all
lower case and capitalised.

Shared by the Phonon-2 engines.  numpy only.  An empty list builds nothing, and the decoders then run their
unbiased loop unchanged.
"""
from __future__ import annotations

from collections import deque

import numpy as np

#: Logit-unit bonus per advancing piece (the `hotword_lambda` default).
DEFAULT_LAMBDA = 2.0
#: Words per decode; extra words are dropped (in the order given) with a note from the caller.
MAX_WORDS = 25
#: Segmentations kept per casing variant (shortest first).
MAX_SEGMENTATIONS = 6
#: Hard cap on automaton states; a 25-word list lands in the hundreds.
MAX_STATES = 8192
WORD_BOUNDARY = "▁"


def _is_special_piece(piece: str) -> bool:
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


def clean_words(words, limit: int | None = MAX_WORDS) -> tuple[list, int]:
    """Strings (or ``{"word": ..., "spoken": [...]}`` dicts) -> the distinct entries in order, whitespace collapsed,
    at most ``limit``.  Returns (entries, how many distinct entries were given)."""
    out, seen = [], set()
    for e in words or ():
        if isinstance(e, dict):
            w = " ".join(str(e.get("word", "")).split())
            spoken = [" ".join(str(x).split()) for x in (e.get("spoken") or []) if str(x).strip()]
            item = {"word": w, "spoken": spoken} if spoken else w
        else:
            w = " ".join(str(e).split()); item = w
        if w and w.lower() not in seen:
            seen.add(w.lower()); out.append(item)
    n = len(out)
    return (out[:limit] if limit is not None else out), n


def parse_list(text: str) -> list[str]:
    """A vocabulary list written as text: comma-separated when it has a comma (so multi-word terms survive:
    "Ada Lovelace, Quillon"), else whitespace-separated ("Ada Quillon Neutrino").  Newlines and semicolons
    separate like commas."""
    t = str(text or "").replace("\n", ",").replace(";", ",")
    parts = t.split(",") if "," in t else t.split()
    out: list[str] = []
    for p in parts:
        p = " ".join(p.split())
        if p and p not in out:
            out.append(p)
    return out


def casing_variants(phrase: str) -> list[str]:
    """as given, all lower, each word capitalised -- de-duplicated, order kept."""
    phrase = " ".join(str(phrase).split())
    if not phrase:
        return []
    cap = " ".join(w[:1].upper() + w[1:] for w in phrase.split())
    out = []
    for v in (phrase, phrase.lower(), cap):
        if v not in out:
            out.append(v)
    return out


class PieceIndex:
    """piece string -> id, specials excluded; plus the length bound for the segmenter."""

    def __init__(self, vocabulary):
        self.ids = {}
        for i, p in enumerate(vocabulary):
            if not _is_special_piece(p) and p not in self.ids:
                self.ids[p] = i
        self.max_len = max((len(p) for p in self.ids), default=1)
        self.size = len(vocabulary)


def segmentations(text: str, index: PieceIndex, limit: int = MAX_SEGMENTATIONS) -> list[tuple[int, ...]]:
    """Every concatenation of vocabulary pieces equal to ``text`` (word boundaries already written as U+2581),
    shortest first, at most ``limit``."""
    n = len(text)
    memo: dict[int, list[tuple[int, ...]]] = {n: [()]}

    def rec(pos: int) -> list[tuple[int, ...]]:
        if pos in memo:
            return memo[pos]
        out = []
        for end in range(min(n, pos + index.max_len), pos, -1):
            pid = index.ids.get(text[pos:end])
            if pid is None:
                continue
            for tail in rec(end):
                out.append((pid,) + tail)
                if len(out) >= 4 * limit:
                    break
            if len(out) >= 4 * limit:
                break
        out.sort(key=len)
        memo[pos] = out[:limit]
        return memo[pos]

    return rec(0)


def phrase_surface(phrase: str) -> str:
    return WORD_BOUNDARY + WORD_BOUNDARY.join(phrase.split())


class HotwordAutomaton:
    """Sparse Aho-Corasick tables over piece ids (states x trie pieces, not states x vocabulary).

    toks[k]      the K distinct piece ids that occur in any phrase (sorted)
    col[t]       column of piece t in toks, or -1 (every other piece leads back to the root); length vocab_size + 1
    table[s, k]  next state after piece toks[k] in state s (failure links folded in)
    depth[s]     pieces matched so far in state s; the bonus is lam * max(0, depth[next] - depth[s])
    """

    def __init__(self, words, vocabulary, *, lam: float = DEFAULT_LAMBDA, max_words: int = MAX_WORDS):
        self.lam = float(lam)
        index = PieceIndex(vocabulary)
        self.vocab_size = index.size                      # token scores are [0 .. vocab_size] (blank = vocab_size)
        entries, _ = clean_words(words, max_words)
        self.words = [e["word"] if isinstance(e, dict) else e for e in entries]
        seqs = set()
        for e in entries:
            surfaces = [e["word"]] + list(e["spoken"]) if isinstance(e, dict) else [e]
            for surface in surfaces:
                for v in casing_variants(surface):
                    for seq in segmentations(phrase_surface(v), index):
                        seqs.add(seq)
        self.phrases = sorted(seqs)
        goto = [dict()]; depth = [0]
        for seq in self.phrases:
            s = 0
            for t in seq:
                if t not in goto[s]:
                    goto.append({}); depth.append(depth[s] + 1)
                    goto[s][t] = len(goto) - 1
                s = goto[s][t]
        S = len(goto)
        if S > MAX_STATES:
            raise ValueError(f"hotword automaton too large: {S} states")
        fail = [0] * S; order = []
        q = deque(goto[0].values())
        while q:
            s = q.popleft(); order.append(s)
            for t, c in goto[s].items():
                f = fail[s]
                while f and t not in goto[f]:
                    f = fail[f]
                fail[c] = goto[f][t] if (t in goto[f] and goto[f][t] != c) else 0
                q.append(c)
        self.n_states = S
        toks = sorted({t for g in goto for t in g})
        K = max(1, len(toks)); colmap = {t: k for k, t in enumerate(toks)}
        table = np.zeros((S, K), dtype=np.int32)
        for t, c in goto[0].items():
            table[0, colmap[t]] = c
        for s in order:                                   # BFS order: table[fail[s]] is already complete
            table[s] = table[fail[s]]
            for t, c in goto[s].items():
                table[s, colmap[t]] = c
        col = np.full((self.vocab_size + 1,), -1, dtype=np.int32)
        for t, k in colmap.items():
            col[t] = k
        self.toks = np.array(toks if toks else [0], dtype=np.int32)
        self.col = col
        self.table = np.ascontiguousarray(table)
        self.depth = np.array(depth, dtype=np.int32)

    # -- reference math (numpy): the tests and the Python decode loops use these -------------------------------
    def bonus(self, state: int) -> np.ndarray:
        """(vocab_size + 1,) float32 bias on the token scores in ``state``; blank (the last entry) is always 0."""
        gain = (self.depth[self.table[state]] - self.depth[state]).astype(np.float32)   # (K,)
        b = np.zeros((self.vocab_size + 1,), dtype=np.float32)
        b[self.toks] = self.lam * np.maximum(gain, 0.0)
        b[self.vocab_size] = 0.0
        return b

    def step(self, state: int, token: int) -> int:
        """Next state after emitting ``token`` (blank or an out-of-range id leaves the state alone; a piece outside
        the automaton returns to the root)."""
        if token < 0 or token >= self.vocab_size:
            return state
        k = int(self.col[token])
        return int(self.table[state, k]) if k >= 0 else 0

    @property
    def empty(self) -> bool:
        return self.n_states <= 1

    def describe(self) -> dict:
        return {"words": list(self.words), "phrases": len(self.phrases), "states": self.n_states,
                "pieces": int(self.toks.shape[0]), "lambda": self.lam}


def build(words, vocabulary, *, lam: float = DEFAULT_LAMBDA, max_words: int = MAX_WORDS) -> HotwordAutomaton | None:
    """None when there is nothing to bias with (no words, no spellable word, lam <= 0): the decoder then takes its
    unbiased path, byte-identical to a decode without hotwords."""
    if not words or float(lam) <= 0:
        return None
    try:
        auto = HotwordAutomaton(words, vocabulary, lam=lam, max_words=max_words)
    except ValueError:
        return None
    return None if auto.empty else auto


class HotwordState:
    """The per-model hotword setting shared by the three Phonon-2 engines: the words, lambda, and a cached automaton
    for the model's vocabulary (rebuilt only when the words or lambda change)."""

    def __init__(self):
        self.words: list = []
        self.lam: float = DEFAULT_LAMBDA
        self._cache: dict = {}                            # (words, lam, vocab size) -> automaton, most recent last

    def set(self, words, lam: float | None = None) -> tuple[int, int]:
        """-> (active words, distinct words given)."""
        entries, given = clean_words(words, MAX_WORDS)
        self.words = entries
        if lam is not None:
            lam = float(lam)
            if not (lam >= 0) or lam > 100:
                raise ValueError(f"hotword lambda must be between 0 and 100, got {lam}")
            self.lam = lam
        return len(entries), given

    def automaton(self, vocabulary) -> HotwordAutomaton | None:
        if not self.words or self.lam <= 0:
            return None
        key = (repr(self.words), self.lam, len(vocabulary))
        if key in self._cache:
            auto = self._cache.pop(key)
        else:
            auto = build(self.words, vocabulary, lam=self.lam)
        self._cache[key] = auto                           # a server alternating a few lists keeps them all built
        while len(self._cache) > 32:
            self._cache.pop(next(iter(self._cache)))
        return auto

    def plain_words(self) -> list[str]:
        return [e["word"] if isinstance(e, dict) else e for e in self.words]


class Phonon2HotwordsMixin:
    """`set_hotwords` / per-call hotwords / describe fields for the Phonon-2 engines (MLX, CPU).  Mixed in ahead of
    `engine.SpeechModel`; the engine's `_decode_single` asks `_hotword_automaton(vocabulary)` for the tables
    (None = no hotwords = the unbiased loop)."""

    def _hw_state(self) -> HotwordState:
        st = getattr(self, "_hw", None)
        if st is None:
            st = self._hw = HotwordState()
        return st

    @property
    def hotword_lambda(self) -> float:
        return self._hw_state().lam

    def set_hotwords(self, words, strength=None, *, lam=None) -> int:
        """Names and terms the decoder should favour (at most MAX_WORDS; strings, or {"word", "spoken"} dicts whose
        spoken forms are favoured too).  `lam` is the bonus per matching piece (default 2.0); `strength` is
        Phonon-1's scale and is not used here.  [] turns hotwords off.  Returns how many words are active."""
        n, given = self._hw_state().set(words, lam)
        if given > n:
            import sys
            from fermion._brand import BRAND
            print(f"[{BRAND}] note: {given} hotwords given; the first {n} are used", file=sys.stderr)
        self.hotwords = self._hw_state().plain_words()
        return n

    def _hotword_snapshot(self):
        st = self._hw_state()
        return (list(st.words), st.lam)

    def _hotword_restore(self, snap) -> None:
        st = self._hw_state()
        st.words, st.lam = list(snap[0]), snap[1]
        self.hotwords = st.plain_words()

    def _hotword_automaton(self, vocabulary):
        return self._hw_state().automaton(vocabulary)

    def _hotword_describe(self) -> dict:
        st = self._hw_state()
        return {"hotwords": st.plain_words(), "hotword_lambda": st.lam if st.words else None,
                "hotword_bias": f"bonus per matching piece on the decoder's token scores; blank never biased; "
                                f"at most {MAX_WORDS} words"}

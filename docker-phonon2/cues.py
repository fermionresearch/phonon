"""Transcript segments from word timings.

The decoder times every word; this module groups those words into sentence-
and pause-sized segments with absolute start and end times, a unit that is
easy to cite, caption or align against.

THE RULE (the same cue rules a subtitle export uses, so a segment and an
SRT / VTT cue are cut at the same places):

1. A segment closes after a word that ends a sentence (". ? !", optionally
   followed by a closing quote or bracket) and before a pause of at least
   `CUE_PAUSE_S` between two words.
2. A segment longer than `CUE_MAX_S`, or wider than `CUE_LINES` lines of
   `CUE_LINE_CHARS` characters (wrapped at spaces), is split at its best
   point: clause punctuation, then the widest pause, then the middle;
   recursively, until every piece fits. A word is never split.
3. A segment shorter than `CUE_MIN_S` is merged with a neighbour when the
   merge still fits and no real pause separates them.
4. A segment's `start` is its first word's start; its `end` is its last
   word's end, held to at least `CUE_MIN_S` after the start but never into
   the next segment and never past the end of the audio it came from.

INVARIANTS (tested): segments are in time order and never overlap; every
segment has `end > start`; joining the segment texts with single spaces
gives back the transcript, byte for byte. Segments are built inside each
decoded window, so they never cross a window boundary; when a window's
words do not join back to the window's text (no decoder does this in
practice), or when a degenerate timing would break an invariant, that
window becomes one segment with the window's own text and bounds, so the
transcript identity and the ordering can never break.
"""
from __future__ import annotations

import re
import zlib

#: Characters per line and lines per segment (a subtitle's two lines).
CUE_LINE_CHARS = 42
CUE_LINES = 2
#: Shortest and longest segment, in seconds.
CUE_MIN_S = 1.0
CUE_MAX_S = 7.0
#: A gap between two words at least this long closes a segment.
CUE_PAUSE_S = 0.8

_SENTENCE_END = re.compile(r"[.?!][\"'”’)\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–][\"'”’)\]]*$")


def _wrap_lines(text: str, width: int = CUE_LINE_CHARS) -> list:
    """Greedy wrap at spaces; a word longer than `width` stands alone on its line (never split)."""
    lines, cur = [], ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fits(words: list) -> bool:
    if words[-1]["end"] - words[0]["start"] > CUE_MAX_S + 1e-9:
        return False
    return len(_wrap_lines(" ".join(w["text"] for w in words))) <= CUE_LINES


def _split(words: list) -> list:
    """Cut a run of timed words into runs that each fit: at the best point
    (clause punctuation, then the widest pause, then balance), recursively.
    A single word always fits."""
    if len(words) <= 1 or _fits(words):
        return [words]
    n = len(words)
    best_k, best_score = 1, None
    for k in range(1, n):
        gap = max(0.0, words[k]["start"] - words[k - 1]["end"])
        score = (2.0 if _CLAUSE_END.search(words[k - 1]["text"]) else 0.0) \
            + 3.0 * min(gap, CUE_PAUSE_S) / CUE_PAUSE_S \
            - abs(k - n / 2.0) / (n / 2.0)
        if best_score is None or score > best_score:
            best_k, best_score = k, score
    return _split(words[:best_k]) + _split(words[best_k:])


def build_cues(words: list, limit: float | None = None) -> list:
    """`[{start, end, text}]` from `[{text, start, end}]` words (seconds, in
    order). `limit` is the latest time any segment may end (the end of the
    window the words came from); a segment never ends after it."""
    if not words:
        return []
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 0.0
        if _SENTENCE_END.search(w["text"]) or gap >= CUE_PAUSE_S:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    runs = []
    for g in groups:
        runs.extend(_split(g))
    # Merge a short run into a neighbour when the merge still fits and no real pause separates them.
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for i, run in enumerate(runs):
            if run[-1]["end"] - run[0]["start"] >= CUE_MIN_S:
                continue
            for j in (i + 1, i - 1):
                if not 0 <= j < len(runs):
                    continue
                lo, hi = min(i, j), max(i, j)
                left, right = runs[lo], runs[hi]
                if right[0]["start"] - left[-1]["end"] < CUE_PAUSE_S and _fits(left + right):
                    runs[lo] = left + right
                    del runs[hi]
                    merged = True
                    break
            if merged:
                break
    cues = []
    for i, run in enumerate(runs):
        start = run[0]["start"]
        end = max(run[-1]["end"], start + CUE_MIN_S)          # a short segment is held for CUE_MIN_S ...
        if i + 1 < len(runs):
            nxt = runs[i + 1][0]["start"]
            if nxt > start:
                end = min(end, nxt)                           # ... but never into the next one
            else:
                end = max(run[-1]["end"], start + 0.001)
        if limit is not None:
            end = min(end, limit)                             # ... and never past the audio
            start = min(start, limit - 0.001)                 # (a zero-length word decoded on the last frame)
        cues.append({"start": round(start, 3), "end": round(end, 3),
                     "text": " ".join(w["text"] for w in run)})
    return monotone(cues)


def monotone(segments: list) -> list:
    """In place: no segment starts before the one before it ends (rounding,
    or two words decoded on the same frame, must not create an overlap) and
    every segment keeps a positive length."""
    for i, c in enumerate(segments):
        if i:
            p = segments[i - 1]
            if c["start"] < p["end"]:
                if c["start"] > p["start"]:
                    p["end"] = c["start"]
                else:
                    c["start"] = p["end"]
        if c["end"] <= c["start"]:
            c["end"] = round(c["start"] + 0.001, 3)
    return segments


def window_segments(words: list, text: str, start: float, end: float) -> list:
    """Segments for one decoded window: its words (file time) grouped by
    `build_cues`, or, when the words do not join back to the window's `text`,
    the whole window as one segment. An empty window gives no segment."""
    text = (text or "").strip()
    if not text:
        return []
    lo, hi = round(start, 3), round(end, 3)
    if words and " ".join(w["text"] for w in words) == text:
        cues = build_cues(words, limit=hi)
        if _valid(cues, lo, hi):
            return cues
    return [{"start": lo, "end": hi, "text": text}]


def _valid(segments: list, lo: float, hi: float) -> bool:
    """In order, no overlap, positive lengths, inside [lo, hi]."""
    prev_end = lo
    for s in segments:
        if s["start"] < prev_end - 1e-9 or not s["end"] > s["start"] or s["end"] > hi + 1e-9:
            return False
        prev_end = s["end"]
    return bool(segments)


def _compression_ratio(text: str) -> float:
    """Whisper's definition: UTF-8 bytes over zlib-compressed bytes."""
    data = text.encode("utf-8")
    return round(len(data) / len(zlib.compress(data)), 3) if data else 0.0


def openai_segments(segments: list) -> list:
    """The OpenAI `verbose_json` segment shape, every field typed as the
    OpenAI SDKs declare it (numbers, never null, so a client that validates
    the response, or compares `no_speech_prob` / `avg_logprob` against a
    threshold, works unmodified). `seek` is the segment's start in 10 ms
    frames; `compression_ratio` is computed from the text as Whisper does;
    the decoder reports no log-probabilities, so `avg_logprob` is 0, and a
    segment only exists where words were decoded, so `no_speech_prob` is 0;
    `tokens` is an empty list and `temperature` 0 (decoding is greedy)."""
    return [{"id": i, "seek": int(round(s["start"] * 100)),
             "start": s["start"], "end": s["end"], "text": s["text"],
             "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
             "compression_ratio": _compression_ratio(s["text"]),
             "no_speech_prob": 0.0}
            for i, s in enumerate(segments)]

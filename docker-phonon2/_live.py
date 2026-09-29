"""Rolling live-transcription session for the CUDA runtime.

A port of the Mac engine's `fermion._speech.live.LiveSession` — the same
constants, the same energy-gated segmentation, the same partial cadence —
with the decode call swapped for the container's gated dense Torch path.
The session logic is deliberately kept line-for-line comparable with the Mac
module so the two backends segment identically.

ONE SESSION, TWO FEEDS: the long-file transcription path and the WebSocket
streaming endpoint both push audio through `feed_pcm()`, which re-blocks
arbitrary-sized input into fixed 50 ms blocks before segmentation. Identical
audio therefore produces the identical block sequence — and, with the
deterministic gated decode, identical finals — regardless of how the bytes
arrived. That is what makes the streaming `done` transcript equal the file
transcript byte for byte.

Partial decodes never mutate the segment buffer, so a session with partials
disabled (the file path) finalizes exactly the same segments as one with
partials enabled (the streaming path).
"""
from __future__ import annotations

SAMPLE_RATE = 16_000

# ------------------------------------------------------------- the cadence
# Matches the fermion CLI's live-dictation cadence (fermion/_speech/live.py).
#: Audio in the segment before the FIRST partial decode.
FIRST_PARTIAL_S = 0.35
#: Audio added between partial decodes after that.
PARTIAL_EVERY_S = 0.50
#: Sub-threshold trailing audio that finalizes the segment ("they stopped").
SILENCE_CLOSE_S = 0.70
#: A segment is finalized at this length no matter what.
MAX_SEGMENT_S = 30.0
#: RMS at or below this is room tone (the app's CONVERSATION_NOISE_FLOOR)...
NOISE_FLOOR_RMS = 0.004
#: ...as is anything this far under the segment's own peak (its GATE_RATIO).
GATE_RATIO = 0.18
#: Feed granularity: 50 ms blocks, for files and streams alike.
BLOCK_S = 0.05

_BLOCK = int(BLOCK_S * SAMPLE_RATE)


class LiveSession:
    """Rolling-decode state for one long-file or streaming session.

    `transcribe` is a callable `float32 mono 16 kHz array -> str` (the gated
    decode). `on_partial(text)` fires with the CURRENT WHOLE HYPOTHESIS for
    the in-flight segment (each call replaces the last); `on_final(text)`
    fires once per finalized segment. Both receive stripped, non-empty text
    only. `transcript` joins the finals with single spaces.
    """

    def __init__(self, transcribe, *, on_partial=None, on_final=None,
                 partials: bool = True):
        self._transcribe = transcribe
        self.on_partial = on_partial or (lambda text: None)
        self.on_final = on_final or (lambda text: None)
        self.partials = partials
        self.finals: list[str] = []
        self._tail = None  # sub-block remainder of arbitrary-size input
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._blocks: list = []       # every block of the in-flight segment
        self._samples = 0             # segment length, in samples
        self._voiced = False          # has anything above the gate been heard
        self._last_voice = 0          # sample count when voice was last heard
        self._peak_rms = 0.0          # loudest block so far (adaptive gate)
        self._next_partial = int(FIRST_PARTIAL_S * SAMPLE_RATE)

    # ------------------------------------------------------------- feeding
    def feed_pcm(self, samples) -> None:
        """Accept ANY amount of 16 kHz mono float32 audio.

        Input is re-blocked into fixed 50 ms blocks so segmentation cannot
        depend on how a client framed its audio.
        """
        import numpy as np

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        if self._tail is not None and self._tail.size:
            samples = np.concatenate((self._tail, samples))
        n_whole = (samples.size // _BLOCK) * _BLOCK
        for i in range(0, n_whole, _BLOCK):
            self._feed_block(samples[i:i + _BLOCK])
        self._tail = samples[n_whole:]

    def _feed_block(self, block) -> None:
        """One <= 50 ms block — the Mac `LiveSession.feed`, verbatim logic."""
        import numpy as np

        if block.size == 0:
            return
        self._blocks.append(block)
        self._samples += block.size

        rms = float(np.sqrt(np.mean(block * block)))
        self._peak_rms = max(self._peak_rms, rms)
        gate = max(NOISE_FLOOR_RMS, GATE_RATIO * self._peak_rms)
        if rms > gate:
            self._voiced = True
            self._last_voice = self._samples

        trailing = self._samples - self._last_voice
        if self._voiced and trailing >= int(SILENCE_CLOSE_S * SAMPLE_RATE):
            self._finalize()
        elif self._samples >= int(MAX_SEGMENT_S * SAMPLE_RATE):
            if self._voiced:
                self._finalize()
            else:
                # Pure room tone: keep the last second so a word that starts
                # at the boundary keeps its onset; drop the rest.
                keep = np.concatenate(self._blocks)[-SAMPLE_RATE:]
                self._reset_segment()
                self._blocks = [keep]
                self._samples = keep.size
        elif (self.partials and self._voiced
              and self._samples >= self._next_partial):
            text = self._transcribe(np.concatenate(self._blocks))
            if text:
                self.on_partial(text)
            # Self-clocking: schedule off the buffer as it stands NOW, after
            # the decode, so slow re-decodes of a long segment cannot queue.
            self._next_partial = self._samples + int(
                PARTIAL_EVERY_S * SAMPLE_RATE)

    # ---------------------------------------------------------- finalizing
    def _finalize(self) -> None:
        import numpy as np

        audio = np.concatenate(self._blocks)
        self._reset_segment()
        text = self._transcribe(audio)
        if text:
            self.finals.append(text)
            self.on_final(text)

    def finish(self) -> str:
        """Flush the sub-block tail, finalize what is in flight, return the
        full transcript."""
        if self._tail is not None and self._tail.size:
            tail, self._tail = self._tail, None
            self._feed_block(tail)
        if self._samples and self._voiced:
            self._finalize()
        else:
            self._reset_segment()
        return self.transcript

    @property
    def transcript(self) -> str:
        return " ".join(self.finals)

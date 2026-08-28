"""Optimized inference path for Phonon artifacts.

This module never rewrites weights.  Its default behaviour is one execution
change, verified byte-identical to the reference path on a 400-utterance gate:

* prompt prefill without projecting every audio position through the 151,936-row
  tied vocabulary head; only the final prompt position needs logits.

Two further changes stay opt-in and off by default:

* ``audio_blocks`` independently encodes/caches closed 8-second audio-attention
  blocks — small gain, transcript drift;
* ``compiled_decode`` runs a shapeless compiled fixed-capacity one-token decoder
  step — exact but measurably slower.

The optional fused QMV flag remains experimental and is gated separately.

Batch and streaming decoding share a single ``_iter_tokens`` generator, so a
streamed transcript is byte-identical to the batch transcript by construction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    StreamingResult,
    split_audio_into_chunks,
)
from mlx_audio.stt.utils import load_audio
from mlx_lm.sample_utils import make_logits_processors

from v18_runtime import load_v18


@dataclass
class AudioCacheStats:
    hits: int = 0
    misses: int = 0


class ClosedBlockAudioCache:
    """Cache independent Qwen audio-attention blocks for a growing utterance.

    Qwen's audio tower chunks 10-ms mel frames in groups of 100 and applies a
    block-diagonal attention mask across eight such chunks.  A completed
    800-frame block can therefore be reused when later microphone partials add
    audio to the tail.  Short utterances keep exactly one block.
    """

    feature_block = 800

    def __init__(self):
        self._entries: dict[int, tuple[bytes, mx.array]] = {}
        self._last_frames = 0
        self.stats = AudioCacheStats()

    def reset(self) -> None:
        self._entries.clear()
        self._last_frames = 0
        self.stats = AudioCacheStats()

    @staticmethod
    def _fingerprint(features: np.ndarray) -> bytes:
        return hashlib.sha256(features.tobytes(order="C")).digest()

    def encode(self, model, input_features: mx.array, attention_mask: mx.array) -> mx.array:
        frames = int(np.asarray(attention_mask.sum(axis=-1))[0])
        if frames < self._last_frames:
            self.reset()
        self._last_frames = frames

        outputs: list[mx.array] = []
        live_indices: set[int] = set()
        for index, start in enumerate(range(0, frames, self.feature_block)):
            end = min(frames, start + self.feature_block)
            block_np = np.asarray(input_features[:, :, start:end])
            fingerprint = self._fingerprint(block_np)
            cached = self._entries.get(index)
            if cached is not None and cached[0] == fingerprint:
                encoded = cached[1]
                self.stats.hits += 1
            else:
                block = mx.array(block_np)
                block_mask = mx.ones((1, end - start), dtype=attention_mask.dtype)
                encoded = model.get_audio_features(block, block_mask)
                mx.eval(encoded)
                self._entries[index] = (fingerprint, encoded)
                self.stats.misses += 1
            outputs.append(encoded)
            live_indices.add(index)

        for index in tuple(self._entries):
            if index not in live_indices:
                del self._entries[index]
        return mx.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]


class CompiledFixedCapacityDecoder:
    """Compiled one-token decoder backed by fixed-capacity KV arrays.

    The earlier experiment concatenated one K/V row per generated token.  That
    was exact, but it discarded the native cache's 256-token preallocation and
    made the compiled graph return and carry an ever-growing tensor. A later
    full-capacity masked-attention experiment changed the SDPA reduction path
    and failed the quality gate. This version keeps fixed-capacity backing
    buffers outside the graph, accepts only their active prefix, returns only
    each layer's new one-token K/V rows, and uses the same unmasked SDPA path as
    native MLX. Capacity follows the native 256-token slabs.
    """

    def __init__(self, model):
        self.model = model
        self._compiled_step = mx.compile(self._step, shapeless=True)

    def _step(
        self,
        token: mx.array,
        keys: list[mx.array],
        values: list[mx.array],
        position: mx.array,
    ) -> tuple[mx.array, list[mx.array], list[mx.array]]:
        hidden = self.model.model.embed_tokens(token.reshape((1, 1)))
        next_keys: list[mx.array] = []
        next_values: list[mx.array] = []

        # Call the underlying native RoPE primitive directly.  Unlike the
        # ``nn.RoPE`` wrapper, it accepts a scalar array offset, so the compiled
        # graph keeps the position dynamic without reimplementing RoPE math (or
        # changing BF16 rounding).
        head_dim = self.model.config.text_config.head_dim
        base = self.model.config.text_config.rope_theta

        def apply_rope(x: mx.array) -> mx.array:
            return mx.fast.rope(
                x,
                head_dim,
                traditional=False,
                base=base,
                scale=1.0,
                offset=position,
            )

        for index, layer in enumerate(self.model.model.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attn = layer.self_attn
            batch, length, _ = normed.shape

            queries = attn.q_proj(normed)
            new_keys = attn.k_proj(normed)
            new_values = attn.v_proj(normed)
            queries = queries.reshape(
                batch, length, attn.num_heads, attn.head_dim
            )
            new_keys = new_keys.reshape(
                batch, length, attn.num_kv_heads, attn.head_dim
            )
            new_values = new_values.reshape(
                batch, length, attn.num_kv_heads, attn.head_dim
            )
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            new_keys = attn.k_norm(new_keys).transpose(0, 2, 1, 3)
            new_values = new_values.transpose(0, 2, 1, 3)

            queries = apply_rope(queries)
            new_keys = apply_rope(new_keys)
            # Prefix concatenation is temporary attention input only. We
            # return just ``new_keys``/``new_values``; the caller writes those
            # rows into its preallocated backing buffers.
            all_keys = mx.concatenate((keys[index], new_keys), axis=2)
            all_values = mx.concatenate((values[index], new_values), axis=2)
            attended = mx.fast.scaled_dot_product_attention(
                queries,
                all_keys,
                all_values,
                scale=attn.scale,
                mask=None,
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, length, -1
            )
            hidden = residual + attn.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            next_keys.append(new_keys)
            next_values.append(new_values)

        hidden = self.model.model.norm(hidden)
        if self.model.lm_head is not None:
            logits = self.model.lm_head(hidden)[:, -1, :]
        else:
            logits = self.model.model.embed_tokens.as_linear(hidden)[:, -1, :]
        return logits, next_keys, next_values

    def __call__(self, token, keys, values, position):
        return self._compiled_step(token, keys, values, position)


class TieredVocabularyHead:
    """Two-tier greedy argmax over the tied 151,936-row vocabulary head.

    The BF16 tied head is the single most expensive object in a decode step
    (2.57 ms, 40.6% of a parity step) purely because greedy argmax has to read
    all 311 MB of it.  ``mx.argmax`` itself is free -- fused into the projection
    graph it costs nothing measurable (round 3 §4), so there is no fused-kernel
    win available.  What *is* available is reading fewer bytes for the scan:

    1. score every row against a low-bit copy of the head (87 MB at 4 bits,
       derived at load time -- the artifact on disk is unchanged);
    2. take the top ``k`` candidates;
    3. re-score only those ``k`` rows against the exact BF16 head and take the
       argmax there.

    In exact arithmetic this returns the true argmax whenever it is inside the
    low-bit top ``k``.  On real decode traces the true argmax never fell below
    rank 1 of the 4-bit scan, so k=32 is a wide margin -- but the scheme is
    still an approximation, is WER-gated, and is never the parity default.

    The re-score tier is whatever the profile's own head already is: BF16 for
    ``parity``, and the profile's 8-bit affine head for the compact family (its
    candidate rows are dequantized on the fly, which is 32 rows, not 151,936).
    So the scheme never *adds* head error to a profile -- it only changes which
    rows get read at full precision.
    """

    def __init__(self, model, *, bits: int = 4, group_size: int = 64, k: int = 32):
        embed = model.model.embed_tokens
        if model.lm_head is not None:
            raise TypeError("tiered head requires a tied output head")
        self.group_size = group_size
        self.bits = bits
        self.k = k
        self.quantized_reference = isinstance(embed, nn.QuantizedEmbedding)
        if self.quantized_reference:
            # The profile's own 8-bit head is the exact reference.  Dequantize
            # it once to build the cheap scan tier, then release the BF16 copy.
            self.ref_q = embed.weight
            self.ref_scales = embed.scales
            self.ref_biases = embed.biases
            self.ref_group_size = embed.group_size
            self.ref_bits = embed.bits
            dense = mx.dequantize(
                self.ref_q, self.ref_scales, self.ref_biases,
                group_size=self.ref_group_size, bits=self.ref_bits, mode="affine",
            )
        elif isinstance(embed, nn.Embedding):
            self.weight = embed.weight
            dense = self.weight
        else:
            raise TypeError(f"unsupported tied embedding type {type(embed)!r}")
        self.q, self.scales, self.biases = mx.quantize(
            dense, group_size=group_size, bits=bits, mode="affine"
        )
        mx.eval(self.q, self.scales, self.biases)
        if self.quantized_reference:
            del dense
            mx.clear_cache()

    @property
    def extra_bytes(self) -> int:
        return int(self.q.nbytes + self.scales.nbytes + self.biases.nbytes)

    def _exact_rows(self, candidates: mx.array) -> mx.array:
        if not self.quantized_reference:
            return self.weight[candidates]
        return mx.dequantize(
            self.ref_q[candidates],
            self.ref_scales[candidates],
            self.ref_biases[candidates],
            group_size=self.ref_group_size,
            bits=self.ref_bits,
            mode="affine",
        )

    def argmax(self, hidden: mx.array) -> mx.array:
        approx = mx.quantized_matmul(
            hidden,
            self.q,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )[0, 0]
        candidates = mx.argpartition(-approx, self.k)[: self.k]
        exact = hidden[0, 0] @ self._exact_rows(candidates).T
        return candidates[mx.argmax(exact, axis=-1)].astype(mx.int32)


class OptimizedV18:
    """Drop-in adapter used only by explicit experimental backends.

    ``audio_blocks`` and ``compiled_decode`` are independent switches so each
    execution change can be quality/performance gated separately.  The default
    deliberately enables neither: it is the exact prompt-head-elision control.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        model=None,
        fused_qmv: bool = False,
        audio_blocks: bool = False,
        compiled_decode: bool = False,
        fold_bits: int | None = None,
        tiered_head_bits: int | None = None,
        tiered_head_k: int = 32,
    ):
        if model is None:
            if path is None:
                raise ValueError("either a model directory or a loaded model is required")
            model = load_v18(path, use_fused_qmv=fused_qmv, fold_bits=fold_bits)
        self.model = model
        self.tiered_head = (
            TieredVocabularyHead(model, bits=tiered_head_bits, k=tiered_head_k)
            if tiered_head_bits
            else None
        )
        self.audio_cache = ClosedBlockAudioCache()
        self.audio_blocks = audio_blocks
        self.compiled_decode = compiled_decode
        self.decoder = (
            CompiledFixedCapacityDecoder(self.model) if compiled_decode else None
        )

    def reset_stream(self) -> None:
        self.audio_cache.reset()

    def parameters(self):
        return self.model.parameters()

    @staticmethod
    def _apply_processors(
        token_history: mx.array,
        logits: mx.array,
        processors: Iterable | None,
    ) -> mx.array:
        if processors:
            for processor in processors:
                logits = processor(token_history, logits)
        return logits

    def _iter_tokens(
        self,
        audio: np.ndarray,
        *,
        max_tokens: int,
        language: str | None,
        repetition_penalty: float | None,
        repetition_context_size: int,
        system_prompt: str | None = None,
        prefill_step_size: int = 2048,
        prompt_length: list[int] | None = None,
    ):
        """Yield generated token ids one at a time.

        Both the batch and streaming entry points consume this single
        generator, so a streamed transcript cannot diverge from the batch
        transcript: there is only one decode loop.
        """
        m = self.model
        features, feature_mask, num_audio_tokens = m._preprocess_audio(audio)
        audio_features = (
            self.audio_cache.encode(m, features, feature_mask)
            if self.audio_blocks
            else m.get_audio_features(features, feature_mask)
        )
        input_ids = m._build_prompt(num_audio_tokens, language, system_prompt)
        inputs = m._build_inputs_embeds(input_ids, audio_features)[0]
        prompt = input_ids[0]
        if prompt_length is not None:
            prompt_length.append(int(prompt.shape[0]))
        if max_tokens <= 0:
            return

        cache = m.make_cache()
        # Do not project the audio-prefix positions through the enormous
        # vocabulary head: only their hidden states/KV entries are needed.
        # Chunk exactly like ``mlx_lm.generate_step`` so a very long single
        # chunk sees the same Metal shapes as the native path.
        remaining = inputs.shape[0] - 1
        consumed = 0
        while remaining > 0:
            n = min(prefill_step_size, remaining)
            m.model(inputs_embeds=inputs[consumed : consumed + n][None], cache=cache)
            mx.eval([entry.state for entry in cache])
            consumed += n
            remaining -= n
            mx.clear_cache()
        hidden = m.model(inputs_embeds=inputs[consumed:][None], cache=cache)
        hidden = hidden[:, -1:, :]
        tiered = self.tiered_head if repetition_penalty in (None, 0) else None
        logits = None
        if tiered is None:
            logits = (
                m.lm_head(hidden)[:, -1, :]
                if m.lm_head is not None
                else m.model.embed_tokens.as_linear(hidden)[:, -1, :]
            )
        if self.compiled_decode:
            # Reuse the native prefill buffers at their existing 256-token
            # capacity. Unlike the failed max-budget experiment, this does not
            # force a short active prefix through hundreds of masked slots.
            keys = [entry.keys for entry in cache]
            values = [entry.values for entry in cache]
            mx.eval(keys, values)
            position_int = int(cache[0].offset)
            position = mx.array(position_int, dtype=mx.int32)

        processors = (
            make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if repetition_penalty
            else None
        )
        token_history = prompt[-1:]
        if tiered is None:
            logits = self._apply_processors(token_history, logits, processors)
            token = mx.argmax(logits, axis=-1).astype(mx.int32)
        else:
            token = tiered.argmax(hidden)
        mx.async_eval(token)

        eos = m._eos_token_ids()
        for step in range(max_tokens):
            # Launch the next GPU step before synchronously reading the current
            # token.  This is the same overlap pattern used by mlx_lm's
            # generate_step; reading first serializes Python and Metal and is a
            # severe latency regression even when the math is identical.
            token_history = mx.concatenate((token_history, token.reshape((-1,))))
            if self.compiled_decode:
                if position_int >= keys[0].shape[2]:
                    # Mirror mlx_lm.KVCache exactly: grow all layers by one
                    # 256-token slab, then keep the next compiled steps fixed.
                    grown_keys = []
                    grown_values = []
                    for key, value in zip(keys, values):
                        grown_keys.append(
                            mx.concatenate(
                                (
                                    key,
                                    mx.zeros(
                                        (*key.shape[:2], 256, key.shape[3]),
                                        dtype=key.dtype,
                                    ),
                                ),
                                axis=2,
                            )
                        )
                        grown_values.append(
                            mx.concatenate(
                                (
                                    value,
                                    mx.zeros(
                                        (*value.shape[:2], 256, value.shape[3]),
                                        dtype=value.dtype,
                                    ),
                                ),
                                axis=2,
                            )
                        )
                    keys, values = grown_keys, grown_values
                    mx.eval(keys, values)
                active_keys = [key[..., :position_int, :] for key in keys]
                active_values = [value[..., :position_int, :] for value in values]
                logits, new_keys, new_values = self.decoder(
                    token, active_keys, active_values, position
                )
                for index, (new_key, new_value) in enumerate(
                    zip(new_keys, new_values)
                ):
                    keys[index][..., position_int : position_int + 1, :] = new_key
                    values[index][..., position_int : position_int + 1, :] = new_value
                position_int += 1
                position = position + 1
            elif tiered is not None:
                hidden = m.model(
                    inputs_embeds=m.model.embed_tokens(token.reshape((1, 1))),
                    cache=cache,
                )
                next_token = tiered.argmax(hidden[:, -1:, :])
                mx.async_eval(next_token)
                token_id = int(token.item())
                if token_id in eos:
                    break
                yield token_id
                token = next_token
                if step and step % 256 == 0:
                    mx.clear_cache()
                continue
            else:
                logits = m._forward_with_embeds(
                    m.model.embed_tokens(token.reshape((1, 1))), cache
                )[:, -1, :]
            logits = self._apply_processors(token_history, logits, processors)
            next_token = mx.argmax(logits, axis=-1).astype(mx.int32)
            mx.async_eval(next_token)

            token_id = int(token.item())
            if token_id in eos:
                break
            yield token_id
            token = next_token
            if step and step % 256 == 0:
                mx.clear_cache()

    def _prepare(self, audio, chunk_duration: float, min_chunk_duration: float):
        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio))
        elif isinstance(audio, mx.array):
            audio = np.asarray(audio)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return split_audio_into_chunks(
            audio,
            sr=16_000,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 8192,
        language: str | None = None,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 100,
        chunk_duration: float = 1200.0,
        min_chunk_duration: float = 1.0,
        prefill_step_size: int = 2048,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        **kwargs,
    ):
        if temperature != 0.0 or kwargs.get("sampler") is not None:
            # Prompt-head elision is a greedy-decoding optimisation.  Anything
            # that needs a real sampler falls back to the untouched native
            # implementation rather than losing a capability.
            return self.model.generate(
                audio,
                max_tokens=max_tokens,
                language=language,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
                prefill_step_size=prefill_step_size,
                system_prompt=system_prompt,
                temperature=temperature,
                stream=stream,
                **kwargs,
            )

        chunks = self._prepare(audio, chunk_duration, min_chunk_duration)
        if stream:
            return self._stream(
                chunks,
                max_tokens=max_tokens,
                language=language,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                system_prompt=system_prompt,
                prefill_step_size=prefill_step_size,
            )

        started = time.perf_counter()
        texts: list[str] = []
        segments: list[dict] = []
        prompt_tokens = 0
        generation_tokens = 0
        for chunk, offset in chunks:
            if max_tokens - generation_tokens <= 0:
                break
            prompt_length: list[int] = []
            tokens = list(
                self._iter_tokens(
                    chunk,
                    max_tokens=max_tokens - generation_tokens,
                    language=language,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    system_prompt=system_prompt,
                    prefill_step_size=prefill_step_size,
                    prompt_length=prompt_length,
                )
            )
            text = self.model._tokenizer.decode(tokens, skip_special_tokens=True)
            chunk_language = language
            if chunk_language is None:
                chunk_language, text = self.model.extract_language(text)
            texts.append(text)
            prompt_tokens += prompt_length[0] if prompt_length else 0
            generation_tokens += len(tokens)
            segments.append(
                {
                    "text": text,
                    "language": chunk_language,
                    "start": offset,
                    "end": offset + len(chunk) / 16_000,
                }
            )
            mx.clear_cache()
        elapsed = time.perf_counter() - started
        return STTOutput(
            text=" ".join(texts),
            segments=segments,
            language=[row["language"] for row in segments],
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            total_time=elapsed,
            prompt_tps=prompt_tokens / elapsed if elapsed else 0,
            generation_tps=generation_tokens / elapsed if elapsed else 0,
        )

    def _stream(
        self,
        chunks,
        *,
        max_tokens: int,
        language: str | None,
        repetition_penalty: float | None,
        repetition_context_size: int,
        system_prompt: str | None,
        prefill_step_size: int,
    ):
        """Emit one ``StreamingResult`` per token, matching the native
        ``stream_transcribe`` contract.  The token sequence comes from the same
        ``_iter_tokens`` generator the batch path uses, so a streamed decode
        cannot diverge from the batch decode."""
        tokenizer = self.model._tokenizer
        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens
        language_accumulator = ""

        for index, (chunk, offset) in enumerate(chunks):
            is_last_chunk = index == len(chunks) - 1
            duration = len(chunk) / 16_000
            token_count = 0
            prompt_length: list[int] = []
            for i, token in enumerate(
                self._iter_tokens(
                    chunk,
                    max_tokens=remaining_tokens,
                    language=language,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    system_prompt=system_prompt,
                    prefill_step_size=prefill_step_size,
                    prompt_length=prompt_length,
                )
            ):
                if not total_prompt_tokens and prompt_length:
                    total_prompt_tokens += prompt_length[0]
                text = tokenizer.decode([token])
                if language is None and i <= 2:
                    language_accumulator += text
                    if "<asr_text>" in language_accumulator:
                        language, _ = self.model.extract_language(language_accumulator)
                    continue
                previous = token_count / max(remaining_tokens, 1)
                token_count += 1
                current = min(token_count / max(remaining_tokens, 1), 1.0)
                yield StreamingResult(
                    text=text,
                    is_final=False,
                    start_time=offset + duration * previous,
                    end_time=offset + duration * current,
                    language=language or "English",
                )
            if prompt_length and not total_prompt_tokens:
                total_prompt_tokens += prompt_length[0]
            total_generation_tokens += token_count
            remaining_tokens -= token_count
            yield StreamingResult(
                text="",
                is_final=is_last_chunk or remaining_tokens <= 0,
                start_time=offset,
                end_time=offset + duration,
                language=language or "English",
                prompt_tokens=total_prompt_tokens,
                generation_tokens=total_generation_tokens,
            )
            mx.clear_cache()
            if remaining_tokens <= 0:
                break


def load_optimized_v18(
    path: Path | None = None,
    *,
    model=None,
    fused_qmv: bool = False,
    audio_blocks: bool = False,
    compiled_decode: bool = False,
    fold_bits: int | None = None,
    tiered_head_bits: int | None = None,
    tiered_head_k: int = 32,
) -> OptimizedV18:
    return OptimizedV18(
        path,
        model=model,
        fused_qmv=fused_qmv,
        audio_blocks=audio_blocks,
        compiled_decode=compiled_decode,
        fold_bits=fold_bits,
        tiered_head_bits=tiered_head_bits,
        tiered_head_k=tiered_head_k,
    )

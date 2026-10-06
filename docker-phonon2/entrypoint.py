#!/usr/bin/env python3
"""phonon-cuda — Phonon-2 on NVIDIA.

Two modes:

  phonon-cuda transcribe <audio...> [--model NAME] [--model-dir DIR] [--json] [--verbose]
      Transcribe 16 kHz recordings to stdout (--json: one object per file
      with segment and word timestamps). Single utterances (<= 30 s)
      decode in one call; longer recordings are split at pauses by an
      energy gate and the finals joined with spaces.

  phonon-cuda serve [--host 127.0.0.1] [--port 8000] [--api-key KEY]
                    [--max-queue N] [--model NAME | --model-dir DIR]
      OpenAI-compatible endpoints: POST /v1/audio/transcriptions (the
      Whisper API multipart shape), GET /v1/audio/stream (RFC 6455
      WebSocket, the same protocol as `fermion serve`),
      GET /health, GET /.

The image serves Phonon-2 (`--model phonon-2`, the default). Without
`--model-dir` the model's release archive is downloaded from Hugging Face,
verified against its published SHA-256 pin, and unpacked; with
`--model-dir` a local copy is used and its manifest must match the
requested model.

Decoding is greedy and deterministic. Long audio and streams are segmented
(never resampled, never truncated) and every segment is decoded the same
way.
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = "1.0.7"
BRAND = "phonon-cuda"

#: The published models this image serves. Release facts (archive filename,
#: SHA-256, byte count) are pinned per release; the profile key names the
#: artifact storage layout the loader verifies against the directory.
CATALOG = {
    "phonon-2": {
        "name": "Phonon-2",
        "repo": "FermionResearch/Phonon-2",
        "profile": "five-value",
        "filename": "phonon-2.bps.tar.zst",
        "sha256": "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e",
        "download_bytes": 163_515_201,
        "aliases": ("phonon-2", "phonon2", "phonon"),
    },
}
#: The model served when no `--model` is given.
DEFAULT_MODEL = "phonon-2"

#: Names any server accepts in the `model` form field regardless of which
#: model it is serving (family names, not model-distinguishing ones).
GENERIC_MODEL_NAMES = {"", "phonon", "phonon-cuda"}
SAMPLE_RATE = 16_000
MAX_SECONDS = 30.0          # longest audio decoded as a single utterance
MAX_LONG_SECONDS = 2 * 3600  # sanity cap for the long-audio path
MAX_BODY = 32 * 1024 * 1024  # 32 MB request cap
EOS_IDS = (151643, 151645)
WS_IDLE_TIMEOUT_S = 90.0
DEFAULT_MAX_QUEUE = 8


def log(message: str) -> None:
    print(f"[{BRAND}] {message}", file=sys.stderr, flush=True)


def fail(message: str) -> "NoReturn":  # noqa: F821
    log(message)
    raise SystemExit(2)


# --------------------------------------------------------------------- audio
def decode_audio(source, label: str):
    """Decode an audio file/bytes to float32 mono 16 kHz, or raise ValueError.

    No resampling: inputs at other rates are refused with an actionable
    message rather than run through an unvalidated path.
    """
    import numpy as np
    import soundfile as sf

    try:
        wav, sr = sf.read(source, dtype="float32")
    except Exception as exc:
        raise ValueError(f"could not decode audio {label!r}: {exc}") from exc
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{label}: sample rate is {sr} Hz; this runtime requires 16 kHz. "
            f"Convert first, e.g.: ffmpeg -i input -ar 16000 -ac 1 out.wav")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1 or not len(wav):
        raise ValueError(f"{label}: no audio samples decoded")
    duration = len(wav) / SAMPLE_RATE
    if duration > MAX_LONG_SECONDS:
        raise ValueError(
            f"{label}: {duration / 3600:.1f} h of audio; the long-audio path "
            f"is capped at {MAX_LONG_SECONDS // 3600} hours per file.")
    return np.ascontiguousarray(wav, dtype=np.float32), duration


# --------------------------------------------------------------------- model
def resolve_model_name(spec: str | None) -> str:
    """Alias or repo id (any case) -> canonical catalog key, or refuse."""
    if not spec:
        return DEFAULT_MODEL
    wanted = str(spec).strip().lower()
    for key, entry in CATALOG.items():
        if wanted in entry["aliases"] or wanted == entry["repo"].lower():
            return key
    raise SystemExit(
        f"[{BRAND}] unknown model {spec!r}; this image serves "
        f"{', '.join(CATALOG)} (and their aliases)")


def profile_to_model(profile: str) -> str:
    return next(k for k, v in CATALOG.items() if v["profile"] == profile)


def model_cache_root() -> Path:
    override = os.environ.get("PHONON_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "phonon"


def _directory_profile(model_dir: Path, claimed: str | None) -> str:
    """The profile a model directory holds (Phonon-2: `packed_manifest.json` format fermion-five-value-parakeet-v1 + model.fermion)."""
    man = model_dir / "packed_manifest.json"
    if not man.is_file() or not (model_dir / "model.fermion").is_file() or not (model_dir / "config.json").is_file():
        fail(f"{model_dir} does not look like a Phonon-2 model directory (packed_manifest.json + model.fermion + config.json)")
    fmt = json.loads(man.read_text()).get("format", "")
    if fmt != "fermion-five-value-parakeet-v1":
        fail(f"{model_dir}: manifest format {fmt!r} is not a Phonon-2 container")
    if claimed is not None and claimed != "five-value":
        fail(f"the requested model is the {claimed!r} build but the directory holds a Phonon-2 container")
    return "five-value"


def resolve_model(args) -> tuple[str, Path]:
    """-> (canonical model key, model directory). Downloads if needed."""
    spec = args.model or getattr(args, "repo", None)
    if args.model and getattr(args, "repo", None) \
            and resolve_model_name(args.model) != resolve_model_name(args.repo):
        fail("--model and --repo name different models; pass one of them")
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        claimed = CATALOG[resolve_model_name(spec)]["profile"] if spec else None
        profile = _directory_profile(model_dir, claimed)
        return profile_to_model(profile), model_dir

    key = resolve_model_name(spec)
    entry = CATALOG[key]
    import _archive
    dest = model_cache_root() / "models" / key
    try:
        model_dir = _archive.ensure_model(
            entry["repo"], entry["filename"], entry["sha256"],
            entry["download_bytes"], dest, log=log)
    except Exception as exc:
        fail(f"could not fetch {entry['repo']}: {exc}")
    return key, model_dir


from phonon2_cuda_engine import Transcriber  # noqa: E402  (the Phonon-2 CUDA engine)


# ---------------------------------------------------------------- transcribe
def cmd_transcribe(args) -> None:
    paths = [Path(p) for p in args.audio]
    for path in paths:
        try:
            missing = not path.is_file(); unreadable = not missing and not os.access(path, os.R_OK)
        except PermissionError:                     # a 0700 parent directory: stat itself is refused
            missing, unreadable = False, True
        if missing:
            fail(f"no such file: {path}")
        if unreadable:
            fail(f"{path} is not readable by the container's user (uid {os.getuid()}). The image runs as an unprivileged "
                 f"user: make the file and its directory world-readable (chmod a+rX), or run with --user $(id -u).")
    waveforms = []
    for path in paths:
        try:
            wav, duration = decode_audio(str(path), path.name)
        except ValueError as exc:
            fail(str(exc))
        waveforms.append((path, wav, duration))

    words = _cli_hotwords(args)
    t_load = time.perf_counter()
    engine = Transcriber(*resolve_model(args))
    load_s = time.perf_counter() - t_load
    if words or args.hotword_lambda is not None:
        n = engine.set_hotwords(words, args.hotword_lambda)
        log(f"{n} hotword{'s' if n != 1 else ''} active (lambda {engine.hw.lam:g})")
    for path, wav, duration in waveforms:
        t0 = time.perf_counter()
        text, segments, timed_words = engine.transcribe_long_timed(wav)
        decode_s = time.perf_counter() - t0
        n_seg = len(segments)
        log(f"{path.name}: {duration:.1f}s audio, {n_seg} segment"
            f"{'s' if n_seg != 1 else ''}, decoded in "
            f"{decode_s:.2f}s" + (f" ({duration / decode_s:.0f}x real time, decode only; model load {load_s:.1f}s)"
                                  if args.verbose and decode_s > 0 else ""))
        if args.json:
            # One JSON object per file, one line each. `segments` is the list of {id, start, end, text} (seconds from
            # the start of the file; the texts joined with single spaces equal `text`), `segment_count` its length,
            # `words` one {text, start, end} per word -- the same shapes as `fermion transcribe --json`.
            print(json.dumps({"text": text, "file": str(path), "duration_s": round(duration, 3),
                              "decode_s": round(decode_s, 4), "load_s": round(load_s, 2),
                              "segment_count": n_seg, "segments": segments, "words": timed_words}), flush=True)
        else:
            print(text, flush=True)


def _cli_hotwords(args) -> list:
    """`--hotwords "a, b" --hotwords @terms.txt --hotwords-file terms.txt` -> the distinct terms, in order."""
    from hotwords import parse_list
    out: list = []
    chunks = list(getattr(args, "hotwords", None) or [])
    chunks += ["@" + f for f in (getattr(args, "hotwords_file", None) or [])]
    for chunk in chunks:
        chunk = str(chunk)
        if chunk.startswith("@"):
            try:
                text = Path(chunk[1:]).expanduser().read_text(encoding="utf-8")
            except OSError as exc:
                fail(f"--hotwords file {chunk[1:]}: {exc}")
            terms = [" ".join(l.split()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        else:
            terms = parse_list(chunk) if "," in chunk else [" ".join(chunk.split())]
        for t in terms:
            if t and t not in out:
                out.append(t)
    return out


def _form_hotwords(form):
    """(terms or None, lambda or None) from a transcription form: `hotwords` (a comma-separated string, a JSON list string,
    or repeated `hotwords[]` parts) wins; else `prompt` read as a vocabulary list (comma-separated when it has a comma,
    else whitespace-separated).  None = the field is absent (the engine's own list applies)."""
    from hotwords import parse_list
    terms = None
    for key in ("hotwords[]", "hotwords"):
        v = form.get(key)
        if v is None:
            continue
        terms = [] if terms is None else terms
        vals = v if isinstance(v, list) else [v]
        for item in vals:
            if not isinstance(item, str):
                continue
            t = item.strip()
            if t.startswith("["):
                try:
                    val = json.loads(t)
                    terms.extend(" ".join(str(x).split()) for x in (val if isinstance(val, list) else [val]))
                    continue
                except ValueError:
                    pass
            terms.extend(parse_list(t) if key == "hotwords" else [" ".join(t.split())])
    if terms is None and isinstance(form.get("prompt"), str):
        terms = parse_list(form["prompt"])
    if terms is not None:
        seen = []
        for t in terms:
            if t and t not in seen:
                seen.append(t)
        terms = seen
    lam = form.get("hotword_lambda")
    if isinstance(lam, str) and lam.strip():
        lam = float(lam)
    else:
        lam = None
    return terms, lam


# --------------------------------------------------------------------- serve

def _granularities(form) -> list:
    """`timestamp_granularities` values from a parsed form: `timestamp_granularities[]` parts (the SDKs), a plain or
    comma-separated string, or a JSON list string; lower-cased, de-duplicated, in order."""
    raw: list = []
    for key in ("timestamp_granularities[]", "timestamp_granularities"):
        v = form.get(key)
        if isinstance(v, list):
            raw.extend(v)
        elif isinstance(v, str):
            t = v.strip()
            if t.startswith("["):
                try:
                    val = json.loads(t); raw.extend(val if isinstance(val, list) else [val]); continue
                except ValueError:
                    pass
            raw.extend(t.split(","))
    out: list = []
    for g in raw:
        g = str(g).strip().strip('"').lower()
        if g and g not in out:
            out.append(g)
    return out


class DrainingHTTPServer(ThreadingHTTPServer):
    # Wait for in-flight handler threads on close (graceful SIGTERM drain).
    daemon_threads = False
    block_on_close = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"{BRAND}/{VERSION}"
    sys_version = ""

    @property
    def engine(self) -> Transcriber:
        return self.server.engine  # type: ignore[attr-defined]

    def log_message(self, fmt, *a):
        sys.stderr.write(f"[{BRAND}] {self.address_string()} {fmt % a}\n")

    def _send_json(self, status: int, obj, extra_headers=()) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str,
             type_: str = "invalid_request_error", param=None,
             code=None, extra_headers=()) -> None:
        self._send_json(status, {"error": {
            "message": message, "type": type_, "param": param,
            "code": code}}, extra_headers)

    def _bearer_ok(self) -> bool:
        key = self.server.api_key  # type: ignore[attr-defined]
        if not key:
            return True
        hdr = self.headers.get("Authorization", "")
        return hdr.startswith("Bearer ") and hmac.compare_digest(hdr[7:], key)

    def _ws_authorised(self, query: str) -> bool:
        """WS auth: bearer header OR ?api_key= (browser WebSocket()
        constructors cannot set headers)."""
        key = self.server.api_key  # type: ignore[attr-defined]
        if not key:
            return True
        if self._bearer_ok():
            return True
        supplied = (parse_qs(query).get("api_key") or [""])[0]
        return bool(supplied) and hmac.compare_digest(supplied, key)

    def _drain_body(self) -> None:
        """Consume an unread body before an early-error reply (keep-alive
        desync guard)."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            return
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked" \
                or n > MAX_BODY:
            self.close_connection = True
            return
        try:
            while n > 0:
                chunk = self.rfile.read(min(n, 1 << 16))
                if not chunk:
                    self.close_connection = True
                    return
                n -= len(chunk)
        except Exception:
            self.close_connection = True

    # ------------------------------------------------------------ queueing
    def _take_queue_slot(self) -> bool:
        srv = self.server
        with srv.q_lock:  # type: ignore[attr-defined]
            if srv.q_depth >= 1 + srv.max_queue:  # type: ignore[attr-defined]
                return False
            srv.q_depth += 1  # type: ignore[attr-defined]
            return True

    def _release_queue_slot(self) -> None:
        srv = self.server
        with srv.q_lock:  # type: ignore[attr-defined]
            srv.q_depth -= 1  # type: ignore[attr-defined]

    def _queue_full(self) -> None:
        depth = self.server.q_depth  # type: ignore[attr-defined]
        retry = max(1, 2 * depth)
        self._err(503, f"the request queue is full ({depth} in flight; "
                       f"--max-queue {self.server.max_queue}). "  # type: ignore[attr-defined]
                       f"Retry after ~{retry}s.",
                  type_="server_error", code="queue_full",
                  extra_headers=(("Retry-After", str(retry)),))

    # ------------------------------------------------------------------ GET
    def do_GET(self):  # noqa: N802
        split = urlsplit(self.path)
        route = split.path.rstrip("/") or "/"
        if route in ("/health", "/v1/health", "/healthz"):
            srv = self.server
            eng = self.engine
            return self._send_json(200, {
                "status": "draining" if srv.draining else "ok",  # type: ignore[attr-defined]
                "service": f"{BRAND} serve",
                "version": VERSION,
                "uptime_seconds": round(time.monotonic() - srv.started, 1),  # type: ignore[attr-defined]
                "queue": {"depth": srv.q_depth,  # type: ignore[attr-defined]
                          "max": srv.max_queue},  # type: ignore[attr-defined]
                "stats": dict(srv.stats),  # type: ignore[attr-defined]
                **eng.describe()})
        if route == "/v1/audio/stream":
            return self._stream(split.query)
        if route == "/":
            return self._send_json(200, {
                "service": f"{BRAND} serve", "version": VERSION,
                "model": self.engine.model_key,
                "endpoints": ["/v1/audio/transcriptions", "/v1/audio/stream",
                              "/health"]})
        return self._err(404, f"unknown route {route!r}; this server mounts "
                              f"POST /v1/audio/transcriptions, "
                              f"GET /v1/audio/stream (WebSocket) and "
                              f"GET /health", code="not_found")

    # ----------------------------------------------------------------- POST
    def do_POST(self):  # noqa: N802
        route = urlsplit(self.path).path.rstrip("/") or "/"
        if not self._bearer_ok():
            self._drain_body()
            return self._err(401, "missing or invalid API key",
                             code="invalid_api_key")
        if route != "/v1/audio/transcriptions":
            self._drain_body()
            return self._err(
                404, f"{route} is not served here. "
                     f"{self.engine.entry['name']} is a speech model; this "
                     f"server mounts POST /v1/audio/transcriptions.",
                code="not_found")
        if self.server.draining:  # type: ignore[attr-defined]
            self._drain_body()
            return self._err(503, "server is draining for shutdown",
                             type_="server_error", code="draining",
                             extra_headers=(("Retry-After", "5"),))
        try:
            self._transcriptions()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001
            self.server.stats["errors"] += 1  # type: ignore[attr-defined]
            sys.stderr.write(f"[{BRAND}] internal error: {exc!r}\n")
            self._err(500, f"internal error: {exc}", type_="server_error")

    def _transcriptions(self) -> None:
        """POST /v1/audio/transcriptions — the Whisper API shape."""
        import _multipart

        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.close_connection = True
            return self._err(400, "chunked request bodies are not supported")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._err(400, "invalid Content-Length")
        if n <= 0:
            return self._err(400, "an audio file is required",
                             param="file", code="missing_file")
        if n > MAX_BODY:
            self.close_connection = True
            return self._err(
                413, f"request body too large (limit {MAX_BODY // 2**20} MB). "
                     f"Split longer recordings across requests.")
        raw = self.rfile.read(n)

        try:
            form = _multipart.parse(raw, self.headers.get("Content-Type", ""))
        except _multipart.MultipartError as exc:
            return self._err(
                400, f"{exc}. /v1/audio/transcriptions takes a "
                     f"multipart/form-data body with a `file` part, exactly "
                     f"like the OpenAI audio API.")

        part = form.get("file")
        if not isinstance(part, dict) or not part.get("content"):
            return self._err(400, "no `file` part in the form",
                             param="file", code="missing_file")

        # `model` is accepted and checked, never silently ignored.
        want = form.get("model")
        if isinstance(want, str) and want.strip() \
                and want.strip().lower() not in self.engine.accepted_names:
            return self._err(
                404, f"model {want.strip()!r} is not served here; this "
                     f"process serves {self.engine.model_key}",
                param="model", code="model_not_found")

        fmt = form.get("response_format") or "json"
        if not isinstance(fmt, str):
            fmt = "json"
        fmt = fmt.strip().lower() or "json"
        if fmt not in ("json", "text", "verbose_json"):
            return self._err(
                400, f"response_format {fmt!r} is not supported; use json "
                     f"(default), text, or verbose_json",
                param="response_format")

        for field, why in (
                ("language", "the model is English-only; `language` is "
                             "ignored"),
                ("temperature", "transcription is deterministic "
                                "(temperature 0)")):
            if field in form and field not in self.server.warned_fields:  # type: ignore[attr-defined]
                self.server.warned_fields.add(field)  # type: ignore[attr-defined]
                sys.stderr.write(f"[{BRAND}] note: `{field}` — {why}\n")
        # `timestamp_granularities` selects `words` (word) in verbose_json; `segments` is always filled (OpenAI's default).
        granularities = _granularities(form)
        for g in granularities:
            key = "timestamp_granularities:" + g
            if g not in ("word", "segment") and key not in self.server.warned_fields:  # type: ignore[attr-defined]
                self.server.warned_fields.add(key)  # type: ignore[attr-defined]
                sys.stderr.write(f"[{BRAND}] note: `timestamp_granularities` value {g!r} is not supported; use word and/or segment\n")
        want_words = "word" in granularities
        try:
            hot, hot_lam = _form_hotwords(form)
        except ValueError:
            return self._err(400, "hotword_lambda must be a number", param="hotword_lambda")
        if hot_lam is not None and not (0 <= hot_lam <= 100):
            return self._err(400, "hotword_lambda must be between 0 and 100", param="hotword_lambda")
        if hot is not None and len(hot) > 25 and "hotwords:cap" not in self.server.warned_fields:  # type: ignore[attr-defined]
            self.server.warned_fields.add("hotwords:cap")  # type: ignore[attr-defined]
            sys.stderr.write(f"[{BRAND}] note: more than 25 hotwords in a request; the first 25 are used\n")

        try:
            wav, duration = decode_audio(
                io.BytesIO(part["content"]), part.get("filename") or "upload")
        except ValueError as exc:
            return self._err(400, str(exc), param="file",
                             code="invalid_audio")

        if not self._take_queue_slot():
            return self._queue_full()
        try:
            t0 = time.perf_counter()
            text, segments, words = self.engine.transcribe_long_timed(
                wav, lock=self.server.gpu_lock,  # type: ignore[attr-defined]
                hotwords=hot, hotword_lambda=hot_lam)
            decode_s = time.perf_counter() - t0
        finally:
            self._release_queue_slot()
        stats = self.server.stats  # type: ignore[attr-defined]
        stats["requests"] += 1
        stats["audio_seconds"] = round(stats["audio_seconds"] + duration, 3)
        stats["decode_seconds"] = round(stats["decode_seconds"] + decode_s, 3)

        if fmt == "text":
            body = (text + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        out = {"text": text}
        if fmt == "verbose_json":
            import cues
            out.update({
                "task": "transcribe", "language": "english",
                "duration": round(duration, 3),
                # OpenAI's segment shape, one per sentence- or pause-sized stretch (cues.py, as in the pip package)
                "segments": cues.openai_segments(segments),
            })
            if want_words:
                out["words"] = [{"word": w["text"], "start": w["start"], "end": w["end"]} for w in words]
            out["x_phonon"] = {"decode_seconds": round(decode_s, 3),
                               "audio_segments": len(segments),
                               "segment_count": len(segments),
                               **self.engine.describe()}
        self._send_json(200, out)

    # ------------------------------------------------------------ WebSocket
    def _stream(self, query: str) -> None:
        """GET /v1/audio/stream — the same protocol as `fermion serve`.

        JSON config frame, then binary PCM frames; partial/final/done/error
        JSON events back; one stream at a time per GPU worker (close 1013);
        90 s idle timeout (close 1001)."""
        import _ws

        if not self._ws_authorised(query):
            return self._err(401, "missing or invalid API key "
                                  "(Authorization: Bearer or ?api_key=)",
                             code="invalid_api_key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        key = self.headers.get("Sec-WebSocket-Key")
        if upgrade != "websocket" or not key:
            return self._err(
                426, "/v1/audio/stream is a WebSocket endpoint: send an "
                     "RFC 6455 upgrade handshake (Upgrade: websocket, "
                     "Sec-WebSocket-Key, Sec-WebSocket-Version: 13)",
                code="upgrade_required")

        # Complete the upgrade.
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _ws.accept_key(key))
        self.end_headers()
        self.close_connection = True  # a WS never reuses the connection

        def send_event(obj) -> None:
            self.wfile.write(_ws.encode_frame(_ws.TEXT, json.dumps(obj)))
            self.wfile.flush()

        def close(code: int, reason: str = "") -> None:
            try:
                self.wfile.write(_ws.encode_frame(
                    _ws.CLOSE, _ws.close_payload(code, reason)))
                self.wfile.flush()
            except Exception:
                pass

        srv = self.server
        if srv.draining:  # type: ignore[attr-defined]
            send_event({"type": "error",
                        "message": "server is draining for shutdown"})
            return close(1001, "going away")
        if not srv.stream_lock.acquire(blocking=False):  # type: ignore[attr-defined]
            send_event({"type": "error",
                        "message": "engine busy — one stream at a time"})
            return close(1013, "try again later")
        try:
            self._stream_session(send_event, close)
        finally:
            srv.stream_lock.release()  # type: ignore[attr-defined]

    def _stream_session(self, send_event, close) -> None:
        import socket

        import numpy as np

        import _ws

        srv = self.server
        state = {"segment": 0}

        def on_partial(text: str) -> None:
            send_event({"type": "partial", "text": text})

        def on_final(text: str) -> None:
            state["segment"] += 1
            send_event({"type": "final", "text": text,
                        "segment": state["segment"]})

        session = self.engine.session(
            partials=True, on_partial=on_partial, on_final=on_final,
            lock=srv.gpu_lock)  # type: ignore[attr-defined]

        fmt = "pcm_f32le"
        configured = False
        self.connection.settimeout(WS_IDLE_TIMEOUT_S)
        try:
            while True:
                if srv.draining:  # type: ignore[attr-defined]
                    break
                try:
                    fin, opcode, payload = _ws.read_frame(
                        self.rfile, require_mask=True)
                except socket.timeout:
                    send_event({"type": "error",
                                "message": f"no audio for "
                                           f"{WS_IDLE_TIMEOUT_S:.0f}s — "
                                           f"closing idle stream"})
                    return close(1001, "idle timeout")
                except _ws.ProtocolError as exc:
                    send_event({"type": "error", "message": str(exc)})
                    return close(1002, "protocol error")
                except _ws.ConnectionClosed:
                    break  # finalize what we have

                if not fin or opcode == _ws.CONTINUATION:
                    send_event({"type": "error",
                                "message": "fragmented WebSocket messages "
                                           "are not supported; send each "
                                           "message in a single frame"})
                    return close(1002, "fragmentation not supported")
                if opcode == _ws.PING:
                    self.wfile.write(_ws.encode_frame(_ws.PONG, payload))
                    self.wfile.flush()
                    continue
                if opcode == _ws.PONG:
                    continue
                if opcode == _ws.CLOSE:
                    break
                if opcode == _ws.TEXT:
                    try:
                        msg = json.loads(payload.decode("utf-8"))
                    except Exception:
                        send_event({"type": "error",
                                    "message": "text frames must be JSON"})
                        return close(1008, "bad message")
                    if not configured and isinstance(msg, dict) \
                            and msg.get("type") != "end":
                        rate = msg.get("sample_rate", SAMPLE_RATE)
                        if rate != SAMPLE_RATE:
                            send_event({"type": "error",
                                        "message": f"sample_rate must be "
                                                   f"{SAMPLE_RATE}; resample "
                                                   f"client-side"})
                            return close(1008, "unsupported sample rate")
                        want = msg.get("format", "pcm_f32le")
                        if want not in ("pcm_f32le", "pcm_s16le"):
                            send_event({"type": "error",
                                        "message": "format must be pcm_f32le "
                                                   "or pcm_s16le"})
                            return close(1008, "unsupported format")
                        fmt = want
                        configured = True
                        continue
                    if isinstance(msg, dict) and msg.get("type") == "end":
                        break
                    send_event({"type": "error",
                                "message": f"unexpected message {msg!r}"})
                    return close(1008, "bad message")
                if opcode == _ws.BINARY:
                    configured = True
                    if fmt == "pcm_s16le":
                        samples = np.frombuffer(payload, dtype="<i2") \
                            .astype(np.float32) / 32768.0
                    else:
                        samples = np.frombuffer(payload, dtype="<f4") \
                            .astype(np.float32)
                    session.feed_pcm(samples)
                    continue
                send_event({"type": "error",
                            "message": f"unsupported opcode {opcode}"})
                return close(1002, "protocol error")

            text = session.finish()
            send_event({"type": "done", "text": text})
            srv.stats["streams"] += 1  # type: ignore[attr-defined]
            close(1000, "")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            srv.stats["errors"] += 1  # type: ignore[attr-defined]
            sys.stderr.write(f"[{BRAND}] stream internal error: {exc!r}\n")
            try:
                send_event({"type": "error",
                            "message": f"internal error: {exc}"})
            except Exception:
                pass
            close(1011, "internal error")


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


def cmd_serve(args) -> None:
    api_key = args.api_key or os.environ.get("PHONON_API_KEY") or None
    if not _is_loopback(args.host) and not api_key:
        fail(f"refusing to bind {args.host} without authentication: this "
             f"would expose an open transcription endpoint. Pass --api-key "
             f"(or set PHONON_API_KEY), or bind 127.0.0.1. Inside Docker, "
             f"`--host 0.0.0.0 --api-key <key>` plus "
             f"`docker run -p 127.0.0.1:8000:8000` is the recommended shape.")

    engine = Transcriber(*resolve_model(args))
    server = DrainingHTTPServer((args.host, args.port), Handler)
    server.engine = engine  # type: ignore[attr-defined]
    server.api_key = api_key  # type: ignore[attr-defined]
    server.gpu_lock = threading.Lock()  # type: ignore[attr-defined]
    server.stream_lock = threading.Lock()  # type: ignore[attr-defined]
    server.q_lock = threading.Lock()  # type: ignore[attr-defined]
    server.q_depth = 0  # type: ignore[attr-defined]
    server.max_queue = max(0, args.max_queue)  # type: ignore[attr-defined]
    server.draining = False  # type: ignore[attr-defined]
    server.started = time.monotonic()  # type: ignore[attr-defined]
    server.stats = {"requests": 0, "streams": 0, "errors": 0,  # type: ignore[attr-defined]
                    "audio_seconds": 0.0, "decode_seconds": 0.0}
    server.warned_fields = set()  # type: ignore[attr-defined]

    def _sigterm(_signum, _frame):
        log("SIGTERM: draining — no new requests; finishing in-flight work")
        server.draining = True  # type: ignore[attr-defined]
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)

    log(f"serving {engine.model_key} on http://{args.host}:{args.port} "
        f"(auth={'bearer' if api_key else 'none, loopback only'}, "
        f"max-queue={server.max_queue}) — "  # type: ignore[attr-defined]
        f"POST /v1/audio/transcriptions, GET /v1/audio/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.server_close()
        log("drained; exiting")


# ---------------------------------------------------------------------- main
def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default=None,
        help="which published model to run: phonon-2 (the only model this image serves)")
    parser.add_argument(
        "--model-dir",
        help="local unpacked model directory (mount it into the container, "
             "e.g. -v /path:/model); its manifest must match --model")
    parser.add_argument(
        "--repo", default=None,
        help="alternative to --model: a published Hugging "
             "Face repo id, downloaded when --model-dir is not given")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=BRAND,
        description="Phonon-2 on NVIDIA: greedy, deterministic transcription; long audio is split at pauses by an energy gate.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser(
        "transcribe", help="transcribe 16 kHz recordings to stdout")
    p_tr.add_argument("audio", nargs="+", help="16 kHz audio file(s)")
    p_tr.add_argument("--json", action="store_true",
                      help="print one JSON object per file: text, timings, segments (start, end, text) and words")
    p_tr.add_argument("--verbose", action="store_true",
                      help="also print the decode speed and the model load time on stderr")
    p_tr.add_argument("--hotwords", action="append", default=None, metavar="WORDS",
                      help='names and terms the decoder should favour, comma-separated (repeatable; @FILE reads one '
                           'term per line); at most 25. Default: none (the decode is unchanged)')
    p_tr.add_argument("--hotwords-file", action="append", default=None, metavar="FILE",
                      help="read hotwords from FILE, one term per line")
    p_tr.add_argument("--hotword-lambda", type=float, default=None, metavar="LAMBDA",
                      help="bonus per matching piece on the decoder's scores (default 2.0)")
    _add_model_args(p_tr)
    p_tr.set_defaults(func=cmd_transcribe)

    p_sv = sub.add_parser(
        "serve", help="OpenAI-shaped transcription + streaming server")
    p_sv.add_argument("--host", default="127.0.0.1")
    p_sv.add_argument("--port", type=int, default=8000)
    p_sv.add_argument("--api-key", default=None,
                      help="require `Authorization: Bearer <key>` "
                           "(or set PHONON_API_KEY)")
    p_sv.add_argument("--max-queue", type=int, default=DEFAULT_MAX_QUEUE,
                      help=f"waiting requests beyond the one decoding before "
                           f"503 + Retry-After (default "
                           f"{DEFAULT_MAX_QUEUE})")
    _add_model_args(p_sv)
    p_sv.set_defaults(func=cmd_serve)

    args = parser.parse_args(_model_first(sys.argv[1:]))
    args.func(args)


def _model_first(argv: list) -> list:
    """`transcribe phonon-2 /audio/x.wav` and `serve phonon-2 ...` (the form the docs use, and the pip CLI's) ->
    `--model phonon-2`; the older `transcribe /audio/x.wav --model phonon-2` keeps working."""
    if len(argv) >= 2 and argv[0] in ("transcribe", "serve") and not argv[1].startswith("-") and "--model" not in argv:
        name = argv[1]
        if argv[0] == "serve" or (not Path(name).exists() and len(argv) >= 3):
            return [argv[0], *argv[2:], "--model", name]
    return argv


if __name__ == "__main__":
    main()

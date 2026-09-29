"""Hand-rolled RFC 6455 (WebSocket) framing — standard library only.

`fermion serve` speaks HTTP through `http.server` with no wheel dependency,
and the streaming speech endpoint (`GET /v1/audio/stream`) keeps that rule:
this module is the server end of RFC 6455 written directly on the handler's
socket file objects, because pulling in `websockets`/`wsproto`/an ASGI stack
to move small binary frames over one loopback socket would break the package's
stdlib-only serving doctrine for no capability we need.

WHAT IS IMPLEMENTED — the subset a streaming-transcription client exercises:

  * the opening handshake digest (`accept_key`: SHA-1 of key+GUID, base64);
  * frame DECODE: FIN, opcodes text/binary/ping/pong/close, client masking,
    all three payload-length encodings (7-bit, 126 -> u16, 127 -> u64);
  * frame ENCODE: server frames (unmasked, per RFC) and client frames
    (masked — used by the selftest's reference client);
  * close payloads (u16 status code + UTF-8 reason).

WHAT IS REFUSED, DELIBERATELY: extensions (any RSV bit is a protocol error —
none are ever negotiated) and message fragmentation. A fragmented message is
detected honestly (`fin` is returned to the caller) and the serve layer
answers it with a clean in-band error instead of silently mis-assembling
audio. Live dictation clients send small self-contained frames; v1 does not
need reassembly, and a wrong reassembly would corrupt the audio it carries.

The byte-level cases (length classes both directions, masking, control-frame
limits) are pinned by `fermion/_speech/selftest.py`, including the RFC 6455
§1.3 known-answer handshake vector.
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct

#: RFC 6455 §1.3 — the fixed GUID every conforming handshake concatenates.
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes (RFC 6455 §5.2).
CONTINUATION, TEXT, BINARY = 0x0, 0x1, 0x2
CLOSE, PING, PONG = 0x8, 0x9, 0xA
CONTROL_OPS = (CLOSE, PING, PONG)

#: Frame payload cap — the same 32 MB the HTTP routes enforce via MAX_BODY,
#: for the same reason: a length field is a promise the peer can make us
#: allocate against, so it is bounded before a single byte is read.
MAX_FRAME = 32 * 1024 * 1024


class ProtocolError(Exception):
    """The peer sent bytes that are not legal RFC 6455 for this session."""


class ConnectionClosed(Exception):
    """The socket ended mid-frame (EOF), with no close handshake."""


def accept_key(key: str) -> str:
    """`Sec-WebSocket-Accept` for a client's `Sec-WebSocket-Key`."""
    digest = hashlib.sha1((key.strip() + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def close_payload(code: int = 1000, reason: str = "") -> bytes:
    """A close frame body: u16 status code + UTF-8 reason."""
    return struct.pack(">H", int(code)) + reason.encode("utf-8")


def parse_close(payload: bytes) -> tuple[int | None, str]:
    """(code, reason) out of a close frame body; (None, '') when empty."""
    if len(payload) < 2:
        return None, ""
    (code,) = struct.unpack(">H", payload[:2])
    return code, payload[2:].decode("utf-8", "replace")


def encode_frame(opcode: int, payload: bytes = b"", *, fin: bool = True,
                 mask: bool = False) -> bytes:
    """One frame, wire-ready. Server frames use the default `mask=False`
    (RFC 6455 §5.1: a server MUST NOT mask); `mask=True` is the client side,
    used by the selftest's reference client."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    head = bytearray([(0x80 if fin else 0x00) | (opcode & 0x0F)])
    n = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if n < 126:
        head.append(mask_bit | n)
    elif n < (1 << 16):
        head.append(mask_bit | 126)
        head += struct.pack(">H", n)
    else:
        head.append(mask_bit | 127)
        head += struct.pack(">Q", n)
    if not mask:
        return bytes(head) + payload
    key = os.urandom(4)
    return bytes(head) + key + _mask(payload, key)


def _mask(payload: bytes, key: bytes) -> bytes:
    """XOR-mask (its own inverse), done wide instead of per byte."""
    if not payload:
        return payload
    reps = -(-len(payload) // 4)
    pad = (key * reps)[:len(payload)]
    return (int.from_bytes(payload, "little")
            ^ int.from_bytes(pad, "little")).to_bytes(len(payload), "little")


def _read_exact(rfile, n: int) -> bytes:
    """`n` bytes or ConnectionClosed — a frame is all-or-nothing."""
    buf = b""
    while len(buf) < n:
        chunk = rfile.read(n - len(buf))
        if not chunk:
            raise ConnectionClosed("the peer closed the socket mid-frame")
        buf += chunk
    return buf


def read_frame(rfile, *, require_mask: bool = False,
               max_len: int = MAX_FRAME) -> tuple[bool, int, bytes]:
    """-> (fin, opcode, payload), with the payload already unmasked.

    `require_mask=True` is the SERVER position: RFC 6455 §5.1 requires every
    client frame to be masked, and an unmasked one is a protocol error, not a
    tolerable dialect. Length promises above `max_len` are refused before any
    payload is read.
    """
    b1, b2 = _read_exact(rfile, 2)
    fin = bool(b1 & 0x80)
    if b1 & 0x70:
        raise ProtocolError("RSV bits set, but no extension was negotiated")
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    n = b2 & 0x7F
    if opcode in CONTROL_OPS:
        if not fin:
            raise ProtocolError("fragmented control frame")
        if n > 125:
            raise ProtocolError("control frame payload over 125 bytes")
    if n == 126:
        (n,) = struct.unpack(">H", _read_exact(rfile, 2))
    elif n == 127:
        (n,) = struct.unpack(">Q", _read_exact(rfile, 8))
    if n > max_len:
        raise ProtocolError(f"frame of {n} bytes exceeds the "
                            f"{max_len}-byte cap")
    if require_mask and not masked:
        raise ProtocolError("client frames must be masked (RFC 6455 §5.1)")
    key = _read_exact(rfile, 4) if masked else None
    payload = _read_exact(rfile, n) if n else b""
    if key is not None:
        payload = _mask(payload, key)
    return fin, opcode, payload

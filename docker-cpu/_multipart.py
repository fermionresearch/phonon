"""Minimal `multipart/form-data` parser — stdlib only, byte-exact.

WHY THIS EXISTS AT ALL
----------------------
`/v1/audio/transcriptions` is the Whisper API shape: a `multipart/form-data`
POST with a binary `file` part and a text `model` part. `fermion/server.py` is
"standard library only (`http.server` + `threading`): no FastAPI, no uvicorn,
no new wheel dependency", and adding one for a form parser would break that for
every Neutrino user who will never transcribe anything.

WHY NOT `email` OR `cgi`
------------------------
`cgi.FieldStorage` was removed from the standard library in Python 3.13, and
this package supports 3.10-3.13+. `email.parser.BytesParser` survives and can
do it, but it reaches the payload through a text round-trip
(`raw-unicode-escape`) that has to be exactly undone by
`get_payload(decode=True)`; for a 25 MB binary upload that is both a copy too
many and a correctness argument nobody should have to make. Splitting on the
boundary directly is ~60 lines, allocates once per part, and is trivially
testable against adversarial payloads (a body containing CRLFs, and a body
containing the boundary bytes themselves in the middle of the audio).

WHAT IT DELIBERATELY DOES NOT DO
---------------------------------
No nested multipart, no `multipart/mixed`, no base64/quoted-printable transfer
encodings, no RFC 2231 continuation filenames. Nothing that sends audio to a
Whisper-shaped endpoint uses any of them, and each one is a parsing surface
that would need its own gate.
"""
from __future__ import annotations

_CRLF = b"\r\n"


class MultipartError(ValueError):
    """The body is not a form this parser will accept."""


def boundary_from(content_type: str) -> bytes:
    """Extract and validate the boundary from a Content-Type header."""
    if not content_type:
        raise MultipartError("missing Content-Type")
    main, _, rest = content_type.partition(";")
    if main.strip().lower() != "multipart/form-data":
        raise MultipartError(
            f"expected multipart/form-data, got {main.strip()!r}")
    for param in rest.split(";"):
        key, _, value = param.partition("=")
        if key.strip().lower() != "boundary":
            continue
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if not value:
            break
        # RFC 2046: 1-70 chars from a restricted set. We only need it to be
        # non-empty and ASCII; a boundary we cannot encode is not one the
        # client can have sent.
        try:
            return value.encode("ascii")
        except UnicodeEncodeError:
            raise MultipartError("non-ASCII multipart boundary") from None
    raise MultipartError("multipart/form-data with no boundary parameter")


def _split_headers(chunk: bytes) -> tuple[dict, bytes]:
    head, sep, body = chunk.partition(_CRLF + _CRLF)
    if not sep:
        # Tolerate a bare-LF client (curl always sends CRLF, but a hand-rolled
        # client may not). Falling over on this would be an unhelpful 400.
        head, sep, body = chunk.partition(b"\n\n")
        if not sep:
            # Almost always one cause: the client picked a boundary string that
            # also occurs inside the file it is uploading, so the split landed
            # mid-payload. RFC 2046 forbids that outright, and no correct
            # client does it — but "a form part had no header block" is a
            # useless thing to read, so say what actually happened.
            raise MultipartError(
                "a form part had no header block — this usually means the "
                "multipart boundary also occurs inside the uploaded file, "
                "which RFC 2046 does not allow; have your client pick a "
                "random boundary")
    headers = {}
    for line in head.replace(b"\r\n", b"\n").split(b"\n"):
        if not line.strip():
            continue
        name, _, value = line.partition(b":")
        headers[name.strip().lower().decode("latin-1")] = \
            value.strip().decode("latin-1")
    return headers, body


def _disposition_params(value: str) -> dict:
    out = {}
    for param in value.split(";")[1:]:
        key, _, raw = param.partition("=")
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1].replace('\\"', '"')
        out[key.strip().lower()] = raw
    return out


def parse(body: bytes, content_type: str, *, max_parts: int = 32) -> dict:
    """Parse a form body into `{field_name: value}`.

    A part with a `filename` yields `{"filename": str, "content": bytes,
    "content_type": str}`; a plain field yields a `str` decoded as UTF-8 (with
    `replace`, because a mis-encoded `model` field must produce a 400 about the
    model, not a UnicodeDecodeError traceback).

    Duplicate field names keep the FIRST occurrence, matching what every
    OpenAI-compatible server does with a repeated `model`.
    """
    boundary = boundary_from(content_type)
    delim = b"--" + boundary
    # A well-formed body starts with the delimiter; tolerate leading preamble
    # (RFC 2046 allows it) by finding the first delimiter instead of assuming.
    start = body.find(delim)
    if start < 0:
        raise MultipartError("no multipart boundary found in the request body")
    segments = body[start:].split(_CRLF + delim)
    # `segments[0]` still carries the opening delimiter; strip it and its CRLF.
    segments[0] = segments[0][len(delim):]
    if segments[0].startswith(_CRLF):
        segments[0] = segments[0][2:]
    elif segments[0].startswith(b"\n"):
        segments[0] = segments[0][1:]

    fields: dict = {}
    for index, segment in enumerate(segments):
        if segment.startswith(b"--"):        # the closing delimiter: done
            break
        if index and segment.startswith(_CRLF):
            segment = segment[2:]
        elif index and segment.startswith(b"\n"):
            segment = segment[1:]
        if not segment.strip():
            continue
        if len(fields) >= max_parts:
            raise MultipartError(f"more than {max_parts} form parts")
        headers, content = _split_headers(segment)
        disposition = headers.get("content-disposition", "")
        if "form-data" not in disposition.lower():
            continue
        params = _disposition_params(disposition)
        name = params.get("name")
        if not name:
            continue
        if name in fields:
            continue
        if "filename" in params:
            fields[name] = {
                "filename": params["filename"],
                "content": content,
                "content_type": headers.get("content-type",
                                            "application/octet-stream"),
            }
        else:
            fields[name] = content.decode("utf-8", "replace")
    return fields

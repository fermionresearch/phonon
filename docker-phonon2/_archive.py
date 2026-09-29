"""Download, pin-verify and unpack a published Phonon release archive.

Each model repo publishes one byte-plane-split archive
(``phonon-<profile>.bps.tar.zst``) plus its reference unpacker. This module
reconstructs the model directory from that archive with the same algorithm
the reference unpacker documents, using the ``zstandard`` stream reader
(robust to symlinked download paths), and proves byte-identity twice:

* the archive itself is checked against the SHA-256 + byte-count pin this
  image ships for the release, before unpacking;
* every reconstructed member is checked against the per-file SHA-256 the
  archive's own manifest carries, and every model shard is re-checked
  against ``packed_manifest.json`` after unpacking.

A mismatch at any stage refuses — nothing partially-verified is ever left
where the loader would find it (unpacking happens in a ``.partial``
directory that is renamed only after every check passes).
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_planes(data: bytes, meta: dict) -> bytes:
    """Invert the byte-plane-split transport transform for one member."""
    raw = np.frombuffer(data, dtype=np.uint8)
    base = meta["base"]
    out = np.empty(meta["original_bytes"], dtype=np.uint8)
    out[:base] = raw[:base]
    src = base
    cursor = 0
    for start, end, itemsize in meta["plan"]:
        if start > cursor:
            width = start - cursor
            out[base + cursor: base + start] = raw[src: src + width]
            src += width
        n = end - start
        per = n // itemsize
        block = raw[src: src + n].reshape(itemsize, per)
        out[base + start: base + end] = block.T.reshape(-1)
        src += n
        cursor = end
    remaining = meta["payload_bytes"] - cursor
    if remaining:
        out[base + cursor:] = raw[src: src + remaining]
    return out.tobytes()


def unpack_archive(archive: Path, dest: Path) -> int:
    """Reconstruct a model directory from a release archive. Returns the
    number of files written; raises on any checksum mismatch."""
    import zstandard

    dest.mkdir(parents=True, exist_ok=True)
    manifest = None
    index: dict = {}
    written: list[str] = []
    with open(archive, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                blob = handle.read()
                if member.name == "bps_manifest.json":
                    manifest = json.loads(blob)
                    index = {row["path"]: row for row in manifest["files"]}
                    continue
                if manifest is None:
                    raise RuntimeError(
                        "bps_manifest.json must be the first archive member")
                rel = (member.name[:-4] if member.name.endswith(".bps")
                       else member.name)
                row = index[rel]
                data = (_join_planes(blob, row["transform"])
                        if "transform" in row else blob)
                if (hashlib.sha256(data).hexdigest() != row["original_sha256"]
                        or len(data) != row["original_bytes"]):
                    raise RuntimeError(f"checksum mismatch on {rel}")
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written.append(rel)
    missing = {row["path"] for row in manifest["files"]} - set(written)
    if missing:
        raise RuntimeError(f"archive is missing members: {sorted(missing)}")
    return len(written)


def verify_shards(model_dir: Path) -> int:
    """Re-check every model shard against packed_manifest.json."""
    manifest = json.loads((model_dir / "packed_manifest.json").read_text())
    for row in manifest.get("shards", []):
        path = model_dir / row["name"]
        if (not path.is_file() or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]):
            raise RuntimeError(f"model shard failed verification: {row['name']}")
    return len(manifest.get("shards", []))


def ensure_model(repo: str, filename: str, sha256: str, nbytes: int,
                 dest: Path, log=None) -> Path:
    """Download + verify + unpack one published model, once.

    ``dest`` is the final model directory. If it already holds a verified
    layout marker it is reused without touching the network.
    """
    log = log or (lambda message: None)
    dest = Path(dest)
    if (dest / "packed_manifest.json").is_file() and \
            (dest / "config.json").is_file():
        return dest

    from huggingface_hub import hf_hub_download

    # The root config.json is fetched first: it is the request the Hub
    # counts as a download, and it lets a client cross-check the release
    # facts before a multi-hundred-megabyte transfer.
    try:
        hf_hub_download(repo, "config.json")
    except Exception:
        pass  # the archive pin below stays authoritative either way

    log(f"downloading {repo}/{filename} "
        f"({nbytes / 1e6:.0f} MB; set --model-dir to use a local copy) ...")
    archive = Path(hf_hub_download(repo, filename)).resolve()
    got_sha = sha256_file(archive)
    got_bytes = archive.stat().st_size
    if got_sha != sha256 or got_bytes != nbytes:
        raise RuntimeError(
            f"{repo}/{filename} does not match its release pin "
            f"(sha {got_sha[:16]}…, {got_bytes} bytes) — refusing to unpack. "
            f"Update this image if a new release has shipped.")
    log(f"archive verified (sha256 {got_sha[:16]}…); unpacking ...")

    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    files = unpack_archive(archive, partial)
    shards = verify_shards(partial)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(dest)
    log(f"unpacked {files} files, {shards} shards verified -> {dest}")
    return dest

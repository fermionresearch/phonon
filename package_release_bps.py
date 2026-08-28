#!/usr/bin/env python3
"""Byte-plane-split release packer/unpacker.

A BF16 array interleaves a highly predictable exponent byte with a near-random
mantissa byte.  Compressed interleaved, zstd models neither population well.
Split into per-byte-position planes, the exponent plane compresses hard.  U8
code planes are left untouched.

This is a *transport* transform only:

* the transform is applied per tensor, using the dtype in the safetensors
  header, so it never has to guess;
* bytes not covered by any tensor (the header, alignment padding) are copied
  verbatim;
* ``unpack`` reconstructs the original file and the manifest carries the
  original SHA-256 and length of every member, so an install proves
  byte-identity before anything is used.

Runtime tensors are therefore provably unchanged and accuracy cannot move.

    python package_release_bps.py pack   parity --level 19
    python package_release_bps.py unpack local_stt/releases_bps/phonon-parity.bps.tar.zst DEST
    python package_release_bps.py verify parity          # full roundtrip proof
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "parity": ROOT / "model_v18_mlx_quint5",
    "micro": ROOT / "model_v18_mlx_hybrid4_quint5",
    "audio6": ROOT / "model_v18_mlx_head8audio6_quint5",
}
ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
            "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8, "F8_E4M3": 1,
            "F8_E5M2": 1}
BLOCK = 16 << 20
FORMAT = "phonon-byteplane-tar-zstd-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_spans(raw: np.ndarray):
    """Return (base, [(start, end, itemsize), ...]) sorted, non-overlapping."""
    header_len = int.from_bytes(raw[:8].tobytes(), "little")
    header = json.loads(raw[8:8 + header_len].tobytes())
    base = 8 + header_len
    spans = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        itemsize = ITEMSIZE[meta["dtype"]]
        if itemsize > 1 and (end - start) % itemsize == 0:
            spans.append((int(start), int(end), itemsize))
    spans.sort()
    merged = []
    last_end = 0
    for start, end, itemsize in spans:
        if start < last_end:                       # overlapping/aliased tensors
            continue
        merged.append((start, end, itemsize))
        last_end = end
    return base, merged


def split_file(path: Path) -> tuple[bytes, dict]:
    raw = np.fromfile(path, dtype=np.uint8)
    base, spans = tensor_spans(raw)
    out = io.BytesIO()
    out.write(raw[:base].tobytes())               # header verbatim
    cursor = 0
    plan = []
    for start, end, itemsize in spans:
        if start > cursor:                         # padding / uncovered bytes
            out.write(raw[base + cursor: base + start].tobytes())
        chunk = raw[base + start: base + end]
        for i in range(itemsize):
            out.write(chunk[i::itemsize].tobytes())
        plan.append([start, end, itemsize])
        cursor = end
    tail = raw[base + cursor:]
    if tail.size:
        out.write(tail.tobytes())
    meta = {
        "base": base,
        "plan": plan,
        "payload_bytes": int(raw.size - base),
        "original_bytes": int(raw.size),
        "original_sha256": sha256_bytes(raw.tobytes()),
    }
    return out.getvalue(), meta


def join_file(data: bytes, meta: dict) -> bytes:
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


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def pack(profile: str, level: int, out_dir: Path) -> dict:
    source = PROFILES[profile]
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"phonon-{profile}.bps.tar.zst"

    members = sorted(p for p in source.rglob("*") if p.is_file())
    manifest = {
        "release_format": FORMAT,
        "profile": profile,
        "compression": {"codec": "zstd", "level": level},
        "transform": "byte-plane-split-per-tensor-v1",
        "files": [],
    }
    payloads: list[tuple[str, bytes]] = []
    for path in members:
        rel = str(path.relative_to(source))
        blob = path.read_bytes()
        entry = {"path": rel, "original_bytes": len(blob),
                 "original_sha256": sha256_bytes(blob)}
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            transformed, meta = split_file(path)
            entry["transform"] = meta
            entry["stored_bytes"] = len(transformed)
            payloads.append((rel + ".bps", transformed))
        else:
            entry["stored_bytes"] = len(blob)
            payloads.append((rel, blob))
        manifest["files"].append(entry)

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    started = time.perf_counter()
    args = ["zstd", "-q", f"-{level}", "-T0", "-f", "-o", str(archive), "-"]
    if level >= 20:
        args.insert(1, "--ultra")
    process = subprocess.Popen(args, stdin=subprocess.PIPE)
    assert process.stdin is not None
    with tarfile.open(fileobj=process.stdin, mode="w|") as tar:
        tar.addfile(tar_info("bps_manifest.json", len(manifest_bytes)),
                    io.BytesIO(manifest_bytes))
        for name, blob in payloads:
            tar.addfile(tar_info(name, len(blob)), io.BytesIO(blob))
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("zstd failed")
    pack_s = time.perf_counter() - started
    return {"profile": profile, "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "source_bytes": sum(p.stat().st_size for p in members),
            "level": level, "pack_seconds": pack_s,
            "archive_sha256": sha256_file(archive)}


def unpack(archive: Path, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(archive)],
                               stdout=subprocess.PIPE)
    assert process.stdout is not None
    manifest = None
    written = []
    with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
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
                raise RuntimeError("bps_manifest.json must be the first member")
            rel = member.name[:-4] if member.name.endswith(".bps") else member.name
            row = index[rel]
            data = join_file(blob, row["transform"]) if "transform" in row else blob
            got = sha256_bytes(data)
            if got != row["original_sha256"] or len(data) != row["original_bytes"]:
                raise RuntimeError(f"checksum mismatch on {rel}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written.append(rel)
    if process.wait() != 0:
        raise RuntimeError("zstd decompression failed")
    missing = {row["path"] for row in manifest["files"]} - set(written)
    if missing:
        raise RuntimeError(f"archive is missing members: {sorted(missing)}")
    return {"files": len(written), "unpack_seconds": time.perf_counter() - started}


def verify(profile: str, level: int, out_dir: Path) -> dict:
    """Pack, unpack to a temporary directory, and prove byte-identity."""
    packed = pack(profile, level, out_dir)
    source = PROFILES[profile]
    with tempfile.TemporaryDirectory() as tmp:
        stats = unpack(Path(packed["archive"]), Path(tmp))
        mismatched = []
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            rel = path.relative_to(source)
            other = Path(tmp) / rel
            if not other.exists() or sha256_file(other) != sha256_file(path):
                mismatched.append(str(rel))
    packed.update(stats)
    packed["roundtrip_byte_identical"] = not mismatched
    packed["mismatched"] = mismatched
    return packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("pack", "unpack", "verify"))
    ap.add_argument("target")
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--level", type=int, default=19)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "releases_bps")
    args = ap.parse_args()

    if args.command == "unpack":
        print(json.dumps(unpack(Path(args.target), Path(args.dest)), indent=2))
        return
    fn = pack if args.command == "pack" else verify
    result = fn(args.target, args.level, args.out_dir)
    ratio = 100 * result["archive_bytes"] / result["source_bytes"]
    result["percent_of_source"] = ratio
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"{args.target}: {result['source_bytes']/1e6:.1f} MB -> "
          f"{result['archive_bytes']/1e6:.1f} MB ({ratio:.2f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify a Phonon deployment artifact without loading or mutating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = {
    "parity": ROOT / "model_v18_mlx_quint5",
    "audio6": ROOT / "model_v18_mlx_head8audio6_quint5",
    "micro": ROOT / "model_v18_mlx_hybrid4_quint5",
}
# The three published models, and the only values shown in `--help`.
PUBLIC_PROFILES = ("parity", "audio6", "micro")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    # Every key in MODELS stays valid; only the three published models are
    # advertised.  `metavar` controls the help text, `choices` controls what is
    # accepted, so the unpublished ones remain checkable by name.
    parser.add_argument(
        "--profile",
        choices=MODELS,
        metavar="{" + ",".join(PUBLIC_PROFILES) + "}",
        default="audio6",
    )
    args = parser.parse_args()
    model = MODELS[args.profile]
    if platform.machine() != "arm64":
        raise RuntimeError("the optimized local runtime requires Apple Silicon")
    manifest = json.loads((model / "packed_manifest.json").read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("packed manifest is not PASS")
    if manifest.get("source_checkpoint_sha256") != (
        "27f01f214a0c0916944118458d0f43791b5377431fb6a230d7a2f4248368a49e"
    ):
        raise RuntimeError("this artifact does not match the published Phonon-1 release checkpoint")
    if len(manifest.get("modules", [])) != 196:
        raise RuntimeError("expected 196 packed decoder layers")
    total = 0
    for row in manifest["shards"]:
        path = model / row["name"]
        if path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"size mismatch: {path.name}")
        actual = sha256(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path.name}")
        total += path.stat().st_size
        print(f"PASS {path.name} {actual[:16]}…")
    if total != manifest["total_bytes"]:
        raise RuntimeError("packed byte total mismatch")
    print(
        f"PASS Phonon {args.profile} model: {len(manifest['modules'])} linears, "
        f"{total / 1e9:.3f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

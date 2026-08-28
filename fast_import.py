"""Trim the V18 inference cold start from ~1.4 s of imports to ~0.1 s.

Measured: importing the V18
runtime costs 1.34-1.62 s, of which ``import mlx_lm`` alone is 1.307 s, while
loading the weights is only 237-317 ms.  None of that 1.3 s is work the
inference path uses.  Two eager package ``__init__`` files are responsible:

* ``mlx_audio/stt/models/__init__.py`` imports all 15 STT model families
  (cohere_asr, whisper, voxtral, ...) to reach ``qwen3_asr``;
* ``mlx_lm/__init__.py`` imports ``convert`` / ``generate`` / ``utils`` ->
  ``tokenizer_utils`` -> ``transformers`` -> ``torch`` -> ``torch._dynamo``,
  to reach ``mlx_lm.models.base``, which itself imports only ``inspect``,
  ``dataclasses``, ``typing`` and ``mlx``.

``enable()`` pre-registers lightweight stand-ins for exactly those two package
``__init__`` bodies, with ``__path__`` intact so ordinary submodule imports keep
working.  Nothing upstream is patched or written to.

This is a **strictly optional accelerator**:

* it refuses to act if any affected package is already imported, so it can never
  half-apply;
* it refuses to act unless every module file it expects is present, so a
  different upstream version simply falls back to the normal import;
* ``verify()`` re-imports the concrete symbols the runtime needs and returns
  False if any is missing, so a caller can fall back before doing real work.

Callers that do not opt in are completely unaffected.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

# Package __init__ bodies that are skipped, and one module inside each that must
# exist for the skip to be safe.
_STUBBED = {
    "mlx_lm": "models/base.py",
    "mlx_lm.models": "base.py",
    "mlx_audio.stt.models": "qwen3_asr/qwen3_asr.py",
}
# Symbols the V18 inference path actually imports out of the stubbed packages.
_REQUIRED = (
    ("mlx_lm.models.base", ("create_attention_mask", "scaled_dot_product_attention")),
    ("mlx_lm.sample_utils", ("make_logits_processors",)),
    ("mlx_audio.stt.models.qwen3_asr", ("Model", "ModelConfig")),
    ("mlx_audio.stt.models.base", ("STTOutput",)),
)


def _package_dir(name: str) -> Path | None:
    """Locate a package directory without executing its ``__init__``."""
    parts = name.split(".")
    search = None
    path = None
    for index, part in enumerate(parts):
        full = ".".join(parts[: index + 1])
        module = sys.modules.get(full)
        if module is not None and getattr(module, "__path__", None):
            path = Path(list(module.__path__)[0])
            search = [str(path)]
            continue
        spec = importlib.util.find_spec(full, None) if search is None else \
            importlib.machinery.PathFinder.find_spec(part, search)
        if spec is None or not spec.submodule_search_locations:
            return None
        path = Path(list(spec.submodule_search_locations)[0])
        search = [str(path)]
    return path


def available() -> bool:
    """True when the fast path can be installed cleanly right now."""
    if any(name in sys.modules for name in _STUBBED):
        return False
    for name, probe in _STUBBED.items():
        directory = _package_dir(name)
        if directory is None or not (directory / probe).exists():
            return False
    return True


def enable() -> bool:
    """Register the stub packages.  Returns True if the fast path is active."""
    if not available():
        return False
    # mlx_lm/__init__ sets this before importing transformers; preserve it so
    # behaviour does not change for anything that imports transformers later.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    installed: list[str] = []
    try:
        for name in _STUBBED:
            directory = _package_dir(name)
            module = types.ModuleType(name)
            module.__path__ = [str(directory)]
            module.__package__ = name
            module.__spec__ = importlib.machinery.ModuleSpec(
                name, None, origin=str(directory / "__init__.py"), is_package=True
            )
            module.__spec__.submodule_search_locations = module.__path__
            module.__fast_import_stub__ = True
            sys.modules[name] = module
            installed.append(name)
            parent, _, leaf = name.rpartition(".")
            if parent and parent in sys.modules:
                setattr(sys.modules[parent], leaf, module)
    except Exception:
        for name in installed:
            sys.modules.pop(name, None)
        return False
    return True


def verify() -> bool:
    """Import every symbol the runtime needs; False means fall back."""
    try:
        for module_name, symbols in _REQUIRED:
            module = importlib.import_module(module_name)
            for symbol in symbols:
                getattr(module, symbol)
    except Exception:
        return False
    return True


def disable() -> None:
    """Remove the stubs (and anything imported through them)."""
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in ("mlx_lm", "mlx_audio"):
            del sys.modules[name]


def enable_and_verify() -> bool:
    """Install the fast path, proving it works; restore normal imports if not."""
    if not enable():
        return False
    if verify():
        return True
    disable()
    return False


def active() -> bool:
    return any(
        getattr(sys.modules.get(name), "__fast_import_stub__", False)
        for name in _STUBBED
    )

"""Python ownership wrapper for the optional Phonon CUDA fold4 library.

This is a backend boundary, not a complete ASR engine. The complete CUDA
decoder will compose these matrices with the portable audio tower, attention,
KV cache, tokenizer, and generation loop. Until those pieces and transcript
gates pass, the public CLI remains on its existing Apple/MLX path.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np

from phonon_cuda_artifact import Fold4Matrix, load_fold4_matrix


ABI_VERSION: Final = 1


def _default_library() -> Path:
    override = os.environ.get("PHONON_CUDA_LIBRARY")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().with_name("libphonon_fold4_cuda.so")


class CudaKernelLibrary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_library()
        if not self.path.is_file():
            raise RuntimeError(
                f"Phonon CUDA kernel library is not installed: {self.path}")
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.phonon_fold4_cuda_abi_version.restype = ctypes.c_int
        self.lib.phonon_fold4_cuda_create.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
        )
        self.lib.phonon_fold4_cuda_create.restype = ctypes.c_void_p
        self.lib.phonon_fold4_cuda_run_device.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self.lib.phonon_fold4_cuda_run_device.restype = ctypes.c_int
        self.lib.phonon_fold4_cuda_run_host.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        )
        self.lib.phonon_fold4_cuda_run_host.restype = ctypes.c_int
        self.lib.phonon_fold4_cuda_destroy.argtypes = (ctypes.c_void_p,)
        self.lib.phonon_fold4_cuda_destroy.restype = None
        actual = self.lib.phonon_fold4_cuda_abi_version()
        if actual != ABI_VERSION:
            raise RuntimeError(
                f"Phonon CUDA ABI {actual} != Python ABI {ABI_VERSION}")

    def matrix(self, weights: Fold4Matrix) -> "Fold4CudaMatrix":
        return Fold4CudaMatrix(self, weights)

    def matrix_from_artifact(
        self, model_dir: str | Path, module_name: str
    ) -> "Fold4CudaMatrix":
        return self.matrix(load_fold4_matrix(model_dir, module_name))


class Fold4CudaMatrix:
    def __init__(self, library: CudaKernelLibrary, weights: Fold4Matrix):
        self.library = library
        self.weights = weights
        self._handle = library.lib.phonon_fold4_cuda_create(
            weights.rows,
            weights.cols,
            weights.codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            weights.scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weights.centers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if not self._handle:
            raise RuntimeError(
                f"CUDA matrix creation failed for {weights.name} "
                f"[{weights.rows}, {weights.cols}]")

    def close(self) -> None:
        if self._handle:
            self.library.lib.phonon_fold4_cuda_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Fold4CudaMatrix":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - best-effort process shutdown path
        try:
            self.close()
        except Exception:
            pass

    def run_host(self, x) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (self.weights.cols,):
            raise ValueError(
                f"input shape {x.shape} != ({self.weights.cols},)")
        y = np.empty(self.weights.rows, dtype=np.float32)
        status = self.library.lib.phonon_fold4_cuda_run_host(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status:
            raise RuntimeError(f"CUDA fold4 host run failed: status {status}")
        return y

    def run_device(
        self, x_pointer: int, y_pointer: int, stream_pointer: int = 0
    ) -> None:
        if not x_pointer or not y_pointer:
            raise ValueError("device x/y pointers must be nonzero")
        status = self.library.lib.phonon_fold4_cuda_run_device(
            self._handle,
            ctypes.c_void_p(x_pointer),
            ctypes.c_void_p(y_pointer),
            ctypes.c_void_p(stream_pointer),
        )
        if status:
            raise RuntimeError(f"CUDA fold4 device run failed: status {status}")

"""ctypes ownership wrapper for the Phonon CPU fold4 kernel library.

FP32 in/out. Variants: 0 = f32 SIMD dequantization, 1 = int8 activations,
2 = table lookup over 12-bit activations — the shipped-quality default and
the only variant this image runs.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

V_F32, V_I8DOT, V_LUT = 0, 1, 2
VARIANTS = {"f32": V_F32, "i8dot": V_I8DOT, "lut": V_LUT}


def default_library() -> Path:
    override = os.environ.get("PHONON_CPU_LIBRARY")
    if override:
        return Path(override).expanduser()
    name = ("libphonon_cpu-macos-arm64.dylib"
            if os.uname().sysname == "Darwin"
            else "libphonon_cpu-linux-x86_64.so")
    return Path(__file__).resolve().parent / "bin" / name


class CpuKernelLibrary:
    def __init__(self, path: str | Path | None = None, *,
                 nthreads: int = 6, max_cols: int = 3072,
                 spin_iters: int = 20000):
        self.path = Path(path) if path is not None else default_library()
        if not self.path.is_file():
            raise RuntimeError(f"Phonon CPU kernel library missing: {self.path}")
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.phonon_cpu_init.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_long)
        self.lib.phonon_cpu_init.restype = ctypes.c_int
        self.lib.phonon_cpu_matrix_create.argtypes = (
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8))
        self.lib.phonon_cpu_matrix_create.restype = ctypes.c_long
        self.lib.phonon_cpu_matvec.argtypes = (
            ctypes.c_long, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
        self.lib.phonon_cpu_matvec.restype = None
        # Raw-pointer binding without per-call ctypes.cast (hot decode path).
        self._matvec_raw = ctypes.CFUNCTYPE(
            None, ctypes.c_long, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p)(
            ("phonon_cpu_matvec", self.lib))
        if self.lib.phonon_cpu_init(nthreads, max_cols, spin_iters) != 0:
            raise RuntimeError("phonon_cpu_init failed")
        self.nthreads = nthreads

    def matrix(self, rows: int, cols: int, codes: np.ndarray,
               scales: np.ndarray, centers: np.ndarray) -> int:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        centers = np.ascontiguousarray(centers, dtype=np.uint8)
        assert codes.shape == (rows, cols // 2)
        handle = self.lib.phonon_cpu_matrix_create(
            rows, cols,
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            centers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if handle < 0:
            raise RuntimeError("phonon_cpu_matrix_create failed")
        return int(handle)

    def matvec_ptr(self, handle: int, variant: int, x_ptr: int, y_ptr: int) -> None:
        """Raw-pointer path for torch tensors (avoids numpy round-trips)."""
        self._matvec_raw(handle, variant, x_ptr, y_ptr)

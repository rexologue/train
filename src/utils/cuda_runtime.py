from __future__ import annotations

import ctypes
import os
from pathlib import Path
import site
import sys
from typing import Iterable


def _candidate_roots() -> Iterable[Path]:
    seen: set[Path] = set()
    raw_roots = [Path(sys.prefix), Path(sys.base_prefix)]
    raw_roots.extend(Path(p) for p in site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        raw_roots.append(Path(user_site))

    for root in raw_roots:
        if root in seen:
            continue
        seen.add(root)
        yield root


def _candidate_libcudart_paths() -> list[Path]:
    candidates: list[Path] = []

    relative_paths = (
        "lib/libcudart.so.12",
        "nvidia/cuda_runtime/lib/libcudart.so.12",
    )

    for root in _candidate_roots():
        for rel in relative_paths:
            path = root / rel
            if path.exists():
                candidates.append(path)

    for root in _candidate_roots():
        try:
            candidates.extend(root.glob("**/nvidia/cuda_runtime/lib/libcudart.so.12"))
        except OSError:
            continue

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def preload_cuda_runtime() -> Path | None:
    """Load libcudart from the PyPI NVIDIA runtime wheel before CUDA extensions import.

    Some CUDA extension wheels import before torch has loaded its CUDA runtime libraries.
    In that case the extension loader can fail with:
        ImportError: libcudart.so.12: cannot open shared object file

    Loading the runtime by absolute path with RTLD_GLOBAL makes the symbol provider visible
    to later dlopen() calls without relying on LD_LIBRARY_PATH being set by the shell.
    """

    if os.name == "nt":
        return None

    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    for path in _candidate_libcudart_paths():
        try:
            ctypes.CDLL(str(path), mode=mode)
            return path
        except OSError:
            continue

    return None

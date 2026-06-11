from __future__ import annotations


def bf16_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except ImportError:
        return False


from __future__ import annotations


def assert_packing_disabled(enabled: bool) -> None:
    if enabled:
        raise NotImplementedError("packing requires separate boundary-preserving label tests before use")


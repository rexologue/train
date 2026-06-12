from __future__ import annotations


def assert_resume_hashes_match(expected: dict[str, str], actual: dict[str, str]) -> None:
    mismatched = {key: (expected.get(key), actual.get(key)) for key in expected if expected.get(key) != actual.get(key)}
    if mismatched:
        raise ValueError(f"resume hash mismatch: {mismatched}")


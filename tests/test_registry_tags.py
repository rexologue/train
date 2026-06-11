from __future__ import annotations

import pytest

from qwen35_tuning.registry.tags import candidate_alias, validate_training_aliases


def test_candidate_alias_is_explicit_and_never_champion():
    assert candidate_alias("candidate-{candidate_index:06d}", 1) == "candidate-000001"
    with pytest.raises(ValueError):
        validate_training_aliases(["candidate-000001", "champion"])


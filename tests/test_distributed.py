from __future__ import annotations

from types import SimpleNamespace

import pytest
from accelerate.utils import DistributedType

from trainer.distributed import validate_training_runtime


def test_training_runtime_requires_fsdp():
    with pytest.raises(RuntimeError, match="requires Accelerate/FSDP"):
        validate_training_runtime(SimpleNamespace(distributed_type=DistributedType.NO))


def test_training_runtime_accepts_fsdp():
    validate_training_runtime(SimpleNamespace(distributed_type=DistributedType.FSDP))

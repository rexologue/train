from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from registry.modelctl_client import ModelctlRegisterRequest
from registry.selection import RegistrationDecision


def build_candidate_registration_request(config: Any, decision: RegistrationDecision) -> ModelctlRegisterRequest:
    """Build an explicit candidate registration request for an adapter checkpoint."""

    checkpoint = decision.checkpoint
    training_tags = {
        "training.registry_role": "candidate",
        "training.candidate_index": decision.candidate_index,
        "training.global_step": checkpoint.global_step,
        "training.checkpoint_index": checkpoint.checkpoint_index,
        "training.selection_metric": config.registry.selection.metric,
        "training.selection_metric_value": checkpoint.metric_value,
    }
    general_tags = {
        "artifact.kind": "peft_adapter_checkpoint",
        "artifact.contains_merged_model": False,
    }
    tag_dir = checkpoint.path.parent / "modelctl_tags" / checkpoint.path.name
    tag_dir.mkdir(parents=True, exist_ok=True)
    training_tags_path = tag_dir / "training_tags.json"
    general_tags_path = tag_dir / "general_tags.json"
    write_json(training_tags_path, training_tags)
    write_json(general_tags_path, general_tags)
    return ModelctlRegisterRequest(
        model_name=config.project.name,
        source_dir=checkpoint.path / "adapter",
        aliases=tuple(decision.aliases),
        training_tags_json=training_tags_path,
        general_tags_json=general_tags_path,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a stable JSON object."""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

from __future__ import annotations

from types import SimpleNamespace

import torch

from data.ref_cache import (
    RefLogpCache,
    load_ref_logp_cache,
    load_reusable_entries,
    read_cache_entries,
    ref_cache_paths,
    reference_signature,
    write_ref_logp_cache,
)
from losses.dpo import dpo_loss
from trainer.trainer import RoutedTrainer
from conftest import example_config


class AdapterAwareTinyLogitModel:
    """Reference test double: adapter-on and adapter-off produce different logits."""

    def __init__(self) -> None:
        self.adapter_enabled = True
        self.call_input_shapes: list[tuple[int, ...]] = []

    def disable_adapter(self):
        model = self

        class DisableAdapterContext:
            def __enter__(self):
                self.previous = model.adapter_enabled
                model.adapter_enabled = False
                return model

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                model.adapter_enabled = self.previous
                return False

        return DisableAdapterContext()

    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.call_input_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device)
        if self.adapter_enabled:
            logits[..., 5] = 3.0
            logits[..., 6] = 1.0
        else:
            logits[..., 5] = 2.0
            logits[..., 6] = 2.0
        return SimpleNamespace(logits=logits)


class DummyAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def _dpo_batch() -> dict[str, object]:
    return {
        "loss_kind": "dpo_target",
        "chosen_input_ids": torch.tensor([[1, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1]]),
        "chosen_labels": torch.tensor([[-100, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6]]),
        "rejected_attention_mask": torch.tensor([[1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6]]),
        "chosen_render_hash": ["chosen-hash"],
        "rejected_render_hash": ["rejected-hash"],
    }


def test_ref_logp_cache_lookup_hits_and_misses() -> None:
    cache = RefLogpCache({"a": -1.0, "b": -2.0}, signature="sig")

    assert cache.lookup(["a", "b"]) == [-1.0, -2.0]
    assert cache.lookup(["a", "missing"]) is None
    assert cache.lookup([None]) is None  # type: ignore[list-item]
    assert "a" in cache and "missing" not in cache
    assert len(cache) == 2


def test_dpo_cached_reference_matches_on_the_fly_and_skips_forward() -> None:
    batch = _dpo_batch()

    on_the_fly_model = AdapterAwareTinyLogitModel()
    on_the_fly = dpo_loss(
        on_the_fly_model,
        batch,
        beta=0.1,
        ignore_index=-100,
        accelerator=DummyAccelerator(),
    )
    assert on_the_fly.metrics["dpo/ref_cache_used"] == 0.0
    # policy forward + reference forward
    assert len(on_the_fly_model.call_input_shapes) == 2

    ref_chosen = on_the_fly.metrics["dpo/ref_chosen_logp"]
    ref_rejected = on_the_fly.metrics["dpo/ref_rejected_logp"]

    cached_model = AdapterAwareTinyLogitModel()
    cached = dpo_loss(
        cached_model,
        batch,
        beta=0.1,
        ignore_index=-100,
        accelerator=DummyAccelerator(),
        ref_chosen_logp=torch.tensor([ref_chosen]),
        ref_rejected_logp=torch.tensor([ref_rejected]),
    )

    assert cached.metrics["dpo/ref_cache_used"] == 1.0
    # only the policy forward runs when the reference is supplied
    assert len(cached_model.call_input_shapes) == 1
    assert torch.allclose(cached.loss, on_the_fly.loss)


def test_trainer_uses_cache_when_hashes_present_and_falls_back_otherwise() -> None:
    config = example_config()
    cache = RefLogpCache({"chosen-hash": -0.5, "rejected-hash": -0.9}, signature="sig")

    cached_trainer = RoutedTrainer(
        config=config,
        accelerator=DummyAccelerator(),
        cadence=None,
        ref_logp_cache=cache,
    )
    cached_model = AdapterAwareTinyLogitModel()
    cached_loss = cached_trainer.compute_loss(cached_model, _dpo_batch())
    assert torch.isfinite(cached_loss)
    assert cached_trainer.last_loss_metrics["dpo/ref_cache_used"] == 1.0
    assert len(cached_model.call_input_shapes) == 1

    miss_batch = _dpo_batch()
    miss_batch["chosen_render_hash"] = ["unknown-hash"]
    miss_model = AdapterAwareTinyLogitModel()
    miss_loss = cached_trainer.compute_loss(miss_model, miss_batch)
    assert torch.isfinite(miss_loss)
    assert cached_trainer.last_loss_metrics["dpo/ref_cache_used"] == 0.0
    assert len(miss_model.call_input_shapes) == 2


def test_trainer_without_cache_uses_on_the_fly_reference() -> None:
    trainer = RoutedTrainer(config=example_config(), accelerator=DummyAccelerator(), cadence=None)
    model = AdapterAwareTinyLogitModel()

    loss = trainer.compute_loss(model, _dpo_batch())

    assert torch.isfinite(loss)
    assert trainer.last_loss_metrics["dpo/ref_cache_used"] == 0.0
    assert len(model.call_input_shapes) == 2


def _fake_model_source(source_dir_hash: str = "sha256:model") -> SimpleNamespace:
    return SimpleNamespace(
        effective_model_id="local-dir",
        ref="models:/m@candidate",
        source_dir_hash=source_dir_hash,
        expected_payload_hash=source_dir_hash,
        resolved_version="1",
    )


def test_reference_signature_tracks_model_and_precision(tmp_path) -> None:
    config = example_config(project={"output_dir": str(tmp_path)})
    source = _fake_model_source()

    base = reference_signature(config, source)
    assert base == reference_signature(config, source)
    assert base != reference_signature(config, _fake_model_source("sha256:other-model"))

    other_precision = example_config(project={"output_dir": str(tmp_path)}, model={"precision": "fp16"})
    assert base != reference_signature(other_precision, source)


def test_ref_cache_write_load_roundtrip_and_signature_gating(tmp_path) -> None:
    config = example_config(project={"output_dir": str(tmp_path)})
    source = _fake_model_source()
    signature = reference_signature(config, source)
    entries = {"h1": -1.25, "h2": -3.5}

    write_ref_logp_cache(
        config,
        entries=entries,
        signature=signature,
        model_source=source,
        splits=["train", "valid"],
    )

    assert read_cache_entries(ref_cache_paths(config)) == entries

    loaded = load_ref_logp_cache(config, expected_signature=signature)
    assert loaded is not None
    assert loaded.lookup(["h1", "h2"]) == [-1.25, -3.5]

    assert load_ref_logp_cache(config, expected_signature="different-signature") is None
    assert load_reusable_entries(config, expected_signature=signature) == entries
    assert load_reusable_entries(config, expected_signature="different-signature") == {}


def test_ref_cache_load_detects_corrupted_parquet(tmp_path) -> None:
    config = example_config(project={"output_dir": str(tmp_path)})
    source = _fake_model_source()
    signature = reference_signature(config, source)

    write_ref_logp_cache(
        config,
        entries={"h1": -1.0},
        signature=signature,
        model_source=source,
        splits=["train"],
    )

    # Corrupt the parquet so its checksum no longer matches the manifest.
    paths = ref_cache_paths(config)
    paths.parquet.write_bytes(b"not a parquet file")

    assert load_ref_logp_cache(config, expected_signature=signature) is None

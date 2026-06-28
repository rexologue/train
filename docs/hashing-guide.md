# Hashing Guide

This document maps every config field and external artifact to the hash checks
that depend on it, so you know exactly which operation will break if you change
something — and how to unblock it intentionally.

---

## Overview of hash contexts

There are four independent hash contexts. Each can be triggered separately.

| Context | Stored in | Config flag | Fails at |
|---|---|---|---|
| **Preprocessing signature** | `pretokenized/manifest.json` → `preprocessing_signatures` | n/a (always checked) | `sft-dpo-prepare` cache reuse |
| **Training contract** | checkpoint `manifest.json` → `config_hashes.training_contract` | `resume.strict_config` | `sft-dpo-train` resume |
| **Dataset contract** | checkpoint `manifest.json` → `config_hashes.{dataset,data_contract}` | `resume.strict_dataset_hash` | `sft-dpo-train` resume |
| **Template hash** | checkpoint `manifest.json` → `config_hashes.template` | `resume.strict_template_hash` | `sft-dpo-train` resume |
| **Checkpoint file integrity** | checkpoint `checksums.json` | always enforced | `sft-dpo-train` resume |

---

## 1. Preprocessing cache — `preprocessing_signature`

`sft-dpo-prepare` reuses an existing pretokenized split when three conditions
all hold:

1. `file_sha256(raw_parquet) == manifest["splits"][split]` — raw data unchanged.
2. `file_sha256(pretokenized_parquet) == manifest["pretokenized"][split]` — cache file not corrupt.
3. `preprocessing_signature == manifest["preprocessing_signatures"][split]` — processing contract unchanged.

The signature is `stable_hash` of the following structure:

```
schema_version          (hardcoded int, bumped on breaking pipeline changes)
preprocessing.sequence  (max_seq_len, truncation, packing)
preprocessing.rendering (use_system, reject_raw_special_markers)
preprocessing.reasoning (enable_thinking)
preprocessing.masking   (ignore_index, require_positive_supervised_tokens, policies)
preprocessing.quality   (max_rejected_fraction, min_processed_rows_per_loss_kind, min_supervised_tokens)
tokenizer:
  use_fast
  add_special_tokens
  class name
  chat_template_hash      (sha256 of the template string)
  artifact_hashes:
    tokenizer.json
    tokenizer_config.json  ← also contains chat_template
    special_tokens_map.json
    vocab.json
    merges.txt             (if present)
    tokenizer.model        (if present)
```

**What rebuilds the cache:**

| Change | Effect |
|---|---|
| Raw parquet file bytes change | That split is rebuilt (per-split check) |
| Any `preprocessing.*` config field | All splits rebuilt |
| `tokenizer.use_fast` or `add_special_tokens` | All splits rebuilt |
| Any tokenizer artifact file on disk | All splits rebuilt |
| `tokenizer_config.json` (contains chat_template) | All splits rebuilt |
| Tokenizer class (different model family) | All splits rebuilt |
| `PREPROCESSING_SCHEMA_VERSION` bump in code | All splits rebuilt |

**What does NOT invalidate the cache:**

| Change | Why ignored |
|---|---|
| `model.cache_dir` (if tokenizer files are identical) | Cache is tokenizer-content-keyed, not model-path-keyed |
| Model weights, registry ref, payload hash | Weights do not affect tokenization |
| `training.*`, `lora.*`, `distributed.*` | Not part of the preprocessing contract |
| `project.seed` | Seed only affects sampler; masking seed is in `preprocessing.masking.policies.sft_target` |
| `--workers`, `--worker-chunk-size` | Parallelism setting; output is deterministic |

---

## 2. Training contract — `training_contract` / `strict_config`

Saved at every checkpoint. Checked on resume when `resume.strict_config: true`.

The hash covers:

```
model:
  name, alias, cache_dir, precision
  attn_implementation, experts_implementation
  gradient_checkpointing, freeze_router
  expected_payload_hash        (from modelctl verify; null if modelctl unavailable)
tokenizer:  (entire tokenizer section from config)
lora:       (entire lora section from config)
loss_routing: (entire loss_routing section from config)
training:   (entire training section from config)
distributed: (entire distributed section from config)
```

**Changes that break resume with `strict_config: true`:**

| Field | Why it matters |
|---|---|
| `model.name` / `alias` / `cache_dir` | Different model identity |
| `model.precision` | Base dtype changes forward math |
| `model.attn_implementation` | Numerics and memory layout differ |
| `model.experts_implementation` | MoE expert dispatch changes outputs |
| `model.gradient_checkpointing` | Changes activation memory semantics |
| `model.freeze_router` | Changes which parameters receive gradients |
| Any `lora.*` | Adapter architecture and scale |
| Any `loss_routing.*` | Which routes are trained and DPO beta |
| Any `training.*` | Learning rate, schedule, batch size, etc. |
| Any `distributed.*` | FSDP sharding, wrap policy, etc. |
| Any `tokenizer.*` | Affects rendering and forward pass input |
| Model payload hash | Different base weights |

**Changes that do NOT block resume (not in training_contract):**

| Field | How to handle |
|---|---|
| `project.name`, `project.run_name` | Safe to change; purely metadata |
| `mlflow.*` | Safe to change; only affects tracking |
| `eval.*` | Safe to change; eval runs don't affect gradients |
| `checkpointing.*` | Safe to change; save cadence is not training-critical |
| `registry.*` | Safe to change |
| `progress.*` | Safe to change |
| `model.checks.*` | Safe to change; only gates preflight verification |
| `model.trust_remote_code` | Safe to change; loading option |

---

## 3. Dataset contract — `dataset` + `data_contract` / `strict_dataset_hash`

Two sub-hashes are checked together:

**`dataset`** = `file_sha256(pretokenized/manifest.json)`

Changes whenever any split is rebuilt by `sft-dpo-prepare`. This catches raw
data changes, preprocessing config changes, and tokenizer changes — all of
which force a prepare re-run that writes a new manifest.

**`data_contract`** = `stable_hash` of:

```
pretokenized_manifest:  file_sha256(manifest.json)
sampler:
  batch_size:   training.per_device_train_batch_size
  drop_last:    training.drop_last
  seed:         project.seed
```

This catches changes that affect batch membership and order even when the
raw data is unchanged.

**Changes that break resume with `strict_dataset_hash: true`:**

| Change | Sub-hash affected |
|---|---|
| Raw parquet replaced (any split) | Both `dataset` and `data_contract` |
| Any `preprocessing.*` field (triggers prepare rebuild) | Both |
| `training.per_device_train_batch_size` | `data_contract` only |
| `training.drop_last` | `data_contract` only |
| `project.seed` | `data_contract` only |

---

## 4. Template hash / `strict_template_hash`

`template` = `sha256_text(tokenizer.chat_template)`

This is the actual chat template string loaded from the tokenizer at training
start, not the file on disk. It is an independent check from the preprocessing
signature: the preprocessing cache also incorporates the template, but
`strict_template_hash` is what protects the training run from the template
changing mid-run (for example, after a model re-pull that replaces
`tokenizer_config.json`).

**What breaks it:**

Any change to the chat template string in the active tokenizer — whether by
editing `tokenizer_config.json` directly, by pulling a new model version with
a modified template, or by swapping `model.cache_dir` to a different model.

---

## 5. Checkpoint file integrity — always enforced

Before any resume, the code verifies:
- `READY` marker is present (proves the save completed atomically).
- Every file in the checkpoint directory matches `checksums.json`.

Missing files, extra files, and byte-level changes all fail loudly. This check
is unconditional and has no config flag.

---

## Decision table: what to do after a change

| You changed | Preprocessing cache | Resume? |
|---|---|---|
| Raw parquet only | Rebuild (prepare again) | OK with `strict_dataset_hash: false`, or re-run prepare and start new `output_dir` |
| A `preprocessing.*` field | Rebuild (prepare again) | Must rebuild dataset and start fresh `output_dir` or set all `strict_*: false` |
| `tokenizer.*` or tokenizer files | Rebuild (prepare again) | Breaks `strict_config` + `strict_template_hash` |
| `lora.*` or `training.*` | No impact | Breaks `strict_config` |
| `eval.*`, `mlflow.*`, `registry.*` | No impact | Does not break resume |
| `project.seed` | No impact | Breaks `strict_dataset_hash` (sampler seed) |
| `training.per_device_train_batch_size` | No impact | Breaks `strict_dataset_hash` (batch shape) |
| Source model re-pulled with different payload hash | Rebuild (prepare, new tokenizer hash) | Breaks `strict_config` (payload hash in training_contract) |

**To resume intentionally after a training-critical change**, set the
corresponding flag to `false` in the config:

```yaml
checkpointing:
  resume:
    enabled: true
    strict_config: false         # allow model/lora/training/distributed changes
    strict_dataset_hash: false   # allow data or seed changes
    strict_template_hash: false  # allow template changes
```

Disabling a strict flag means you accept responsibility for the changed
contract. Always log why you did it.

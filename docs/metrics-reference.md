# Metrics Reference

All metrics are logged to MLflow. This document describes every metric key,
what it measures, how it is computed, and when it is emitted.

---

## Timing

| Phase | Cadence |
|---|---|
| Preprocessing metrics | Once, at training start (after prepare results are loaded) |
| Dataloader metrics | Once, at training start (after dataloaders are built) |
| Training step metrics | Every optimizer step (not every micro-batch) |
| Validation metrics | Every `eval.every_train_steps` optimizer steps |
| BFCL metrics | Every `eval.bfcl.run_every_n_validations` validation events |

All metrics except preprocessing/dataloader metrics include an MLflow `step`
equal to `global_step` (the optimizer step count).

---

## Preprocessing metrics

Logged once when training starts. Prefix: `preprocessing/{split}/`.

| Key | Value |
|---|---|
| `preprocessing/{split}/rows_raw` | Total rows read from raw parquet before any filtering |
| `preprocessing/{split}/rows_processed` | Rows that passed all quality checks and produced a valid pretokenized sample |
| `preprocessing/{split}/rows_rejected` | Rows dropped for any reason (overlength, zero supervised tokens, template injection, etc.) |
| `preprocessing/{split}/tokens` | Total token count in processed samples (if available in split stats) |
| `preprocessing/{split}/supervised_tokens` | Total supervised-token count across processed samples (if available) |

`{split}` is one of `train`, `valid`, `test`. Metrics are logged for each
configured split.

The full manifest JSON is also logged as `preprocessing/manifest.json`
artifact. If `mlflow.log_rendered_samples: true`, the `debug.jsonl` audit file
(sample renderings with masks) is logged as an artifact under `preprocessing/`.

---

## Dataloader metrics

Logged once when training starts. Prefix: `dataloader/{split}/`.

| Key | Value |
|---|---|
| `dataloader/{split}/rows` | Number of pretokenized samples in the dataloader for this split |
| `dataloader/{split}/batches` | Total number of route-homogeneous batches the sampler will produce per epoch |
| `dataloader/{split}/short_batches` | Batches that are smaller than `per_device_train_batch_size` (padding/drop_last artifact) |
| `dataloader/{split}/loss_kind/{kind}` | Number of batches for each loss route (`sft_target`, `sft_tool`, `dpo_target`) |

These numbers reflect the actual batch schedule, including any replica-group
padding the routed sampler applies to align batch counts across ranks.

---

## Training step metrics

Logged every optimizer step (after gradient accumulation). These are
cross-rank aggregated via `accelerator.gather_for_metrics` before logging.

### Core

| Key | Value |
|---|---|
| `train/loss` | Average per-micro-batch loss over the current accumulation window. Computed as `sum(raw_losses) / (num_micro_batches × num_processes)`. Unit depends on the active route: CE (nats/token) for SFT, logsigmoid (per pair) for DPO. Since routed batches are homogeneous, this is always a single-objective value per step. |
| `train/lr` | Current learning rate from the first optimizer param group |
| `train/samples_per_second` | Sequences processed per wall-clock second across all ranks, measured from step start to optimizer step |
| `train/tokens_per_second` | `attention_mask.sum()` tokens (all non-padding positions) per second. For DPO batches: sum of chosen and rejected mask tokens. |
| `train/supervised_tokens_per_second` | Label-selected tokens (positions where `labels != ignore_index`) per second. This is the signal actually used by the loss, excluding prompt and padding. |

### DPO route metrics (only on `dpo_target` steps)

These are accumulated over the accumulation window and averaged, then logged
under `train/`:

| Key | Value |
|---|---|
| `train/dpo/policy_chosen_logp` | Mean log-probability of the chosen completion under the current policy (LoRA adapter active) |
| `train/dpo/policy_rejected_logp` | Mean log-probability of the rejected completion under the current policy |
| `train/dpo/ref_chosen_logp` | Mean log-probability of the chosen completion under the reference policy (LoRA adapter disabled) |
| `train/dpo/ref_rejected_logp` | Mean log-probability of the rejected completion under the reference policy |
| `train/dpo/reward_chosen` | `beta × (policy_chosen_logp − ref_chosen_logp)` — implicit reward for the chosen response |
| `train/dpo/reward_rejected` | `beta × (policy_rejected_logp − ref_rejected_logp)` — implicit reward for the rejected response |
| `train/dpo/reward_margin` | `reward_chosen − reward_rejected`. Positive margin means the policy prefers chosen over rejected. Tracking this over training reveals whether DPO is working. |
| `train/dpo/accuracy` | Fraction of pairs where `reward_chosen > reward_rejected` within the batch. 0.5 is random; convergence to >0.7 is a healthy training signal. |

---

## Validation metrics

Logged every `eval.every_train_steps` optimizer steps, on the validation split.
Prefix: `eval/`.

### Shared

| Key | Value |
|---|---|
| `eval/batches` | Number of batches evaluated (capped by `eval.standard.max_batches` if set) |
| `eval/tokens` | Total tokens in evaluated batches (all non-ignored label positions plus padding) |
| `eval/supervised_tokens` | Total supervised positions across evaluated batches |

### Aggregate loss (single-objective splits only)

`eval/loss` and `eval/ppl` are emitted **only when the validation split
contains exactly one objective** (all SFT, or all DPO — not a mixture).
Blending per-token CE and per-pair logsigmoid is a category error, so for
mixed splits these keys are suppressed and only per-route metrics are emitted.

| Key | Value |
|---|---|
| `eval/loss` | Weighted average loss. For SFT: weighted by supervised token count. For DPO: weighted by pair count. |
| `eval/ppl` | `exp(eval/loss)`, clipped at `exp(20)` to avoid numeric overflow. Only for SFT-only splits. |

### SFT route (present when at least one SFT batch was evaluated)

| Key | Value |
|---|---|
| `eval/sft/loss` | Supervised-token-weighted CE loss across all SFT batches (`sft_target` + `sft_tool`) |
| `eval/sft/ppl` | `exp(eval/sft/loss)`, clipped at `exp(20)` |
| `eval/sft/batches` | Number of SFT batches |
| `eval/sft/tokens` | Total label positions (supervised + ignored) in SFT batches |
| `eval/sft/supervised_tokens` | Total supervised positions in SFT batches |

### DPO route (present when at least one DPO batch was evaluated)

| Key | Value |
|---|---|
| `eval/dpo/loss` | Pair-count-weighted mean DPO loss across all DPO batches |
| `eval/dpo/accuracy` | Pair-count-weighted mean accuracy (`reward_chosen > reward_rejected`) |
| `eval/dpo/reward_margin` | Pair-count-weighted mean reward margin (`reward_chosen − reward_rejected`) |
| `eval/dpo/batches` | Number of DPO batches |
| `eval/dpo/pairs` | Total chosen/rejected pairs evaluated |
| `eval/dpo/tokens` | Total label positions in DPO batches (chosen + rejected, all positions) |
| `eval/dpo/supervised_tokens` | Total supervised positions in DPO batches (chosen + rejected completion tokens) |

### Notes on eval accuracy

- `eval/dpo/accuracy` and `eval/dpo/reward_margin` are the most interpretable
  DPO training signals. They do not require a reference model — they are
  computed from the on-the-fly reference pass identical to training.
- `eval/sft/loss` (not `eval/loss`) is the correct metric for registry
  selection on mixed SFT+DPO splits.

---

## BFCL metrics

Logged on validation events where
`validation_index % eval.bfcl.run_every_n_validations == 0`.
BFCL is disabled by default (`eval.bfcl.enabled: false`).

### Accuracy

| Key | Value |
|---|---|
| `eval/bfcl/accuracy` | Fraction of non-skipped samples where all predicted tool calls match the ground truth (pass/fail per sample, averaged). Computed over the **surviving denominator** (loaded − skipped). |
| `eval/bfcl/total` | Total non-skipped samples evaluated |
| `eval/bfcl/passed` | Samples where all tool calls matched |
| `eval/bfcl/failed` | Samples that did not pass |

### Coverage

| Key | Value |
|---|---|
| `eval/bfcl/loaded_total` | Total samples in the filtered BFCL dataset (after category/limit filtering) |
| `eval/bfcl/skipped_total` | Samples excluded because the tokenizer or template refused to render them |
| `eval/bfcl/skipped_fraction` | `skipped_total / loaded_total`. High or drifting skip fraction means accuracy is computed over a shrinking denominator and comparisons across checkpoints are unreliable. |

### Per-reason skip counts

For each skip reason encountered, a separate counter is logged:

| Key | Value |
|---|---|
| `eval/bfcl/skipped/no_user_query` | Template raised "No user query found in messages" — the eval turn's last message is a tool response with no following user message |
| `eval/bfcl/skipped/system_message` | Template refused due to a system message issue |
| `eval/bfcl/skipped/unexpected_role` | Template refused due to an unexpected message role |
| `eval/bfcl/skipped/{other}` | Other tokenizer/render errors, keyed by exception class name |

### Per-category accuracy

For each category present in the evaluated dataset:

| Key | Value |
|---|---|
| `eval/bfcl/{category}/accuracy` | Accuracy for this category |
| `eval/bfcl/{category}/total` | Total non-skipped samples in this category |

---

## MLflow artifacts

In addition to metrics, the following files are logged as MLflow artifacts:

| Path | When | Content |
|---|---|---|
| `config/effective_config.json` | Run start | Full resolved config dict |
| `lineage/code.json` | Run start | Git commit, branch, dirty status of training code |
| `lineage/dvc.json` | Run start | DVC metadata for the training dataset (if `data.dvc` found) |
| `model/source_resolution.json` | Run start | modelctl resolution result: ref, payload hash, pull status |
| `preprocessing/manifest.json` | Run start | Full preprocessing manifest with split hashes and row counts |
| `preprocessing/manifest.json` (artifact) | Run start | Same file, logged as a raw artifact |
| `preprocessing/debug.jsonl` | Run start (if `log_rendered_samples: true`) | Sample renderings with mask annotations for audit |
| `data/dataloaders.json` | Run start | Per-split dataloader summaries (batch counts, route distribution) |
| `eval/bfcl_rows.jsonl` | Each BFCL eval | Per-sample BFCL predictions and pass/fail decisions |

---

## MLflow params

At run start, every config field is logged as a flat MLflow param under its
dotted key path (e.g. `training.learning_rate`, `lora.r`). Keys containing
sensitive substrings (`password`, `secret`, `token`, `auth`, etc.) are
silently dropped. Values are truncated at 500 characters.

A `config_hash` param is also logged — a stable SHA256 of the full resolved
config dict for exact reproducibility lookup.

## MLflow tags

| Tag | Value |
|---|---|
| `stage` | `training_pipeline` |
| `project.name` | `project.name` from config |
| `project.run_name` | `project.run_name` from config |
| `model.registry_name` | `model.name` from config |
| `model.registry_alias` | `model.alias` from config |
| `model.resolved_model_id` | Local path to the resolved model payload |
| `model.registry_source` | `true` when resolved via modelctl |
| `model.registry_ref` | Full modelctl ref (e.g. `models:/source-llm@champion`) |
| `model.resolved_version` | Registry version number |
| `model.expected_payload_hash` | Payload hash from modelctl verify |
| `code.git_commit` | HEAD commit SHA of the training code repository |
| `code.git_dirty` | `true` if the working tree has uncommitted changes |
| `data.git_commit` | HEAD commit of the DVC data repository (if found) |
| `data.git_dirty` | Dirty status of the data repository |
| `data.dvc.data.md5` | DVC md5 of the first tracked data output (if found) |

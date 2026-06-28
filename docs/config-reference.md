# Config Reference

Complete description of every field in the training config. For the working
template with typical production values, see `configs/config.example.yaml`.

Unknown keys are rejected at load time — a typo is a hard error, not a silent
fallback to a default. Paths are resolved to absolute at load time.

---

## `project`

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | MLflow experiment name. Also the destination model name in the candidate registry (`models:/<name>`). Use a stable model-family name, not a per-run label. |
| `run_name` | string | no | MLflow run name. Free-form. Safe to change between attempts. |
| `seed` | int | yes | Global random seed for Python, PyTorch, preprocessing short-reply sampling, and sampler shuffling. Changing this changes batch order (breaks `strict_dataset_hash`). |
| `output_dir` | path | yes | Root for all generated artifacts. The trainer writes `{output_dir}/pretokenized/`, `{output_dir}/checkpoints/`, and `{output_dir}/eval/bfcl_rows.jsonl` under it. Use a fresh directory for clean experiments; reuse to get cache/resume behavior. |

---

## `model`

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Registry model name. Resolved as `models:/<name>@<alias>` via modelctl. |
| `alias` | string | yes | Registry alias to resolve (e.g. `champion`). |
| `cache_dir` | path | yes | Local directory for the resolved model payload. Main process verifies or pulls into this directory at startup. On shared machines use a path outside `output_dir` to avoid re-downloading across runs. |

### `model.checks`

Controls how strictly the local model cache is verified against the registry.

| Field | Type | Default | Description |
|---|---|---|---|
| `verify_local_hash` | bool | — | When `true`, modelctl verifies the local payload hash before using `cache_dir`. Set `false` only for local experiments without complete registry metadata. |
| `verify_remote_ref` | bool | — | When `true`, contacts the registry to verify the ref before local verification. Usually `false` is enough because modelctl info/pull/verify already protects the payload. |
| `require_registry_metadata` | bool | — | When `true`, fails if modelctl cannot provide payload hash metadata (e.g. legacy registry entries). |

### `model` continued

| Field | Type | Required | Description |
|---|---|---|---|
| `precision` | `bf16` \| `fp16` \| `fp32` | yes | Base model load dtype. `bf16` is the standard choice for Ampere+ GPUs. `fp16` can be numerically less stable for large models. `fp32` is for debugging only and is usually too memory-heavy. Enters the training contract hash. |
| `trust_remote_code` | bool | yes | Pass `trust_remote_code=True` to Transformers. Required for custom modeling code shipped in the model payload. |
| `attn_implementation` | string \| null | no | Attention backend string passed to `from_pretrained`. Typical values: `flash_attention_2`, `eager`, `sdpa`. Enters the training contract hash. |
| `experts_implementation` | `eager` \| `batched_mm` \| `grouped_mm` \| null | no | MoE expert kernel backend. `eager` is the safest. `grouped_mm`/`batched_mm` require compatible kernel versions and should be validated before use at scale. Enters the training contract hash. |
| `gradient_checkpointing` | bool | yes | Model-level gradient checkpointing (Transformers). Cannot be `true` simultaneously with `distributed.fsdp.activation_checkpointing` — the schema rejects this combination. Enters the training contract hash. |
| `freeze_router` | bool | yes | Freeze the MoE router/gate parameters in addition to vocab modules. Vocab modules are always frozen (required for on-the-fly DPO reference). Enters the training contract hash. |

---

## `tokenizer`

Tokenizer is always loaded from `model.cache_dir`. These fields control how
it is instantiated.

| Field | Type | Required | Description |
|---|---|---|---|
| `use_fast` | bool | yes | Use the Rust-backed HuggingFace fast tokenizer. Required for offset-aware masking and throughput. Set `false` only for compatibility issues with a specific tokenizer. Enters both preprocessing signature and training contract hash. |
| `add_special_tokens` | bool | yes | Pass `add_special_tokens` to the tokenizer when encoding. Keep `false` — chat templates already manage special tokens; adding them again produces duplicate BOS/EOS/control tokens. Enters both preprocessing signature and training contract hash. |
| `padding_side` | `left` \| `right` | yes | Padding direction for collation. `right` is the default for causal LM training with the current SFT collator. BFCL generation uses `left` padding internally regardless. Enters the training contract hash. |

---

## `lora`

PEFT LoRA adapter configuration. All fields enter the training contract hash.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `r` | int | > 0 | LoRA rank. Higher rank increases adapter capacity and optimizer memory. Common sweep values: 8, 16, 32, 64. |
| `alpha` | int | > 0 | LoRA scaling factor. Effective scale is `alpha / r`. A common stable default is `2 × r`. |
| `dropout` | float | [0, 1) | Adapter dropout. Use `0.0` for small datasets or deterministic ablations. |
| `bias` | `none` \| `all` \| `lora_only` | — | Whether to train bias parameters. `none` is the standard adapter-only setting. `all` or `lora_only` increases trainable state and changes checkpoint semantics. |
| `target_modules` | list[string] | non-empty | Module name patterns for PEFT to wrap with LoRA. Must not include vocab projection modules (the schema rejects this). Typical: `[q_proj, k_proj, v_proj, o_proj]`. |
| `modules_to_save` | list[string] | may be empty | Non-LoRA modules to save/train alongside the adapter. Keep empty for pure LoRA. Do not include vocab modules. |

---

## `preprocessing`

### `preprocessing.raw`

| Field | Type | Required | Description |
|---|---|---|---|
| `train_path` | path | yes | Training split raw parquet. Each row: `type` (route) + `data` (JSON payload). |
| `valid_path` | path | yes | Validation split raw parquet. |
| `test_path` | path | no | Optional test split. Pretokenized when present but not used by the training loop. |
| `test_required` | bool | yes | When `false`, a missing `test_path` or missing file is silently skipped. When `true`, a missing test split is a hard error. |

### `preprocessing.workers`

| Field | Type | Default | Description |
|---|---|---|---|
| `num_workers` | int | 1 | CPU process workers for `sft-dpo-prepare`. `1` preserves the deterministic single-process path. Increase to 4–8 for large datasets after checking host RAM — each worker loads its own tokenizer. |
| `chunk_size` | int | 512 | Rows per worker task. Larger chunks reduce multiprocessing overhead but increase peak memory per returned chunk. Neither field enters the preprocessing signature — changing them does not invalidate the cache. |

### `preprocessing.sequence`

| Field | Type | Constraints | Description |
|---|---|---|---|
| `max_seq_len` | int | > 0 | Maximum token length after rendering. Samples exceeding this are rejected when `truncation: false`. Enters the preprocessing signature. |
| `truncation` | bool | must be `false` | Truncation is disabled: overlength samples are rejected, not silently cut. The schema enforces this — set `max_seq_len` high enough or curate long samples. |
| `packing` | bool | must be `false` | Sequence packing is not implemented. The schema enforces this. |

### `preprocessing.rendering`

| Field | Type | Required | Description |
|---|---|---|---|
| `use_system` | bool | yes | Include system messages when rendering via the chat template. Must match the target serving template. Changing this changes supervised tokens and invalidates the preprocessing cache. Enters the preprocessing signature. |
| `reject_raw_special_markers` | bool | yes | Reject raw payload text containing model-owned special tokens (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, etc.). Keep `true` to prevent template injection. Enters the preprocessing signature. |

### `preprocessing.reasoning`

| Field | Type | Required | Description |
|---|---|---|---|
| `enable_thinking` | bool | yes | When `false`, `<think>…</think>` scaffolding is masked out of the supervised loss (the template pre-fills the empty scaffold as prompt). When `true`, the think block is inside the supervised span and the model trains on its own reasoning. Flipping this changes supervised tokens and invalidates the preprocessing cache. Enters the preprocessing signature. |

### `preprocessing.masking`

| Field | Type | Required | Description |
|---|---|---|---|
| `ignore_index` | int | yes | Label value that CE loss and DPO sequence logprob ignore. Must match PyTorch CE default (`-100`). Enters the preprocessing signature. |
| `require_positive_supervised_tokens` | bool | yes | Reject samples that render to zero supervised tokens after masking. Keep `true` for production — a zero-supervised sample wastes memory and may signal a masking bug. Enters the preprocessing signature. |

#### `preprocessing.masking.policies.sft_target`

Controls which assistant turns in `sft_target` dialogs are supervised.

| Field | Type | Description |
|---|---|---|
| `min_guaranteed_assistant_chars` | int ≥ 0 | Assistant turns longer than this character count are always supervised regardless of probability. This prevents long, informative replies from being randomly dropped. |
| `loss_on_short_assistant_reply_prob` | float [0, 1] | Probability that a short reply (below `min_guaranteed_assistant_chars`) is kept. Sampling uses a deterministic hash keyed on `{sample_id}:{turn_index}:{short_response_sampling_seed}` — independent of dataloader order and RNG state. `0.0` drops all short replies; `1.0` keeps all of them. |
| `short_response_sampling_seed` | int | Seed for the deterministic short-reply hash. Changing this changes which short replies are kept — it is part of the preprocessing signature. |
| `require_user_anchor` | bool (default `false`) | When `false` (default), assistant turns that open a dialog without a preceding user message are supervised. This is correct for agent-speaks-first / outbound-call data. When `true`, only assistant turns immediately preceded by a user turn are supervised — suitable for pure responder models. |

All four fields enter the preprocessing signature.

### `preprocessing.quality`

These checks are enforced after preprocessing and fail hard if violated.

| Field | Type | Description |
|---|---|---|
| `max_rejected_fraction` | float [0, 1] | Maximum allowed fraction of rejected rows per split. If more than this fraction of raw rows are rejected, `sft-dpo-prepare` fails. Tighten for clean production data; loosen during dataset migration. Enters the preprocessing signature (tightening changes which cache is valid). |
| `min_processed_rows_per_loss_kind` | dict[str, int] | Minimum processed row count for each route (`sft_target`, `sft_tool`, `dpo_target`). Set a route to `0` to make it optional for this run. A route not present in the raw data fails if its minimum is > 0. Enters the preprocessing signature. |
| `min_supervised_tokens` | int | Minimum total supervised tokens across the processed split. This is a low-bar sanity check — a near-zero value catches template or masking bugs that make training a no-op. Enters the preprocessing signature. |

---

## `loss_routing`

### `loss_routing.routes`

Defines which raw `type` values are active and which loss function they use.

| Field | Type | Description |
|---|---|---|
| `sft_target.type` | `sft_ce` | Routes `sft_target` samples through masked cross-entropy loss. |
| `sft_tool.type` | `sft_ce` | Routes `sft_tool` samples through masked cross-entropy loss. |
| `dpo_target.type` | `dpo` | Routes `dpo_target` samples through on-the-fly DPO loss. |

Route frequency is determined entirely by dataset composition — there are no
sampler weights. To change the SFT/DPO balance, change the dataset.

### `loss_routing.dpo`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `beta` | float | > 0 | DPO temperature. Controls the strength of the preference signal. Higher beta pushes harder on chosen-vs-rejected margins and can destabilize noisy preference data. Typical sweep range: 0.05–0.2. The reference logprobs are computed on-the-fly from the same model with the LoRA adapter disabled. |

All `loss_routing` fields enter the training contract hash.

---

## `training`

All `training` fields enter the training contract hash. Changing any of them
blocks resume with `strict_config: true`.

| Field | Type | Constraint | Description |
|---|---|---|---|
| `num_epochs` | int | > 0 | Full passes over the routed train dataloader. Total optimizer steps = `ceil(dataloader_batches / (world_size × grad_accum)) × num_epochs`. |
| `per_device_train_batch_size` | int | > 0 | Sequences per rank per micro-batch. The main knob to reduce on OOM. With long context, `1` is the safest default. Also enters the dataset contract hash (affects sampler). |
| `drop_last` | bool | — | When `false`, incomplete route replica groups are padded by duplicating the last batch so all ranks see the same route. When `true`, incomplete groups are discarded. `false` is recommended for small datasets to avoid data waste. Also enters the dataset contract hash. |
| `gradient_accumulation_steps` | int | > 0 | Micro-batches per optimizer step. The effective global batch size is `per_device_train_batch_size × world_size × gradient_accumulation_steps`. |
| `learning_rate` | float | > 0 | Peak learning rate for AdamW. |
| `adamw_betas` | [float, float] | — | AdamW momentum parameters β1, β2. |
| `weight_decay` | float | ≥ 0 | L2 weight decay. Typically `0.0` for LoRA adapters. |
| `warmup_ratio` | float | [0, 1] | Fraction of total optimizer steps used for linear LR warmup. |
| `lr_scheduler_type` | string | — | Scheduler name passed to `transformers.get_scheduler`. Common values: `cosine`, `linear`, `constant`. |
| `max_grad_norm` | float | ≥ 0 | Gradient clipping threshold. `0.0` disables clipping. `1.0` is a conservative guardrail for mixed SFT/DPO runs. |

---

## `progress`

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Enable console progress bars. Disable in cluster log environments where tqdm output is noisy. |

---

## `distributed`

All `distributed` fields enter the training contract hash.

### `distributed.fsdp`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `sharding_strategy` | string | — | FSDP sharding mode. `full_shard` shards parameters, gradients, and optimizer state across all ranks. Single-GPU runs fall back to `NO_SHARD` automatically. |
| `mixed_precision` | string | — | FSDP mixed precision policy. `bf16` reduces peak memory on modern GPUs. This is the FSDP policy, intentionally separate from `Accelerator(mixed_precision)`, which the code keeps at `"no"` to avoid fp32 FlatParameter upcast. |
| `cpu_offload` | bool | — | Offload parameters to CPU when not in use. Reduces VRAM at the cost of PCIe bandwidth. Keep `false` unless GPU memory is exhausted after all other options. |
| `activation_checkpointing` | bool | — | FSDP-level activation checkpointing. Trades compute for lower activation memory. Cannot be `true` simultaneously with `model.gradient_checkpointing`. |
| `state_dict_type` | `sharded_state_dict` | required | Sharded state dict avoids materializing the full optimizer state on one rank during save/resume. The schema enforces this value. |
| `use_orig_params` | bool | must be `true` | Required for PEFT LoRA with frozen base parameters. With `false`, FSDP requires uniform `requires_grad` within a flatten group and fails on mixed frozen/trainable tensors. The schema enforces `true`. |
| `limit_all_gathers` | bool | — | Throttle all-gather scheduling to reduce peak memory spikes. Keep `true` for memory-bound runs. |
| `auto_wrap_policy` | string | — | FSDP unit wrapping strategy. `transformer_based_wrap` wraps each decoder block as one FSDP unit. `no_wrap` is not viable at large model scale. |
| `transformer_cls_names_to_wrap` | list[string] | non-empty | Exact Python class names of decoder layer modules to wrap. Must match the model's `modeling_*.py` class names precisely. A wrong name silently wraps nothing and causes OOM. The trainer validates that at least one class matches before building the FSDP graph. For `Qwen3_5MoeForConditionalGeneration`: `Qwen3_5MoeTextDecoderLayer`. |
| `cpu_ram_efficient_loading` | bool | — | Load pretrained weights on rank 0 only and broadcast to other ranks. Requires `sync_module_states: true`. Prevents peak RAM spike proportional to number of GPUs during model load. |
| `sync_module_states` | bool | — | Synchronize module states across ranks after FSDP initialization. Required when `cpu_ram_efficient_loading: true`. |

---

## `eval`

Eval fields do not enter the training contract hash and can be changed on resume.

| Field | Type | Constraint | Description |
|---|---|---|---|
| `every_train_steps` | int | > 0 | Run validation every N optimizer steps. Also determines checkpoint cadence (see `checkpointing.save_every_n_validations`). |

### `eval.standard`

| Field | Type | Description |
|---|---|---|
| `max_batches` | int \| null | Cap ordinary validation at this many batches. `null` evaluates the full validation split. Use a small number (e.g. 50) for fast smoke runs. |

### `eval.bfcl`

BFCL is the tool-calling eval on the bundled RU BFCL dataset. Generation runs
a manual FSDP-safe forward decode loop (not `model.generate`) so the model
stays sharded.

| Field | Type | Constraint | Description |
|---|---|---|---|
| `enabled` | bool | — | Disabled by default. Enable for candidate-quality gates. Much more expensive than ordinary validation because it generates tokens. |
| `path` | path \| null | — | Custom BFCL dataset path. `null` uses the bundled `src/eval/ru_bfcl/data/bfcl_eval.jsonl`. |
| `run_every_n_validations` | int | > 0 | Run BFCL every N validation events. When `registry.selection.metric` is a BFCL metric, every checkpoint boundary must include BFCL — the schema validates this alignment. |
| `include_multi_turn` | bool | — | Include multi-turn tool-call test cases in addition to single-turn. |
| `categories` | list[string] \| null | — | Optional category filter. `null` includes all categories in the dataset. |
| `limit` | int \| null | > 0 | Cap the number of BFCL samples. `null` uses the full filtered set. Use 10–100 for smoke runs. |

#### `eval.bfcl.generation`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `max_new_tokens` | int | > 0 | Token budget per BFCL turn. Increase if tool-call XML is being truncated in responses. |
| `temperature` | float | ≥ 0 | Sampling temperature. `0.0` for deterministic greedy decoding (recommended for registry selection). |
| `top_p` | float | (0, 1] | Nucleus sampling threshold. Only active when `do_sample: true`. |
| `do_sample` | bool | — | Enable token sampling. Keep `false` for deterministic eval; `true` only for robustness experiments. |

---

## `checkpointing`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `save_every_n_validations` | int | > 0 | Save a checkpoint every N validation events. Checkpoint saving triggers candidate window selection and async registry work. |
| `save_total_limit` | int \| null | > 0; ≥ `registry.register_every_n_checkpoints` | Keep only the newest N checkpoint directories. Checkpoints still needed by an active registry-selection window or pending async registration are never deleted regardless of this limit. `null` disables pruning. |

### `checkpointing.resume`

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | When `true`, the trainer searches for the latest valid checkpoint under `{output_dir}/checkpoints` and resumes from it. |
| `strict_config` | bool | When `true`, the training contract hash (`model`, `lora`, `training`, `distributed`, `tokenizer`, `loss_routing`) must match the value saved at the checkpoint. Prevents accidental continuation after architecture or hyperparameter changes. |
| `strict_dataset_hash` | bool | When `true`, the preprocessing manifest hash and sampler contract (`batch_size`, `drop_last`, `seed`) must match. Prevents resuming with different data or batch order. |
| `strict_template_hash` | bool | When `true`, the active chat template string must match what was used when the checkpoint was saved. Prevents resuming after a model re-pull that modifies `tokenizer_config.json`. |

See `docs/hashing-guide.md` for the complete breakdown of what each flag covers.

---

## `mlflow`

MLflow tracking settings do not enter any hash and are safe to change on resume.

| Field | Type | Description |
|---|---|---|
| `tracking_uri` | string | MLflow tracking server URI. Use `http://localhost:5000` for local dev; production should point to the shared server. |
| `resume_run_id` | string \| null | Append to an existing MLflow run. `null` creates a new run. Use when resuming from a checkpoint to keep metrics in one continuous run view. |

### `mlflow.async_logging`

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Run MLflow logging and modelctl registration in a background thread so they do not block training steps. Keep `true` unless debugging tracking failures. |
| `queue_max_items` | int | Capacity of the async job queue. Increase only if tracking bursts faster than the worker thread can flush. |
| `flush_timeout_seconds` | float | Maximum seconds to wait for the queue to drain at shutdown and on flush calls. |
| `fail_on_worker_error` | bool | When `true`, a failed async MLflow or modelctl job raises at the next flush, which is visible in training logs. Keep `true` for production so failures are not silently ignored. |

| Field | Type | Description |
|---|---|---|
| `log_rendered_samples` | bool | Log the preprocessing debug JSONL (rendered samples with mask annotations) as an MLflow artifact. Useful for mask and template audits. Disable if artifact volume becomes too large. |

---

## `registry`

Registry settings do not enter any hash and are safe to change on resume.

| Field | Type | Constraint | Description |
|---|---|---|---|
| `register_every_n_checkpoints` | int | > 0 | A candidate window covers this many consecutive checkpoints. When a window completes, the best checkpoint (by `selection.metric`) is registered with a `candidate-NNNNNN` alias. |

### `registry.selection`

| Field | Type | Description |
|---|---|---|
| `metric` | string | Metric key used to rank checkpoints within a window. Must be one of the allowed `REGISTRY_SELECTION_METRICS` set (see schema). Use route-specific metrics for mixed SFT/DPO datasets — `eval/loss` is rejected for mixed splits because it blends incommensurable objectives. |
| `mode` | `min` \| `max` | `min` for loss/perplexity metrics, `max` for accuracy/reward_margin/pass metrics. |

#### Valid `registry.selection.metric` values

```
eval/loss                  eval/sft/loss             eval/dpo/loss
eval/ppl                   eval/sft/ppl              eval/dpo/accuracy
eval/batches               eval/sft/batches          eval/dpo/reward_margin
eval/tokens                eval/sft/tokens           eval/dpo/batches
eval/supervised_tokens     eval/sft/supervised_tokens eval/dpo/pairs
                                                      eval/dpo/tokens
                                                      eval/dpo/supervised_tokens
eval/bfcl/accuracy
eval/bfcl/total
eval/bfcl/passed
eval/bfcl/failed
```

BFCL metrics are only valid when `eval.bfcl.enabled: true` and every
checkpoint boundary runs BFCL (validated by the schema).

---

## Cross-field constraints enforced by the schema

| Constraint | Rule |
|---|---|
| Double checkpointing | `model.gradient_checkpointing` and `distributed.fsdp.activation_checkpointing` cannot both be `true` |
| FSDP state dict type | `distributed.fsdp.state_dict_type` must be `sharded_state_dict` |
| FSDP orig params | `distributed.fsdp.use_orig_params` must be `true` |
| Efficient loading | `distributed.fsdp.cpu_ram_efficient_loading: true` requires `sync_module_states: true` |
| LoRA target modules | Must be non-empty; vocab projection modules are rejected |
| Pruning and registry window | `checkpointing.save_total_limit` must be ≥ `registry.register_every_n_checkpoints` |
| Truncation/packing | Both must be `false` until explicit implementations exist |
| BFCL selection cadence | Using a BFCL metric for registry selection requires `eval.bfcl.enabled: true` and BFCL running at every checkpoint boundary |

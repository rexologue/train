# sft-dpo-trainer

Production LoRA/FSDP fine-tuning for causal LLMs on mixed SFT, tool-calling,
and DPO routes. Target architecture: `Qwen3_5MoeForConditionalGeneration`.

## Pipeline

```
modelctl registry model resolution
  → pretokenized cache (render + mask + tokenize to parquet)
  → routed FSDP LoRA training (SFT CE + on-the-fly DPO reference)
  → ordinary validation (+ optional BFCL tool-calling eval)
  → atomic checkpoint
  → MLflow logging
  → candidate registry
```

Two separate commands map to the two phases. Preprocessing is always done
before distributed training so it does not hold NCCL process groups open.

```bash
sft-dpo-prepare --config configs/config.yaml
accelerate launch --use_fsdp --num_processes <N> sft-dpo-train --config configs/config.yaml
```

---

## Installation

CUDA 12.6 with flash-attn/causal-conv1d kernels:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1"
python -m pip install -c constraints/cuda126_kernels.txt -e ".[cuda-kernels]"
```

Development extras:

```bash
python -m pip install -e ".[dev]"
```

`modelctl` is installed from the `modelctl-mlflow` PyPI package and expected
in `PATH`. The constraint file `constraints/cuda126_kernels.txt` pins kernel
package versions for the CUDA 12.6 build.

---

## Configuration

Copy the annotated template and edit for your run:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Minimum fields to set for a new run:

```yaml
project:
  name: my-model
  run_name: v1
  output_dir: runs/my-model/v1

model:
  name: source-llm
  alias: champion
  cache_dir: artifacts/model_cache/source-llm/champion

preprocessing:
  raw:
    train_path: data/train.parquet
    valid_path: data/valid.parquet

mlflow:
  tracking_uri: http://my-mlflow-server:5000
```

All derived paths are rooted at `project.output_dir`:

```
{output_dir}/pretokenized/   — tokenized cache and manifest
{output_dir}/checkpoints/    — step-NNNNNN checkpoint directories
{output_dir}/eval/bfcl_rows.jsonl
```

For detailed field documentation see [`docs/config-reference.md`](docs/config-reference.md).

---

## Prepare

`sft-dpo-prepare` resolves the model, verifies/pulls the payload, and builds
the pretokenized cache:

1. Validates and loads the config.
2. Resolves `models:/<model.name>@<model.alias>` via modelctl. Pulls into
   `model.cache_dir` if missing or hash mismatch. Writes a sidecar file next
   to `model.cache_dir` with the verified payload hash.
3. Loads the tokenizer from the resolved model directory.
4. For each configured split: computes a raw parquet hash and a preprocessing
   signature (tokenizer files + config contract). Reuses an existing
   pretokenized split when all three checks pass; rebuilds otherwise.
5. Validates quality thresholds (rejection rate, minimum rows, minimum
   supervised tokens).

Force-rebuild ignoring all hashes:

```bash
sft-dpo-prepare --config configs/config.yaml --force
```

Parallel tokenization (overrides YAML workers for this run):

```bash
sft-dpo-prepare --config configs/config.yaml --workers 8 --worker-chunk-size 1024
```

`--workers` and `--worker-chunk-size` do not enter the preprocessing signature
and do not invalidate an already valid cache.

---

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --use_fsdp --num_processes 4 \
  sft-dpo-train --config configs/config.yaml
```

Train expects `sft-dpo-prepare` to have completed first. It does not re-run
model pulls or preprocessing. It reads the model source sidecar and the
pretokenized manifest; both must exist.

Key behaviors:

- **Routed batches** are homogeneous per optimizer step: every micro-batch
  in a gradient accumulation window shares one `loss_kind`. The sampler
  pads incomplete replica groups so all FSDP ranks run the same route.
- **DPO reference** is on-the-fly: the same FSDP model runs twice per DPO
  step — once with the active LoRA adapter (policy), once with it disabled
  (reference). No second model, no cached logprobs.
- **Vocab modules** are always frozen and excluded from FSDP flattening,
  because the DPO reference pass requires disabling the adapter on the same
  model instance.
- **Checkpoints** are written atomically: everything goes to `step-NNNNNN.tmp/`,
  the `READY` marker is written last, then the directory is renamed. A partial
  save is never resumed from.
- **Resume** verifies file checksums, then optionally verifies config and data
  contract hashes (controlled by `checkpointing.resume.strict_*` flags).

Single-process debug mode (no distributed):

```bash
PYTHONPATH=src python -m train --config configs/config.yaml
```

NCCL debug environment:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
NCCL_DEBUG=INFO \
accelerate launch --use_fsdp --num_processes 4 \
  sft-dpo-train --config configs/config.yaml
```

---

## Dataset format

Raw parquet with two columns per row:

```python
{
    "type": "sft_target" | "sft_tool" | "dpo_target",   # authoritative route
    "data": "<JSON string payload>",
}
```

`type` is the only authoritative route column. A `target` column is a schema
error. Payload shapes:

```python
# sft_target / sft_tool
{"messages": [...], "tools": [...], "parallel_tool_calls": bool}

# dpo_target
{"prompt": [...], "chosen": {...}, "rejected": {...}}
```

---

## Loss mask

Only selected assistant completions contribute to the loss.

Not supervised: system, user, tool messages; tool schemas; role headers; padding;
non-selected assistant turns; `<think>…</think>` when `enable_thinking: false`.

Supervised: selected `sft_target` replies; all assistant turns in `sft_tool`
(including tool call output and final answer); DPO chosen/rejected completion
tokens; stop/end-of-message tokens of supervised turns.

---

## Evaluation

**Ordinary validation** runs every `eval.every_train_steps` optimizer steps on
the validation split. Per-route metrics (`eval/sft/loss`, `eval/dpo/accuracy`,
etc.) are always correct. A blended `eval/loss` is only emitted for
single-objective splits — mixing per-token CE and per-pair logsigmoid is a
category error.

**BFCL** (`eval.bfcl.enabled: false` by default) runs the bundled RU BFCL
tool-calling dataset using a manual FSDP-safe generation loop. Malformed turns
are quarantined and counted by reason; watch `eval/bfcl/skipped_fraction`.

For the complete metric catalog see [`docs/metrics-reference.md`](docs/metrics-reference.md).

---

## Checkpoints and resume

A checkpoint directory contains:

```
step-NNNNNN/
├── adapter/             — PEFT adapter weights only
├── accelerate_state/    — FSDP-sharded optimizer + scheduler
├── trainer_state.json   — global_step, consumed_batches, validation_index
├── manifest.json        — config hashes, adapter path
├── metrics.json         — metrics at save time
├── checksums.json       — sha256 of every file in this directory
└── READY                — written last; absence means incomplete save
```

Auto-resume finds the newest `step-NNNNNN` directory with a `READY` marker and
passes all integrity checks. Resume restores: adapter, optimizer, scheduler,
RNG state, `global_step`, sampler epoch, and the in-progress registry window.

```yaml
checkpointing:
  resume:
    enabled: true
    strict_config: true        # model / lora / training / distributed
    strict_dataset_hash: true  # dataset manifest + sampler contract
    strict_template_hash: true # chat template string
```

If a training-critical field changes and you want to continue from a checkpoint,
set the relevant `strict_*` flag to `false` deliberately. For everything that
affects hashes and resume see [`docs/hashing-guide.md`](docs/hashing-guide.md).

---

## Registry

Training registers candidate aliases only — `candidate-NNNNNN` and
`candidate-latest`. Promotion to `baseline` or `champion` is a separate
manual step; the training loop never auto-assigns those.

`CandidateWindowSelector` observes `register_every_n_checkpoints` consecutive
checkpoints and registers the best by `registry.selection.metric`. For mixed
SFT/DPO splits, use a route-specific metric (`eval/sft/loss`,
`eval/dpo/accuracy`, etc.) rather than the blended `eval/loss`.

Registry operations are best-effort and run in a background thread. A modelctl
outage logs and continues; it never kills a training run.

---

## Tests

```bash
python -m pytest
```

---

## Reference documentation

| Document | Contents |
|---|---|
| [`docs/config-reference.md`](docs/config-reference.md) | Every config field: type, constraints, semantics, which hashes it enters |
| [`docs/metrics-reference.md`](docs/metrics-reference.md) | Every MLflow metric and artifact: definition, computation, cadence |
| [`docs/hashing-guide.md`](docs/hashing-guide.md) | What breaks the preprocessing cache and what blocks resume, and how to unblock intentionally |
| `configs/config.example.yaml` | Annotated production template with typical values |

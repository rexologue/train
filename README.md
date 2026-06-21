# estadel-trainer

Production-контур для LoRA-дообучения Qwen3.5 в диалоговом домене с SFT,
tool-calling loss и DPO.

Поддержанный runtime:

```text
MLflow/modelctl registry model
  -> bundled tokenizer
  -> PEFT LoRA
  -> Transformers
  -> Accelerate/FSDP
  -> RoutedTrainer
  -> MLflow/modelctl candidate registry
```

Ключевое правило запуска: подготовка данных и модели выполняется отдельной
CPU-стадией до distributed training.

```text
estadel-prepare --config configs/config.yaml
accelerate launch --use_fsdp --num_processes <N> estadel-train --config configs/config.yaml
```

Так дорогой preprocessing не держит NCCL/FSDP process group открытым и не
может упасть по distributed timeout, пока rank0 один долго токенизирует
датасет.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`modelctl` приходит из PyPI-пакета `modelctl-mlflow` и вызывается из `PATH`.
Путь к executable в YAML не задается.

Опциональные extras:

```bash
# CUDA kernels вроде flash-attn. Обычно ставится после torch.
python -m pip install --no-build-isolation -e ".[cuda-kernels]"

# Только если окружению нужен Transformers main.
python -m pip install -e ".[transformers-main]"
```

## Конфиг

Production template:

```bash
cp configs/config.example.yaml configs/config.yaml
```

`configs/config.example.yaml` является самодокументированным шаблоном. README
ниже объясняет рабочую схему, а конкретные поля и дефолты смотрите в YAML.

Минимум, который нужно задать под новый запуск:

```yaml
project:
  name: dialog-tuning
  run_name: qwen35-lora-sft-dpo-v1
  output_dir: artifacts/runs/qwen35-lora-sft-dpo-v1

model:
  name: Qwen_3.5_35B_A3B
  alias: champion
  cache_dir: artifacts/model_cache/qwen35/champion

preprocessing:
  raw:
    train_path: artifacts/data/train.parquet
    valid_path: artifacts/data/valid.parquet

  workers:
    num_workers: 1
    chunk_size: 512

mlflow:
  tracking_uri: http://...
```

Derived paths всегда выводятся из `project.output_dir`:

```text
{project.output_dir}/pretokenized
{project.output_dir}/ref_logprobs/<signature>
{project.output_dir}/checkpoints
{project.output_dir}/eval/bfcl_rows.jsonl
```

`project.name` используется как MLflow experiment и destination model для
candidate registration. `project.run_name` используется как MLflow run name.

## Двухфазный Запуск

### 1. Prepare

```bash
estadel-prepare --config configs/config.yaml
```

Prepare делает только локальную подготовку:

- читает YAML и валидирует config schema;
- резолвит source model как `models:/<model.name>@<model.alias>`;
- вызывает `modelctl info`;
- если `model.cache_dir` уже содержит payload, вызывает `modelctl verify`;
- если payload отсутствует или hash не совпал, вызывает `modelctl pull` и
  повторный `modelctl verify`;
- пишет sidecar рядом с `model.cache_dir`;
- грузит tokenizer из resolved model directory;
- считает preprocessing signature: tokenizer files, chat template hash,
  preprocessing contract, model source hash;
- для каждого split считает raw parquet hash;
- переиспользует split cache, если raw hash, pretokenized parquet hash и
  preprocessing signature совпали;
- перестраивает только отсутствующий или несогласованный split.
- при `preprocessing.workers.num_workers > 1` делит decoded rows на chunks и
  обрабатывает их в CPU process pool.

Принудительный rebuild всех pretokenized split caches:

```bash
estadel-prepare --config configs/config.yaml --force
```

`--force` не нужен для обычного обновления данных или конфига: если raw parquet,
tokenizer, source model или preprocessing settings изменились, prepare сам
увидит mismatch и перестроит нужные splits. `--force` нужен, когда вы хотите
пересоздать кеш несмотря на совпадающие hashes.

Prepare не стартует MLflow training run, не создает Accelerator и не
инициализирует CUDA/NCCL.

CPU parallel prepare:

```bash
estadel-prepare --config configs/config.yaml --workers 8 --worker-chunk-size 1024
```

`--workers` и `--worker-chunk-size` переопределяют только текущий prepare run.
Если CLI flags не заданы, используются `preprocessing.workers.num_workers` и
`preprocessing.workers.chunk_size` из YAML. Число workers и размер chunk не
входят в preprocessing signature: они не меняют tokenization/masking contract,
а только способ вычисления. Поэтому смена `--workers` не инвалидирует уже
валидный cache.

Практические ориентиры:

- `num_workers: 1` - лучший режим для отладки и минимальной RAM;
- `4-8` workers обычно дают хороший прирост на больших parquet splits;
- каждый worker грузит свой tokenizer и держит один processed chunk, поэтому
  не ставьте `num_workers=$(nproc)` вслепую;
- увеличивайте `chunk_size`, если overhead multiprocessing заметен;
- уменьшайте `chunk_size`, если samples длинные и RAM растет слишком сильно.

### 2. Train

```bash
accelerate launch --use_fsdp --num_processes <GPU_COUNT> \
  estadel-train --config configs/config.yaml
```

`estadel-train` запускается уже внутри Accelerate/FSDP и не строит
pretokenized cache. Он ожидает, что prepare был выполнен заранее.

Train делает:

- создает Accelerator/FSDP runtime;
- читает model source sidecar из `model.cache_dir`;
- читает `{project.output_dir}/pretokenized/manifest.json`;
- падает быстро, если sidecar, manifest или split parquet отсутствуют;
- логирует model source, lineage и preprocessing manifest в MLflow;
- строит routed dataloaders с route-homogeneous batches;
- при DPO строит или переиспользует ref-logprob cache;
- запускает training, ordinary eval, optional BFCL eval;
- сохраняет adapter-only checkpoints;
- регистрирует candidate aliases в registry.

Train не выполняет `modelctl pull` и не пересобирает preprocessing. Если
изменились source model, tokenizer, raw parquet или preprocessing settings,
сначала снова запускайте `estadel-prepare`.

Strict resume остается в training контуре: checkpoint resume сверяет effective
config, dataset manifest, data/training contracts, actual tokenizer chat
template hash и verified source model payload hash. Поэтому training не
доверяет голым путям без manifest/sidecar.

## Что Проверять Перед Дорогим Запуском

1. Prepare завершился строкой `prepare complete`.
2. Есть sidecar рядом с `model.cache_dir`:

```text
<model.cache_dir>.estadel_registry.json
```

3. Есть preprocessing manifest:

```text
{project.output_dir}/pretokenized/manifest.json
```

4. В manifest есть нужные splits и `preprocessing_signatures`.
5. `estadel-train` стартует, строит dataloaders и доходит до
   `loading tokenizer, model, LoRA adapter, optimizer, scheduler`.

Если падает `pretokenized manifest not found`, это не NCCL проблема: запустите
`estadel-prepare --config ...`.

Если падает `model source sidecar not found`, prepare не был выполнен для
текущего `model.cache_dir`.

Если raw parquet был заменен, запускайте prepare еще раз. Валидные splits будут
переиспользованы, измененные будут перестроены.

## Dataset Contract

Raw parquet split содержит одну строку на sample:

```python
{
    "data": "{\"messages\": [...], ...}",
    "type": "sft_target" | "sft_tool" | "dpo_target",
}
```

`type` является единственной authoritative route column. Колонка `target` не
поддерживается и считается schema error. `loss_kind` не выводится из JSON
payload.

SFT/tool payload:

```python
{
    "messages": [...],
    "tools": [...],              # optional
    "parallel_tool_calls": true, # optional
}
```

DPO payload:

```python
{
    "prompt": [...],
    "chosen": {"role": "assistant", ...},
    "rejected": {"role": "assistant", ...},
}
```

## Loss Mask

Loss считается только на выбранных assistant completions.

Не обучаются:

- system/user/tool messages;
- tool schemas и role headers;
- padding;
- невыбранные assistant turns;
- `<think>...</think>` при `preprocessing.reasoning.enable_thinking=false`.

Обучаются:

- выбранные `sft_target` replies;
- все assistant messages в `sft_tool`;
- final assistant answer after tool response;
- stop/end-of-message token выбранной completion;
- selected DPO chosen/rejected completion tokens при расчете sequence logprobs.

## DPO

`dpo_target` является активным route.

Pipeline:

```text
raw dpo_target
  -> prompt+chosen / prompt+rejected render
  -> branch-local labels
  -> routed DPO collator
  -> optional ref-logprob cache
  -> DPO loss in RoutedTrainer
```

Reference logprobs считаются в training phase, потому что им нужна модель.
Стадия distributed-safe: все ranks участвуют после создания Accelerator.

```yaml
loss_routing:
  routes:
    dpo_target:
      type: dpo
  dpo:
    beta: 0.1
    reference:
      cache_enabled: true
      cache_refresh: false
      cache_required: false
```

`cache_refresh=true` пересчитывает ref-logprob cache. `cache_required=true`
превращает incomplete/missing cache для DPO rows в ошибку.

## FSDP И Память

Training всегда запускается через Accelerate/FSDP:

```yaml
distributed:
  fsdp:
    sharding_strategy: full_shard
    mixed_precision: bf16
    activation_checkpointing: true
    state_dict_type: sharded_state_dict
    use_orig_params: true
    cpu_ram_efficient_loading: true
    sync_module_states: true
```

Практические правила:

- `use_orig_params: true` нужен для LoRA + frozen base/frozen embeddings. При
  `use_orig_params: false` FSDP требует uniform `requires_grad` внутри flatten
  group и будет падать на смешанных frozen/trainable tensors.
- `mixed_precision: bf16` относится к FSDP mixed precision policy. Это не то
  же самое, что запуск всего обучения в fp32.
- `model.precision: bf16` управляет dtype загрузки модели.
- `activation_checkpointing: true` снижает memory pressure ценой compute.
- `cpu_ram_efficient_loading: true` и `sync_module_states: true` помогают не
  материализовать полный payload независимо на каждом rank во время загрузки.
- `per_device_train_batch_size` и `preprocessing.sequence.max_seq_len` сильнее
  всего влияют на VRAM во время forward/backward.

Если нужно проверить, что distributed sampler не расходится по routes, смотрите
в логи dataloader:

```text
replica_group_size=<num_processes>
padded_replica_batches=<...>
loss_kinds={'sft_target': ..., 'sft_tool': ..., 'dpo_target': ...}
```

## Evaluation И Registry

Validation boundary:

```text
ordinary eval -> optional BFCL eval -> checkpoint -> candidate selection
```

Ordinary metrics:

```text
eval/loss
eval/ppl
eval/batches
eval/tokens
eval/supervised_tokens
```

BFCL включается через `eval.bfcl.enabled`. RU BFCL dataset bundled в
`src/eval/ru_bfcl/data/bfcl_eval.jsonl`; путь не конфигурируется.

Registry selection:

```yaml
registry:
  register_every_n_checkpoints: 5
  selection:
    metric: eval/loss
    mode: min
```

Training регистрирует только candidate aliases:

```text
candidate-000001
candidate-latest
```

Promotion в `baseline` или `champion` training loop не выполняет.

## Checkpoints И Resume

Checkpoints сохраняются атомарно:

```text
{project.output_dir}/checkpoints/step-000012/
├── adapter/
├── accelerate_state/
├── trainer_state.json
├── metrics.json
├── manifest.json
└── checksums.json
```

`adapter/` содержит только trainable PEFT adapter parameters. Full base model
state dict не материализуется в checkpoint package.

Auto-resume ищет последний валидный `step-NNNNNN` в
`{project.output_dir}/checkpoints`. Strict resume контролируется:

```yaml
checkpointing:
  resume:
    enabled: true
    strict_config: true
    strict_dataset_hash: true
    strict_template_hash: true
    strict_model_source_hash: true
```

Если меняете датасет, tokenizer, source model или training-critical config,
ожидайте strict resume error и начинайте новый `project.output_dir` либо
осознанно меняйте resume policy.

## Типовые Команды

Prepare:

```bash
estadel-prepare --config configs/config.yaml
```

Force prepare:

```bash
estadel-prepare --config configs/config.yaml --force
```

Parallel prepare:

```bash
estadel-prepare --config configs/config.yaml --workers 8 --worker-chunk-size 1024
```

Train на 2 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --use_fsdp --num_processes 2 \
  estadel-train --config configs/config.yaml
```

С явным debug для NCCL:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,COLL \
accelerate launch --use_fsdp --num_processes 2 \
  estadel-train --config configs/config.yaml
```

Source-tree module fallback в editable checkout:

```bash
PYTHONPATH=src python -m prepare --config configs/config.yaml
PYTHONPATH=src python -m prepare --config configs/config.yaml --workers 8
PYTHONPATH=src accelerate launch --use_fsdp --num_processes 2 -m train --config configs/config.yaml
```

## RU BFCL CLI

Standalone validator:

```bash
ru-bfcl-eval --help
```

Training loop использует bundled evaluator напрямую через `eval.bfcl`.

## Тесты

```bash
python -m pytest
```

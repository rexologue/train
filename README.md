# estadel-trainer

Production-grade контур для SFT LoRA дообучения Qwen3.5-35B-A3B с отдельной проверкой tool calling.

Проект сейчас включает preprocessing, routed DataLoaders, custom training loop, Accelerate/FSDP, ordinary eval, RU BFCL eval, MLflow tracking, async registry logging, adapter-only checkpoints, strict resume и checkpoint pruning.

## Статус

Реализовано:

- raw parquet -> canonical rows -> official chat template -> tokenization -> labels mask;
- pretokenized parquet cache с split-local reuse по raw split hash;
- routed homogeneous batches по `loss_kind`;
- SFT CE loss для `sft_target` и `sft_tool`;
- Accelerate/FSDP-only training runtime;
- AdamW optimizer с YAML-configurable `learning_rate`, `adamw_betas`, `weight_decay`;
- scheduler через `transformers.get_scheduler`;
- progress bar с validation/checkpoint phase messages;
- ordinary eval с token-weighted validation loss;
- RU BFCL offline eval с bundled `jsonl`;
- adapter-only atomic checkpoints;
- strict resume по config/data/template hashes;
- `checkpointing.save_total_limit` pruning;
- MLflow tracking и async CPU-only worker;
- candidate registration через `modelctl` без champion/baseline promotion.

DPO статус:

- DPO preprocessing/collation частично готовы (`dpo_target`, `chosen_*`, `rejected_*`);
- DPO trainer loss route пока не реализован;
- включать `dpo_target` в active `loss_routing.routes` пока нельзя.

## Быстрый запуск

Локальное окружение, используемое в проекте:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m pytest
```

Preprocessing/DataLoader startup на текущем debug config:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m train --config configs/config.preprocess.yaml --force-preprocess
```

С inspection random batch:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m train --config configs/config.preprocess.yaml --force-preprocess --inspect-random-batch --inspect-split train --inspect-token-limit 32
```

Принудительный training run на debug config:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m train --config configs/config.preprocess.yaml --train
```

`configs/config.preprocess.yaml` сейчас имеет `training.enabled: false`, поэтому без `--train` обучение не начнется.

## Конфиги

В `configs/` два основных файла:

- `config.example.yaml` — самодокументированный production template с комментариями по каждому важному полю.
- `config.preprocess.yaml` — локальный рабочий debug config с machine-specific paths.

Training-critical значения должны жить в YAML и валидироваться в `src/config/schema.py`.

Ключевые секции:

- `project` — имя run family, seed, output directory.
- `model` — base model/source registry/local settings, precision, freeze policy.
- `tokenizer` — tokenizer source и chat template guardrails.
- `lora` — PEFT LoRA параметры.
- `preprocessing` — raw paths, cache root, max length, thinking mode, masking policies.
- `loss_routing` — active loss routes.
- `training` — optimizer steps, batch size, grad accumulation, AdamW params, scheduler, grad clipping.
- `distributed.fsdp` — единственный supported distributed runtime.
- `eval` — ordinary eval и RU BFCL cadence/settings.
- `checkpointing` — atomic adapter checkpoint root, save cadence, pruning, strict resume.
- `mlflow` — tracking и async logging.
- `lineage` — DVC metadata tracking.
- `registry` — candidate model registration.

## Training step semantics

`training.max_steps` считается в optimizer steps.

Один optimizer step состоит из:

```text
training.gradient_accumulation_steps
```

micro-batches на каждом GPU/process.

Effective global samples per optimizer step:

```text
world_size * training.per_device_train_batch_size * training.gradient_accumulation_steps
```

Например, на 2 GPU при `per_device_train_batch_size=1` и `gradient_accumulation_steps=16` один optimizer step соответствует 32 samples, если нет короткого хвостового batch-а.

## FSDP

Проект поддерживает только Accelerate/FSDP training path.

FSDP оборачивает decoder layers и shard-ит параметры между GPU. При `full_shard` shard-ятся параметры, градиенты и optimizer state. Перед forward конкретного wrapped block-а FSDP временно all-gather-ит нужные параметры, считает block, затем освобождает/reshard-ит их. На backward аналогично собираются нужные параметры и reduce-scatter-ятся gradients.

Текущий локальный model cache:

```text
model_type: qwen3_5_moe
architecture: Qwen3_5MoeForConditionalGeneration
decoder layer class: Qwen3_5MoeDecoderLayer
```

Поэтому актуальный FSDP wrap config:

```yaml
distributed:
  fsdp:
    auto_wrap_policy: transformer_based_wrap
    transformer_cls_names_to_wrap:
      - Qwen3_5MoeDecoderLayer
```

## Checkpoints и resume

Checkpoint создается на validation boundary:

```text
ordinary eval -> BFCL eval -> checkpoint save
```

Сохраняется adapter-only package:

```text
{checkpointing.root_dir}/step-000012/
├── adapter/
├── accelerate_state/
├── trainer_state.json
├── metrics.json
├── manifest.json
└── checksums.json
```

Сохранение атомарное:

```text
step-000012.tmp -> step-000012
```

Auto-resume игнорирует `.tmp` и видит только final directories вида `step-\d+` с `manifest.json`.

Strict resume сравнивает текущие hashes с checkpoint manifest до загрузки state:

- config hash;
- pretokenized dataset manifest hash;
- tokenizer chat template hash.

Позиция данных восстанавливается через `trainer_state.consumed_batches`: dataloader fast-forward-ится на количество уже потребленных micro-batches.

`checkpointing.save_total_limit` удаляет старые final checkpoint-ы, оставляя последние N. Checkpoint-и, которые еще нужны registry window или async candidate registration, временно защищаются от pruning.

## RU BFCL

RU BFCL модуль лежит здесь:

```text
src/eval/ru_bfcl/
```

Bundled eval data:

```text
src/eval/ru_bfcl/data/bfcl_eval.jsonl
```

Training eval entrypoint:

```text
src/eval/bfcl.py
```

BFCL eval строит prompts через tokenizer chat template, генерирует tool calls, нормализует predictions и сравнивает с expected calls offline.

## Структура

```text
src/
├── train.py
├── checkpointing/
├── config/
├── data/
├── eval/
├── losses/
├── preprocessing/
├── registry/
├── tracking/
├── trainer/
└── utils/
```

Удаленные placeholder areas:

- `src/dpo`;
- `src/sampling`;
- unused DPO/logprob/metrics loss helpers;
- unused eval report/summarize helpers.

## Тесты

Полный suite:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m pytest
```

Текущий результат:

```text
82 passed
```

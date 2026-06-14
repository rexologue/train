# estadel-trainer

Production-контур для LoRA-дообучения Qwen3.5-35B-A3B:

- source model и tokenizer только из Model Registry;
- official tokenizer chat template и проверяемая loss mask;
- homogeneous batches по `loss_kind`;
- Accelerate/FSDP-only training;
- ordinary validation и опциональный bundled RU BFCL;
- MLflow tracking, adapter-only checkpoints и candidate registration;
- strict resume и автоматический DVC lineage.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`modelctl` устанавливается вместе с проектом и должен быть доступен в `PATH`.

## Конфигурация

Единственный шаблон конфигурации: `configs/config.example.yaml`.

```bash
cp configs/config.example.yaml configs/config.yaml
```

Минимально перед запуском задаются:

- `project.name`: MLflow experiment и destination model в registry;
- `project.run_name`: имя MLflow run;
- `project.output_dir`: единый корень локальных результатов;
- `model.name`, `model.alias`, `model.cache_dir`: source model из registry;
- `preprocessing.raw.train_path` и `preprocessing.raw.valid_path`;
- `training.num_epochs`;
- `mlflow.tracking_uri`.

Tokenizer всегда загружается из resolved model directory. Отдельный tokenizer
source, revision и template hash в конфиге не поддерживаются.

Все локальные output paths выводятся из `project.output_dir`:

```text
{project.output_dir}/
├── pretokenized/
├── checkpoints/
└── eval/
    └── bfcl_rows.jsonl
```

## Запуск

Поддержан один entrypoint и один CLI-параметр:

```bash
accelerate launch --use_fsdp --num_processes <GPU_COUNT> \
  -m train --config configs/config.yaml
```

Запуск всегда выполняет полный контур: registry resolution, data cache,
training, validation, checkpointing и candidate registration. Отдельных
preprocess/debug/train режимов нет.

## Model Registry

Source model задается alias-ссылкой:

```yaml
model:
  name: estadel-llm
  alias: champion
  cache_dir: artifacts/model_cache/estadel-llm/champion
  checks:
    verify_local_hash: true
    verify_remote_ref: false
    require_registry_metadata: true
```

Candidate checkpoints регистрируются в model с именем `project.name`.
Aliases формируются автоматически:

```text
candidate-000001
candidate-latest
```

Promotion до `baseline` или `champion` не выполняется training loop-ом.

## Epochs И Batches

Продолжительность обучения задается через `training.num_epochs`.
Число optimizer steps вычисляется автоматически из train DataLoader,
числа процессов и `gradient_accumulation_steps`. Неполный accumulation-хвост
завершается на границе каждой эпохи без повторного чтения первых batches.

Частота route batches определяется только составом датасета.
Sampler weights и искусственное повышение/понижение вероятности routes
не поддерживаются.

## Evaluation И Selection

Ordinary validation возвращает:

```text
eval/loss
eval/ppl
eval/batches
eval/tokens
eval/supervised_tokens
```

Bundled RU BFCL включается через `eval.bfcl.enabled`. Путь к dataset не
настраивается: используется `src/eval/ru_bfcl/data/bfcl_eval.jsonl`.

BFCL возвращает:

```text
eval/bfcl/accuracy
eval/bfcl/total
eval/bfcl/passed
eval/bfcl/failed
eval/bfcl/<category>/accuracy
eval/bfcl/<category>/total
```

Registry selection задается явно:

```yaml
registry:
  register_every_n_checkpoints: 5
  selection:
    metric: eval/loss
    mode: min
```

Для запуска без BFCL:

```yaml
eval:
  bfcl:
    enabled: false

registry:
  selection:
    metric: eval/loss
    mode: min
```

## Dataset И DVC

Raw parquet row:

```python
{
    "data": "{\"messages\": [...], ...}",
    "type": "sft_target" | "sft_tool" | "dpo_target",
}
```

`target` поддерживается как alias для `type`; конфликт между ними является
schema error. `loss_kind` никогда не выводится из содержимого `data`.

DVC не имеет отдельной секции конфигурации. Проект ищет `data.dvc` сначала
рядом с `preprocessing.raw.train_path`, затем на один уровень выше. Если файл
не найден, run продолжается без DVC metadata.

## Checkpoints И Resume

Checkpoints сохраняются атомарно в:

```text
{project.output_dir}/checkpoints/step-000012/
├── adapter/
├── accelerate_state/
├── trainer_state.json
├── metrics.json
├── manifest.json
└── checksums.json
```

Resume всегда ищет последний валидный `step-NNNNNN` в derived checkpoint
directory. Strict checks сравнивают config, dataset manifest и actual tokenizer
chat template hashes.

## DPO

DPO preprocessing и collation существуют, но DPO loss route пока не
реализован. `dpo_target` нельзя добавлять в active `loss_routing.routes`.

## Тесты

```bash
python -m pytest
```

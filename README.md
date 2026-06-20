# estadel-trainer

Production-контур для LoRA-дообучения Qwen3.5-35B-A3B в диалоговом домене:

- source model и tokenizer берутся из MLflow/modelctl registry;
- tokenizer всегда загружается из resolved model directory;
- SFT и DPO идут через единый routed training loop;
- batches однородны по `loss_kind`;
- training запускается только через Accelerate/FSDP;
- ordinary validation и опциональный bundled RU BFCL;
- MLflow tracking, async side effects, adapter-only checkpoints и candidate registration;
- strict resume с проверкой config, dataset manifest, tokenizer chat template и source model hash;
- DVC lineage ищется автоматически рядом с train parquet.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`modelctl` приходит зависимостью `modelctl-mlflow` из PyPI и вызывается из
`PATH`. Путь к executable в YAML не конфигурируется.

Для окружений, где нужны CUDA kernels:

```bash
python -m pip install --no-build-isolation -e ".[cuda-kernels]"
```

Для окружений, которым нужен Transformers main:

```bash
python -m pip install -e ".[transformers-main]"
```

## Конфигурация

Единственный production template: `configs/config.example.yaml`.

```bash
cp configs/config.example.yaml configs/config.yaml
```

Минимально перед запуском задаются:

- `project.name`: MLflow experiment и destination registry model;
- `project.run_name`: MLflow run name;
- `project.output_dir`: единый корень локальных результатов;
- `model.name`, `model.alias`, `model.cache_dir`: source model из registry;
- `preprocessing.raw.train_path` и `preprocessing.raw.valid_path`;
- `training.num_epochs`;
- `mlflow.tracking_uri`.

Training-critical значения живут в YAML и валидируются в `src/config/schema.py`.
Отдельный tokenizer source, revision и template hash в конфиге не поддерживаются.

Derived paths:

```text
{project.output_dir}/pretokenized
{project.output_dir}/ref_logprobs/<signature>
{project.output_dir}/checkpoints
{project.output_dir}/eval/bfcl_rows.jsonl
```

## Запуск

Основной entrypoint из package:

```bash
accelerate launch --use_fsdp --num_processes <GPU_COUNT> \
  estadel-train --config configs/config.yaml
```

`train` также остается Python module в `src`, но production-документация
ориентируется на console script из `pyproject.toml`.

Запуск выполняет полный контур: model source resolution, preprocessing cache,
routed dataloaders, DPO ref-logprob cache, training, validation,
checkpointing и candidate registration.

## Model Registry

Source model задается alias-ссылкой:

```yaml
model:
  name: estadel-llm
  alias: champion
  cache_dir: artifacts/model_cache/estadel-llm/champion
```

Main process вызывает `modelctl info`, проверяет локальный cache через
`modelctl verify`, при mismatch или отсутствии payload выполняет `modelctl pull`
и повторную verification. Результат пишется в sidecar рядом с cache directory.

Candidate checkpoints регистрируются в registry model с именем `project.name`.
Training регистрирует только candidate aliases:

```text
candidate-000001
candidate-latest
```

Promotion до `baseline` или `champion` training loop не выполняет.

## Dataset

Raw parquet split содержит одну строку на sample:

```python
{
    "data": "{\"messages\": [...], ...}",
    "type": "sft_target" | "sft_tool" | "dpo_target",
}
```

`type` является единственным authoritative route column на raw boundary.
`loss_kind` не выводится из содержимого `data`. Колонка `target` в raw parquet
сейчас не поддерживается и считается schema error.

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
- selected chosen/rejected completion tokens для DPO sequence logprobs.

## DPO

`dpo_target` является поддержанным route:

- preprocessing рендерит `prompt + chosen` и `prompt + rejected` отдельно;
- collator паддит chosen/rejected branches независимо;
- trainer считает DPO loss через `losses.dpo.dpo_loss`;
- reference policy берется из source base model до подключения LoRA;
- reference logprobs считаются отдельным pre-training stage после dataloaders и
  кэшируются в `{project.output_dir}/ref_logprobs/<signature>`;
- DPO loss использует только precomputed reference logprobs из batch cache.

Relevant YAML:

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

## Epochs И Batches

Продолжительность задается через `training.num_epochs`. Число optimizer steps
вычисляется из train DataLoader, числа процессов и
`gradient_accumulation_steps`.

Route frequency определяется составом датасета. Sampler weights не
поддерживаются. `RoutedBatchSampler` строит route-local batches и затем
детерминированно перемешивает список batches seed-ом `project.seed`.

## Evaluation И Selection

Validation boundary:

```text
ordinary eval -> optional BFCL eval -> checkpoint -> candidate selection
```

Ordinary validation возвращает:

```text
eval/loss
eval/ppl
eval/batches
eval/tokens
eval/supervised_tokens
```

Bundled RU BFCL включается через `eval.bfcl.enabled`. Dataset bundled в
`src/eval/ru_bfcl/data/bfcl_eval.jsonl`; путь не конфигурируется.

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

Auto-resume ищет последний валидный `step-NNNNNN` в derived checkpoint
directory. Strict resume сравнивает effective config, dataset manifest,
data/training contracts, actual tokenizer chat template hash и verified source
model payload hash.

## RU BFCL CLI

Standalone validator entrypoint:

```bash
ru-bfcl-eval --help
```

Training loop использует bundled evaluator напрямую через `eval.bfcl`.

## Тесты

```bash
python -m pytest
```

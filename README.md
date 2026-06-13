# estadel-trainer

Проект находится в состоянии разработки. Этап preprocessing считается решенным: raw parquet split-ы читаются, official Qwen chat template применяется, thinking switch учитывается, tokenization и `labels`/loss mask строятся, pretokenized cache сохраняется и переиспользуется по hash raw split-а.

Полный training loop для 35B LoRA/FSDP еще не является текущей готовой стадией: `src/train.py` пока доходит до подготовки pretokenized split-ов и останавливается перед будущим этапом model training.

## Быстрый запуск текущего этапа

В `configs/` сейчас оставлены только два файла:

- `config.example.yaml` — полный шаблон со всеми полями и dummy-значениями.
- `config.preprocess.yaml` — текущий рабочий debug-конфиг для preprocessing DVC-backed `train.parquet`/`valid.parquet`.

Запуск полного реализованного на данный момент процесса для configured `train`/`valid` split-ов:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m train --config configs/config.preprocess.yaml --force-preprocess
```

Что произойдет:

1. Будет прочитан `configs/config.preprocess.yaml`.
2. Будет открыт MLflow run и залоггированы config, code/DVC lineage и стартовые params.
3. Будет загружен tokenizer из resolved model source.
4. Будут прочитаны raw parquet split-ы из `/home/duka/data/llm-call-center/llm-call-center/data/`.
5. Каждая строка `data` будет распарсена как JSON string, а `type`/`target` будет использован как authoritative `loss_kind`.
6. Для SFT/tool/DPO samples будет применен chat template, построены supervised spans и `labels`.
7. Результат будет записан в `artifacts/pretokenized/render_valid/`.
8. Будут построены routed DataLoader-ы и залоггированы их summaries.

Если нужно только проверить reuse уже готового cache, убери `--force-preprocess`:

```bash
PYTHONPATH=src:vendor/modelctl-mlflow /home/duka/miniforge3/envs/train/bin/python -m train --config configs/config.preprocess.yaml
```

## Execution Path

Основной запуск идет через такие файлы:

```text
src/train.py
  -> config.load_config
     src/config/{loader.py,schema.py,hashing.py}
  -> tracking.ExperimentTracker
     MLflow run, DVC lineage, registry/local model source resolution
  -> preprocessing.pipeline.prepare_pretokenized_splits
     src/preprocessing/pipeline.py
  -> preprocessing.pipeline.load_tokenizer
     tokenizer: tokenizer.source=model -> model.resolved_model_id
  -> preprocessing.io.resolve_split_paths/read_raw_dataframe/dataframe_to_rows
     src/preprocessing/io.py
  -> preprocessing.pipeline._preprocess_sft_dataset_row / _preprocess_dpo_dataset_row
     src/preprocessing/pipeline.py
  -> preprocessing.rendering.reject_forbidden_raw_markers
     src/preprocessing/rendering.py
  -> preprocessing.masking.tokenize_with_offsets / build_labels
     src/preprocessing/masking.py
  -> preprocessing.io.write_split_cache
     artifacts/pretokenized/render_valid/{train.parquet,valid.parquet,debug.jsonl,manifest.json}
```

## Входы

Текущий raw parquet contract простой:

```text
data: JSON string одного sample
type или target: sft_target | sft_tool | dpo_target
```

Одна строка parquet равна одному sample. `type`/`target` является authoritative `loss_kind`; тип не выводится из содержимого `data`.

Для текущего debug run используется:

```text
configs/config.preprocess.yaml
/home/duka/data/llm-call-center/llm-call-center/data/train.parquet
/home/duka/data/llm-call-center/llm-call-center/data/valid.parquet
model.source.local_dir: /mnt/e/Downloads/tokenizer
tokenizer.source: model
MLflow: http://31.192.108.203:5000/
DVC repo: /home/duka/data/llm-call-center/llm-call-center
```

Preprocessing-related настройки теперь собраны под одним YAML блоком:

```yaml
preprocessing:
  raw:
    train_path: /home/duka/data/llm-call-center/llm-call-center/data/train.parquet
    valid_path: /home/duka/data/llm-call-center/llm-call-center/data/valid.parquet
    test_path: /home/duka/data/llm-call-center/llm-call-center/data/test.parquet
    test_required: false
  output:
    root_dir: artifacts/pretokenized/render_valid
    reuse_if_hash_matches: true
    debug_examples_per_loss_kind: 5
  sequence:
    max_seq_len: 81920
    truncation: false
    packing: false
  rendering:
    reject_raw_special_markers: true
  reasoning:
    enable_thinking: false
  masking:
    ignore_index: -100
    require_positive_supervised_tokens: true
    policies:
      sft_target:
        min_guaranteed_assistant_chars: 80
        loss_on_short_assistant_reply_prob: 0.3
        short_response_sampling_seed: 42
      sft_tool: {}
```

Источник tokenizer намеренно следует за источником модели:

```yaml
model:
  source:
    kind: local_or_hf
    local_dir: /mnt/e/Downloads/tokenizer

tokenizer:
  source: model
  tokenizer_id: null
```

`tokenizer` остается отдельной секцией только для tokenizer-specific параметров вроде `use_fast`, `add_special_tokens`, `padding_side` и expected template hash. Если для отдельного debug run понадобится отвязать tokenizer от модели, надо выставить `tokenizer.source=explicit` и указать `tokenizer.tokenizer_id`.

Что здесь за что отвечает:

- `raw`: входные parquet split-ы. `train_path`, `valid_path`, `test_path` — пути к raw parquet; `test_required=false` значит, что отсутствующий test split можно пропустить.
- `output`: pretokenized cache. В `root_dir` пишутся `{split}.parquet`, общий `debug.jsonl` и `manifest.json`; `reuse_if_hash_matches` включает reuse по hash raw split-а; `debug_examples_per_loss_kind` ограничивает число debug rows на каждый `loss_kind`.
- `sequence`: ограничения длины. `max_seq_len` сравнивается с tokenizer/model context и с длиной каждого tokenized row; при `truncation=false` overlong rows отклоняются; `packing=false` фиксирует, что packing сейчас не применяется.
- `rendering`: проверки до chat template. Сейчас используется только `reject_raw_special_markers`, чтобы raw content не содержал template-owned маркеры вроде `<|im_start|>`.
- `reasoning`: один флаг `enable_thinking`. Если `false`, `<think>...</think>` не попадает в loss; если `true`, попадает.
- `masking`: построение `labels`. `ignore_index` — значение для токенов без loss; `require_positive_supervised_tokens` отклоняет строки с нулем supervised tokens; `sft_target` задает deterministic short-reply sampling; `sft_tool: {}` означает текущую фиксированную политику: loss на всех assistant completions.

MLflow/DVC lineage живет в отдельном tracking слое, а не внутри `preprocessing`: dataset git commit, DVC `.dvc` hash, split hashes, config hash, preprocessing manifest и DataLoader summaries логгируются в MLflow, но не управляют render/tokenize/mask проходом.

## Выходы

После запуска с `configs/config.preprocess.yaml` пишется:

```text
artifacts/pretokenized/render_valid/
├── train.parquet
├── valid.parquet
├── debug.jsonl
└── manifest.json
```

Split parquet содержит tokenized rows с `input_ids`, `attention_mask`, `labels`, `loss_kind` и metadata/hash полями.

`debug.jsonl` содержит ограниченный audit sample: максимум `debug_examples_per_loss_kind` примеров на `loss_kind` для split-а. Это файл для ручного просмотра rendered text и loss-only участков, а не полный dump датасета.

`manifest.json` намеренно минимален. Cache reuse идет по hash конкретного raw split. Его структура:

```json
{
  "debug": {
    "examples_per_loss_kind_per_split": 5,
    "num_rows": 10,
    "path": "artifacts/pretokenized/render_valid/debug.jsonl"
  },
  "splits": {
    "train": "sha256:...",
    "valid": "sha256:..."
  },
  "pretokenized": {
    "train": "sha256:...",
    "valid": "sha256:..."
  },
  "rows": {
    "train": {
      "raw": 0,
      "processed": 0,
      "rejected": 0
    },
    "valid": {
      "raw": 0,
      "processed": 0,
      "rejected": 0
    }
  }
}
```

Если изменился только `valid.parquet`, пересчитывается только `valid`. Если cache отсутствует или hash split-а не совпадает, split пересобирается.

`artifacts/rendered_debug/*` — старый render-only artifact от предыдущего ручного probe. Текущий startup preprocessing не пишет туда и не читает оттуда; актуальный debug output — только `artifacts/pretokenized/.../debug.jsonl`.

## Реализованные правила preprocessing

- `sft_tool`: loss на всех assistant completions.
- `sft_target`: assistant completions длиннее `min_guaranteed_assistant_chars` всегда идут в loss; короткие идут в loss с deterministic probability `loss_on_short_assistant_reply_prob`, по умолчанию `0.3`.
- `dpo_target`: loss/logprob labels только на `chosen` и `rejected` completions.
- `assistant.tool_calls[].function.arguments` из JSON string нормализуется в object перед Qwen template rendering.
- `preprocessing.reasoning.enable_thinking` передается в `apply_chat_template`, если tokenizer это поддерживает; если не поддерживает, preprocessing делает fallback и пишет unsupported kwarg в debug/stats.
- Если `enable_thinking=false`, все `<think>...</think>` blocks вырезаются из supervised loss.
- Если `enable_thinking=true`, `<think>...</think>` blocks остаются в supervised loss.
- Raw text с template markers вроде `<|im_start|>` отклоняется.
- `preprocessing.sequence.max_seq_len` берется из YAML, сверяется с tokenizer/model context и применяется к каждому tokenized row. При `truncation=false` overlong rows отклоняются.
- `debug.jsonl` loss-only text строится из финальных `input_ids` и `labels`: декодируются только позиции, где `labels[i] != ignore_index`.

## Что еще не готово

Текущий `src/train.py` пока не запускает реальное обучение модели. После успешного preprocessing, MLflow tracking и DataLoader build он логирует:

```text
startup preprocessing complete; model training is the next pipeline stage
```

Следующие крупные этапы проекта: training dataset/dataloader, routed homogeneous batches, SFT/DPO loss routes, Accelerate/FSDP trainer, checkpoints/resume, MLflow/registry и BFCL-like eval.

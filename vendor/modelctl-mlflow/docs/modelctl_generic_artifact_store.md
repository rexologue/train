# modelctl generic artifact store

Этот документ описывает философию, использование и внутреннюю реализацию `modelctl` в режиме `generic`.

`generic` — основной режим для model registry, когда нужно хранить и доставать **любой формат модели** через MLflow Artifact Store и MLflow Model Registry.

## 1. Философия инструмента

`modelctl` создавался не как inference framework и не как попытка заставить все модели быть MLflow-native flavor.

Его задача проще:

```text
зарегистрировать локальную директорию модели
→ сохранить её в artifact store
→ создать версию в MLflow Registry
→ повесить aliases
→ дать простой pull с любого сервера
```

То есть `modelctl` — это тонкая дисциплинированная оболочка над MLflow Registry.

Основной принцип:

```text
artifact store хранит байты модели
registry хранит имя, версию, aliases, tags и ссылку на artifact
```

За счёт этого можно работать не с путями вида:

```text
/mnt/nfs_share/models-pool/baselines/Qwen3.5-35B-A3B
```

а с логическими ссылками:

```text
qwen35_35b_a3b_cc@baseline
qwen35_35b_a3b_cc@champion
qwen35_35b_a3b_cc@candidate
qwen35_35b_a3b_cc:7
```

Потребителю модели не нужно знать, где модель была обучена, с какого сервера она была зарегистрирована и где лежала исходная папка. Потребителю достаточно сделать `modelctl pull`.

## 2. Почему не external path

Для некоторых систем достаточно хранить только ссылку на директорию:

```text
source_path=/mnt/nfs_share/models-pool/baselines/Qwen3.5-35B-A3B
```

Но это не решает твою задачу.

Требование `modelctl` другое:

```text
сервер без доступа к исходной шаре должен иметь возможность скачать модель
```

Поэтому `generic` не должен быть metadata-only записью. Он должен физически положить payload в MLflow artifact store.

Идеальная модель ответственности такая:

```text
исходная папка          -> временный источник регистрации
MLflow artifact store   -> долговременное хранилище модели
MLflow registry         -> каталог версий и aliases
любой другой сервер     -> скачивает через modelctl pull
```

После успешного `register` исходную папку можно удалить, перенести или потерять. Модель всё равно должна оставаться доступной через artifact store.

## 3. Роли компонентов

### PostgreSQL / MLflow backend store

PostgreSQL хранит:

```text
experiments
runs
registered models
model versions
aliases
tags
params
metadata records
```

PostgreSQL **не должен хранить 67 GB весов модели**.

Ему всё равно нужен свободный локальный диск для WAL, checkpoint-ов и обычной работы БД. Но объём весов модели относится к artifact store, а не к PostgreSQL.

### MLflow artifact store

Artifact store хранит реальные файлы:

```text
safetensors
bin
json
tokenizer files
configs
adapters
ONNX files
custom bundles
```

В твоём Docker Compose варианте контейнер MLflow видит artifact store как:

```text
/mlflow/artifacts
```

а host path берётся из `.env`:

```text
MLFLOW_ARTIFACTS_DIR=/mnt/nfs_share/mlflow/artifacts
```

То есть payload физически оказывается на NFS, но потребителю не обязательно иметь прямой mount этой NFS. Он обращается к MLflow server.

### MLflow Model Registry

Registry хранит человеко-понятный слой:

```text
model name
version
alias
source URI
tags
created_at
source_dir_hash
```

Пример:

```text
qwen35_35b_a3b_cc/1
qwen35_35b_a3b_cc@baseline -> 1
qwen35_35b_a3b_cc@champion -> 1
```

## 4. Что такое generic

`generic` — это режим, который принимает любую директорию и не пытается интерпретировать её содержимое.

Ему не важно, что внутри:

```text
Hugging Face checkpoint
LoRA adapter
ONNX bundle
Torch checkpoint
custom C++ inference bundle
архив конфигов
папка с tokenizer + weights
```

Он делает директорию версионируемым artifact payload.

Минимальная команда:

```bash
modelctl register . qwen35_35b_a3b_cc --alias baseline --alias champion
```

Явно указать kind можно так:

```bash
modelctl register . qwen35_35b_a3b_cc --kind generic --alias baseline --alias champion
```

`generic` является default mode, поэтому `--kind generic` обычно не нужен.

## 5. Artifact layout

Generic-версия сохраняется в artifact store так:

```text
model/
├── MLmodel
├── manifest.json
├── metadata/
│   ├── general_tags.json
│   └── training_tags.json
└── payload/
    └── ... исходное содержимое source_dir ...
```

### `payload/`

`payload/` — это исходная модель.

Если ты регистрировал:

```text
Qwen3.5-35B-A3B/
├── config.json
├── tokenizer.json
├── model-00001-of-00016.safetensors
└── ...
```

то после `modelctl pull qwen35_35b_a3b_cc@baseline ./model` ты получишь:

```text
./model/
├── config.json
├── tokenizer.json
├── model-00001-of-00016.safetensors
└── ...
```

То есть default `pull` возвращает именно payload, а не wrapper-папку.

### `manifest.json`

`manifest.json` — техническое описание зарегистрированного payload-а.

Примерная структура:

```json
{
  "schema_version": "1.0",
  "created_by": "modelctl",
  "created_at": "2026-06-14T09:00:00Z",
  "model_name": "qwen35_35b_a3b_cc",
  "kind": "generic",
  "source_basename": "Qwen3.5-35B-A3B",
  "source_dir_hash": "sha256:...",
  "payload_path": "payload",
  "general_tags_path": "metadata/general_tags.json",
  "training_tags_path": "metadata/training_tags.json",
  "general_tags": {},
  "training_tags": {},
  "hf_task": null,
  "pytorch_file": null
}
```

### `MLmodel`

`MLmodel` нужен для самодокументирования artifact package.

Он описывает не inference flavor, а modelctl generic layout:

```yaml
artifact_path: model
flavors:
  modelctl_generic:
    schema_version: 1.0
    payload_path: payload
    manifest_path: manifest.json
    model_name: "qwen35_35b_a3b_cc"
    source_dir_hash: "sha256:..."
modelctl_kind: generic
modelctl_schema_version: 1.0
```

Этот файл полезен при ручном просмотре artifact-а и в MLflow UI.

## 6. Register workflow

Команда:

```bash
modelctl register ./Qwen3.5-35B-A3B qwen35_35b_a3b_cc --alias baseline --alias champion
```

Внутри происходит следующее.

### Шаг 1. Проверка source_dir

`modelctl` проверяет, что путь существует и является директорией.

Ошибки на этом этапе:

```text
Source path does not exist
Source path must be a directory
Registered model name cannot be empty
```

### Шаг 2. Настройка MLflow tracking URI

По умолчанию:

```text
http://localhost:5000
```

Можно переопределить:

```bash
modelctl register ./model my_model --tracking-uri http://mlflow.internal:5000
```

### Шаг 3. Хеширование source_dir

`modelctl` считает стабильный SHA256 по всем файлам директории.

В hash входят:

```text
relative file path
file bytes
```

В hash не входят:

```text
mtime
owner
group
permissions
absolute path
```

Это сделано специально: одна и та же модель на разных серверах и mount points должна давать один и тот же digest.

Для больших моделей `modelctl` пишет coarse progress в stderr:

```text
[modelctl] hashing source directory: /mnt/nfs_share/models-pool/baselines/Qwen3.5-35B-A3B
[modelctl] hashed 5.0 GiB so far
[modelctl] hashed 10.0 GiB so far
[modelctl] hashed total 67.0 GiB
[modelctl] source hash computed: sha256:...
```

### Шаг 4. Создание technical run

Все регистрации складываются в отдельный experiment:

```text
__model_registry_uploads__
```

Это не training experiment. Это технический experiment для операций registry upload.

Run name выглядит так:

```text
register:<model_name>:<kind>
```

Пример:

```text
register:qwen35_35b_a3b_cc:generic
```

### Шаг 5. Логирование metadata

Перед payload-ом `modelctl` логирует небольшие JSON artifacts:

```text
modelctl_metadata/general_tags.json
modelctl_metadata/training_tags.json
modelctl_metadata/manifest.json
```

Также metadata попадает внутрь основного model artifact:

```text
model/manifest.json
model/metadata/general_tags.json
model/metadata/training_tags.json
model/MLmodel
```

### Шаг 6. Логирование payload

Крупная операция одна:

```text
source_dir -> model/payload в MLflow artifact store
```

`modelctl` не создаёт полную локальную копию source_dir перед upload/copy.

Это принципиально для больших LLM-директорий:

```text
локальный root disk не должен иметь свободное место размером с модель только ради staging copy
```

Нужно, чтобы место было в artifact store.

### Шаг 7. Создание model version

После artifact logging создаётся Model Registry version:

```text
name=qwen35_35b_a3b_cc
source=runs:/<run_id>/model
run_id=<run_id>
tags=...
```

### Шаг 8. Tags и aliases

`modelctl` пишет служебные tags:

```text
modelctl.managed=true
modelctl.schema_version=1.0
modelctl.kind=generic
modelctl.source_dir_hash=sha256:...
modelctl.created_at=...
```

Пользовательские metadata tags flatten-ятся в namespace:

```text
general.*
training.*
```

Aliases назначаются после создания версии.

Default behavior:

```text
первая версия -> baseline + champion
следующие версии -> candidate
```

## 7. Pull workflow

Команда:

```bash
modelctl pull qwen35_35b_a3b_cc@champion ./Qwen3.5-35B-A3B --overwrite
```

Внутри происходит:

```text
1. ref qwen35_35b_a3b_cc@champion resolves to concrete model version
2. modelctl reads model version source URI
3. modelctl downloads runs:/<run_id>/model/payload
4. download goes to staging directory next to output_dir
5. staging directory is moved to output_dir
```

Почему staging рядом с `output_dir`:

```text
чтобы не использовать системный /tmp для 50-100+ GB моделей
```

Пример staging path:

```text
./.modelctl_download_Qwen3.5-35B-A3B_<uuid>/
```

После успешного move staging удаляется.

Default `pull` для generic отдаёт payload-only:

```text
modelctl pull name@alias ./out
```

Результат:

```text
./out/
└── ... original source_dir contents ...
```

Если нужен весь wrapper package:

```bash
modelctl pull name@alias ./out_package --full-package --overwrite
```

Результат:

```text
./out_package/
├── MLmodel
├── manifest.json
├── metadata/
└── payload/
```

## 8. Metadata philosophy

`modelctl` хранит metadata в двух формах.

### Полные JSON artifacts

Это source-of-truth metadata:

```text
modelctl_metadata/general_tags.json
modelctl_metadata/training_tags.json
model/metadata/general_tags.json
model/metadata/training_tags.json
```

Туда можно класть вложенные dict/list структуры.

### Searchable MLflow tags

MLflow tags плоские, поэтому `modelctl` flatten-ит вложенные структуры.

Пример:

```json
{
  "metrics": {
    "bfcl_accuracy": 0.91
  }
}
```

становится:

```text
training.metrics.bfcl_accuracy=0.91
```

Это удобно для просмотра в UI и поиска версий.

## 9. Promotion philosophy

`promote` не копирует модель.

Команда:

```bash
modelctl promote qwen35_35b_a3b_cc 3 champion
```

делает только:

```text
alias champion -> version 3
```

Это быстрый rollback/promotion mechanism.

Например:

```text
qwen35_35b_a3b_cc@champion -> version 2
```

после promotion:

```text
qwen35_35b_a3b_cc@champion -> version 3
```

Потребитель, который всегда pull-ит `@champion`, автоматически получает новую версию при следующем pull.

## 10. Typical workflows

### Зарегистрировать baseline

```bash
modelctl register /mnt/nfs_share/models-pool/baselines/Qwen3.5-35B-A3B qwen35_35b_a3b_cc --alias baseline --alias champion
```

### Зарегистрировать candidate

```bash
modelctl register /mnt/nfs_share/models-pool/candidates/qwen35_sft_run_042 qwen35_35b_a3b_cc --alias candidate
```

### Посмотреть версии

```bash
modelctl list qwen35_35b_a3b_cc
```

### Посмотреть одну ссылку

```bash
modelctl info qwen35_35b_a3b_cc@candidate
```

### Продвинуть candidate в champion

```bash
modelctl promote qwen35_35b_a3b_cc 2 champion
```

### Откатиться на baseline

```bash
modelctl promote qwen35_35b_a3b_cc 1 champion
```

### Скачать champion на сервер без NFS

```bash
modelctl pull qwen35_35b_a3b_cc@champion /models/qwen35_35b_a3b_cc --overwrite --tracking-uri http://mlflow.example.com:5000
```

## 11. Implementation details

Ключевая функция:

```text
register_model_directory(...)
```

Логика:

```text
validate source path
configure MLflow
ensure registered model exists
select aliases
build manifest
build version tags
start MLflow run
log metadata
log model by kind
create model version
set tags
set aliases
return RegisterResult
```

Для `kind=generic` вызывается:

```text
log_generic_model(...)
```

Она всегда использует direct artifact layout:

```text
log small metadata directory -> artifact_path=model
log source_path directory    -> artifact_path=model/payload
```

Псевдокод:

```python
with tempfile.TemporaryDirectory(prefix="modelctl_meta_") as temp_dir:
    model_dir = Path(temp_dir) / "model"
    write_json(model_dir / "manifest.json", manifest)
    write_json(model_dir / "metadata/general_tags.json", general_tags)
    write_json(model_dir / "metadata/training_tags.json", training_tags)
    write_text(model_dir / "MLmodel", build_generic_mlmodel_text(manifest))
    mlflow.log_artifacts(str(model_dir), artifact_path="model")

mlflow.log_artifacts(str(source_path), artifact_path="model/payload")
```

Важное свойство:

```text
локально staging-ятся только маленькие metadata-файлы
payload не copytree-ится в системный temp
```

## 12. Failure model

### Ошибка до создания model version

Если upload payload прошёл частично, но `create_model_version` не успел выполниться, в MLflow может остаться technical run без registry version.

Проверка:

```bash
modelctl list qwen35_35b_a3b_cc
```

Если версия не появилась, регистрацию нужно повторить после устранения причины.

### PostgreSQL recovery / 503

Если MLflow отвечает 503, а в логах Postgres есть:

```text
No space left on device
```

то проблема не в том, что Postgres хранит модель. Проблема в том, что backend DB не может записать свои служебные файлы.

Проверить:

```bash
df -h /
```

```bash
docker logs --tail=200 mlflow-postgres
```

### Artifact store full

Если заполнен artifact store, регистрация оборвётся на upload/copy payload-а.

Проверить:

```bash
df -h /mnt/nfs_share
```

## 13. How to confirm that the new implementation is used

Импорт должен проходить без побочных предупреждений от generic-реализации:

```bash
python -c "import modelctl, modelctl.core; print(modelctl.core.__file__)"
```

Проверить, откуда запускается CLI:

```bash
which modelctl && python -c "import sys, modelctl, modelctl.cli, modelctl.core; print('python=', sys.executable); print('modelctl=', modelctl.__file__); print('cli=', modelctl.cli.__file__); print('core=', modelctl.core.__file__)"
```

## 14. What to monitor during large register

В отдельном терминале:

```bash
watch -n 2 'df -h / /mnt/nfs_share; du -sh /mnt/nfs_share/mlflow/artifacts 2>/dev/null'
```

Нормальная картина:

```text
/ не теряет десятки GB из-за modelctl staging
/mnt/nfs_share/mlflow/artifacts растёт во время upload/copy
stderr modelctl показывает hash/upload/model-version stages
```

## 15. Practical recommendation for large LLMs

Для 30-100+ GB моделей нормальный workflow такой:

```text
1. Сложить модель во временную source_dir, например на NFS или локальный большой диск.
2. Запустить modelctl register source_dir name --alias candidate.
3. Проверить modelctl list name.
4. Проверить modelctl pull name@candidate на другом сервере.
5. После проверки продвинуть alias champion.
6. Старую исходную source_dir можно удалить, если artifact store считается canonical.
```

Главное правило:

```text
artifact store должен быть долговременным хранилищем, а source_dir — только источником регистрации
```

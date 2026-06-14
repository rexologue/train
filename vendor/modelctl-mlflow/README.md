# modelctl-mlflow

`modelctl` is a small CLI wrapper around MLflow Model Registry. It gives a project one stable interface for registering, promoting and pulling model directories through MLflow, without forcing every model to have the same framework format.

The default and most important mode is `generic`: an arbitrary local directory is stored as an opaque payload in the MLflow artifact store and registered as a Model Registry version.

Detailed documentation for the generic artifact-store workflow is available here:

```text
docs/modelctl_generic_artifact_store.md
```

## Core idea

`modelctl` treats MLflow as two separate systems working together:

```text
MLflow backend store / PostgreSQL -> registry metadata, runs, versions, aliases, tags
MLflow artifact store             -> actual model files and payload directories
```

The database is not expected to store model weights. It stores metadata only. The artifact store is the durable source of truth for model files.

That is the reason `generic` exists: it lets you put any model bundle into the artifact store and later pull it from another server through MLflow, even if that server does not have direct access to the original training machine or original NFS/source path.

## Supported kinds

### `generic`

Default mode. Works with any local directory.

Use it for:

```text
Hugging Face checkpoints
raw PyTorch checkpoint folders
ONNX exports
tokenizer folders
LoRA adapter folders
custom inference bundles
configs + weights
any arbitrary model directory
```

Generic mode stores the original directory under this artifact layout:

```text
model/
├── MLmodel
├── manifest.json
├── metadata/
│   ├── general_tags.json
│   └── training_tags.json
└── payload/
    └── ... original directory contents ...
```

The large payload is logged directly into the configured MLflow artifact store under `model/payload`. `modelctl` does not create a full local temporary copy of the source directory before upload.

### `hf`

Native Hugging Face Transformers logging through MLflow Transformers flavor.

Use this only when you specifically want MLflow's Transformers flavor. For large production LLM bundles where the goal is artifact-store durability and simple `pull`, `generic` is usually simpler.

### `pytorch`

Native MLflow PyTorch logging for TorchScript models.

This mode needs a scripted/traced model file loadable with `torch.jit.load`. A raw `.pth` state dict is not enough because the Python model class cannot be reconstructed from weights alone. For raw checkpoint folders, use `generic`.

## Installation

From the project directory:

```bash
pip install -e .
```

With optional ML framework dependencies:

```bash
pip install -e '.[all]'
```

## MLflow connection

By default, `modelctl` connects to:

```text
http://localhost:5000
```

Override it with:

```bash
modelctl list my_model --host 127.0.0.1 --port 5000
```

Or pass a full URI:

```bash
modelctl list my_model --tracking-uri http://127.0.0.1:5000
```

MLflow Basic Auth credentials are read by MLflow itself from environment variables:

```bash
export MLFLOW_TRACKING_USERNAME=my_user
```

```bash
export MLFLOW_TRACKING_PASSWORD=my_password
```

`modelctl` does not parse or store passwords.

## Register a generic model folder

Minimal command:

```bash
modelctl register ./exported_model sentiment_text_classifier
```

Explicit kind:

```bash
modelctl register ./exported_model sentiment_text_classifier --kind generic
```

What happens:

```text
1. modelctl connects to MLflow.
2. modelctl creates/uses the technical experiment __model_registry_uploads__.
3. modelctl computes a stable SHA256 hash of the source directory.
4. modelctl starts a short MLflow run.
5. modelctl writes manifest.json, MLmodel and metadata JSON files.
6. modelctl logs the original source directory to model/payload in the artifact store.
7. modelctl creates a new MLflow Model Registry version whose source is runs:/<run_id>/model.
8. modelctl attaches aliases and searchable tags to the new version.
```

Default alias behavior:

```text
first version of a registered model -> baseline + champion
later versions -> candidate
```

So the first registration creates something like:

```text
sentiment_text_classifier/1
sentiment_text_classifier@baseline -> version 1
sentiment_text_classifier@champion -> version 1
```

A later registration without explicit aliases creates:

```text
sentiment_text_classifier/2
sentiment_text_classifier@candidate -> version 2
```

## Register with explicit aliases

Aliases can be repeated:

```bash
modelctl register ./exported_model sentiment_text_classifier --alias candidate
```

```bash
modelctl register ./baseline_model sentiment_text_classifier --alias baseline --alias champion
```

## Register with metadata tags

Tags are optional. There are two namespaces:

```text
general  - stable model information
training - training, dataset, metrics and experiment information
```

The full dictionaries are logged as JSON artifacts. A flattened searchable projection is also written to MLflow Model Version tags.

Example `general.json`:

```json
{
  "task": "sentiment-classification",
  "owner": "duka",
  "labels": ["negative", "neutral", "positive"]
}
```

Example `training.json`:

```json
{
  "dataset_version": "v4",
  "git_sha": "abc123",
  "metrics": {
    "f1_weighted": 0.914
  }
}
```

Register with JSON metadata:

```bash
modelctl register ./exported_model sentiment_text_classifier --general-tags-json general.json --training-tags-json training.json
```

Inline metadata also works:

```bash
modelctl register ./exported_model sentiment_text_classifier --general-tag task=sentiment --training-tag dataset_version=v4 --training-tag f1_weighted=0.914
```

Inline values are parsed as JSON when possible, so numbers, booleans and lists work.

## Register a Hugging Face model

```bash
modelctl register ./hf_model sentiment_text_classifier --kind hf --hf-task text-classification --alias candidate
```

## Register a TorchScript PyTorch model

```bash
modelctl register ./torchscript_bundle image_classifier --kind pytorch --pytorch-file model.pt --alias candidate
```

## List versions

```bash
modelctl list sentiment_text_classifier
```

Example output:

```json
[
  {
    "aliases": ["champion"],
    "created_at": "2026-06-14T09:00:00Z",
    "kind": "generic",
    "name": "sentiment_text_classifier",
    "run_id": "...",
    "source": "runs:/.../model",
    "source_dir_hash": "sha256:...",
    "status": "READY",
    "version": "3"
  }
]
```

## Show one model ref

```bash
modelctl info sentiment_text_classifier@champion
```

Supported refs:

```text
name@alias
name:version
models:/name@alias
models:/name/version
```

## Promote aliases

Promotion is alias reassignment. It does not copy files and does not modify model artifacts.

```bash
modelctl promote sentiment_text_classifier 3 champion
```

After that, consumers pulling `sentiment_text_classifier@champion` receive version `3`.

## Pull a model

Pull the payload only, which is the default for `generic` models:

```bash
modelctl pull sentiment_text_classifier@champion ./local_model --overwrite
```

For a generic model, the output directory receives the original registered payload contents, not the small MLflow wrapper metadata:

```text
./local_model/
└── ... original directory contents ...
```

To download the full MLflow artifact package with `MLmodel`, `manifest.json`, metadata and `payload`, use:

```bash
modelctl pull sentiment_text_classifier@champion ./local_package --full-package --overwrite
```

`pull` downloads into a staging directory placed next to the final output directory and then moves the result into place. This avoids using system `/tmp` for large model downloads.

## Status output

`register` prints coarse status lines to stderr. This makes long operations visible while keeping stdout as machine-readable JSON.

Example:

```text
[modelctl] hashing source directory: /mnt/nfs_share/models-pool/baselines/Qwen3.5-35B-A3B
[modelctl] hashed 5.0 GiB so far
[modelctl] hashed 10.0 GiB so far
[modelctl] source hash computed: sha256:...
[modelctl] starting MLflow run: register:qwen35_35b_a3b_cc:generic
[modelctl] logging generic payload directory: ... -> model/payload
[modelctl] creating MLflow model version: name=qwen35_35b_a3b_cc, source=runs:/.../model
```

## Operational notes

For large model directories, make sure the MLflow artifact store has enough space. PostgreSQL only needs space for metadata and WAL/checkpoint files, but it still must have some free local disk space because it is the MLflow backend store.

For the Docker Compose setup in this repository, the intended split is:

```text
PostgreSQL volume -> local Docker volume, metadata only
MLflow artifacts  -> /mlflow/artifacts inside container
Host artifact dir -> MLFLOW_ARTIFACTS_DIR from .env
```

If `.env` contains:

```text
MLFLOW_ARTIFACTS_DIR=/mnt/nfs_share/mlflow/artifacts
```

then the actual payload files are stored under the NFS-mounted artifact directory on the host.

## Troubleshooting

Check that the CLI is importing the expected source tree:

```bash
which modelctl && python -c "import sys, modelctl, modelctl.cli, modelctl.core; print('python=', sys.executable); print('modelctl=', modelctl.__file__); print('cli=', modelctl.cli.__file__); print('core=', modelctl.core.__file__)"
```

Check backend and artifact store free space:

```bash
df -h / /mnt/nfs_share
```

Check PostgreSQL readiness:

```bash
docker exec mlflow-postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
```

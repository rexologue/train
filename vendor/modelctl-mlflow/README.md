# modelctl-mlflow

`modelctl` is a small CLI wrapper around MLflow Model Registry. It gives projects one simple way to version model folders, regardless of whether the payload is a raw directory, a Hugging Face Transformers checkpoint, or a TorchScript PyTorch model.

The utility always creates technical MLflow runs in one dedicated experiment:

```text
__model_registry_uploads__
```

You do not need to think about that experiment during normal usage. It exists only to keep registry upload runs separate from real training experiments.

## What this solves

Instead of manually creating folders like this:

```text
models/model_v1
models/model_v2
models/final_best
```

use stable MLflow Registry names and aliases:

```text
sentiment_text_classifier@baseline
sentiment_text_classifier@candidate
sentiment_text_classifier@champion
```

A model consumer can pull or load `sentiment_text_classifier@champion` and does not care which exact version number is behind it.

## Supported kinds

### `generic`

Default mode. Works with any local directory.

The utility creates a generic PyFunc MLflow model by itself. You do not write any PyFunc class. The original directory is stored under the modelctl payload package.

Use this for:

```text
raw PyTorch checkpoints
ONNX exports
tokenizer folders
custom inference bundles
configs + weights
any arbitrary model directory
```

### `hf`

Native Hugging Face Transformers logging through MLflow Transformers flavor.

The source directory should be a valid local Transformers checkpoint directory, usually containing `config.json`.

### `pytorch`

Native MLflow PyTorch logging for TorchScript models.

This mode needs a scripted/traced model file loadable with `torch.jit.load`. A raw `.pth` state dict is not enough because the Python model class cannot be reconstructed from weights alone. For raw PyTorch checkpoint folders, use `generic`.

## Installation

From the project directory:

```bash
pip install -e .
```

With Hugging Face / PyTorch extras:

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

## Authentication

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

What happens:

```text
1. modelctl connects to MLflow at http://localhost:5000
2. modelctl creates/uses experiment __model_registry_uploads__
3. modelctl creates a short technical run
4. modelctl computes a stable SHA256 hash of the source directory
5. modelctl generates manifest.json
6. modelctl logs the payload as a generic PyFunc MLflow model
7. modelctl creates a new Model Registry version
8. modelctl sets aliases
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

A later registration without explicit alias creates:

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

Tags are optional.

There are two metadata namespaces:

```text
general  - general model information
training - training/experiment/dataset information
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

For text generation:

```bash
modelctl register ./qwen_export qwen_chat_model --kind hf --hf-task text-generation --alias candidate
```

## Register a TorchScript PyTorch model

If the directory contains `model.pt`:

```bash
modelctl register ./torchscript_export face_detector --kind pytorch --alias candidate
```

If the file has another name:

```bash
modelctl register ./torchscript_export face_detector --kind pytorch --pytorch-file traced_model.pt --alias candidate
```

For arbitrary PyTorch checkpoint folders, use generic mode:

```bash
modelctl register ./checkpoint_dir face_detector --kind generic --alias candidate
```

## Promote a version

Promotion is just alias movement.

```bash
modelctl promote sentiment_text_classifier 2 champion
```

After that:

```text
sentiment_text_classifier@champion -> version 2
```

## Pull a model

Pull current champion:

```bash
modelctl pull sentiment_text_classifier@champion ./models/sentiment_text_classifier --overwrite
```

Pull exact version:

```bash
modelctl pull sentiment_text_classifier:2 ./models/sentiment_text_classifier_v2 --overwrite
```

For generic models, the default is to copy only the original payload directory into the destination. This is usually what project code wants.

To copy the full MLflow model package:

```bash
modelctl pull sentiment_text_classifier@champion ./models/sentiment_full_package --full-package --overwrite
```

## List versions

```bash
modelctl list sentiment_text_classifier
```

## Show info

```bash
modelctl info sentiment_text_classifier@champion
```

```bash
modelctl info sentiment_text_classifier:2
```

## Internal artifact structure for generic models

A generic model is stored as an MLflow PyFunc model. Inside the modelctl package, the structure is:

```text
package/
  manifest.json
  metadata/
    general_tags.json
    training_tags.json
  payload/
    ... original source directory ...
```

The generated `manifest.json` contains:

```json
{
  "schema_version": "1.0",
  "created_by": "modelctl",
  "model_name": "sentiment_text_classifier",
  "kind": "generic",
  "source_dir_hash": "sha256:...",
  "payload_path": "payload",
  "general_tags_path": "metadata/general_tags.json",
  "training_tags_path": "metadata/training_tags.json"
}
```

## Design rules

```text
Every register creates one MLflow model version.
Every register creates one technical run in __model_registry_uploads__.
Generic mode accepts any directory.
Native HF mode is explicit: --kind hf.
Native PyTorch mode is explicit: --kind pytorch.
All metadata tags are optional.
Full metadata dictionaries are stored as JSON artifacts.
Searchable metadata is stored as flattened MLflow tags.
Production code should consume aliases, not latest versions.
```

## Recommended workflow

First version:

```bash
modelctl register ./model_export sentiment_text_classifier --general-tag task=sentiment
```

New candidate:

```bash
modelctl register ./new_model_export sentiment_text_classifier --alias candidate --training-tag dataset_version=v5 --training-tag f1_weighted=0.927
```

Check it:

```bash
modelctl pull sentiment_text_classifier@candidate ./tmp/candidate --overwrite
```

Promote it:

```bash
modelctl promote sentiment_text_classifier 2 champion
```

Use it in a project:

```bash
modelctl pull sentiment_text_classifier@champion ./models/sentiment_text_classifier --overwrite
```

## Notes

`modelctl` is intentionally small. It does not replace training tracking. Real training runs should stay in their own MLflow experiments. Registry upload runs are separate technical records that connect a model version to the exact artifact package, source hash and metadata used during registration.

## Notes about `list`

`modelctl list` prints registered model versions newest first. Internally it enriches the raw MLflow search results with `get_model_version` and the registered model alias map. This avoids empty aliases/tags in MLflow backends where `search_model_versions` returns partially populated entities.

Example:

```bash
modelctl list sentiment_text_classifier
```

Expected output shape:

```json
[
  {
    "name": "sentiment_text_classifier",
    "version": "2",
    "aliases": ["candidate"],
    "kind": "generic",
    "source_dir_hash": "sha256:..."
  }
]
```

If old versions were registered before modelctl metadata was written, fields such as `kind` and `source_dir_hash` can remain `null`. New registrations write these fields explicitly as Model Version tags.

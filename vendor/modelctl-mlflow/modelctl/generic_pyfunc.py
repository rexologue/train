"""Generic PyFunc wrapper used for arbitrary model directories.

The goal of this wrapper is intentionally modest: it makes an arbitrary folder
look like a valid MLflow PyFunc model without forcing the project owner to write
custom inference code.

This is useful when a team needs a single registry workflow for many model
formats: raw PyTorch checkpoints, ONNX exports, tokenizer folders, inference
configs, custom C++ bundles, Hugging Face snapshots, and so on.

The wrapper is not a universal inference runtime. For real online inference the
consumer should normally use ``modelctl pull`` and then load the payload with the
project-specific serving code. The PyFunc ``predict`` method returns metadata
about the packaged payload so that the model is still loadable through MLflow's
PyFunc API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is provided by MLflow pyfunc envs.
    pd = None  # type: ignore[assignment]

import mlflow.pyfunc


class GenericDirectoryPyFunc(mlflow.pyfunc.PythonModel):
    """A tiny PyFunc model for arbitrary directory payloads.

    MLflow's Model Registry expects model versions to point to an MLflow Model
    directory. A random folder with weights/configs is only an artifact until it
    is wrapped in a model flavor. This class provides that wrapper.

    The wrapper receives one artifact named ``package``. The package directory is
    created by ``modelctl`` and has the following structure::

        package/
          manifest.json
          metadata/
            general_tags.json
            training_tags.json
          payload/
            ... original user directory ...

    ``load_context`` stores paths to these files. ``predict`` returns a metadata
    table instead of running model inference. This keeps the model loadable and
    inspectable while preserving the exact original payload.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load paths and manifest from MLflow artifacts.

        Parameters
        ----------
        context:
            Runtime context supplied by MLflow. It contains materialized artifact
            paths. ``modelctl`` always logs a single artifact key named
            ``package`` for generic models.
        """

        package_path = Path(context.artifacts["package"])
        manifest_path = package_path / "manifest.json"
        payload_path = package_path / "payload"

        self.package_path = package_path
        self.manifest_path = manifest_path
        self.payload_path = payload_path
        self.manifest = self._read_json(manifest_path)

    def predict(self, context, model_input, params=None):
        """Return generic model metadata.

        The generic flavor deliberately does not assume how to execute the
        payload. The method returns one metadata row per input row if a pandas
        DataFrame-like input is provided; otherwise it returns a single row.

        Parameters
        ----------
        context:
            MLflow runtime context. It is unused here because paths are loaded in
            ``load_context``. The argument is intentionally left untyped: recent
            MLflow versions infer model signatures from Python type hints and
            warn when a generic PyFunc uses unsupported ``Any`` annotations.
        model_input:
            Any PyFunc-compatible input. Only its length is used.
        params:
            Optional PyFunc parameters. They are accepted for API compatibility.

        Returns
        -------
        pandas.DataFrame | list[dict[str, Any]]
            Metadata containing the packaged payload path and manifest summary.
        """

        row = {
            "kind": self.manifest.get("kind", "generic"),
            "model_name": self.manifest.get("model_name"),
            "payload_path": str(self.payload_path),
            "manifest_path": str(self.manifest_path),
            "source_dir_hash": self.manifest.get("source_dir_hash"),
            "message": "generic MLflow wrapper; use payload_path with project-specific loader",
        }
        n_rows = self._infer_row_count(model_input)
        rows = [row.copy() for _ in range(n_rows)]
        if pd is not None:
            return pd.DataFrame(rows)
        return rows

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read a JSON object from disk and return an empty dict if absent."""

        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _infer_row_count(model_input: Any) -> int:
        """Infer how many metadata rows should be returned by ``predict``."""

        if model_input is None:
            return 1
        try:
            count = len(model_input)  # type: ignore[arg-type]
        except Exception:
            return 1
        return max(int(count), 1)

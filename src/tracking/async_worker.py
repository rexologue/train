from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import tempfile
import threading
from typing import Any, Callable

from mlflow.tracking import MlflowClient

from registry.modelctl_client import ModelctlClient, ModelctlRegisterRequest


JobFn = Callable[[], None]
ClientFactory = Callable[[str | None], Any]


@dataclass(frozen=True)
class AsyncJobError:
    name: str
    error: BaseException


class AsyncTrackingWorker:
    """Run MLflow and registry side effects outside the training step path."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        run_id: str | None = None,
        queue_max_items: int = 1024,
        flush_timeout_seconds: float = 300,
        fail_on_worker_error: bool = True,
        client_factory: ClientFactory | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.flush_timeout_seconds = float(flush_timeout_seconds)
        if self.flush_timeout_seconds <= 0:
            raise ValueError("flush_timeout_seconds must be positive")
        self.fail_on_worker_error = fail_on_worker_error
        self.client_factory = client_factory
        self._jobs: queue.Queue[tuple[str, JobFn] | None] = queue.Queue(maxsize=queue_max_items)
        self._errors: list[AsyncJobError] = []
        self._errors_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def errors(self) -> list[AsyncJobError]:
        """Return background job errors collected so far."""

        with self._errors_lock:
            return list(self._errors)

    def start(self) -> "AsyncTrackingWorker":
        """Start the worker thread on first use."""

        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="async-tracking-worker", daemon=True)
            self._thread.start()
        return self

    def enqueue(self, name: str, fn: JobFn) -> None:
        """Queue a named background job."""

        if self._closed:
            raise RuntimeError("async tracking worker is closed")
        self.start()
        try:
            self._jobs.put((name, fn), timeout=self.flush_timeout_seconds)
        except queue.Full as exc:
            raise TimeoutError(f"async tracking queue remained full for {self.flush_timeout_seconds:g}s") from exc

    def flush(self) -> None:
        """Wait until all previously queued jobs have run and surface errors."""

        completed = threading.Event()
        self.enqueue("worker.flush", completed.set)
        if not completed.wait(self.flush_timeout_seconds):
            raise TimeoutError(f"async tracking flush timed out after {self.flush_timeout_seconds:g}s")
        self._raise_if_needed()

    def close(self) -> None:
        """Flush pending jobs and stop the worker thread."""

        if self._closed:
            return
        flush_error: BaseException | None = None
        try:
            self.flush()
        except BaseException as exc:  # noqa: BLE001 - close must still stop the worker.
            flush_error = exc
        self._closed = True
        try:
            self._jobs.put(None, timeout=self.flush_timeout_seconds)
        except queue.Full:
            if flush_error is None:
                flush_error = TimeoutError("async tracking worker could not enqueue shutdown")
        if self._thread is not None:
            self._thread.join(timeout=self.flush_timeout_seconds)
            if self._thread.is_alive() and flush_error is None:
                flush_error = TimeoutError("async tracking worker did not stop before timeout")
        if flush_error is not None:
            raise flush_error
        self._raise_if_needed()

    def __enter__(self) -> "AsyncTrackingWorker":
        """Enter the worker context and start the thread."""

        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close the worker context."""

        self.close()

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log metrics to the active MLflow run in the background."""

        run_id = self._require_run_id()

        def task() -> None:
            client = self._new_client()
            for key, value in metrics.items():
                client.log_metric(run_id, key, float(value), step=step)

        self.enqueue("mlflow.log_metrics", task)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary artifact to the active MLflow run in the background."""

        run_id = self._require_run_id()

        def task() -> None:
            client = self._new_client()
            artifact = Path(artifact_file)
            with tempfile.TemporaryDirectory(prefix="estadel-mlflow-") as tmp:
                local_path = Path(tmp) / artifact.name
                local_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                artifact_path = None if str(artifact.parent) == "." else str(artifact.parent)
                client.log_artifact(run_id, str(local_path), artifact_path=artifact_path)

        self.enqueue("mlflow.log_dict", task)

    def log_artifact(self, local_path: str | Path, *, artifact_path: str | None = None) -> None:
        """Log a local artifact to the active MLflow run in the background."""

        run_id = self._require_run_id()

        def task() -> None:
            self._new_client().log_artifact(run_id, str(local_path), artifact_path=artifact_path)

        self.enqueue("mlflow.log_artifact", task)

    def run_modelctl_register(self, request: ModelctlRegisterRequest) -> None:
        """Register a modelctl payload in the background."""

        def task() -> None:
            client = ModelctlClient(
                tracking_uri=self.tracking_uri,
                timeout_seconds=self.flush_timeout_seconds,
            )
            client.register(request)

        self.enqueue("modelctl.register", task)

    def _run(self) -> None:
        """Run queued jobs until a shutdown sentinel is received."""

        while True:
            item = self._jobs.get()
            try:
                if item is None:
                    return
                name, fn = item
                try:
                    fn()
                except BaseException as exc:  # noqa: BLE001 - preserved for surfacing on flush.
                    with self._errors_lock:
                        self._errors.append(AsyncJobError(name=name, error=exc))
            finally:
                self._jobs.task_done()

    def _new_client(self) -> Any:
        """Create an MLflow client for worker-side calls."""

        if self.client_factory is not None:
            return self.client_factory(self.tracking_uri)

        return MlflowClient(tracking_uri=self.tracking_uri)

    def _require_run_id(self) -> str:
        """Return the active MLflow run id or fail."""

        if not self.run_id:
            raise ValueError("run_id is required for asynchronous MLflow logging")
        return self.run_id

    def _raise_if_needed(self) -> None:
        """Raise the first collected worker error when configured to fail fast."""

        with self._errors_lock:
            errors = list(self._errors)
        if self.fail_on_worker_error and errors:
            first = errors[0]
            raise RuntimeError(f"async tracking job failed: {first.name}") from first.error

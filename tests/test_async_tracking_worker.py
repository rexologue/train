from __future__ import annotations

import threading

import pytest

from tracking.async_worker import AsyncTrackingWorker


class FakeMlflowClient:
    def __init__(self):
        self.metrics = []
        self.artifacts = []

    def log_metric(self, run_id, key, value, step=None):
        self.metrics.append((run_id, key, value, step))

    def log_artifact(self, run_id, local_path, artifact_path=None):
        with open(local_path, encoding="utf-8") as f:
            content = f.read()
        self.artifacts.append((run_id, artifact_path, content))


def test_async_tracking_worker_logs_metrics_and_dict():
    client = FakeMlflowClient()
    worker = AsyncTrackingWorker(
        tracking_uri="http://mlflow.local",
        run_id="run-1",
        client_factory=lambda tracking_uri: client,
    )

    with worker:
        worker.log_metrics({"train/loss": 1.25}, step=3)
        worker.log_dict({"ok": True}, "reports/summary.json")
        worker.flush()

    assert client.metrics == [("run-1", "train/loss", 1.25, 3)]
    assert len(client.artifacts) == 1
    assert client.artifacts[0][0] == "run-1"
    assert client.artifacts[0][1] == "reports"
    assert '"ok": true' in client.artifacts[0][2]


def test_async_tracking_worker_preserves_job_order():
    seen: list[int] = []
    worker = AsyncTrackingWorker(fail_on_worker_error=True)

    with worker:
        worker.enqueue("first", lambda: seen.append(1))
        worker.enqueue("second", lambda: seen.append(2))
        worker.flush()

    assert seen == [1, 2]


def test_async_tracking_worker_surfaces_job_error_and_closes():
    worker = AsyncTrackingWorker(fail_on_worker_error=True)

    worker.start()
    worker.enqueue("broken", lambda: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(RuntimeError, match="async tracking job failed: broken"):
        worker.close()

    assert worker.errors[0].name == "broken"


def test_async_tracking_worker_flush_times_out_instead_of_hanging():
    release = threading.Event()
    worker = AsyncTrackingWorker(flush_timeout_seconds=0.05)
    worker.enqueue("blocked", release.wait)

    with pytest.raises(TimeoutError, match="flush timed out"):
        worker.flush()

    release.set()
    worker.close()

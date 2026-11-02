from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_STORE_PATH = Path("models/experiments/runs.json")


class ExperimentTracker:
    def __init__(self, experiment_name: str = "indonesia-dengue-forecast") -> None:
        self.experiment_name = experiment_name
        self.current_run_id: str | None = None
        self.current_run: dict[str, Any] | None = None
        self.mlflow_available = False

        # Attempt to import mlflow if available in environment
        try:
            import mlflow

            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            self.mlflow = None

    def start_run(self, run_name: str | None = None) -> str:
        self.current_run_id = str(uuid.uuid4())[:8]
        self.current_run = {
            "run_id": self.current_run_id,
            "run_name": run_name or f"run-{self.current_run_id}",
            "experiment_name": self.experiment_name,
            "start_time": datetime.now(UTC).isoformat(),
            "status": "RUNNING",
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

        if self.mlflow_available and self.mlflow is not None:
            with contextlib.suppress(Exception):
                self.mlflow.set_experiment(self.experiment_name)
                self.mlflow.start_run(run_name=self.current_run["run_name"])

        return self.current_run_id

    def log_param(self, key: str, value: Any) -> None:
        if self.current_run:
            self.current_run["params"][key] = value
        if self.mlflow_available and self.mlflow is not None:
            with contextlib.suppress(Exception):
                self.mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float) -> None:
        if self.current_run:
            self.current_run["metrics"][key] = round(float(value), 4)
        if self.mlflow_available and self.mlflow is not None:
            with contextlib.suppress(Exception):
                self.mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, artifact_path: Path | str) -> None:
        path_str = str(artifact_path)
        if self.current_run:
            self.current_run["artifacts"].append(path_str)
        if self.mlflow_available and self.mlflow is not None:
            with contextlib.suppress(Exception):
                self.mlflow.log_artifact(path_str)

    def end_run(self, status: str = "COMPLETED") -> dict[str, Any]:
        if not self.current_run:
            raise RuntimeError("No active run to end.")

        self.current_run["end_time"] = datetime.now(UTC).isoformat()
        self.current_run["status"] = status

        # Save to runs.json
        EXPERIMENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        if EXPERIMENT_STORE_PATH.exists():
            with contextlib.suppress(Exception):
                runs = json.loads(EXPERIMENT_STORE_PATH.read_text(encoding="utf-8"))

        runs.append(self.current_run)
        EXPERIMENT_STORE_PATH.write_text(
            json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        completed_run = self.current_run
        self.current_run = None
        self.current_run_id = None

        if self.mlflow_available and self.mlflow is not None:
            with contextlib.suppress(Exception):
                self.mlflow.end_run()

        return completed_run


# Global default tracker
tracker = ExperimentTracker()

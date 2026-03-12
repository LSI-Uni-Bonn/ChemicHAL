"""chemagent.ml.training_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for background model training and job management.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
train_model             — non-blocking: submit a training job, return job_id
check_training          — poll a background job until completed/failed

Internal helpers
----------------
_jobs                   — shared dict of background job state
_default_model_path     — resolve default output path for a trained model
_run_job_in_background  — thread launcher that writes results into _jobs
build_model_from_split_file — blocking pipeline from a split .pkl
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.servers.server_helpers import _run_pipeline, _workspace_root
from chemagent.session_utils import get_session_logger as _get_session_logger


# ---------------------------------------------------------------------------
# Shared job state  (ephemeral — lost on server restart)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model_path(algorithm: str, stem: str = "") -> str:
    session_logger = _get_session_logger()
    out_dir = session_logger.session_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{algorithm}.pkl" if stem else f"trained_model_{algorithm}.pkl"
    return str(out_dir / name)


def _run_job_in_background(job_id: str, fn, *args, **kwargs) -> None:
    """Run *fn* in a daemon thread; write result/error into _jobs[job_id]."""
    session_logger = _get_session_logger()

    def _worker():
        t_start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            session_logger.log_event(
                "background_job_completed",
                job_id=job_id,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
            session_logger.copy_artifacts_from_result(result)
        except Exception as exc:  # noqa: BLE001
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"]  = str(exc)
            session_logger.log_event(
                "background_job_failed",
                job_id=job_id,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
        finally:
            _jobs[job_id]["finished_at"] = time.time()

    _jobs[job_id] = {
        "status":      "running",
        "result":      None,
        "error":       None,
        "started_at":  time.time(),
        "finished_at": None,
    }
    threading.Thread(target=_worker, daemon=True).start()


def build_model_from_split_file(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Blocking pipeline: tune+train+eval from a split .pkl (internal helper used by run_pipeline)."""
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    split   = joblib.load(split_path)
    X_train = np.array(split["train_features"])
    y_train = np.array(split["train_labels"])
    X_test  = np.array(split["test_features"])
    y_test  = np.array(split["test_labels"])
    has_val = "val_features" in split and len(split["val_features"]) > 0
    X_val   = np.array(split["val_features"]) if has_val else None
    y_val   = np.array(split["val_labels"])   if has_val else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm, stem=split_path.stem)

    return _run_pipeline(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )


# ===========================================================================
# MCP tool functions
# ===========================================================================

def train_model(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Train and tune a model from a split .pkl file. Non-blocking — returns a job_id immediately.

    Workflow: split_dataset → THIS TOOL → check_training(job_id, model_save_path=model_save_path)

    Poll check_training(job_id, model_save_path=model_save_path) every 30 s until
    status='completed' or 'failed'. Always pass model_save_path — it enables on-disk
    fallback detection if the MCP server restarts and in-memory job state is lost.
    The full evaluation result is in check_training(...)[\"result\"] when done.

    Args:
        split_file_path: Path to .pkl produced by split_dataset(). Must contain
                         train_features, train_labels, test_features, test_labels.
        algorithm: "RFC", "RFR", or "SVC" (default "RFC"). See get_ml_info().
        task: "classification" (default), "classification-cw", or "regression".
        cv_fold: GridSearchCV folds (default 5).
        opt_metric: Scoring metric for GridSearchCV (default "balanced_accuracy").
        random_seed: Random seed (default 42).
        model_save_path: Output .pkl path. Defaults to session models/ dir.

    Returns:
        job_id, status="running", model_save_path, message with polling instruction.
    """
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    split   = joblib.load(split_path)
    X_train = np.array(split["train_features"])
    y_train = np.array(split["train_labels"])
    X_test  = np.array(split["test_features"])
    y_test  = np.array(split["test_labels"])
    has_val = "val_features" in split and len(split["val_features"]) > 0
    X_val   = np.array(split["val_features"]) if has_val else None
    y_val   = np.array(split["val_labels"])   if has_val else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm, stem=split_path.stem)

    job_id = str(uuid.uuid4())
    _run_job_in_background(
        job_id, _run_pipeline,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )
    return {
        "job_id":          job_id,
        "status":          "running",
        "model_save_path": model_save_path,
        "message": (
            f"Training started in the background. "
            f"Call check_training('{job_id}', model_save_path='{model_save_path}') to poll for completion. "
            "Poll every 30 seconds until status is 'completed' or 'failed'."
        ),
    }


def check_training(job_id: str, model_save_path: Optional[str] = None) -> dict[str, Any]:
    """Poll a background training job started by train_model().

    Workflow: train_model → THIS TOOL (poll until done) → plot_classification_results

    Call repeatedly every 30 seconds until status is 'completed' or 'failed'.
    The full pipeline result (best_params, train/test metrics, model_path) is in
    the 'result' key when completed.

    Always pass the model_save_path returned by train_model() so that if the MCP
    server restarts (clearing in-memory job state), this tool can detect the saved
    model on disk and return status='completed' instead of raising an error.

    Args:
        job_id: The job_id returned by train_model().
        model_save_path: Path to the expected .pkl model file (returned by train_model()).
                         Used as a fallback when the job is no longer in memory.

    Returns:
        job_id, status ("running" | "completed" | "failed"), elapsed_seconds,
        and either result (on success) or error (on failure).
    """
    if job_id not in _jobs:
        if model_save_path and Path(model_save_path).exists():
            return {
                "job_id":  job_id,
                "status":  "completed",
                "message": (
                    "Job state was lost (MCP server restarted), but the model file "
                    f"already exists on disk at '{model_save_path}'. "
                    "Training had completed successfully before the restart. "
                    "You can proceed with plot_classification_results or export_predictions."
                ),
                "model_path": model_save_path,
            }
        raise ValueError(
            f"Job '{job_id}' not found. "
            "Job IDs are ephemeral — lost if the MCP server restarts. "
            "If you have the model_save_path, pass it to check_training() to detect the "
            "saved model on disk. Otherwise re-run train_model() to create a new job."
        )
    job = _jobs[job_id]
    elapsed = round(time.time() - job["started_at"], 1)
    response: dict[str, Any] = {
        "job_id":          job_id,
        "status":          job["status"],
        "elapsed_seconds": elapsed,
    }
    if job["status"] == "running":
        response["message"] = (
            f"Still training ({elapsed}s elapsed). "
            f"Call check_training('{job_id}', model_save_path='{model_save_path}') again in 30 seconds."
        )
    elif job["status"] == "completed":
        response["result"]  = job["result"]
        response["message"] = "Training completed successfully. See 'result' for full evaluation."
    elif job["status"] == "failed":
        response["error"]   = job["error"]
        response["message"] = "Training failed. See 'error' for details."
    return response

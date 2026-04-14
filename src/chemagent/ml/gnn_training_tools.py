"""chemagent.ml.gnn_training_tools — MCP tools for GNN training workflows.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Important workflow note for LLM tool users
-----------------------------------------
For GNN pipelines, use ``prepare_gnn_dataset`` (and then
``train_gnn_model_mcp``). Do **not** call ``compute_features`` for GNN
training: molecular fingerprint featurization (ECFP/MACCS/etc.) is for
standard tabular ML models, while GNNs build graph representations directly
from SMILES.

Functions
---------
prepare_gnn_dataset     — prepare train/val/test datasets from split .pkl and SMILES
train_gnn_model_mcp     — train a GNN model on prepared dataset (non-blocking job)
check_gnn_training      — poll a background GNN training job
load_gnn_model_mcp      — load a trained GNN model from disk and validate

Internal helpers
----------------
_gnn_jobs               — shared dict of background GNN job state
_run_gnn_job_in_background — thread launcher for GNN training
_GNN_MODEL_MAP          — mapping of model names to classes
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import joblib

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.ml.gnn_models import GCN, GAT, GC_GNN, GIN, GINE, GraphSAGE
from chemagent.ml.gnn_training import load_and_prepare_gnn_dataset, train_gnn_model, load_gnn_model as _load_gnn_model_impl
from chemagent.session_utils import get_session_logger as _get_session_logger, resolve_path as _resolve_path


# Shared job state (lost on server restart)
_gnn_jobs: dict[str, dict[str, Any]] = {}

# Map model names to classes
_GNN_MODEL_MAP = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "GC_GNN": GC_GNN,
    "GINE": GINE,
    "GIN": GIN,
}


def _run_gnn_job_in_background(job_id: str, fn, *args, **kwargs) -> None:
    """Run *fn* in a daemon thread; write result/error into _gnn_jobs[job_id]."""
    session_logger = _get_session_logger()

    def _worker():
        t_start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            _gnn_jobs[job_id]["status"] = "completed"
            _gnn_jobs[job_id]["result"] = result
            session_logger.log_event(
                "gnn_training_completed",
                job_id=job_id,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
        except Exception as exc:  # noqa: BLE001
            _gnn_jobs[job_id]["status"] = "failed"
            _gnn_jobs[job_id]["error"] = str(exc)
            session_logger.log_event(
                "gnn_training_failed",
                job_id=job_id,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
        finally:
            _gnn_jobs[job_id]["finished_at"] = time.time()

    _gnn_jobs[job_id] = {
        "status": "running",
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    threading.Thread(target=_worker, daemon=True).start()


def prepare_gnn_dataset(
    split_file_path: str,
    smiles_csv_path: str,
    smiles_column: str = "smiles",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare train/val/test GNN datasets from a split file and SMILES CSV.

    This is the dataset-preparation step for GNN workflows. It converts SMILES
    to graph objects and caches them for GNN training.

    Use this instead of ``compute_features`` when training GNN models.
    Fingerprint generation (ECFP/MACCS/...) is only needed for standard ML
    models (RFC/XGBoost/SVM/etc.), not for graph neural networks.

    Reads SMILES from CSV, uses indices from split .pkl to create graph datasets.

    Args:
    split_file_path :
        Path to .pkl split file with train/test indices + labels.
    smiles_csv_path :
        Path to CSV file with SMILES strings.
    smiles_column :
        Column name in CSV containing SMILES (default "smiles").
    test_size :
        Validation split fraction (default 0.2).
    seed :
        Random seed (default 42).

    Returns:
    Dict with:
        - "status": "completed" or "failed"
        - "train_dataset_path": path to train dataset cache
        - "val_dataset_path": path to val dataset cache
        - "test_dataset_path": path to test dataset cache
        - "num_train": number of training graphs
        - "num_val": number of validation graphs
        - "num_test": number of test graphs
    """
    import csv
    import pickle

    session_logger = _get_session_logger()

    try:
        split_file_path = _resolve_path(split_file_path)
        smiles_csv_path = _resolve_path(smiles_csv_path)

        # Read SMILES from CSV
        smiles_list = []
        with open(smiles_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles_list.append(row[smiles_column])

        # Load datasets
        train_dataset, val_dataset, test_dataset = load_and_prepare_gnn_dataset(
            split_file_path,
            smiles_list,
            test_size=test_size,
            seed=seed,
        )

        session_logger.log_event(
            "gnn_dataset_prepared",
            split_file_path=split_file_path,
            smiles_csv_path=smiles_csv_path,
            num_train=len(train_dataset),
            num_val=len(val_dataset),
            num_test=len(test_dataset),
        )

        return {
            "status": "completed",
            "train_dataset_path": str(train_dataset.processed_paths[0]),
            "val_dataset_path": str(val_dataset.processed_paths[0]),
            "test_dataset_path": str(test_dataset.processed_paths[0]),
            "num_train": len(train_dataset),
            "num_val": len(val_dataset),
            "num_test": len(test_dataset),
        }
    except Exception as exc:
        session_logger.log_event(
            "gnn_dataset_prep_failed",
            error=str(exc),
        )
        return {
            "status": "failed",
            "error": str(exc),
        }


def train_gnn_model_mcp(
    split_file_path: str,
    smiles_csv_path: str,
    model_class_name: Literal["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"] = "GCN",
    hidden_channels: int = 64,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: Optional[str] = None,
    smiles_column: str = "smiles",
) -> dict[str, Any]:
    """Train a GNN model on SMILES selectivity data (non-blocking background job).

    Expected GNN workflow:
    1) ``load_dataset``
    2) ``split_dataset``
    3) ``prepare_gnn_dataset`` (GNN-specific data prep)
    4) ``train_gnn_model_mcp``

    Do not run ``compute_features`` for this workflow. GNN models consume graph
    data derived from SMILES directly rather than molecular fingerprint vectors.

    Submits training to background thread; use `check_gnn_training()` to poll results.

    Args:
    split_file_path :
        Path to .pkl split file with train/test indices + labels.
    smiles_csv_path :
        Path to CSV file with SMILES strings.
    model_class_name :
        GNN architecture: GCN, GraphSAGE, GAT, GC_GNN, GINE, GIN (default GCN).
    hidden_channels :
        Hidden dimension (default 64).
    epochs :
        Training epochs (default 100).
    lr :
        Learning rate (default 0.001).
    batch_size :
        Batch size (default 32).
    device :
        torch device string (default: auto cuda/cpu).
    smiles_column :
        Column name in CSV for SMILES (default "smiles").

    Returns:
    Dict with:
        - "job_id": unique job identifier
        - "status": "submitted"
        - "model_save_path": where best model will be saved
    """
    import csv

    session_logger = _get_session_logger()

    # Read SMILES from CSV
    smiles_list = []
    try:
        split_file_path = _resolve_path(split_file_path)
        smiles_csv_path = _resolve_path(smiles_csv_path)

        with open(smiles_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles_list.append(row[smiles_column])
    except Exception as exc:
        session_logger.log_event("gnn_training_failed", error=f"SMILES read error: {exc}")
        return {"status": "failed", "error": str(exc)}

    # Resolve model class
    if model_class_name not in _GNN_MODEL_MAP:
        err_msg = f"Unknown model: {model_class_name}. Available: {list(_GNN_MODEL_MAP.keys())}"
        session_logger.log_event("gnn_training_failed", error=err_msg)
        return {"status": "failed", "error": err_msg}

    model_class = _GNN_MODEL_MAP[model_class_name]

    # Default save path
    out_dir = session_logger.session_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = str(out_dir / f"gnn_{model_class_name}.pt")

    # Create job and submit
    job_id = str(uuid.uuid4())
    _run_gnn_job_in_background(
        job_id,
        train_gnn_model,
        split_file_path=split_file_path,
        smiles_list=smiles_list,
        model_class=model_class,
        model_save_path=model_save_path,
        hidden_channels=hidden_channels,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    session_logger.log_event(
        "gnn_training_submitted",
        job_id=job_id,
        model_class=model_class_name,
        hidden_channels=hidden_channels,
        epochs=epochs,
    )

    return {
        "job_id": job_id,
        "status": "submitted",
        "model_save_path": model_save_path,
    }


def check_gnn_training(
    job_id: str,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Poll a background GNN training job.

    Args:
    job_id :
        Job ID from `train_gnn_model_mcp()`.
    model_save_path :
        Optional path to model (fallback to disk if job state lost).

    Returns:
    Dict with:
        - "status": "running", "completed", or "failed"
        - "best_val_acc": best validation accuracy (if completed)
        - "test_acc": test accuracy (if completed)
        - "train_evaluation": train split metrics (if completed)
        - "val_evaluation": validation split metrics (if completed)
        - "test_evaluation": test split metrics (if completed)
        - "model_path": path to saved model (if completed)
        - "error": error message (if failed)
    """
    if job_id not in _gnn_jobs:
        return {
            "status": "unknown",
            "error": f"Job {job_id} not found. (Lost after server restart?)",
            "model_save_path": model_save_path,
        }

    job = _gnn_jobs[job_id]

    if job["status"] == "completed":
        result = job["result"]
        return {
            "status": "completed",
            "best_val_acc": result.get("best_val_acc"),
            "test_acc": result.get("test_acc"),
            "train_evaluation": result.get("train_evaluation"),
            "val_evaluation": result.get("val_evaluation"),
            "test_evaluation": result.get("test_evaluation"),
            "n_train": result.get("n_train"),
            "n_val": result.get("n_val"),
            "n_test": result.get("n_test"),
            "model_path": result.get("model_path"),
        }
    elif job["status"] == "failed":
        return {
            "status": "failed",
            "error": job.get("error"),
        }
    else:  # "running"
        return {
            "status": "running",
            "started_at": job.get("started_at"),
        }


def load_gnn_model_mcp(
    model_class_name: Literal["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"],
    node_features_dim: int,
    hidden_channels: int,
    num_classes: int,
    model_path: str,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Load a trained GNN model from disk and verify it loads correctly.

    Loads a GNN model from a saved state dict and performs a validation step
    to ensure the weights were loaded correctly.

    Args:
    model_class_name :
        GNN architecture name: GCN, GraphSAGE, GAT, GC_GNN, GINE, or GIN.
    node_features_dim :
        Input node feature dimension (typically 4 for atomic features).
    hidden_channels :
        Hidden dimension (must match the training configuration).
    num_classes :
        Number of output classes (must match the training data).
    model_path :
        Path to saved model state dict (.pt file).
    device :
        torch device string ('cuda' or 'cpu'); auto-detects if None.

    Returns:
    Dict with:
        - "status": "completed" or "failed"
        - "model_path": path to the loaded model (if successful)
        - "model_class": architecture name (if successful)
        - "device": device used (if successful)
        - "error": error message (if failed)
    """
    import torch

    session_logger = _get_session_logger()

    try:
        # Resolve model class
        if model_class_name not in _GNN_MODEL_MAP:
            err_msg = (
                f"Unknown model class: {model_class_name}. "
                f"Available: {', '.join(_GNN_MODEL_MAP.keys())}"
            )
            return {"status": "failed", "error": err_msg}

        model_class = _GNN_MODEL_MAP[model_class_name]
        model_path = _resolve_path(model_path)

        # Load model
        model = _load_gnn_model_impl(
            model_class=model_class,
            node_features_dim=node_features_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            model_path=model_path,
            device=device,
        )

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        session_logger.log_event(
            "gnn_model_loaded",
            model_class=model_class_name,
            model_path=model_path,
            device=device,
        )

        return {
            "status": "completed",
            "model_path": str(model_path),
            "model_class": model_class_name,
            "device": device,
        }
    except Exception as exc:
        session_logger.log_event(
            "gnn_model_load_failed",
            error=f"{type(exc).__name__}: {str(exc)}",
        )
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {str(exc)}",
        }


def load_gnn_model(
    model_class_name: Literal["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"],
    node_features_dim: int,
    hidden_channels: int,
    num_classes: int,
    model_path: str,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Alias for `load_gnn_model_mcp` to make LLM tool-calling robust.

    Some MCP clients/LLMs may try the internal-style name `load_gnn_model`.
    This public wrapper intentionally maps that call to the MCP-safe loader.
    """
    return load_gnn_model_mcp(
        model_class_name=model_class_name,
        node_features_dim=node_features_dim,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        model_path=model_path,
        device=device,
    )


__all__ = [
    "prepare_gnn_dataset",
    "train_gnn_model_mcp",
    "check_gnn_training",
    "load_gnn_model",
    "load_gnn_model_mcp",
]

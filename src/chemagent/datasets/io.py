"""
Split file I/O and processed-dataset retrieval helpers.

Handles loading previously saved ``.pkl`` split files and serialising
processed feature/label arrays to be returned to the LLM context.

Usage
-----
    from chemagent.datasets.io import load_split_file, get_ml_ready_data

    split   = load_split_file("data/splits/my_split.pkl")
    payload = get_ml_ready_data(processed, dataset_id="my_ds")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from .loader import resolve_path


# Split file I/O
def load_split_file(file_path: str) -> Dict[str, Any]:
    """Load a split ``.pkl`` file produced by :func:`~chemagent.datasets.splitter.save_split`.

    Parameters
    ----------
    file_path:
        Absolute path or workspace-relative path to the ``.pkl`` file.

    Returns
    -------
    dict
        Keys ``"train"``, ``"val"``, ``"test"``, each containing:
        ``features`` (list), ``labels`` (list), ``n_samples`` (int),
        and optionally ``smiles`` and ``cid``.
        Plus ``"file_path"`` with the resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    data = joblib.load(path)

    def _build(prefix: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "features":  data[f"{prefix}_features"].tolist(),
            "labels":    data[f"{prefix}_labels"].tolist(),
            "n_samples": int(len(data[f"{prefix}_labels"])),
        }
        if f"{prefix}_smiles" in data:
            entry["smiles"] = data[f"{prefix}_smiles"].tolist()
        if f"{prefix}_cid" in data:
            entry["cid"] = data[f"{prefix}_cid"].tolist()
        return entry

    return {
        "train":     _build("train"),
        "val":       _build("val"),
        "test":      _build("test"),
        "file_path": str(path),
    }


# ML-ready payload builder
def get_ml_ready_data(
    processed: Dict[str, Any],
    dataset_id: str,
    as_lists: bool = True,
) -> Dict[str, Any]:
    """Build the JSON-serialisable response for :func:`get_ml_ready_data` MCP tool.

    Parameters
    ----------
    processed:
        Processed-dataset dict from ``_processed_datasets[dataset_id]``.
    dataset_id:
        Identifier string (included in the response for traceability).
    as_lists:
        If ``True`` (default), return feature and label arrays as plain lists.
        If ``False``, return shape / metadata only — avoids large context transfer.

    Returns
    -------
    dict
        ``dataset_id``, ``shape``, ``label_column``, and conditionally
        ``features``, ``labels``, ``smiles``, ``cid``.
    """
    features = processed["features"]
    labels   = processed["labels"]

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "shape": {
            "n_samples":  int(features.shape[0]),
            "n_features": int(features.shape[1]),
        },
        "label_column": processed["label_column"],
    }

    if as_lists:
        result["features"] = features.tolist()
        result["labels"]   = labels.tolist()
        if "smiles" in processed:
            result["smiles"] = processed["smiles"].tolist()
        if "cid" in processed:
            result["cid"] = processed["cid"].tolist()

    return result


# Dataset info builder
def get_dataset_info(
    dataset_id: str,
    loaded_datasets: Dict[str, Any],
    processed_datasets: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the info response dict for a dataset identified by *dataset_id*.

    Parameters
    ----------
    dataset_id:
        Dataset key to inspect.
    loaded_datasets:
        The server's ``_loaded_datasets`` registry.
    processed_datasets:
        The server's ``_processed_datasets`` registry.

    Returns
    -------
    dict
        ``loaded``, ``prepared``, ``raw_data`` (if loaded), ``ml_ready``
        (if prepared).
    """
    import pandas as pd

    info: Dict[str, Any] = {"dataset_id": dataset_id}

    if dataset_id in loaded_datasets:
        df = loaded_datasets[dataset_id]
        info["loaded"] = True
        info["raw_data"] = {
            "n_samples":  len(df),
            "columns":    df.columns.tolist(),
            "label_col":  df.attrs.get("label_col"),
            "smiles_col": df.attrs.get("smiles_col"),
            "id_col":     df.attrs.get("id_col"),
        }
    else:
        info["loaded"] = False

    if dataset_id in processed_datasets:
        proc = processed_datasets[dataset_id]
        info["prepared"] = True
        info["ml_ready"] = {
            "n_samples":    int(proc["features"].shape[0]),
            "n_features":   int(proc["features"].shape[1]),
            "label_column": proc["label_column"],
            "has_smiles":   "smiles" in proc,
            "has_cid":      "cid"    in proc,
        }
    else:
        info["prepared"] = False

    return info

"""
CSV dataset loading, path resolution, and basic statistics.

All functions are pure / stateless — they return data structures that the
caller (e.g. the MCP server) is responsible for caching.

Usage
-----
    from chemagent.datasets.loader import load_csv, label_stats, resolve_path

    df, meta = load_csv("data/datasets/my.csv", label_col="class_label",
                         smiles_col="smiles", id_col="cid")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Path helpers
def workspace_root() -> Path:
    """Return the workspace root (4 levels up from this file)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(file_path: str, directory: str = "") -> Path:
    """Resolve *file_path* to an absolute ``Path``.

    Args:
    file_path:
        Absolute path, or filename / relative path within *directory*.
    directory:
        Optional directory prefix, resolved relative to the workspace root.

    Returns:
    Path
        Absolute resolved path.
    """
    if directory:
        p = Path(directory) / file_path
    else:
        p = Path(file_path)

    if not p.is_absolute():
        # On Windows, paths like "/data/..." are drive-relative, not absolute.
        # Stripping leading separators ensures workspace_root() / p works correctly.
        p = workspace_root() / Path(str(p).lstrip("/\\"))
    return p


def list_csv_files(directory: str = "data/datasets") -> Dict[str, Any]:
    """List ``.csv`` files in *directory*.

    Args:
    directory:
        Workspace-relative or absolute path to search.

    Returns:
    dict
        ``{"datasets": [...], "count": int, "directory": str}``
    """
    path = resolve_path(directory)
    if not path.exists():
        return {
            "datasets": [],
            "count": 0,
            "directory": str(path),
            "error": f"Directory not found: {path}",
        }
    csv_files = sorted(path.glob("*.csv"))
    return {
        "datasets": [f.name for f in csv_files],
        "count": len(csv_files),
        "directory": str(path),
    }


# Label statistics
def label_stats(labels: np.ndarray) -> Dict[str, Any]:
    """Compute basic statistics for a label array.

    Args:
    labels:
        1-D array of numeric labels.

    Returns:
    dict
        mean, std, min, max, unique_values.
    """
    arr = np.asarray(labels, dtype=float)
    return {
        "mean":          float(arr.mean()),
        "std":           float(arr.std()),
        "min":           float(arr.min()),
        "max":           float(arr.max()),
        "unique_values": int(len(np.unique(arr))),
    }


# CSV loading
def load_csv(
    file_path: str,
    label_col: str = "class_label",
    smiles_col: Optional[str] = "smiles",
    id_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    dataset_id: Optional[str] = None,
    directory: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a CSV file and validate requested columns.

    Args:
    file_path:
        Absolute path, filename within *directory*, or workspace-relative path.
    label_col:
        Target / label column name (default ``"class_label"``).
    smiles_col:
        Column containing SMILES strings; ``None`` if not present.
    id_col:
        Column containing sample identifiers; ``None`` if absent.
    feature_cols:
        Explicit list of numeric feature column names.  When supplied (and
        *smiles_col* is ``None``), a processed feature matrix is returned
        immediately via ``meta["features"]``.
    dataset_id:
        Cache key; defaults to the file stem.
    directory:
        Directory prefix applied when *file_path* is not absolute.

    Returns:
    tuple[pd.DataFrame, dict]
        * ``df`` — parsed DataFrame with column-config stored in ``.attrs``.
        * ``meta`` — summary dict ready to return directly as an MCP response.

    Raises:
    FileNotFoundError
        If the resolved path does not exist.
    ValueError
        If any requested column is missing from the CSV.
    """
    full_path = resolve_path(file_path, directory)
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found: {full_path}")

    df = pd.read_csv(full_path)

    if dataset_id is None:
        dataset_id = full_path.stem

    # --- validate required column ---
    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found. "
            f"Available: {df.columns.tolist()}"
        )
    if smiles_col and smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column {smiles_col!r} not found. "
            f"Available: {df.columns.tolist()}"
        )
    if id_col and id_col not in df.columns:
        raise ValueError(
            f"ID column {id_col!r} not found. "
            f"Available: {df.columns.tolist()}"
        )

    # --- store config in df.attrs so downstream tools can retrieve it ---
    df.attrs["label_col"]  = label_col
    df.attrs["smiles_col"] = smiles_col
    df.attrs["id_col"]     = id_col

    has_smiles       = bool(smiles_col and smiles_col in df.columns)
    has_precomputed  = False
    n_features       = 0
    features_arr: Optional[np.ndarray] = None

    # Pre-compute features if explicit columns are given (no SMILES flow)
    if feature_cols and not smiles_col:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        features_arr    = df[feature_cols].values.astype(float)
        has_precomputed = True
        n_features      = features_arr.shape[1]
    elif not smiles_col and not feature_cols:
        # Auto-detect numeric columns
        exclude      = {label_col, id_col} if id_col else {label_col}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        if numeric_cols:
            features_arr    = df[numeric_cols].values.astype(float)
            has_precomputed = True
            n_features      = features_arr.shape[1]

    # Build next-step hint
    if has_smiles and not has_precomputed:
        next_step = (
            f"Call featurize_dataset(dataset_id={dataset_id!r}, method='ECFP', "
            "radius=2, n_bits=2048) to compute fingerprints server-side, then "
            "split_prepared_dataset()."
        )
    elif has_precomputed:
        next_step = (
            f"Dataset is ML-ready. Call split_prepared_dataset({dataset_id!r}) "
            "to create train/test splits."
        )
    else:
        next_step = (
            "No SMILES or numeric features detected. Pass feature_cols explicitly "
            "or ensure the CSV has a SMILES column."
        )

    meta: Dict[str, Any] = {
        "dataset_id":              dataset_id,
        "n_samples":               len(df),
        "columns":                 df.columns.tolist(),
        "label_col":               label_col,
        # label statistics removed — kept out of loader meta to reduce log noise
        "has_smiles":              has_smiles,
        "has_precomputed_features": has_precomputed,
        "loaded":                  True,
        "next_step":               next_step,
    }
    if has_smiles:
        meta["smiles_sample"] = df[smiles_col].head(3).tolist()
    if has_precomputed:
        meta["n_features"]   = n_features
    if features_arr is not None:
        meta["_features_arr"] = features_arr   # consumed by the MCP server

    return df, meta

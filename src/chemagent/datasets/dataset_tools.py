"""chemagent.datasets.dataset_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for dataset handling — loaded, featurized, and split.

These are plain functions (no MCP decorator).  They are imported by
``chemagent_mcp.py`` and registered there via ``_register()``.

In-memory state
---------------
``_loaded_datasets``    — raw DataFrames, keyed by dataset_id
``_processed_datasets`` — featurized + label-ready dicts, keyed by dataset_id

Both dicts are defined here so that ``chemagent_mcp.py`` can import them and
share them with ``run_pipeline`` without circular state duplication.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal, Optional

from chemagent.session_utils import get_session_logger as _get_session_logger, resolve_path as _resolve_path

from .featurizer import featurize_df, build_processed_entry, list_featurizers as _list_featurizers_impl
from .io import get_dataset_info as _get_dataset_info_impl, get_ml_ready_data as _get_ml_ready_data_impl
from .loader import list_csv_files, load_csv
from .splitter import save_split as _save_split, split_processed


# Shared in-memory state  (ephemeral — lost on server restart)
_loaded_datasets:    dict[str, Any] = {}   # dataset_id → pd.DataFrame
_processed_datasets: dict[str, dict[str, Any]] = {}


# Dataset tool functions (registered in chemagent_mcp.py via _register)
def find_datasets(directory: str = "data/datasets") -> dict[str, Any]:
    """List CSV files available for ML in a directory.

    Workflow: THIS TOOL → load_dataset(file_path)

    Args:
        directory: Workspace-relative or absolute path to search (default: "data/datasets").

    Returns:
        datasets (list of filenames), count, directory (resolved path).
    """
    return list_csv_files(directory)


def list_loaded_datasets() -> dict[str, Any]:
    """List datasets currently in server memory (state is ephemeral — lost on restart).

    If lists are empty, re-run load_dataset() and compute_features() before continuing.

    Returns:
        loaded (raw), prepared (featurized + ready to split), totals, note.
    """
    return {
        "loaded":         list(_loaded_datasets.keys()),
        "prepared":       list(_processed_datasets.keys()),
        "total_loaded":   len(_loaded_datasets),
        "total_prepared": len(_processed_datasets),
        "note":           "State is ephemeral — lost on server restart.",
    }


def list_featurizers() -> dict[str, Any]:
    """List all available molecular featurization methods.

    Returns name, parameters, and description for each method.
    Use the name directly as the `method` argument to compute_features().
    """
    return _list_featurizers_impl()


def load_dataset(
    file_path: str,
    label_col: str = "class_label",
    smiles_col: Optional[str] = "smiles",
    id_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    dataset_id: Optional[str] = None,
    directory: str = "",
) -> dict[str, Any]:
    """Load a CSV dataset into server memory for ML.

    Workflow: find_datasets → THIS TOOL → compute_features

    Supports: (1) molecular datasets with SMILES, (2) tabular datasets with
    explicit feature_cols, (3) tabular datasets with auto-detected numeric features.

    Args:
        file_path: Absolute, workspace-relative, or filename within `directory`.
        label_col: Target column (default "class_label").
        smiles_col: SMILES column, or None for non-molecular datasets.
        id_col: Compound/sample ID column (optional).
        feature_cols: Explicit feature column list (ignored when smiles_col is set).
        dataset_id: In-memory cache key (default: CSV filename stem).
        directory: Directory prefix when file_path is a bare filename.

    Returns:
        dataset_id, n_samples, label_col, label_stats, has_smiles, next_step.
    """
    df, meta = load_csv(
        file_path=file_path,
        label_col=label_col,
        smiles_col=smiles_col,
        id_col=id_col,
        feature_cols=feature_cols,
        dataset_id=dataset_id,
        directory=directory,
    )
    ds_id = meta["dataset_id"]
    _loaded_datasets[ds_id] = df
    # Save a CSV copy asynchronously so it doesn't block the response
    threading.Thread(
        target=_get_session_logger().save_dataframe,
        args=(df, ds_id),
        daemon=True,
    ).start()
    if "_features_arr" in meta:
        features_arr = meta.pop("_features_arr")
        _processed_datasets[ds_id] = build_processed_entry(
            df=df, features=features_arr,
            label_col=label_col, smiles_col=smiles_col, id_col=id_col,
        )
    return {k: v for k, v in meta.items() if not k.startswith("_")}


def compute_features(
    dataset_id: str,
    method: str = "ECFP",
    n_bits: int = 2048,
    radius: int = 2,
    label_col: Optional[str] = None,
) -> dict[str, Any]:
    """Compute molecular fingerprints server-side for a loaded dataset.

    Features stay on disk — nothing large is returned to the LLM context.
    Workflow: load_dataset → THIS TOOL → split_dataset

    Args:
        dataset_id: ID from load_dataset().
        method: Fingerprint method (default: "ECFP"). Call list_featurizers() to see all.
        n_bits: Bit vector size (default: 2048).
        radius: Morgan radius (default: 2, i.e. ECFP4).
        label_col: Override label column from load_dataset().

    Returns:
        dataset_id, method, n_samples, n_features, label_stats, prepared=True, next_step.
    """
    if dataset_id not in _loaded_datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")
    df = _loaded_datasets[dataset_id]
    lc = label_col or df.attrs.get("label_col", "class_label")
    if lc not in df.columns:
        raise ValueError(f"Label column '{lc}' not found. Available: {df.columns.tolist()}")
    features = featurize_df(df, method=method, n_bits=n_bits, radius=radius)
    _processed_datasets[dataset_id] = build_processed_entry(df=df, features=features, label_col=lc)
    return {
        "dataset_id": dataset_id,
        "method":     method,
        "n_samples":  int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "prepared":   True,
        "next_step": (
            f"Call split_dataset('{dataset_id}', train_size=0.7, "
            "val_size=0.0, test_size=0.3, stratified=True) to create splits."
        ),
    }


def split_dataset(
    dataset_id: str,
    split_type: Literal["random", "scaffold"] = "random",
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.1,
    seed: Optional[int] = 42,
    stratified: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Split a featurized dataset into train/val/test partitions and save to .pkl.

    Workflow: compute_features → THIS TOOL → train_model(split_file_path)

    Args:
        dataset_id: ID of a featurized dataset (from compute_features).
        split_type: "random" (default) or "scaffold".
        train_size: Training fraction (default 0.8).
        val_size: Validation fraction (default 0.0). Use 0.0 for two-way split.
        test_size: Test fraction (default 0.1).
        seed: Random seed (default 42).
        stratified: Preserve class proportions across splits (supported for both "random" and "scaffold" split types).
        save_path: Output .pkl path. Defaults to session splits/ dir. Pass "" to skip.

    Returns:
        train/val/test metadata, statistics, saved_to path, next_step hint.
    """
    # Accept bool, "true"/"false", "True"/"False", or None (→ False)
    if isinstance(stratified, str):
        stratified_bool = stratified.strip().lower() in ("true", "1", "yes")
    else:
        stratified_bool = bool(stratified)

    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not featurized. "
            "Call compute_features() first."
        )
    processed    = _processed_datasets[dataset_id]
    split_result = split_processed(
        processed=processed, split_type=split_type,
        train_size=train_size, val_size=val_size, test_size=test_size,
        seed=seed, stratified=stratified_bool,
    )
    train_idx  = split_result["train_idx"]
    val_idx    = split_result["val_idx"]
    test_idx   = split_result["test_idx"]
    statistics = split_result["statistics"]

    saved_to = None
    if save_path != "":
        if save_path is None:
            out_dir = _get_session_logger().session_dir / "splits"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"{dataset_id}_{split_type}.pkl")
        else:
            save_path = _resolve_path(save_path)
        saved_to = _save_split(
            save_dict=split_result["save_dict"],
            dataset_id=dataset_id,
            split_type=split_type,
            save_path=save_path,
        )

    def _split_meta(idx):
        meta: dict[str, Any] = {
            "n_samples": len(idx),
            "indices":   idx.tolist() if hasattr(idx, "tolist") else list(idx),
        }
        if "smiles" in processed:
            meta["smiles_sample"] = processed["smiles"][idx[:3]].tolist()
        if "cid" in processed:
            meta["cid_sample"]    = processed["cid"][idx[:3]].tolist()
        return meta

    warnings_list = []
    if len(val_idx) == 0:
        warnings_list.append(
            "val split is empty (val_size=0.0). "
            "Use split='test' when calling predict_from_split_file."
        )

    result: dict[str, Any] = {
        "train":      _split_meta(train_idx),
        "val":        _split_meta(val_idx),
        "test":       _split_meta(test_idx),
        "split_type": split_type,
        "statistics": statistics,
        "seed":       seed,
        "saved_to":   saved_to,
        "next_step": (
            f"Call train_model(split_file_path='{saved_to}', "
            "algorithm='RFC', task='classification', opt_metric='balanced_accuracy') "
            "to train in the background (non-blocking). "
            "Then poll with check_training(job_id) until status='completed'. "
            "Features are on disk — not returned here."
        ) if saved_to else "No file saved. Pass save_path.",
    }
    if warnings_list:
        result["warnings"] = warnings_list
    return result


def get_ml_ready_data(dataset_id: str, as_lists: bool = True) -> dict[str, Any]:
    """Return the processed feature matrix and labels (internal helper)."""
    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not prepared. "
            "Call compute_features() first."
        )
    return _get_ml_ready_data_impl(
        processed=_processed_datasets[dataset_id],
        dataset_id=dataset_id,
        as_lists=as_lists,
    )


def dataset_status(dataset_id: str) -> dict[str, Any]:
    """Return load/prepare status and metadata for a dataset.

    Args:
        dataset_id: Dataset ID to inspect.

    Returns:
        loaded, prepared, raw_data (if loaded), ml_ready (if prepared).
    """
    return _get_dataset_info_impl(
        dataset_id=dataset_id,
        loaded_datasets=_loaded_datasets,
        processed_datasets=_processed_datasets,
    )

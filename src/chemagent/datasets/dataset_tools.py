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
    For ECFP, also generates and saves bit information (required for MolAnchor explainability).
    
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
    import joblib
    from chemagent.session_utils import get_session_logger as _get_session_logger
    
    if dataset_id not in _loaded_datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")
    df = _loaded_datasets[dataset_id]
    lc = label_col or df.attrs.get("label_col", "class_label")
    if lc not in df.columns:
        raise ValueError(f"Label column '{lc}' not found. Available: {df.columns.tolist()}")
    
    # Compute features, optionally with bit info for ECFP
    features_result = featurize_df(
        df, method=method, n_bits=n_bits, radius=radius, return_bit_info=(method == "ECFP")
    )
    
    bit_info = None
    if isinstance(features_result, tuple):
        features, bit_info = features_result
    else:
        features = features_result
    
    _processed_datasets[dataset_id] = build_processed_entry(
        df=df, features=features, label_col=lc, bit_info=bit_info
    )
    
    # Save bit info to session if available
    if bit_info is not None:
        try:
            logger = _get_session_logger()
            bit_info_dir = logger.session_dir / "bit_info"
            bit_info_dir.mkdir(exist_ok=True)
            bit_info_path = bit_info_dir / f"{dataset_id}_bit_info.pkl"
            joblib.dump(bit_info, bit_info_path)
        except Exception as e:
            # Non-critical: bit info couldn't be saved, but featurization succeeded
            pass
    
    return {
        "dataset_id": dataset_id,
        "method":     method,
        "n_samples":  int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "prepared":   True,
        "bit_info_saved": bit_info is not None,
        "next_step": (
            f"Call split_dataset('{dataset_id}', train_size=0.7, "
            "val_size=0.0, test_size=0.3, stratified=True) to create splits."
        ),
    }


def split_dataset(
    dataset_id: str,
    split_type: Literal["random", "scaffold"] = "random",
    mode: Literal["auto", "ml", "gnn"] = "auto",
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.1,
    seed: Optional[int] = 42,
    stratified: bool | str | None = None,
    save_path: Optional[str] = None,
    save_csv: bool = False,
    csv_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Split a dataset into train/val/test partitions and save to .pkl.

    Workflow:
    - Standard ML: compute_features → THIS TOOL → train_model(split_file_path)
    - GNN: load_dataset(smiles_col=...) → THIS TOOL → prepare_gnn_dataset/train_gnn_model_mcp

    Args:
        dataset_id: Dataset ID from load_dataset(). If compute_features was run,
            feature arrays are included in the saved split for standard ML. If
            not, splits still include labels (and smiles/cid when available),
            which is sufficient for GNN workflows.
        mode: Split payload mode:
            - "auto" (default): use ML payload when features are available,
              otherwise use GNN payload from loaded data.
            - "ml": require a featurized dataset (from compute_features).
            - "gnn": force label/smiles/cid payload from loaded data only.
        split_type: "random" (default) or "scaffold".
        train_size: Training fraction (default 0.8).
        val_size: Validation fraction (default 0.0). Use 0.0 for two-way split.
        test_size: Test fraction (default 0.1).
        seed: Random seed (default 42).
        stratified: Preserve class proportions across splits (supported for both "random" and "scaffold" split types).
        save_path: Output .pkl path. Defaults to session splits/ dir. Pass "" to skip.
        save_csv: If True, also export a CSV version of the split (default False).
        csv_save_path: Output .csv path when save_csv=True. Defaults to the same
            stem as the saved .pkl, or session splits/<dataset_id>_<split_type>.csv.

    Returns:
        split_file_path (alias: saved_to), train/val/test metadata,
        statistics, and next_step hint.
    """
    def _split_save_dict_to_dataframe(save_dict: dict[str, Any]):
        import numpy as np
        import pandas as pd

        parts = []
        for prefix in ("train", "val", "test"):
            labels_key = f"{prefix}_labels"
            if labels_key not in save_dict:
                continue

            labels = np.asarray(save_dict[labels_key])
            data: dict[str, Any] = {
                "split": [prefix] * int(len(labels)),
                "label": labels,
            }

            for optional_col in ("smiles", "cid", "core"):
                key = f"{prefix}_{optional_col}"
                if key in save_dict:
                    data[optional_col] = np.asarray(save_dict[key])

            features_key = f"{prefix}_features"
            if features_key in save_dict:
                features = np.asarray(save_dict[features_key])
                if features.ndim == 1:
                    data["feature_0"] = features
                elif features.ndim == 2:
                    for i in range(features.shape[1]):
                        data[f"feature_{i}"] = features[:, i]

            parts.append(pd.DataFrame(data))

        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    # Accept bool, "true"/"false", "True"/"False", or None (→ False)
    if isinstance(stratified, str):
        stratified_bool = stratified.strip().lower() in ("true", "1", "yes")
    else:
        stratified_bool = bool(stratified)

    if mode not in {"auto", "ml", "gnn"}:
        raise ValueError("mode must be one of: 'auto', 'ml', 'gnn'.")

    if mode == "ml":
        if dataset_id not in _processed_datasets:
            raise ValueError(
                f"Dataset '{dataset_id}' is not featurized. "
                "Run compute_features() before split_dataset(..., mode='ml')."
            )
        processed = _processed_datasets[dataset_id]

    elif mode == "gnn":
        if dataset_id not in _loaded_datasets:
            raise ValueError(
                f"Dataset '{dataset_id}' is not loaded. "
                "Run load_dataset(..., smiles_col='smiles') before split_dataset(..., mode='gnn')."
            )
        df = _loaded_datasets[dataset_id]
        label_col = df.attrs.get("label_col", "class_label")
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset '{dataset_id}'."
            )

        processed = {
            "labels": df[label_col].to_numpy(),
            "label_column": label_col,
        }

        smiles_col = df.attrs.get("smiles_col")
        if smiles_col and smiles_col in df.columns:
            processed["smiles"] = df[smiles_col].to_numpy()

        id_col = df.attrs.get("id_col")
        if id_col and id_col in df.columns:
            processed["cid"] = df[id_col].to_numpy()

    elif dataset_id in _processed_datasets:
        processed = _processed_datasets[dataset_id]
    elif dataset_id in _loaded_datasets:
        df = _loaded_datasets[dataset_id]
        label_col = df.attrs.get("label_col", "class_label")
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset '{dataset_id}'."
            )

        processed = {
            "labels": df[label_col].to_numpy(),
            "label_column": label_col,
        }

        smiles_col = df.attrs.get("smiles_col")
        if smiles_col and smiles_col in df.columns:
            processed["smiles"] = df[smiles_col].to_numpy()

        id_col = df.attrs.get("id_col")
        if id_col and id_col in df.columns:
            processed["cid"] = df[id_col].to_numpy()
    else:
        raise ValueError(
            f"Dataset '{dataset_id}' not loaded. "
            "Call load_dataset() first."
        )

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

    csv_saved_to = None
    if save_csv and csv_save_path != "":
        if csv_save_path is None:
            if saved_to is not None:
                csv_path = str(Path(saved_to).with_suffix(".csv"))
            else:
                out_dir = _get_session_logger().session_dir / "splits"
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = str(out_dir / f"{dataset_id}_{split_type}.csv")
        else:
            csv_path = _resolve_path(csv_save_path)

        split_df = _split_save_dict_to_dataframe(split_result["save_dict"])
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(csv_path, index=False)
        csv_saved_to = str(Path(csv_path).resolve())

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
        # Keep this first so LLM tool callers can reliably chain downstream calls.
        "split_file_path": saved_to,
        # Backward-compatible alias used by existing callers.
        "saved_to":   saved_to,
        "split_csv_file_path": csv_saved_to,
        "csv_saved_to": csv_saved_to,
        "train":      _split_meta(train_idx),
        "val":        _split_meta(val_idx),
        "test":       _split_meta(test_idx),
        "split_type": split_type,
        "mode":       mode,
        "has_features": "features" in processed,
        "statistics": statistics,
        "seed":       seed,
        "next_step": (
            f"Call train_model(split_file_path='{saved_to}', "
            "algorithm='RFC', task='classification', opt_metric='balanced_accuracy') "
            "to train in the background (non-blocking). "
            "Then poll with check_training(job_id) until status='completed'. "
            "Features are on disk — not returned here."
        ) if saved_to and "features" in processed else (
            f"Call prepare_gnn_dataset(split_file_path='{saved_to}', smiles_csv_path='...') "
            "for GNN workflows, or run compute_features() first for standard ML training."
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

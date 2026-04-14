"""
Dataset splitting helpers.

Bridges the processed-dataset dict (features + labels) and
``chemagent.splitting``, and handles saving the resulting split to disk.

Usage
-----
    from chemagent.datasets.splitter import split_processed, save_split

    result = split_processed(processed, split_type="random",
                             train_size=0.7, val_size=0.0, test_size=0.3,
                             seed=42, stratified=True)
    saved_to = save_split(result["save_dict"], dataset_id="my_ds",
                          split_type="random")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np

from chemagent.splitting import random_split, scaffold_split, analogue_series_split
from .loader import workspace_root



def split_processed(
    processed: Dict[str, Any],
    split_type: Literal["random", "scaffold", "analogue_series"] = "random",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: Optional[int] = 42,
    stratified: bool = False,
    n_attempts: int = 10,
    n_cpds_tolerance: int = 5,
) -> Dict[str, Any]:
    """Split a processed dataset dict into train / val / test partitions.

    Args:
    processed:
        Dict with key ``labels`` and optionally ``features``, ``smiles``, ``cid``, and ``core`` — produced by
        :func:`~chemagent.datasets.featurizer.build_processed_entry`.
    split_type:
        ``"random"`` (default), ``"scaffold"``, or ``"analogue_series"``.
    train_size, val_size, test_size:
        Split proportions; must sum to 1.0.  For ``"analogue_series"``
        only *test_size* is used (no validation set is produced).
    seed:
        Random seed for reproducibility (default 42).
    stratified:
        Preserve class proportions (random splits only).
    n_attempts:
        ``"analogue_series"`` only — number of random shuffles to try
        before accepting the best result (default 10).
    n_cpds_tolerance:
        ``"analogue_series"`` only — maximum allowed deviation (in
        compounds) between actual and target test size (default 5).

    Returns:
    dict
        Keys:
        ``train_idx``, ``val_idx``, ``test_idx`` (index arrays),
        ``statistics`` (counts / percentages),
        ``save_dict`` (ready to pass to :func:`save_split`).

    Raises:
    ValueError
        If scaffold split is requested but no SMILES are available, or
        analogue_series split is requested but no core column is available.
    """
    labels = np.asarray(processed["labels"])
    features = processed.get("features")
    n_samples = len(labels)

    if split_type == "random":
        split_indices = random_split(
            n_samples=n_samples,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            labels=labels.tolist() if stratified else None,
            stratified=stratified,
        )
    elif split_type == "scaffold":
        if "smiles" not in processed:
            raise ValueError(
                "Scaffold split requires a SMILES array in the processed dict. "
                "Ensure smiles_col was set when calling load_dataset()."
            )
        split_indices = scaffold_split(
            smiles_list=processed["smiles"].tolist(),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            labels=labels.tolist() if stratified else None,
            stratified=stratified,
        )
    elif split_type == "analogue_series":
        if "core" not in processed:
            raise ValueError(
                "Analogue series split requires a 'core' array in the processed dict. "
                "Ensure core_col was set when calling build_processed_entry()."
            )
        split_indices = analogue_series_split(
            cores=processed["core"].tolist(),
            test_size=test_size,
            seed=seed,
            n_attempts=n_attempts,
            n_cpds_tolerance=n_cpds_tolerance,
        )
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    train_idx = np.array(split_indices["train"], dtype=int)
    val_idx   = np.array(split_indices["val"],   dtype=int)
    test_idx  = np.array(split_indices["test"],  dtype=int)

    statistics = {
        "train": {
            "count":      len(train_idx),
            "percentage": round(len(train_idx) / n_samples * 100, 2),
        },
        "val": {
            "count":      len(val_idx),
            "percentage": round(len(val_idx) / n_samples * 100, 2),
        },
        "test": {
            "count":      len(test_idx),
            "percentage": round(len(test_idx) / n_samples * 100, 2),
        },
    }

    # Build the save_dict. Features are optional for GNN-oriented splits.
    save_dict: Dict[str, Any] = {
        "train_labels":   labels[train_idx],
        "val_labels":     labels[val_idx],
        "test_labels":    labels[test_idx],
    }
    if features is not None:
        save_dict["train_features"] = features[train_idx]
        save_dict["val_features"] = features[val_idx]
        save_dict["test_features"] = features[test_idx]
    if "smiles" in processed:
        save_dict["train_smiles"] = processed["smiles"][train_idx]
        save_dict["val_smiles"]   = processed["smiles"][val_idx]
        save_dict["test_smiles"]  = processed["smiles"][test_idx]
    if "cid" in processed:
        save_dict["train_cid"] = processed["cid"][train_idx]
        save_dict["val_cid"]   = processed["cid"][val_idx]
        save_dict["test_cid"]  = processed["cid"][test_idx]
    if "core" in processed:
        save_dict["train_core"] = processed["core"][train_idx]
        save_dict["val_core"]   = processed["core"][val_idx]
        save_dict["test_core"]  = processed["core"][test_idx]

    return {
        "train_idx":  train_idx,
        "val_idx":    val_idx,
        "test_idx":   test_idx,
        "statistics": statistics,
        "save_dict":  save_dict,
    }


def save_split(
    save_dict: Dict[str, Any],
    dataset_id: str,
    split_type: str = "random",
    save_path: Optional[str] = None,
) -> str:
    """Serialise a split dict to a ``.pkl`` file.

    Args:
    save_dict:
        Dict to persist (from :func:`split_processed`).
    dataset_id:
        Used in the default file name.
    split_type:
        Used in the default file name.
    save_path:
        Explicit file path. Defaults to
        ``data/splits/<dataset_id>_<split_type>.pkl`` under the workspace root.

    Returns:
    str
        Absolute path of the saved file.
    """
    if save_path is None:
        out_dir = workspace_root() / "data" / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / f"{dataset_id}_{split_type}.pkl")
    joblib.dump(save_dict, save_path)
    return str(Path(save_path).resolve())

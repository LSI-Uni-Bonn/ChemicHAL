"""chemagent.ml.ml_model_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for ML model information and inference.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
get_ml_info             — reference card for algorithms and recommended metrics
export_predictions      — run inference on a split .pkl, save predictions CSV
compare_exported_predictions — compare model predictions across multiple exported CSVs

Internal helpers
----------------
predict_from_split_file — raw prediction + pkl dump (used by other tools)
build_model_from_arrays — train from raw in-memory arrays
"""

from __future__ import annotations

from collections import Counter
import inspect
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.ml.hyperparameter_tuning import HYPERPARAMETERS
from chemagent.servers.server_helpers import (
    _run_pipeline,
    _to_serialisable,
    _predict,
    _workspace_root,
    evaluate_classification,
    evaluate_regression,
)
from chemagent.session_utils import get_session_logger as _get_session_logger

# Optional imports for GNN model inference
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from chemagent.ml import gnn_models as _gnn_models
    from chemagent.ml.gnn_training import smiles_to_nx_graph, nx_graph_to_pyg_data
except Exception:
    torch = None
    F = None
    PyGDataLoader = None
    _gnn_models = None
    smiles_to_nx_graph = None
    nx_graph_to_pyg_data = None


# Reference tool
def get_ml_info() -> dict[str, Any]:
    """Return all ML reference information in one call: algorithms, hyperparameter grids, and recommended metrics.

    Call once before choosing an algorithm or opt_metric for train_model().

    Returns:
        algorithms: dict of algorithm code → name, task_type, hyperparameter grid,
                    supports_multiclass, description.
        recommended_metrics: dict of task type → optimization and evaluation metric lists.
    """
    algorithms = {
        "RFR": {
            "name": "Random Forest Regressor",
            "task_type": "regression",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("RFR", {})),
            "supports_multiclass": False,
            "description": "Ensemble of decision trees for regression tasks",
        },
        "RFC": {
            "name": "Random Forest Classifier",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("RFC", {})),
            "supports_multiclass": True,
            "description": "Ensemble of decision trees for classification, handles multi-class",
        },
        "SVC": {
            "name": "Support Vector Classifier",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("SVC", {})),
            "supports_multiclass": True,
            "description": "SVM classifier with RBF/linear kernels, probability estimates enabled",
        },
        "DNN": {
            "name": "Feed-forward Neural Network",
            "task_type": "both",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("DNN", {})),
            "supports_multiclass": True,
            "description": "PyTorch MLP wrapped as a scikit-learn estimator; no grid-search hyperparameters configured by default",
        },
        # Graph Neural Networks (PyTorch Geometric)
        "GCN": {
            "name": "Graph Convolutional Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GCN", {})),
            "supports_multiclass": True,
            "description": "Graph Convolutional Network (PyG) for graph-structured molecular data",
        },
        "GraphSAGE": {
            "name": "GraphSAGE",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GraphSAGE", {})),
            "supports_multiclass": True,
            "description": "GraphSAGE inductive representation learner for graphs",
        },
        "GAT": {
            "name": "Graph Attention Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GAT", {})),
            "supports_multiclass": True,
            "description": "Graph Attention Network using attention-based message passing",
        },
        "GIN": {
            "name": "Graph Isomorphism Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GIN", {})),
            "supports_multiclass": True,
            "description": "GIN: powerful message-passing GNN for graph classification",
        },
        "GINE": {
            "name": "GINE",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GINE", {})),
            "supports_multiclass": True,
            "description": "GINE: GIN variant with edge features for molecular graphs",
        },
        "GC_GNN": {
            "name": "GC-GNN",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GC_GNN", {})),
            "supports_multiclass": True,
            "description": "Custom GC_GNN architecture used in the project",
        },
    }
    recommended_metrics = {
        "binary_classification": {
            "optimization": ["f1", "roc_auc", "average_precision", "balanced_accuracy"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "AUC", "BA"],
        },
        "binary_imbalanced": {
            "optimization": ["f1", "average_precision", "roc_auc"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "BA"],
            "note":         "Pass task='classification-cw' to train_model() for auto class-weighting",
        },
        "multiclass": {
            "optimization": ["f1_macro", "f1_weighted", "balanced_accuracy"],
            "evaluation":   ["MCC", "BA", "F1_macro", "F1_weighted", "Accuracy"],
        },
        "regression": {
            "optimization": ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
            "evaluation":   ["RMSE", "MAE", "R2", "Pearson r"],
        },
    }
    return {"algorithms": algorithms, "recommended_metrics": recommended_metrics}



# Inference / export tool
def export_predictions(
    model_path: str,
    split_file_path: str,
    task: Literal["regression", "classification", "classification-cw"] = "classification",
    split: Literal["train", "val", "test"] = "test",
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run inference on a split .pkl and export per-sample predictions to a CSV.

    Loads the model and split from disk, predicts on the chosen partition, and
    writes a CSV with columns: cid (if available), smiles (if available),
    true_label, predicted_label, prob_class_0, prob_class_1 (classification)
    or predicted_value (regression). Also saves aggregated metrics.

    Workflow: check_training → THIS TOOL → plot_classification_results

    Args:
        model_path: Path to .pkl model from train_model() / check_training().
        split_file_path: Path to the .pkl split file from split_dataset().
        task: "classification" (default), "classification-cw", or "regression".
        split: Partition to predict on — "test" (default), "train", or "val".
        save_path: Output CSV path. Defaults to <session>/results/<stem>_<split>_predictions.csv.

    Returns:
        csv_path, metrics_path, metrics dict, n_samples, columns.
    """
    import pandas as pd

    session_logger = _get_session_logger()

    # Handle PyTorch GNN models saved as .pt/.pth
    if str(model_path).lower().endswith((".pt", ".pth")):
        return _export_predictions_gnn(
            model_path=model_path,
            split_file_path=split_file_path,
            split=split,
            save_path=save_path,
            device=None,
        )

    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = _predict(model_path=model_path,
                      features=data[f"{split}_features"].tolist(),
                      reg_class=task)

    model_stem    = Path(model_path).stem
    is_regression = task == "regression"

    cid_key    = f"{split}_cid"
    smiles_key = f"{split}_smiles"
    cid_col    = data[cid_key].tolist()    if cid_key    in data else None
    smiles_col = data[smiles_key].tolist() if smiles_key in data else None

    if is_regression:
        df_out = pd.DataFrame({
            "true_label":      labels,
            "predicted_value": result["predictions"],
        })
    else:
        proba = result.get("probabilities") or []
        n_classes = len(proba[0]) if proba else 0
        df_out = pd.DataFrame({"true_label": labels, "predicted_label": result["predictions"]})
        for c in range(n_classes):
            df_out[f"prob_class_{c}"] = [p[c] for p in proba]

    if smiles_col is not None:
        df_out.insert(0, "smiles", smiles_col)
    if cid_col is not None:
        df_out.insert(0, "cid", cid_col)

    metrics_dict = (
        evaluate_regression(labels=labels, predictions=result["predictions"],
                            model_id=model_stem)
        if is_regression
        else evaluate_classification(labels=labels, predictions=result["predictions"],
                                     probabilities=result.get("probabilities"),
                                     model_id=model_stem)
    )

    if save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base          = f"{model_stem}_{split}"
        save_path     = str(out_dir / f"{base}_predictions.csv")
        metrics_path  = str(out_dir / f"{base}_metrics.pkl")
    else:
        save_path    = str(Path(save_path).resolve())
        metrics_path = str(Path(save_path).with_suffix("").as_posix() + "_metrics.pkl")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    joblib.dump(metrics_dict, metrics_path)

    return {
        "csv_path":     save_path,
        "metrics_path": metrics_path,
        "metrics":      metrics_dict,
        "n_samples":    len(labels),
        "columns":      list(df_out.columns),
    }


def compare_exported_predictions(
    prediction_paths: list[str],
    model_names: Optional[list[str]] = None,
    task: Literal["auto", "classification", "regression"] = "auto",
    match_on: Literal["auto", "cid", "smiles", "row_index"] = "auto",
    include_rows: Literal["all", "disagreements", "agreements"] = "disagreements",
    min_models_per_compound: int = 2,
    max_compounds: int = 100,
    include_probabilities: bool = False,
    top_n_patterns: int = 15,
    regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare predictions from multiple exported prediction CSV files.

    Use this after running ``export_predictions`` for two or more models to
    inspect where models agree, disagree, and how often each disagreement
    pattern occurs.

    Flexible matching:
    - ``match_on='auto'`` prefers ``cid``, then ``smiles``, then row index.
    - Duplicate ``cid``/``smiles`` entries are aligned by occurrence order
      within each file to avoid cartesian merge artifacts.

    Flexible filtering:
    - ``include_rows='disagreements'`` returns only compounds with conflicting
      model predictions (classification) or spread above tolerance (regression).
    - ``include_rows='agreements'`` returns only compounds with model agreement.
    - ``include_rows='all'`` returns all compared compounds.

    Args:
        prediction_paths: Paths to two or more CSV files from export_predictions().
        model_names: Optional labels for each file (same order as prediction_paths).
            Defaults to CSV stem names.
        task: "classification", "regression", or "auto" (infer from columns).
        match_on: Row alignment key: "cid", "smiles", "row_index", or "auto".
        include_rows: Which rows to include in ``compounds`` output.
        min_models_per_compound: Minimum number of models that must have a
            prediction for a compound to be included in comparison stats.
        max_compounds: Maximum number of row-level records returned in ``compounds``.
        include_probabilities: For classification, include per-model probability
            vectors (when present in input CSVs).
        top_n_patterns: Number of most common prediction patterns to include
            (classification only).
        regression_tolerance: In regression, rows with prediction spread greater
            than this value are considered disagreements.

    Returns:
        A summary dictionary with:
        - matching metadata,
        - per-model coverage,
        - pairwise agreement/difference stats,
        - aggregate agreement summary,
        - row-level ``compounds`` records suitable for LLM reasoning.
    """
    import pandas as pd

    valid_tasks = {"auto", "classification", "regression"}
    valid_match = {"auto", "cid", "smiles", "row_index"}
    valid_rows = {"all", "disagreements", "agreements"}

    if task not in valid_tasks:
        raise ValueError(f"Invalid task={task!r}. Expected one of {sorted(valid_tasks)}")
    if match_on not in valid_match:
        raise ValueError(f"Invalid match_on={match_on!r}. Expected one of {sorted(valid_match)}")
    if include_rows not in valid_rows:
        raise ValueError(
            f"Invalid include_rows={include_rows!r}. Expected one of {sorted(valid_rows)}"
        )
    if len(prediction_paths) < 2:
        raise ValueError("Provide at least two prediction CSV paths to compare.")
    if min_models_per_compound < 2:
        raise ValueError("min_models_per_compound must be >= 2.")
    if max_compounds <= 0:
        raise ValueError("max_compounds must be > 0.")
    if top_n_patterns <= 0:
        raise ValueError("top_n_patterns must be > 0.")
    if regression_tolerance < 0:
        raise ValueError("regression_tolerance must be >= 0.")

    def _to_python_scalar(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _coalesce_value(row: pd.Series, columns: list[str]) -> Any:
        for col in columns:
            if col in row and pd.notna(row[col]):
                return _to_python_scalar(row[col])
        return None

    def _resolve_prediction_path(raw_path: str) -> Path:
        p = Path(raw_path)
        if not p.exists():
            p = _workspace_root() / raw_path
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {raw_path}")
        return p.resolve()

    def _prob_sort_key(col: str) -> tuple[int, str]:
        tail = col.rsplit("_", 1)[-1]
        if tail.isdigit():
            return (int(tail), col)
        return (10_000, col)

    resolved_paths = [_resolve_prediction_path(p) for p in prediction_paths]
    frames = [pd.read_csv(path) for path in resolved_paths]

    if model_names is None:
        used: set[str] = set()
        inferred_names: list[str] = []
        for path in resolved_paths:
            base = path.stem
            if base.endswith("_predictions"):
                base = base[: -len("_predictions")]
            candidate = base or "model"
            if candidate in used:
                i = 2
                while f"{candidate}_{i}" in used:
                    i += 1
                candidate = f"{candidate}_{i}"
            used.add(candidate)
            inferred_names.append(candidate)
        model_names = inferred_names
    else:
        if len(model_names) != len(prediction_paths):
            raise ValueError(
                "model_names must have the same length as prediction_paths "
                f"({len(model_names)} != {len(prediction_paths)})."
            )
        if len(set(model_names)) != len(model_names):
            raise ValueError("model_names must be unique.")

    has_pred_label = ["predicted_label" in df.columns for df in frames]
    has_pred_value = ["predicted_value" in df.columns for df in frames]

    inferred_task: Literal["classification", "regression"]
    if task == "auto":
        if all(has_pred_label):
            inferred_task = "classification"
        elif all(has_pred_value):
            inferred_task = "regression"
        else:
            per_file = {
                model: {
                    "has_predicted_label": bool(has_cls),
                    "has_predicted_value": bool(has_reg),
                }
                for model, has_cls, has_reg in zip(model_names, has_pred_label, has_pred_value)
            }
            raise ValueError(
                "Could not infer a common task from input files. "
                "Each file must provide either predicted_label (classification) "
                "or predicted_value (regression). "
                f"Per-file columns: {per_file}"
            )
    else:
        inferred_task = task

    pred_col = "predicted_label" if inferred_task == "classification" else "predicted_value"
    for model, path, df in zip(model_names, resolved_paths, frames):
        if pred_col not in df.columns:
            raise ValueError(
                f"File for model '{model}' ({path}) is missing required column '{pred_col}'."
            )

    if match_on == "auto":
        if all("cid" in df.columns for df in frames):
            match_method: Literal["cid", "smiles", "row_index"] = "cid"
        elif all("smiles" in df.columns for df in frames):
            match_method = "smiles"
        else:
            match_method = "row_index"
    else:
        match_method = match_on

    if match_method in {"cid", "smiles"}:
        missing_models = [
            model for model, df in zip(model_names, frames)
            if match_method not in df.columns
        ]
        if missing_models:
            raise ValueError(
                f"match_on='{match_method}' requested, but missing in models: {missing_models}"
            )

    prepared_frames: list[pd.DataFrame] = []
    probability_cols_by_model: dict[str, list[str]] = {}

    merge_keys = ["__row_index__"] if match_method == "row_index" else [match_method, "__merge_occurrence__"]

    for model, df in zip(model_names, frames):
        work = df.copy()
        work["__row_index__"] = np.arange(len(work), dtype=int)
        if match_method != "row_index":
            work["__merge_occurrence__"] = work.groupby(match_method, dropna=False).cumcount()

        out = work[merge_keys].copy()
        out[f"pred__{model}"] = work[pred_col]

        if "true_label" in work.columns:
            out[f"true__{model}"] = work["true_label"]
        if "cid" in work.columns:
            out[f"cid__{model}"] = work["cid"]
        if "smiles" in work.columns:
            out[f"smiles__{model}"] = work["smiles"]

        probability_cols_by_model[model] = []
        if inferred_task == "classification" and include_probabilities:
            prob_cols = sorted(
                [c for c in work.columns if c.startswith("prob_class_")],
                key=_prob_sort_key,
            )
            for col in prob_cols:
                new_col = f"{col}__{model}"
                out[new_col] = work[col]
                probability_cols_by_model[model].append(new_col)

        prepared_frames.append(out)

    merged = prepared_frames[0]
    for nxt in prepared_frames[1:]:
        merged = merged.merge(nxt, on=merge_keys, how="outer")

    pred_cols = [f"pred__{model}" for model in model_names]
    cid_cols = [f"cid__{model}" for model in model_names if f"cid__{model}" in merged.columns]
    smiles_cols = [f"smiles__{model}" for model in model_names if f"smiles__{model}" in merged.columns]

    merged["n_models_present"] = merged[pred_cols].notna().sum(axis=1)
    coverage_by_model = {
        model: int(merged[f"pred__{model}"].notna().sum())
        for model in model_names
    }

    compared = merged[merged["n_models_present"] >= min_models_per_compound].copy()
    n_compounds_compared = int(len(compared))

    pairwise_comparison: list[dict[str, Any]] = []
    for model_a, model_b in combinations(model_names, 2):
        col_a = f"pred__{model_a}"
        col_b = f"pred__{model_b}"
        mask = compared[col_a].notna() & compared[col_b].notna()
        n_overlap = int(mask.sum())
        if n_overlap == 0:
            pairwise_comparison.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_overlap": 0,
            })
            continue

        if inferred_task == "classification":
            n_agree = int((compared.loc[mask, col_a] == compared.loc[mask, col_b]).sum())
            n_disagree = n_overlap - n_agree
            pairwise_comparison.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_overlap": n_overlap,
                "n_agree": n_agree,
                "n_disagree": n_disagree,
                "agreement_rate": float(n_agree / n_overlap),
            })
        else:
            diffs = (
                compared.loc[mask, col_a].astype(float)
                - compared.loc[mask, col_b].astype(float)
            ).abs()
            pairwise_comparison.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_overlap": n_overlap,
                "mean_abs_difference": float(diffs.mean()),
                "max_abs_difference": float(diffs.max()),
            })

    result: dict[str, Any] = {
        "task": inferred_task,
        "models": model_names,
        "source_files": {
            model: str(path)
            for model, path in zip(model_names, resolved_paths)
        },
        "matching": {
            "method": match_method,
            "merge_keys": merge_keys,
        },
        "filters": {
            "include_rows": include_rows,
            "min_models_per_compound": min_models_per_compound,
            "max_compounds": max_compounds,
            "include_probabilities": include_probabilities,
            "regression_tolerance": regression_tolerance,
        },
        "n_compounds_input_union": int(len(merged)),
        "n_compounds_compared": n_compounds_compared,
        "coverage_by_model": coverage_by_model,
        "pairwise_comparison": pairwise_comparison,
    }

    if compared.empty:
        result["agreement_summary"] = {
            "message": "No compounds satisfy min_models_per_compound filter.",
        }
        result["n_compounds_reported"] = 0
        result["compounds"] = []
        result["next_step"] = (
            "Lower min_models_per_compound, or verify that input CSV files refer "
            "to the same split and matching key (cid/smiles/row index)."
        )
        return _to_serialisable(result)

    if inferred_task == "classification":
        compared["_n_unique_predictions"] = compared[pred_cols].apply(
            lambda row: len({
                _to_python_scalar(v)
                for v in row.tolist()
                if pd.notna(v)
            }),
            axis=1,
        )
        compared["_unanimous"] = compared["_n_unique_predictions"] == 1

        unanimous_count = int(compared["_unanimous"].sum())
        disagreement_count = int(len(compared) - unanimous_count)
        result["agreement_summary"] = {
            "unanimous_count": unanimous_count,
            "disagreement_count": disagreement_count,
            "unanimous_rate": float(unanimous_count / len(compared)),
        }

        pattern_counter: Counter[tuple[Any, ...]] = Counter()
        for _, row in compared[pred_cols].iterrows():
            pattern = tuple(
                _to_python_scalar(v) if pd.notna(v) else None
                for v in row.tolist()
            )
            pattern_counter[pattern] += 1

        top_patterns = pattern_counter.most_common(top_n_patterns)
        result["prediction_pattern_counts"] = [
            {
                "count": int(count),
                "fraction": float(count / len(compared)),
                "predictions_by_model": {
                    model: value
                    for model, value in zip(model_names, pattern)
                },
                "n_unique_predictions": int(len({v for v in pattern if v is not None})),
            }
            for pattern, count in top_patterns
        ]

        if include_rows == "disagreements":
            report_df = compared[~compared["_unanimous"]].copy()
        elif include_rows == "agreements":
            report_df = compared[compared["_unanimous"]].copy()
        else:
            report_df = compared.copy()

        report_df = report_df.sort_values(
            by=["_n_unique_predictions", "n_models_present"],
            ascending=[False, False],
            kind="stable",
        )
        report_df = report_df.head(max_compounds)

        compound_rows: list[dict[str, Any]] = []
        for _, row in report_df.iterrows():
            preds_by_model = {
                model: _to_python_scalar(row[f"pred__{model}"])
                for model in model_names
                if pd.notna(row[f"pred__{model}"])
            }
            models_present = list(preds_by_model.keys())
            vote_counts = Counter(preds_by_model.values())
            max_votes = max(vote_counts.values())
            majority_labels = sorted(
                [label for label, n in vote_counts.items() if n == max_votes],
                key=str,
            )
            majority_set = set(majority_labels)

            true_by_model = {
                model: _to_python_scalar(row[f"true__{model}"])
                for model in model_names
                if f"true__{model}" in row and pd.notna(row[f"true__{model}"])
            }
            true_unique = sorted(set(true_by_model.values()), key=str)

            item: dict[str, Any] = {
                "n_models_present": int(row["n_models_present"]),
                "models_present": models_present,
                "predictions_by_model": preds_by_model,
                "unanimous": bool(row["_unanimous"]),
                "n_unique_predictions": int(row["_n_unique_predictions"]),
                "majority_prediction": (
                    majority_labels[0] if len(majority_labels) == 1 else majority_labels
                ),
                "majority_models": [
                    model for model, pred in preds_by_model.items()
                    if pred in majority_set
                ],
                "minority_models": [
                    model for model, pred in preds_by_model.items()
                    if pred not in majority_set
                ],
                "models_with_true_label": list(true_by_model.keys()),
                "true_label": true_unique[0] if len(true_unique) == 1 else None,
                "true_label_consistent": len(true_unique) <= 1,
            }

            if match_method == "row_index":
                item["row_index"] = int(row["__row_index__"])
            else:
                item[match_method] = _to_python_scalar(row[match_method])
                item["match_occurrence"] = int(row["__merge_occurrence__"])

            cid_value = _coalesce_value(row, cid_cols)
            if cid_value is not None:
                item["cid"] = cid_value
            smiles_value = _coalesce_value(row, smiles_cols)
            if smiles_value is not None:
                item["smiles"] = smiles_value

            if include_probabilities:
                probs_by_model: dict[str, list[float]] = {}
                for model in model_names:
                    model_prob_cols = probability_cols_by_model.get(model, [])
                    if not model_prob_cols:
                        continue
                    if not all(col in row and pd.notna(row[col]) for col in model_prob_cols):
                        continue
                    probs_by_model[model] = [float(row[col]) for col in model_prob_cols]
                if probs_by_model:
                    item["probabilities_by_model"] = probs_by_model

            compound_rows.append(item)

        result["compounds"] = compound_rows
        result["n_compounds_reported"] = int(len(compound_rows))
        result["next_step"] = (
            "Inspect compounds with minority_models to identify systematic model disagreements; "
            "set include_rows='all' to review full agreement context."
        )
        return _to_serialisable(result)

    # Regression path
    def _spread(row_values: list[Any]) -> float:
        vals = [float(v) for v in row_values if pd.notna(v)]
        if len(vals) < 2:
            return 0.0
        return float(max(vals) - min(vals))

    compared["_prediction_spread"] = compared[pred_cols].apply(
        lambda row: _spread(row.tolist()),
        axis=1,
    )
    compared["_within_tolerance"] = compared["_prediction_spread"] <= regression_tolerance

    within_tolerance_count = int(compared["_within_tolerance"].sum())
    disagreement_count = int(len(compared) - within_tolerance_count)
    result["agreement_summary"] = {
        "within_tolerance_count": within_tolerance_count,
        "disagreement_count": disagreement_count,
        "within_tolerance_rate": float(within_tolerance_count / len(compared)),
        "regression_tolerance": float(regression_tolerance),
    }

    if include_rows == "disagreements":
        report_df = compared[~compared["_within_tolerance"]].copy()
    elif include_rows == "agreements":
        report_df = compared[compared["_within_tolerance"]].copy()
    else:
        report_df = compared.copy()

    report_df = report_df.sort_values(
        by=["_prediction_spread", "n_models_present"],
        ascending=[False, False],
        kind="stable",
    )
    report_df = report_df.head(max_compounds)

    compound_rows = []
    for _, row in report_df.iterrows():
        preds_by_model = {
            model: float(row[f"pred__{model}"])
            for model in model_names
            if pd.notna(row[f"pred__{model}"])
        }
        pred_values = list(preds_by_model.values())
        pred_arr = np.array(pred_values, dtype=float)

        true_by_model = {
            model: _to_python_scalar(row[f"true__{model}"])
            for model in model_names
            if f"true__{model}" in row and pd.notna(row[f"true__{model}"])
        }
        true_unique = sorted(set(true_by_model.values()), key=str)

        item = {
            "n_models_present": int(row["n_models_present"]),
            "models_present": list(preds_by_model.keys()),
            "predictions_by_model": preds_by_model,
            "mean_prediction": float(pred_arr.mean()) if len(pred_arr) else None,
            "std_prediction": float(pred_arr.std(ddof=0)) if len(pred_arr) else None,
            "min_prediction": float(pred_arr.min()) if len(pred_arr) else None,
            "max_prediction": float(pred_arr.max()) if len(pred_arr) else None,
            "prediction_spread": float(row["_prediction_spread"]),
            "within_tolerance": bool(row["_within_tolerance"]),
            "models_with_true_label": list(true_by_model.keys()),
            "true_label": true_unique[0] if len(true_unique) == 1 else None,
            "true_label_consistent": len(true_unique) <= 1,
        }

        if match_method == "row_index":
            item["row_index"] = int(row["__row_index__"])
        else:
            item[match_method] = _to_python_scalar(row[match_method])
            item["match_occurrence"] = int(row["__merge_occurrence__"])

        cid_value = _coalesce_value(row, cid_cols)
        if cid_value is not None:
            item["cid"] = cid_value
        smiles_value = _coalesce_value(row, smiles_cols)
        if smiles_value is not None:
            item["smiles"] = smiles_value

        compound_rows.append(item)

    result["compounds"] = compound_rows
    result["n_compounds_reported"] = int(len(compound_rows))
    result["next_step"] = (
        "Inspect compounds with large prediction_spread to identify unstable regions; "
        "tune regression_tolerance to tighten or relax disagreement detection."
    )
    return _to_serialisable(result)


def predict_from_split_file(
    model_path: str,
    split_file_path: str,
    reg_class: Literal["regression", "classification", "classification-cw"] = "classification",
    split: Literal["train", "val", "test"] = "test",
    results_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Internal helper — raw predictions + pkl dump. Use export_predictions() for the MCP tool."""
    session_logger = _get_session_logger()

    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = _predict(model_path=model_path,
                      features=data[f"{split}_features"].tolist(),
                      reg_class=reg_class)
    result["labels"] = labels

    model_stem    = Path(model_path).stem
    is_regression = reg_class == "regression"
    metrics_dict = (
        evaluate_regression(labels=labels, predictions=result["predictions"],
                            model_id=model_stem)
        if is_regression
        else evaluate_classification(labels=labels, predictions=result["predictions"],
                                     probabilities=result.get("probabilities"),
                                     model_id=model_stem)
    )
    result["metrics"] = metrics_dict

    if results_save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base              = f"{model_stem}_{split}"
        results_save_path = str(out_dir / f"{base}_predictions.pkl")
        metrics_save_path = str(out_dir / f"{base}_metrics.pkl")
    else:
        results_save_path = str(Path(results_save_path).resolve())
        metrics_save_path = str(Path(results_save_path).with_name(
            Path(results_save_path).stem + "_metrics.pkl"
        ))

    Path(results_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result,       results_save_path)
    joblib.dump(metrics_dict, metrics_save_path)
    result["results_path"] = results_save_path
    result["metrics_path"] = metrics_save_path
    return result


def _export_predictions_gnn(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
    model_class: Optional[str] = None,
    hidden_channels: int = 64,
) -> dict[str, Any]:
    """Export predictions for PyTorch Geometric GNN models saved as .pt/.pth.

    Expects the split file to contain the per-split smiles arrays (e.g. "test_smiles").

    To avoid architecture-mismatch load errors, this function infers model
    metadata in the following priority:
    1) explicit function args,
    2) checkpoint metadata fields,
    3) filename hints,
    4) state_dict tensor shapes/keys,
    5) safe defaults.
    """
    if torch is None or PyGDataLoader is None or smiles_to_nx_graph is None:
        raise ImportError("PyTorch / PyG not available in this environment")

    session_logger = _get_session_logger()

    # Load split (support pickle or joblib like other helpers)
    split_obj = None
    load_errors: list[str] = []
    try:
        import pickle as _pickle

        with open(split_file_path, "rb") as f:
            split_obj = _pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"pickle: {type(exc).__name__}: {exc}")

    if split_obj is None:
        try:
            split_obj = joblib.load(split_file_path)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"joblib: {type(exc).__name__}: {exc}")

    if split_obj is None:
        raise ValueError(f"Could not load split file '{split_file_path}': {'; '.join(load_errors)}")

    smiles_key = f"{split}_smiles"
    labels_key = f"{split}_labels"
    if smiles_key not in split_obj:
        raise ValueError(f"Split file '{split_file_path}' does not contain '{smiles_key}'. Provide a split file with per-split SMILES.")

    smiles_list = list(split_obj[smiles_key])
    labels = list(split_obj[labels_key]) if labels_key in split_obj else [None] * len(smiles_list)

    # Build Data objects
    data_list = []
    for s, lbl in zip(smiles_list, labels):
        nx_g = smiles_to_nx_graph(s)
        pyg = nx_graph_to_pyg_data(nx_g, lbl if lbl is not None else 0)
        if pyg is not None:
            data_list.append(pyg)

    if not data_list:
        raise ValueError("No valid graph data could be created from the provided SMILES.")

    loader = PyGDataLoader(data_list, batch_size=batch_size)

    model_stem = Path(model_path).stem
    possible = ["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load state/checkpoint first so we can infer architecture dimensions.
    state = torch.load(model_path, map_location=device)
    checkpoint: dict[str, Any] | None = state if isinstance(state, dict) else None
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        sd = None

    # Inference priority: explicit arg -> checkpoint metadata -> filename -> state_dict keys -> fallback.
    if model_class is None and isinstance(checkpoint, dict):
        model_class = checkpoint.get("model_class_name") or checkpoint.get("model_class")
    if model_class is None:
        for name in possible:
            if name.lower() in model_stem.lower():
                model_class = name
                break
    if model_class is None and isinstance(sd, dict):
        state_keys = list(sd.keys())
        if any("att_src" in k or "att_dst" in k for k in state_keys):
            model_class = "GAT"
        elif any("lin_rel" in k or "lin_root" in k for k in state_keys):
            model_class = "GC_GNN"
        elif any("lin_l" in k and "lin_r" in k for k in state_keys):
            model_class = "GraphSAGE"
    if model_class is None:
        model_class = "GCN"

    if not hasattr(_gnn_models, model_class):
        raise ValueError(f"Unknown GNN model class '{model_class}'. Available: {', '.join([n for n in possible if hasattr(_gnn_models, n)])}")

    # Infer model dimensions from checkpoint/state dict when available.
    node_features_dim = 4
    inferred_num_classes = len(sorted(set([lbl for lbl in labels if lbl is not None]))) or 2
    inferred_hidden_channels = hidden_channels
    inferred_num_layers = 4
    inferred_aggregation_method = None

    if isinstance(checkpoint, dict):
        node_features_dim = int(checkpoint.get("node_features_dim", node_features_dim))
        inferred_hidden_channels = int(checkpoint.get("hidden_channels", inferred_hidden_channels))
        inferred_num_classes = int(checkpoint.get("num_classes", inferred_num_classes))
        inferred_num_layers = int(checkpoint.get("num_layers", inferred_num_layers))
        inferred_aggregation_method = checkpoint.get("aggregation_method", inferred_aggregation_method)

    if isinstance(sd, dict) and "lin.weight" in sd:
        lin_w = sd["lin.weight"]
        if hasattr(lin_w, "shape") and len(lin_w.shape) == 2:
            inferred_num_classes = int(lin_w.shape[0])
            inferred_hidden_channels = int(lin_w.shape[1])

    if isinstance(sd, dict):
        # Support newer ModuleList keys (convs.0..., convs.1...) and older
        # fixed-name keys (conv1..., conv2...).
        conv_indices = {
            int(k.split(".")[1])
            for k in sd.keys()
            if k.startswith("convs.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        }
        if conv_indices:
            inferred_num_layers = max(conv_indices) + 1
        else:
            legacy_count = sum(1 for i in range(1, 9) if any(key.startswith(f"conv{i}.") for key in sd.keys()))
            if legacy_count > 0:
                inferred_num_layers = legacy_count

    ModelCls = getattr(_gnn_models, model_class)
    model_kwargs = {
        "node_features_dim": node_features_dim,
        "hidden_channels": inferred_hidden_channels,
        "num_classes": inferred_num_classes,
    }
    model_sig = inspect.signature(ModelCls)
    if "num_layers" in model_sig.parameters:
        model_kwargs["num_layers"] = inferred_num_layers
    if inferred_aggregation_method is not None and "aggregation_method" in model_sig.parameters:
        model_kwargs["aggregation_method"] = inferred_aggregation_method

    model = ModelCls(**model_kwargs).to(device)

    try:
        if isinstance(state, dict):
            # Accept either raw state_dict or checkpoint with 'state_dict'.
            if sd is None:
                raise RuntimeError("Checkpoint dictionary did not include a valid state dict")
            model.load_state_dict(sd)
        else:
            # If a full model object was saved
            model = state.to(device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model state: {exc}")

    # Inference
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1).cpu().tolist()
            preds = out.argmax(dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)

    # Build DataFrame
    import pandas as pd

    df_out = pd.DataFrame({
        "smiles": smiles_list,
        "true_label": labels,
        "predicted_label": all_preds,
    })
    n_classes = len(all_probs[0]) if all_probs else 0
    for c in range(n_classes):
        df_out[f"prob_class_{c}"] = [p[c] for p in all_probs]

    metrics_dict = evaluate_classification(labels=labels, predictions=all_preds, probabilities=all_probs, model_id=model_stem)

    if save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{model_stem}_{split}"
        save_path = str(out_dir / f"{base}_predictions.csv")
        metrics_path = str(out_dir / f"{base}_metrics.pkl")
    else:
        save_path = str(Path(save_path).resolve())
        metrics_path = str(Path(save_path).with_suffix("").as_posix() + "_metrics.pkl")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    joblib.dump(metrics_dict, metrics_path)

    return {
        "csv_path": save_path,
        "metrics_path": metrics_path,
        "metrics": metrics_dict,
        "n_samples": len(labels),
        "columns": list(df_out.columns),
    }


def build_model_from_arrays(
    train_features: list[list[float]],
    train_labels: list[float],
    test_features: list[list[float]],
    test_labels: list[float],
    algorithm: Literal["RFC", "RFR", "SVC", "DNN"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
    val_features: Optional[list[list[float]]] = None,
    val_labels: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Train pipeline from raw arrays (internal helper — use train_model when possible)."""
    from chemagent.ml.training_tools import _default_model_path

    if len(train_features) != len(train_labels):
        raise ValueError("train_features and train_labels length mismatch")
    if len(test_features) != len(test_labels):
        raise ValueError("test_features and test_labels length mismatch")

    X_val = np.array(val_features) if val_features is not None else None
    y_val = np.array(val_labels)   if val_labels  is not None else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm)

    return _run_pipeline(
        X_train=np.array(train_features), y_train=np.array(train_labels),
        X_test=np.array(test_features),   y_test=np.array(test_labels),
        X_val=X_val, y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )

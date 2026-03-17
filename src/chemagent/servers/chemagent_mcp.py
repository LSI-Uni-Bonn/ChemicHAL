"""chemagent_mcp.py — single consolidated FastMCP server (20 tools).

STANDARD WORKFLOW (data stays on disk — preferred):
    find_datasets()                                          # discover CSVs
    load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
    compute_features(dataset_id, method="ECFP", n_bits=2048)
    split_dataset(dataset_id, train_size=0.7, test_size=0.3, stratified=True)
    job = train_model(split_file_path, algorithm="RFC",
                      task="classification", opt_metric="balanced_accuracy")
    result = check_training(job["job_id"], model_save_path=job["model_save_path"])  # poll every 30 s
    export_predictions(result["model_path"], split_file_path)
    plot_classification_results(predictions_path)

SHORTCUT (load+featurize+split synchronously, then trains in background):
    job = run_pipeline("data/datasets/chembl_activity_data_O00329_P42336.csv",
                       algorithm="RFC", task="classification",
                       featurizer_kwargs={"n_bits": 2048, "radius": 2})
    result = check_training(job["job_id"], model_save_path=job["model_save_path"])  # poll every 30 s

TOOLS
─────────────────────────────────────────────
Dataset
  find_datasets          list CSV files in a directory
  list_loaded_datasets   inspect in-memory state
  list_featurizers       discover available fingerprint methods
  load_dataset           load a CSV for ML
  compute_features       compute molecular fingerprints server-side
  split_dataset          create train/test splits, save .pkl
  dataset_status         inspect a dataset's current load/prepare state

ML
  get_ml_info            algorithms, hyperparameter grids, recommended metrics
  train_model            non-blocking train+tune pipeline from split .pkl
  check_training         poll a background training job
  export_predictions     run inference on a split .pkl, save predictions CSV

Plots
  plot_classification_results confusion matrix, ROC, PR, metric bar, threshold (from predictions CSV)
  plot_regression_results     actual vs predicted, residuals, error distribution (from predictions CSV)
  show_plot                   display a saved PNG directly in the chat UI

XAI
  explain_with_shap      compute per-compound SHAP values from a model + split .pkl
  explain_smiles         compute SHAP values for SMILES strings typed directly in chat (no split file needed)
  plot_shap_mol          render atom-level SHAP heatmaps on molecular structures

Utilities
  log_thought            record reasoning in the session log
  start_new_session      start a fresh session directory
  run_pipeline           non-blocking shortcut: load → featurize → split → train
  generate_report        write a Markdown summary of the current session
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make chemagent packages importable when launched from servers/ via uv run
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2]  # …/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Must set backend before any pyplot import (plot modules import pyplot at module level)
import matplotlib
matplotlib.use("Agg")

from chemagent.datasets.dataset_tools import (
    find_datasets,
    list_loaded_datasets,
    list_featurizers,
    load_dataset,
    compute_features,
    split_dataset,
    dataset_status,
)
from chemagent.ml.ml_model_tools import (
    get_ml_info,
    export_predictions,
    run_pipeline,
)
from chemagent.ml.training_tools import (
    train_model,
    check_training,
)
from chemagent.plots.display import show_plot
from chemagent.plots.plot_tools import plot_classification_results, plot_regression_results
from chemagent.explainability.shap_explainer import explain_with_shap, explain_smiles_with_shap, plot_shap_mol
from chemagent.servers.session_tools import (
    mcp,
    session_logger,
    _register,
    log_thought,
    log_answer,
    generate_report,
    generate_pdf_report,
    start_new_session,
)


# ===========================================================================
# DATASET TOOLS  (functions defined in chemagent.datasets.dataset_tools)
# ===========================================================================

_register(find_datasets)
_register(list_loaded_datasets)
_register(list_featurizers)
_register(load_dataset)
_register(compute_features)
_register(split_dataset)
_register(dataset_status)


# ===========================================================================
# ML MODEL TOOLS  (functions defined in chemagent.ml.ml_model_tools)
# ===========================================================================

_register(get_ml_info)
_register(export_predictions)


# ===========================================================================
# MODEL TRAINING TOOLS  (functions defined in chemagent.ml.training_tools)
# ===========================================================================

_register(train_model)
_register(check_training)


# ===========================================================================
# Plot tools
# ===========================================================================

_register(plot_classification_results)
_register(plot_regression_results)


# ===========================================================================
# Shortcut tool
# ===========================================================================

_register(run_pipeline)

# ===========================================================================
# Inline image display
# ===========================================================================

_register(show_plot)

# ===========================================================================
# Inline image display + XAI / Explainability tools
# ===========================================================================

_register(explain_with_shap)
_register(explain_smiles_with_shap)
_register(plot_shap_mol)


# ===========================================================================
# Session / utility tools
# ===========================================================================

_register(log_thought)
_register(log_answer)
_register(generate_report)
_register(generate_pdf_report)
_register(start_new_session)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")

"""chemagent_mcp.py — single consolidated FastMCP server (27 tools).

STANDARD WORKFLOW (data stays on disk — preferred):
    find_datasets()                                          # discover CSVs
    load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
    compute_features(dataset_id, method="ECFP", n_bits=2048) # generates + saves bit info for MolAnchor
    split_dataset(dataset_id, train_size=0.7, test_size=0.3, stratified=True)
    job = train_model(split_file_path, algorithm="RFC",
                      task="classification", opt_metric="balanced_accuracy")
    result = check_training(job["job_id"], model_save_path=job["model_save_path"])  # poll every 30 s
    export_predictions(result["model_path"], split_file_path)
    plot_classification_results(predictions_path)
    explain_with_molanchor(smiles="CCO", model_path="model.pkl", dataset_id=dataset_id)  # use saved bit info

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
  plot_regression_results        actual vs predicted, residuals, error distribution (from predictions CSV)
  show_plot                      display a saved PNG directly in the chat UI

GNN
  prepare_gnn_dataset     convert split .pkl + SMILES CSV to train/val/test graph datasets
  train_gnn_model_mcp     train a GNN (GCN, GraphSAGE, GAT, etc.) on graph datasets (non-blocking job)
  check_gnn_training      poll a background GNN training job
  load_gnn_model_mcp      load a trained GNN model from disk and validate

XAI
  explain_with_shap              compute per-compound SHAP values from a model + split .pkl
  explain_smiles                 compute SHAP values for SMILES strings typed directly in chat (no split file needed)
  plot_shap_mol                  render atom-level SHAP heatmaps on molecular structures
  explain_with_molanchor         identify molecular anchors (critical fragments) for a single prediction
  identify_recurrent_anchor_rules batch MolAnchor + compute substructure & anchor occurrence metrics
  get_molanchor_info             reference information about MolAnchor parameters and methods
  select_compound_for_xai        randomly select a correctly predicted compound of specified class for analysis
  generate_counterfactuals       generate counterfactual molecules that change the model prediction
  visualize_counterfactuals      draw query compound + counterfactuals as a molecule grid image
  explain_with_molce             contrastive R-group + scaffold attribution — why class A and not class B?
  identify_recurrent_molce_rules global MolCE: aggregate top-3 R-group + scaffold rules across a compound class
  explain_gnn_with_edgeshaper    edge-level Shapley values (explainability) for GNN predictions
  visualize_edgeshaper_results   render edge importance heatmaps on molecular structures
  get_edgeshaper_info            reference information about EdgeSHAPer parameters and methods

Utilities
  log_thought            record reasoning in the session log
  start_new_session      start a fresh session directory
  #run_pipeline           non-blocking shortcut: load → featurize → split → train (not used)
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
    #run_pipeline,
)
from chemagent.ml.training_tools import (
    train_model,
    check_training,
)
from chemagent.ml.gnn_training_tools import (
    prepare_gnn_dataset,
    train_gnn_model_mcp,
    check_gnn_training,
    load_gnn_model_mcp,
)
from chemagent.plots.display import show_plot
from chemagent.plots.plot_tools import plot_classification_results, plot_regression_results
from chemagent.explainability.shap_explainer import explain_with_shap, explain_smiles_with_shap, plot_shap_mol
from chemagent.explainability.molanchor_tools import (
    explain_with_molanchor,
    identify_recurrent_anchor_rules,
    get_molanchor_info,
    select_compound_for_xai,
)
from chemagent.explainability.counterfactual_tools import (
    generate_counterfactuals,
    visualize_counterfactuals,
)
from chemagent.explainability.molce_tools import (
    explain_with_molce,
    identify_recurrent_molce_rules,
)
from chemagent.explainability.edgeshaper_tools import (
    explain_gnn_with_edgeshaper,
    visualize_edgeshaper_results,
    get_edgeshaper_info,
)
from chemagent.servers.session_tools import (
    mcp,
    session_logger,
    _register,
    log_thought,
    log_answer,
    generate_report,
    generate_pdf_report,
    export_chat_html,
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
# GNN Training tools
# ===========================================================================

_register(prepare_gnn_dataset)
_register(train_gnn_model_mcp)
_register(check_gnn_training)
_register(load_gnn_model_mcp)


# ===========================================================================
# Plot tools
# ===========================================================================

_register(plot_classification_results)
_register(plot_regression_results)


# ===========================================================================
# Shortcut tool
# ===========================================================================

#_register(run_pipeline)

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
_register(explain_with_molanchor)
_register(identify_recurrent_anchor_rules)
_register(get_molanchor_info)
_register(select_compound_for_xai)
_register(generate_counterfactuals)
_register(visualize_counterfactuals)
_register(explain_with_molce)
_register(identify_recurrent_molce_rules)
_register(explain_gnn_with_edgeshaper)
_register(visualize_edgeshaper_results)
_register(get_edgeshaper_info)


# ===========================================================================
# Session / utility tools
# ===========================================================================

_register(log_thought)
_register(log_answer)
_register(generate_report)
_register(generate_pdf_report)
_register(export_chat_html)
_register(start_new_session)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")

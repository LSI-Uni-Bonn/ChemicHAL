````instructions
# Copilot Instructions

## Project Overview
An AI agent for **compound selectivity prediction and explainability** using the **Model Context Protocol (MCP)**. The agent exposes cheminformatics and ML capabilities as MCP tools that LLMs (e.g. via LM Studio) can call autonomously to predict selectivity between protein target pairs using ChEMBL data.

## Architecture
All capabilities are consolidated into a **single FastMCP server** (`chemagent_mcp.py`) that runs as a subprocess over `stdio` transport. The server lives at `src/chemagent/servers/chemagent_mcp.py` and is registered in `lm_studio_mcp_config.json` as `"chemagent"`.

### MCP Tools exposed by `chemagent_mcp.py`

| Group | Tools |
|---|---|
| **Dataset** | `list_available_datasets`, `list_loaded_datasets`, `list_featurizers`, `load_dataset`, `get_dataset_smiles`, `featurize_dataset`, `prepare_ml_dataset`, `split_prepared_dataset`, `load_split`, `get_ml_ready_data`, `get_dataset_info` |
| **ML model** | `train_model`, `predict`, `evaluate_classification`, `evaluate_regression`, `get_available_algorithms`, `get_recommended_metrics` |
| **Model builder** | `build_model_from_split_file`, `build_model_from_arrays`, `get_hyperparameter_grids` |
| **Dataset plots** | `plot_class_distribution`, `plot_split_statistics`, `plot_column_distribution`, `plot_class_balance_splits`, `plot_dataset_comparison` |
| **Classification plots** | `plot_confusion_matrix`, `plot_roc_curve`, `plot_pr_curve`, `plot_metric_bar`, `plot_feature_importance`, `plot_threshold_metrics` |
| **Regression plots** | `plot_actual_vs_predicted`, `plot_residuals`, `plot_residual_histogram`, `plot_error_distribution` |

`train_on_split_file` and `predict_from_split_file` are internal helpers used by other tools — they are **not** registered as MCP tools.

XAI/SHAP tools are the **next planned implementation step** (`shap` is already installed). Do not add XAI tools without explicit instruction.

## Preferred Data Flow (server-side, no large array transfer)
Prefer this flow — features stay on disk and are never serialised through the LLM context:
```
list_available_datasets()                                              # discover datasets
load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
  → list_featurizers()                                               # discover methods
  → featurize_dataset(dataset_id, method="ECFP", n_bits=2048)       # server-side, no array transfer
  → split_prepared_dataset(dataset_id, train_size=0.7,
                            val_size=0.0, test_size=0.3,
                            stratified=True)                          # saves .pkl to data/splits/
  → get_available_algorithms() / get_recommended_metrics()           # discover options
  → build_model_from_split_file(split_file_path, algorithm="RFC",
                                 task="classification",
                                 opt_metric="balanced_accuracy")      # tune+train+eval in one call
```

## Explicit Data Flow (features transferred through LLM context)
Use only when a split file is unavailable or features come from an external source:
```
load_dataset → get_dataset_smiles
  → featurize_dataset(dataset_id, method="ECFP", n_bits=2048)  # preferred: server-side
    OR prepare_ml_dataset(dataset_id, external_features)        # fallback: external features
  → get_ml_ready_data(dataset_id)                               # retrieve arrays if needed
  → build_model_from_arrays(train_features, train_labels,
                             test_features, test_labels,
                             algorithm="RFC", task="classification")
    OR train_model(features, labels, "RFC", "classification")
       → predict(model_path, features)
       → evaluate_classification(labels, predictions, probabilities)
```

## Running the MCP Server
The primary MCP host is **LM Studio** (config: `lm_studio_mcp_config.json`). Other clients (VS Code, Claude Desktop) may be supported — the config structure is standard MCP and portable.

The server is launched via `uv run` with `--directory` pointing to `src/chemagent/servers`:
```json
{
  "command": "uv",
  "args": ["--directory", "<workspace>/src/chemagent/servers", "run", "chemagent_mcp.py"]
}
```
The server ends with `if __name__ == "__main__": mcp.run(transport="stdio")`.

**`lm_studio_mcp_config.json` contains hardcoded absolute paths** (`C:/Users/janela/...`). Update `--directory` when working on a different machine.

The `--directory` flag ensures `src/` is on the Python path so `from chemagent.datasets import ...` resolves correctly. There is no nested `pyproject.toml` — only one at the workspace root.

## Package Structure
```
src/chemagent/
  datasets/         # loader.py, featurizer.py, splitter.py, io.py
  featurization/    # fingerprints.py (ECFP, MACCS, …), utils.py
  ml/               # models.py, training.py, evaluation.py,
                    # hyperparameter_tuning.py, cross_validation.py, metrics.py
  plots/            # classification.py, dataset.py, regression.py, utils.py
  splitting/        # random_split.py, scaffold_split.py, statistics.py, utils.py
  servers/          # chemagent_mcp.py  ← single MCP server
```

## Environment & Tooling
- **Package manager**: `uv` (not pip). Add deps with `uv add <package>`, defined in `pyproject.toml`.
- **Python**: 3.12+, venv at `.venv/`.
- **Key dependencies**: `scikit-learn`, `rdkit`, `numpy`, `mcp[cli]`, `shap`, `torch`, `seaborn`.
- **Debugging**: `notebooks/debugging_chemagent.ipynb` (Jupyter).

## Key Conventions
- **Workspace root resolution**: Resolved at runtime with `Path(__file__).resolve().parents[3]` (4 levels up from `src/chemagent/servers/`). The `src/` directory is inserted into `sys.path` via `parents[2]`. Adjust offsets if the directory structure changes.
- **In-memory state**: `chemagent_mcp.py` caches loaded DataFrames in `_loaded_datasets` and processed datasets in `_processed_datasets`. State is **ephemeral** — lost on server restart. Re-run `load_dataset()` + `featurize_dataset()` after restarts.
- **Model persistence**: Trained models saved as `.pkl` via `joblib`. Default path: `data/models/<split_stem>_<ALGO>.pkl` (from `build_model_from_split_file`) or `data/models/trained_model_<ALGO>.pkl` (from `train_model`).
- **Dataset naming**: `data/datasets/chembl_activity_data_{UniProtA}_{UniProtB}.csv`. Columns: `smiles`, `class_label` (0/1), `pPot_diff`, `target_pair`, `cid`.
- **Split persistence**: Saved as `.pkl` under `data/splits/` (keys: `train_features`, `train_labels`, `val_*`, `test_*`). Pass `val_size=0.0` for a two-way train/test split.
- **`task` / `reg_class` parameter**: Use `"classification"` as the default. `"classification-cw"` (class-weighted) is available but its utility for these datasets is under review — avoid unless explicitly needed. `"regression"` is for `pPot_diff` prediction tasks.
- **Featurizer extensibility**: Any public `UpperCase` function added to `chemagent/featurization/fingerprints.py` is automatically available as a `method` in `featurize_dataset()` and appears in `list_featurizers()`. Parameters (`n_bits`, `radius`, etc.) are forwarded via `inspect.signature` — only kwargs the function accepts are passed.
- **ML extensibility**: All estimator factories and hyperparameter grids live in `chemagent/ml/models.py` as `build_estimator`, `PARAM_GRIDS`, and `MODEL_INFO` — that is the single source of truth. `HYPERPARAMETERS` in `hyperparameter_tuning.py` is imported from `models.py`.
- **Core ML classes**: `MLModel` (GridSearchCV training) in `chemagent.ml.training`; `Model_Evaluation` (metrics) in `chemagent.ml.evaluation`. Both are exported from `chemagent.ml`.

## Dataset Target Pairs
- `O00329` = PI3Kδ, `P42336` = PI3Kα, `P48736` = PI3Kγ
- Three pairwise datasets cover all three combinations.

````

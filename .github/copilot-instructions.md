````instructions
# Copilot Instructions

## Project Overview
An AI agent for **compound selectivity prediction and explainability** using the **Model Context Protocol (MCP)**. The agent exposes cheminformatics and ML capabilities as MCP tools that LLMs (e.g. via LM Studio) can call autonomously to predict selectivity between protein target pairs using ChEMBL data.

## Architecture
All capabilities are consolidated into a **single FastMCP server** (`chemagent_mcp.py`) that runs as a subprocess over `stdio` transport. The server lives at `src/chemagent/servers/chemagent_mcp.py` and is registered in `lm_studio_mcp_config.json` as `"chemagent"`.

### MCP Tools exposed by `chemagent_mcp.py` (16 total)

| Group | Tools |
|---|---|
| **Dataset** | `find_datasets`, `list_loaded_datasets`, `list_featurizers`, `load_dataset`, `compute_features`, `split_dataset`, `dataset_status` |
| **ML** | `get_ml_info`, `train_model`, `check_training`, `export_predictions` |
| **Plots** | `plot_classification_results`, `plot_regression_results` |
| **Utility** | `log_thought`, `start_new_session`, `run_pipeline` |

The following are **internal Python helpers** — they exist in the server but are **not** registered as MCP tools:
`train_on_split_file`, `predict_from_split_file`, `build_model_from_split_file`, `build_model_from_arrays`, `get_dataset_smiles`, `prepare_ml_dataset`, `load_split`, `get_ml_ready_data`, `_predict`, `evaluate_classification`, `evaluate_regression`

XAI/SHAP tools are the **next planned implementation step** (`shap` is already installed). Do not add XAI tools without explicit instruction.

## Preferred Data Flow (data stays on disk)
Features are never serialised through the LLM context:
```
find_datasets()                                                  # discover CSVs
load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
  → list_featurizers()                                          # discover methods
  → compute_features(dataset_id, method="ECFP", n_bits=2048)   # server-side, no array transfer
  → split_dataset(dataset_id, train_size=0.7,
                  val_size=0.0, test_size=0.3, stratified=True) # saves .pkl to session splits/
  → get_ml_info()                                               # discover algorithms + metrics
  → train_model(split_file_path, algorithm="RFC",
                task="classification",
                opt_metric="balanced_accuracy")                  # non-blocking, returns job_id + model_save_path
  → check_training(job_id, model_save_path=model_save_path)     # poll every 60 s; pass model_save_path for disk fallback
  → export_predictions(model_path, split_file_path)             # optional: export CSV
  → plot_classification_results(model_path, split_file_path)    # all plots in one call
```

## Shortcut (single-call experiments)
```
job = run_pipeline("data/datasets/chembl_activity_data_O00329_P42336.csv",
                   algorithm="RFC", task="classification")
result = check_training(job["job_id"], model_save_path=job["model_save_path"])   # poll every 60 s
```
Steps 1–3 (load → featurize → split) run synchronously; training is submitted as a background job. Returns `job_id` immediately — poll with `check_training()` like `train_model`. Use for quick experiments; use individual steps when intermediate outputs are needed.

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
- **In-memory state**: `chemagent_mcp.py` caches loaded DataFrames in `_loaded_datasets` and processed datasets in `_processed_datasets`. State is **ephemeral** — lost on server restart. Re-run `load_dataset()` + `compute_features()` after restarts.
- **Model persistence**: Trained models saved as `.pkl` via `joblib`. Default path: `<session_dir>/models/<split_stem>_<ALGO>.pkl` (from `train_model`).
- **Dataset naming**: `data/datasets/chembl_activity_data_{UniProtA}_{UniProtB}.csv`. Columns: `smiles`, `class_label` (0/1), `pPot_diff`, `target_pair`, `cid`.
- **Split persistence**: Saved as `.pkl` under `<session_dir>/splits/` (keys: `train_features`, `train_labels`, `val_*`, `test_*`). Pass `val_size=0.0` for a two-way train/test split.
- **`task` parameter**: Use `"classification"` as the default. `"classification-cw"` (class-weighted) is available but its utility for these datasets is under review — avoid unless explicitly needed. `"regression"` is for `pPot_diff` prediction tasks.
- **Featurizer extensibility**: Any public `UpperCase` function added to `chemagent/featurization/fingerprints.py` is automatically available as a `method` in `compute_features()` and appears in `list_featurizers()`. Parameters (`n_bits`, `radius`, etc.) are forwarded via `inspect.signature` — only kwargs the function accepts are passed.
- **ML extensibility**: All estimator factories and hyperparameter grids live in `chemagent/ml/models.py` as `build_estimator`, `PARAM_GRIDS`, and `MODEL_INFO` — that is the single source of truth. `HYPERPARAMETERS` in `hyperparameter_tuning.py` is imported from `models.py`.
- **Core ML classes**: `MLModel` (GridSearchCV training) in `chemagent.ml.training`; `Model_Evaluation` (metrics) in `chemagent.ml.evaluation`. Both are exported from `chemagent.ml`.
- **Plot tools**: The two plot tools (`plot_classification_results`, `plot_regression_results`) load model and split data from disk themselves — never pass raw arrays to them. They accept a `plots` list (or `["all"]`) to select which figures to generate.
- **Background training**: `train_model` is non-blocking and returns a `job_id`. Always poll with `check_training(job_id)` until `status` is `"completed"` or `"failed"`. The full pipeline result is in `check_training(...)[\"result\"]` when done.

## Dataset Target Pairs
- `O00329` = PI3Kδ, `P42336` = PI3Kα, `P48736` = PI3Kγ
- Three pairwise datasets cover all three combinations.

````

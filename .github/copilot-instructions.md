# Copilot Instructions

## Project Overview
An AI agent for **compound selectivity prediction and explainability** using the **Model Context Protocol (MCP)**. The agent exposes cheminformatics and ML capabilities as MCP tools that LLMs (e.g. via LM Studio) can call autonomously to predict selectivity between protein target pairs using ChEMBL data.

## Architecture
Each capability is a standalone **FastMCP server** that runs as a subprocess over `stdio` transport. Servers live under `src/chemagent/servers/` and are registered in `lm_studio_mcp_config.json`:

| MCP Server | Entry point | Key tools |
|---|---|---|
| `dataset-loader` | `dataset_loader_mcp.py` | `list_available_datasets`, `list_featurizers`, `list_loaded_datasets`, `load_dataset`, `get_dataset_smiles`, `featurize_dataset`, `prepare_ml_dataset`, `get_ml_ready_data`, `split_prepared_dataset`, `load_split`, `get_dataset_info` |
| `mol-featurization` | `mol_featurization.py` | `ECFP`, `MACCS` |
| `data-split` | `data_split.py` | `random_split`, `scaffold_split`, `get_split_statistics` |
| `ml-models` | `ml_models_mcp.py` | `train_model`, `train_on_split_file`, `predict`, `predict_from_split_file`, `evaluate_classification`, `evaluate_regression`, `get_available_algorithms`, `get_recommended_metrics` |

The `explainable_ai/` folder is currently empty — XAI/SHAP tools are the **next planned implementation step**. Do not add anything there yet without explicit instruction.

## Preferred Data Flow (server-side, no large array transfer)
Prefer this flow — features stay on the server and are never serialised through the LLM context:
```
list_available_datasets()                                          # discover datasets
load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
  → list_featurizers()                                           # discover methods
  → featurize_dataset(dataset_id, method="ECFP", n_bits=2048)   # server-side, no array transfer
  → split_prepared_dataset(dataset_id)                           # saves .pkl to data/splits/
  → get_available_algorithms() / get_recommended_metrics()       # discover options
  → train_on_split_file(split_file_path, "RFC", "classification")
  → predict_from_split_file(model_path, split_file_path)
  → evaluate_classification(labels, predictions, probabilities)
```

## Explicit Data Flow (features transferred through LLM context)
Use only when the preferred flow cannot be applied (e.g. custom external features):
```
load_dataset → get_dataset_smiles
  → ECFP(smiles, n_bits=2048, radius=2)          # mol-featurization server
  → prepare_ml_dataset(dataset_id, features)
  → get_ml_ready_data(dataset_id)
  → train_model(features, labels, "RFC", "classification")
  → predict(model_path, features) / evaluate_classification(labels, predictions)
```

## Running MCP Servers
The primary MCP host is **LM Studio** (config: `lm_studio_mcp_config.json`). Other clients (VS Code, Claude Desktop) may be supported in the future — the config structure is standard MCP and portable.

All servers are launched via `uv run` with `--directory` pointing to `src/chemagent/servers`:
```json
{ "command": "uv", "args": ["--directory", "<workspace>/src/chemagent/servers", "run", "ml_models_mcp.py"] }
```
Each `*_mcp.py` ends with `if __name__ == "__main__": mcp.run(transport="stdio")`.

**`lm_studio_mcp_config.json` contains hardcoded absolute paths** (`C:/Users/janela/...`). Update all `--directory` values when working on a different machine.

The `--directory` flag sets the working directory so Python resolves sibling imports (e.g. `import mol_featurization`) relative to `servers/`. There is no nested `pyproject.toml` — only one exists at the workspace root.

The standalone `data-split` server (`data_split.py`) is still registered but largely superseded by `split_prepared_dataset` inside `dataset-loader` for the typical workflow.

## Environment & Tooling
- **Package manager**: `uv` (not pip). Add deps with `uv add <package>`, defined in `pyproject.toml`.
- **Python**: 3.12+, venv at `.venv/`.
- **Debugging**: `debugging_new.ipynb` (Jupyter) and `debugging.py` (Marimo) — both in `notebooks/`. Run Marimo with: `marimo edit notebooks/debugging.py`

## Key Conventions
- **Workspace root resolution**: MCP servers resolve the workspace root at runtime with `Path(__file__).resolve().parents[3]` (4 levels up from `src/chemagent/servers/`). Adjust this offset if the directory structure changes.
- **In-memory state**: `dataset_loader_mcp.py` caches loaded DataFrames in `_loaded_datasets` and processed datasets in `_processed_datasets`. State lives only for the lifetime of the MCP server process.
- **Model persistence**: Trained models are saved as `.pkl` via `joblib`. Default path: `data/models/trained_model_<ALGO>.pkl`.
- **Dataset naming**: `data/datasets/chembl_activity_data_{UniProtA}_{UniProtB}.csv`. Columns: `smiles`, `class_label` (0/1), `pPot_diff`, `target_pair`, `cid`.
- **Split persistence**: Pre-computed splits stored as `.pkl` under `data/splits/` (keys: `train_features`, `train_labels`, `val_*`, `test_*`).
- **`reg_class` parameter**: Pass consistently across `train_model`, `predict`, and `evaluate_*`. Use `"classification"` as the default. `"classification-cw"` (class-weighted) is available but its utility for these datasets is under review — avoid using it unless explicitly needed. `"regression"` is for `pPot_diff` prediction tasks.
- **Featurizer extensibility**: Any public `UpperCase` function added to `mol_featurization.py` is automatically available as a `method` in `featurize_dataset()` and appears in `list_featurizers()` — no changes needed elsewhere. Parameters (`n_bits`, `radius`, etc.) are forwarded via `inspect.signature` — only kwargs the function accepts are passed.
- **Core ML classes** (`machine_learning_models_sk.py`): `MLModel` (GridSearchCV training) and `Model_Evaluation` (metrics). These are non-MCP Python classes imported by `ml_models_mcp.py`.

## Dataset Target Pairs
- `O00329` = PI3Kδ, `P42336` = PI3Kα, `P48736` = PI3Kγ
- Three pairwise datasets cover all three combinations.

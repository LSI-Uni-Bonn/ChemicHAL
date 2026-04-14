
<p align="center">
  <img src="docs/assets/chemichal_logo.svg" alt="ChemicHAL logo" width=30%>
</p>

# ChemicHAL: an XAI-enhanced agent for chemoinformatics


## Overview

ChemicHAL is an **LLM-driven agent** for chemoinformatics tasks, powered by the **Model Context Protocol (MCP)** and designed for **explainable AI (XAI)**. It enables autonomous compound selectivity prediction, molecular property modeling, and interpretability through SHAP, MolAnchor, MolCE, and EdgeSHAPer.

## Prerequisites

- Python 3.12+
- `uv` package manager
- RDKit
- PyTorch
- LM Studio (for MCP integration)

## Installation

### 1. Clone and Set Up

```bash
git clone <repository-url>
cd AI-Agent-for-Compound-Prediction-and-Explainability
uv sync
```

### 2. Install LM Studio

Download **LM Studio** from [lmstudio.ai](https://lmstudio.ai) and install it on your system.

### 2. Install LM Studio

Download **LM Studio** from [lmstudio.ai](https://lmstudio.ai) and install it on your system.

> **Note:** LM Studio has been tested as the primary MCP host. Other MCP-compatible interfaces (Ollama, Claude Desktop, etc.) may also be explored for alternative workflows.

### 3. Import MCP Server

1. Open LM Studio and navigate to **Settings** → **MCP Servers**
2. Add a new MCP server with the following configuration:
    ```json
    {
      "name": "chemagent",
      "command": "uv",
      "args": [
         "--directory",
         "<workspace-root>/src/chemagent/servers",
         "run",
         "chemagent_mcp.py"
      ]
    }
    ```
3. Update the `--directory` path to match your workspace root
4. Save and restart LM Studio

## Agent Capabilities

🧪 **Dataset Management** – Discover, load, and preprocess ChEMBL datasets; compute molecular fingerprints (ECFP, MACCS, etc.)

🤖 **ML Modeling** – Train classification and regression models (Random Forest, SVM, XGBoost, etc.) with hyperparameter tuning

🧠 **Graph Neural Networks** – Prepare GNN datasets and train GCN/GAT models for end-to-end molecular learning

📊 **Visualization** – Generate classification/regression plots and model performance summaries

🔍 **Explainability (SHAP)** – Feature importance and decision boundary analysis via SHAP

🧬 **Molecular Anchors** – Identify recurrent structural motifs driving predictions with MolAnchor

🔄 **Counterfactuals** – Generate and visualize molecular modifications for explainability with MolCE

🌐 **GNN Explainability** – Explain graph predictions using EdgeSHAPer to highlight important bonds

📝 **Reporting** – Export results as HTML reports and PDF documents

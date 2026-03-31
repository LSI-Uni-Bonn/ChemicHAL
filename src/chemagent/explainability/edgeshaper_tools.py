"""chemagent.explainability.edgeshaper_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for EdgeSHAPer (edge-level Shapley values) explainability.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
explain_gnn_with_edgeshaper    — compute edge importance scores for a GNN model on a test compound
visualize_edgeshaper_results   — render atom/edge-level SHAP heatmaps on molecular structures

The EdgeSHAPer algorithm computes Shapley value approximations for edge importance
in Graph Neural Networks, enabling edge-level explainability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union
import json
from io import BytesIO

import torch
import numpy as np
import joblib
from rdkit import Chem
from mcp.server.fastmcp import Image as MCPImage

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.explainability.edgeshaper import Edgeshaper
from chemagent.session_utils import get_session_logger as _get_session_logger


def explain_gnn_with_edgeshaper(
    model_path: str,
    graph_data_path: str,
    compound_idx: int = 0,
    M: int = 100,
    target_class: int = 0,
    batch_size: int = 100,
    deviation: Optional[float] = None,
    log_odds: bool = False,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_results: bool = True,
) -> dict[str, Any]:
    """Compute edge importance for a GNN model prediction using EdgeSHAPer (batched).

    Loads a pre-trained GNN model and graph dataset, then computes Shapley value
    approximations for edge importance to explain the model's prediction on a
    specific compound using vectorized batch processing (faster than sequential).

    Parameters
    ----------
    model_path : str
        Path to saved GNN model (.pt file)
    graph_data_path : str
        Path to PyTorch Geometric graph data (.pt file with x, edge_index, etc.)
    compound_idx : int, optional
        Index of the compound in the graph dataset to explain (default: 0)
    M : int, optional
        Number of Monte Carlo sampling steps (default: 100)
    target_class : int, optional
        Class index for classification (default: 0); None for regression
    batch_size : int, optional
        Batch size for vectorized computation; M must be divisible by batch_size
        (will auto-adjust if needed). Higher values = faster but more memory usage (default: 100)
    deviation : float, optional
        **Note:** not supported in batched mode; parameter ignored if provided.
        For early stopping, use sequential `explain()` instead via direct API.
    log_odds : bool, optional
        If True, use log odds instead of softmax probabilities (default: False)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    device : str, optional
        Device to use ('cuda' or 'cpu'); auto-detects CUDA availability
    save_results : bool, optional
        If True, saves results to session directory (default: True)

    Returns
    -------
    dict with:
        - job_id: unique identifier for this explanation
        - status: "completed" or "failed"
        - phi_edges: list of Shapley values (one per edge)
        - pertinent_positive_set: minimal edge subset preserving prediction class
        - minimal_top_k_set: minimal edge subset changing prediction
        - infidelity: fidelity- metric (from pertinent_positive_set)
        - fidelity: fidelity+ metric (from minimal_top_k_set)
        - trustworthiness: harmonic mean of fidelity and (1 - infidelity)
        - num_edges: total number of edges in the graph
        - original_pred_prob: model's predicted probability for target class
        - result_save_path: path to saved results JSON (if save_results=True)
        - error: error message if status is "failed"

    Examples
    --------
    >>> result = explain_gnn_with_edgeshaper(
    ...     model_path="models/gnn_model.pt",
    ...     graph_data_path="data/gnn_dataset/processed/val_processed.pt",
    ...     compound_idx=0,
    ...     M=100,
    ...     batch_size=50,
    ...     target_class=0
    ... )
    >>> if result["status"] == "completed":
    ...     print(f"Found {len(result['phi_edges'])} edges")
    ...     print(f"Trustworthiness: {result['trustworthiness']:.3f}")
    """
    session_logger = _get_session_logger()

    try:
        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = torch.load(model_path, map_location=device)
        model.eval()

        # Load graph data
        graph_data_path = Path(graph_data_path)
        if not graph_data_path.exists():
            raise FileNotFoundError(f"Graph data not found: {graph_data_path}")
        
        graph_data = torch.load(graph_data_path, map_location=device)
        
        # Extract graph components
        if hasattr(graph_data, "x"):
            x = graph_data.x if isinstance(graph_data.x, torch.Tensor) else graph_data.x[compound_idx]
        else:
            raise ValueError("Graph data has no node features (x)")

        if hasattr(graph_data, "edge_index"):
            edge_index = graph_data.edge_index if isinstance(graph_data.edge_index, torch.Tensor) else graph_data.edge_index[compound_idx]
        else:
            raise ValueError("Graph data has no edge_index")

        edge_weight = None
        if hasattr(graph_data, "edge_weight"):
            edge_weight = graph_data.edge_weight if isinstance(graph_data.edge_weight, torch.Tensor) else graph_data.edge_weight[compound_idx]

        # Ensure tensors are on the right device
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Run EdgeSHAPer (batched for speed)
        explainer = Edgeshaper(model, x, edge_index, edge_weight=edge_weight, device=device)
        
        if deviation is not None:
            session_logger.log_event(
                "edgeshaper_deviation_ignored",
                reason="deviation parameter not supported in batched mode; using full M iterations",
            )
        
        phi_edges = explainer.explain_batch(
            M=M,
            target_class=target_class,
            P=None,
            deviation=None,  # Batched version does not support deviation
            log_odds=log_odds,
            seed=seed,
            batch_size=batch_size,
            progress_bar=True,
        )

        # Compute original prediction probability
        explainer.compute_original_predicted_probability()
        original_pred_prob = explainer.original_pred_prob

        # Compute fidelity metrics (classification only)
        pertinent_positive_set = None
        minimal_top_k_set = None
        infidelity = None
        fidelity = None
        trustworthiness = None

        if target_class is not None:
            _, infidelity = explainer.compute_pertinent_positive_set(verbose=False)
            pertinent_positive_set = explainer.pertinent_positive_set.cpu().tolist() if explainer.pertinent_positive_set is not None else None

            _, fidelity = explainer.compute_minimal_top_k_set(verbose=False)
            minimal_top_k_set = explainer.minimal_top_k_set.cpu().tolist() if explainer.minimal_top_k_set is not None else None

            trustworthiness = explainer.compute_trustworthiness(verbose=False)

        result = {
            "job_id": f"edgeshaper_{int(np.random.random() * 1e9)}",
            "status": "completed",
            "phi_edges": [float(v) for v in phi_edges],
            "pertinent_positive_set": pertinent_positive_set,
            "minimal_top_k_set": minimal_top_k_set,
            "infidelity": float(infidelity) if infidelity is not None else None,
            "fidelity": float(fidelity) if fidelity is not None else None,
            "trustworthiness": float(trustworthiness) if trustworthiness is not None else None,
            "num_edges": edge_index.shape[1],
            "original_pred_prob": float(original_pred_prob) if original_pred_prob is not None else None,
            "compound_idx": compound_idx,
        }

        # Save results if requested
        if save_results:
            ws_root = Path(__file__).resolve().parents[3]
            session_dir = ws_root / "session"
            results_dir = session_dir / "edgeshaper_results"
            results_dir.mkdir(parents=True, exist_ok=True)

            result_file = results_dir / f"edgeshaper_{result['job_id']}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            result["result_save_path"] = str(result_file)

        session_logger.log_event(
            "edgeshaper_completed",
            model_path=str(model_path),
            num_edges=result["num_edges"],
            M=M,
            compound_idx=compound_idx,
        )

        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        session_logger.log_event("edgeshaper_failed", error=error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "job_id": f"edgeshaper_failed_{int(np.random.random() * 1e9)}",
        }


def visualize_edgeshaper_results(
    smiles: str,
    phi_edges_json: str,
    edge_index_json: str,
    save_results: bool = True,
) -> dict[str, Any]:
    """Visualize EdgeSHAPer explanations as molecular heatmaps.

    Creates RDKit-based heatmap image(s) showing edge importance on the molecular
    structure. Returns image(s) for inline display in chat.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to visualize
    phi_edges_json : str
        JSON string containing list of Shapley values (one per edge)
    edge_index_json : str
        JSON string containing edge indices as [source_nodes, target_nodes]
    save_results : bool, optional
        If True, saves PNG files to session directory (default: True)

    Returns
    -------
    dict with:
        - status: "completed" or "failed"
        - images: list of MCPImage objects for inline display
        - image_paths: list of filesystem paths to saved PNG files (if save_results=True)
        - error: error message if status is "failed"
        - num_edges: total edges in the GNN graph
        - num_bonds: total bonds in the molecule

    Examples
    --------
    >>> result = visualize_edgeshaper_results(
    ...     smiles="CCO",
    ...     phi_edges_json='[0.1, -0.05, 0.2]',
    ...     edge_index_json='[[0, 1, 2], [1, 2, 0]]'
    ... )
    >>> if result["status"] == "completed":
    ...     for img in result["images"]:
    ...         display(img)  # Or use MCP image display
    """
    try:
        # Parse JSON inputs
        phi_edges = json.loads(phi_edges_json)
        edge_index_list = json.loads(edge_index_json)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Prepare molecule for drawing
        from rdkit.Chem import Draw
        mol_prep = Draw.PrepareMolForDrawing(mol)
        num_bonds = len(mol_prep.GetBonds())
        num_edges = edge_index.shape[1]

        # Map graph edges to molecular bonds
        rdkit_bonds = {}
        for i in range(num_bonds):
            bond = mol_prep.GetBondWithIdx(i)
            init_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            rdkit_bonds[(init_atom, end_atom)] = i
            rdkit_bonds[(end_atom, init_atom)] = i  # Both directions

        # Aggregate edge importance to bonds
        rdkit_bonds_phi = [0.0] * num_bonds
        for i in range(len(phi_edges)):
            phi_value = phi_edges[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()

            if (init_atom, end_atom) in rdkit_bonds:
                bond_idx = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_idx] += phi_value

        # Generate heatmap visualization
        try:
            from chemagent.explainability.edgeshaper_viz_utils.molmapping import mapvalues2mol
            from chemagent.explainability.edgeshaper_viz_utils.utils import transform2png
            import matplotlib.pyplot as plt

            plt.clf()
            canvas = mapvalues2mol(
                mol_prep,
                None,
                rdkit_bonds_phi,
                atom_width=0.2,
                bond_length=0.5,
                bond_width=0.5,
            )
            img_pil = transform2png(canvas.GetDrawingText())
            plt.clf()

            # Convert PIL image to bytes for MCP
            img_bytes = BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            mcp_image = MCPImage(data=img_bytes.getvalue(), format="png")

            result = {
                "status": "completed",
                "images": [mcp_image],
                "num_edges": num_edges,
                "num_bonds": num_bonds,
            }

            # Save PNG to session if requested
            if save_results:
                ws_root = Path(__file__).resolve().parents[3]
                session_dir = ws_root / "session"
                viz_dir = session_dir / "edgeshaper_visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)

                import time
                timestamp = int(time.time() * 1000)
                viz_path = viz_dir / f"edgeshaper_heatmap_{timestamp}.png"
                img_pil.save(viz_path, dpi=(300, 300))
                result["image_paths"] = [str(viz_path)]

            return result

        except ImportError:
            result = {
                "status": "completed",
                "images": [],
                "num_edges": num_edges,
                "num_bonds": num_bonds,
                "note": "EdgeSHAPer visualization dependencies are not available; heatmap visualization skipped. Ensure edgeshaper_viz_utils and its visualization dependencies are installed.",
            }
            return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        return {
            "status": "failed",
            "error": error_msg,
            "images": [],
        }


def get_edgeshaper_info() -> dict[str, Any]:
    """Return reference information about EdgeSHAPer parameters.

    Call once to understand available options and recommended defaults.

    Returns:
        dict with:
            - description: overview of EdgeSHAPer methodology
            - parameters: explanation of key parameters
            - recommended_M: recommended number of Monte Carlo samples
            - devices: available compute devices
    """
    return {
        "description": (
            "EdgeSHAPer computes Shapley value approximations for edge importance "
            "in Graph Neural Networks, enabling fine-grained understanding of which "
            "molecular bonds/connectivity patterns drive predictions."
        ),
        "parameters": {
            "M": {
                "description": "Number of Monte Carlo sampling steps",
                "default": 100,
                "range": "1–500+ (higher = more accurate but slower)",
                "impact": "Accuracy of Shapley value approximation",
            },
            "target_class": {
                "description": "Class index to explain (for classification)",
                "default": 0,
                "note": "Set to None for regression models",
            },
            "deviation": {
                "description": "Early stopping threshold: stops when deviation from true value is ≤ threshold",
                "default": None,
                "note": "If None, uses full M iterations; if set, may terminate early for speed",
            },
            "log_odds": {
                "description": "Use log odds instead of softmax probabilities",
                "default": False,
            },
            "seed": {
                "description": "Random seed for reproducibility",
                "default": 42,
            },
        },
        "metrics": {
            "fidelity": "Fidelity+ — how much removing top-k edges degrades the prediction",
            "infidelity": "Fidelity- — how much keeping only top-k edges preserves the prediction",
            "trustworthiness": "Harmonic mean of fidelity and (1 - infidelity); ranges [0, 1]",
        },
        "recommended_M": {
            "quick": {"M": 50, "description": "Fast exploration; lower accuracy"},
            "standard": {"M": 100, "description": "Balanced speed and accuracy (default)"},
            "thorough": {"M": 200, "description": "High accuracy; slower"},
        },
        "devices": ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
    }

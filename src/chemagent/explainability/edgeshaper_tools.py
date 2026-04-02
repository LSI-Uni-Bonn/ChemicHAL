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

import torch
import numpy as np
import joblib
from rdkit import Chem

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.explainability.edgeshaper import Edgeshaper
from chemagent.session_utils import get_session_logger as _get_session_logger
from chemagent.ml.gnn_models import GCN, GraphSAGE, GAT, GC_GNN, GINE, GIN


def explain_gnn_with_edgeshaper(
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    graph_data_path: Optional[str] = None,
    model_class_name: Literal["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"] = "GCN",
    node_features_dim: int = 4,
    hidden_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
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
    model : torch.nn.Module, optional
        Loaded GNN model object. Must be a full model instance (not a state_dict)
        so that ``model.eval()`` and forward inference can be executed.
        If provided, this takes precedence over ``model_path``.
    model_path : str, optional
        Path to a serialized full GNN model object (.pt). Used only when
        ``model`` is not provided.
    graph_data_path : str
        Path to PyTorch Geometric graph data (.pt file with x, edge_index, etc.)
    model_class_name : str, optional
        Model architecture to use when reconstructing from a state_dict (default: "GCN").
    node_features_dim : int, optional
        Node feature dimension for model reconstruction (default: 4).
    hidden_channels : int, optional
        Hidden channels for model reconstruction. If omitted, inferred from state_dict.
    num_classes : int, optional
        Number of output classes for model reconstruction. If omitted, inferred from state_dict.
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
    Use an already loaded model object:

    >>> loaded_model = load_gnn_model(...)
    >>> result = explain_gnn_with_edgeshaper(
    ...     model=loaded_model,
    ...     graph_data_path="data/gnn_dataset/processed/val_processed.pt",
    ...     compound_idx=0,
    ...     M=100,
    ...     batch_size=50,
    ...     target_class=0
    ... )

    Or load a full serialized model from disk:

    >>> result = explain_gnn_with_edgeshaper(
    ...     model_path="models/gnn_full_model.pt",
    ...     graph_data_path="data/gnn_dataset/processed/val_processed.pt",
    ... )

    Or load from a state_dict checkpoint by providing/reusing architecture params:

    >>> result = explain_gnn_with_edgeshaper(
    ...     model_path="models/gnn_state_dict.pt",
    ...     model_class_name="GCN",
    ...     hidden_channels=64,
    ...     num_classes=2,
    ...     graph_data_path="data/gnn_dataset/processed/val_processed.pt",
    ... )
    >>> if result["status"] == "completed":
    ...     print(f"Found {len(result['phi_edges'])} edges")
    ...     print(f"Trustworthiness: {result['trustworthiness']:.3f}")
    """
    session_logger = _get_session_logger()

    try:
        if not graph_data_path:
            raise ValueError("'graph_data_path' is required.")

        # Resolve model either from a loaded object or from model_path.
        if model is None:
            if not model_path:
                raise ValueError("Provide either 'model' or 'model_path'.")
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")
            loaded_obj = torch.load(model_file, map_location=device, weights_only=False)
            model = loaded_obj

            # If checkpoint is state_dict-like, reconstruct the architecture.
            if isinstance(loaded_obj, dict):
                model_map = {
                    "GCN": GCN,
                    "GraphSAGE": GraphSAGE,
                    "GAT": GAT,
                    "GC_GNN": GC_GNN,
                    "GINE": GINE,
                    "GIN": GIN,
                }
                if model_class_name not in model_map:
                    raise ValueError(
                        f"Unknown model_class_name '{model_class_name}'. "
                        f"Available: {list(model_map.keys())}"
                    )

                inferred_hidden = hidden_channels
                if inferred_hidden is None:
                    for k in ("conv1.bias", "conv1.lin.weight", "conv1.lin_l.weight", "conv1.att_src"):
                        if k in loaded_obj and hasattr(loaded_obj[k], "shape"):
                            inferred_hidden = int(loaded_obj[k].shape[0])
                            break
                if inferred_hidden is None:
                    raise ValueError(
                        "Could not infer hidden_channels from state_dict. "
                        "Pass hidden_channels explicitly."
                    )

                inferred_classes = num_classes
                if inferred_classes is None and "lin.weight" in loaded_obj:
                    inferred_classes = int(loaded_obj["lin.weight"].shape[0])
                if inferred_classes is None:
                    raise ValueError(
                        "Could not infer num_classes from state_dict. "
                        "Pass num_classes explicitly."
                    )

                model = model_map[model_class_name](
                    node_features_dim=node_features_dim,
                    hidden_channels=inferred_hidden,
                    num_classes=inferred_classes,
                )
                model.load_state_dict(loaded_obj)

        # Validate model object: must be a full torch.nn.Module.
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "'model' must be a loaded torch.nn.Module instance with .eval(). "
                f"Got {type(model)!r}."
            )
        model = model.to(device)
        model.eval()

        # Load graph data
        graph_data_path = Path(graph_data_path)
        if not graph_data_path.exists():
            raise FileNotFoundError(f"Graph data not found: {graph_data_path}")
        
        graph_data = torch.load(graph_data_path, map_location=device, weights_only=False)

        # Support both single-graph objects and PyG InMemoryDataset tuples: (data, slices).
        if isinstance(graph_data, tuple) and len(graph_data) == 2:
            data_obj, slices = graph_data
            if "x" not in slices or "edge_index" not in slices:
                raise ValueError("Collated graph data is missing required slices for 'x' or 'edge_index'.")

            x_slices = slices["x"]
            e_slices = slices["edge_index"]

            n_graphs = int(x_slices.numel() - 1)
            if compound_idx < 0 or compound_idx >= n_graphs:
                raise IndexError(
                    f"compound_idx {compound_idx} out of range for dataset of size {n_graphs}."
                )

            x_start = int(x_slices[compound_idx].item())
            x_end = int(x_slices[compound_idx + 1].item())
            e_start = int(e_slices[compound_idx].item())
            e_end = int(e_slices[compound_idx + 1].item())

            x = data_obj.x[x_start:x_end]
            # Reindex edges from global collation space to local node indices.
            edge_index = data_obj.edge_index[:, e_start:e_end] - x_start

            edge_weight = None
            if hasattr(data_obj, "edge_weight") and data_obj.edge_weight is not None:
                edge_weight = data_obj.edge_weight[e_start:e_end]
        else:
            # Extract graph components from a direct Data-like object.
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
            "edge_index": edge_index.detach().cpu().tolist(),
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
            results_dir = session_logger.session_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            result_file = results_dir / f"edgeshaper_{result['job_id']}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            result["result_save_path"] = str(result_file)

        session_logger.log_event(
            "edgeshaper_completed",
            model_class=type(model).__name__,
            model_source=("loaded_object" if model_path is None else "path_or_object"),
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
    structure. Returns a serializable summary plus saved image paths.

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
        - image_paths: list of filesystem paths to saved PNG files (if save_results=True)
        - note: text hint explaining how to display the saved plot with show_plot
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
    ...     print(result["image_paths"][0])
    ...     # Then call show_plot(result["image_paths"][0]) in MCP-compatible chat UIs
    """
    try:
        # Parse JSON inputs
        phi_edges = json.loads(phi_edges_json)
        edge_index_list = json.loads(edge_index_json)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
            )

        num_phi = len(phi_edges)
        num_edges_input = int(edge_index.shape[1])
        if num_phi != num_edges_input:
            raise ValueError(
                "edge_index and phi_edges length mismatch: "
                f"len(phi_edges)={num_phi}, edge_index_edges={num_edges_input}. "
                "Use edge_index returned by explain_gnn_with_edgeshaper for the same explained compound."
            )

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

            result = {
                "status": "completed",
                "num_edges": num_edges,
                "num_bonds": num_bonds,
                "note": "Use show_plot(image_paths[0]) to render the saved PNG inline.",
            }

            # Save PNG to session if requested
            if save_results:
                session_logger = _get_session_logger()
                viz_dir = session_logger.session_dir / "plots"
                viz_dir.mkdir(parents=True, exist_ok=True)

                import time
                timestamp = int(time.time() * 1000)
                viz_path = viz_dir / f"edgeshaper_heatmap_{timestamp}.png"
                img_pil.save(viz_path, dpi=(300, 300))
                result["image_paths"] = [str(viz_path)]
            else:
                result["image_paths"] = []

            return result

        except ImportError:
            result = {
                "status": "completed",
                "num_edges": num_edges,
                "num_bonds": num_bonds,
                "image_paths": [],
                "note": "EdgeSHAPer visualization dependencies are not available; heatmap visualization skipped. Ensure edgeshaper_viz_utils and its visualization dependencies are installed.",
            }
            return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        return {
            "status": "failed",
            "error": error_msg,
            "image_paths": [],
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

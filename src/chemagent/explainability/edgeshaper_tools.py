"""chemagent.explainability.edgeshaper_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for EdgeSHAPer (edge-level Shapley values) explainability.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
select_compound_for_edgeshaper — choose a correctly predicted GNN test compound for EdgeSHAPer
explain_gnn_with_edgeshaper    — compute edge importance scores for a GNN model on a test compound
visualize_edgeshaper_results   — render atom/edge-level SHAP heatmaps on molecular structures

The EdgeSHAPer algorithm computes Shapley value approximations for edge importance
in Graph Neural Networks, enabling edge-level explainability.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union
import json
import pickle
import random

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
from chemagent.ml.gnn_models import GCN, GraphSAGE, GAT, GC_GNN, GINE, GIN
from chemagent.ml.gnn_training import smiles_to_nx_graph, nx_graph_to_pyg_data


_EDGE_GNN_MODEL_MAP = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "GC_GNN": GC_GNN,
    "GINE": GINE,
    "GIN": GIN,
}


def _resolve_edgeshaper_model_class(
    model_class_name: str,
    custom_model_module: Optional[str] = None,
    custom_model_class_name: Optional[str] = None,
) -> tuple[type, str]:
    """Resolve built-in or custom model class for EdgeSHAPer workflows."""
    if model_class_name in _EDGE_GNN_MODEL_MAP:
        return _EDGE_GNN_MODEL_MAP[model_class_name], model_class_name

    class_name = custom_model_class_name or model_class_name
    module_obj = None
    module_hint = custom_model_module

    if module_hint and module_hint.endswith(".py"):
        module_path = Path(module_hint)
        if not module_path.is_absolute():
            module_path = Path(_SRC) / module_path
        module_path = module_path.resolve()
        spec = importlib.util.spec_from_file_location("chemagent_edgeshaper_custom_model", module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import custom module from path: {module_hint}")
        module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_obj)
    elif module_hint:
        module_obj = importlib.import_module(module_hint)

    if module_obj is None and "." in class_name:
        module_part, _, attr_part = class_name.rpartition(".")
        if module_part and attr_part:
            module_obj = importlib.import_module(module_part)
            class_name = attr_part

    if module_obj is None:
        raise ValueError(
            "Unknown model_class_name. For custom models, provide custom_model_module "
            "(module import path or .py path) and custom_model_class_name."
        )

    if not hasattr(module_obj, class_name):
        raise ValueError(
            f"Class '{class_name}' not found in module '{getattr(module_obj, '__name__', str(module_hint))}'."
        )

    return getattr(module_obj, class_name), class_name


def _extract_graph_components(
    graph_data: Any,
    compound_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
    """Extract a single graph by index from common PyG serialization formats."""
    if compound_idx < 0:
        raise IndexError(f"compound_idx must be >= 0, got {compound_idx}.")

    # Format 1: PyG InMemoryDataset-style tuple saved via torch.save((data, slices), ...)
    if isinstance(graph_data, tuple) and len(graph_data) >= 2:
        data_obj, slices = graph_data[0], graph_data[1]
        if not isinstance(slices, dict):
            raise ValueError("Expected slices dict in collated graph tuple.")
        if "x" not in slices or "edge_index" not in slices:
            raise ValueError("Collated graph data is missing required slices for 'x' or 'edge_index'.")

        x_slices = slices["x"]
        e_slices = slices["edge_index"]

        n_graphs = int(x_slices.numel() - 1)
        if compound_idx >= n_graphs:
            raise IndexError(
                f"compound_idx {compound_idx} out of range for dataset of size {n_graphs}."
            )

        x_start = int(x_slices[compound_idx].item())
        x_end = int(x_slices[compound_idx + 1].item())
        e_start = int(e_slices[compound_idx].item())
        e_end = int(e_slices[compound_idx + 1].item())

        x = data_obj.x[x_start:x_end]
        edge_index_raw = data_obj.edge_index[:, e_start:e_end]

        # Some saved collated tuples store edge indices already local to each graph,
        # while others store global node indices and require x_start offset removal.
        n_nodes = x_end - x_start
        edge_index_local = edge_index_raw - x_start

        if edge_index_local.numel() > 0 and int(edge_index_local.min()) >= 0 and int(edge_index_local.max()) < n_nodes:
            edge_index = edge_index_local
        elif edge_index_raw.numel() > 0 and int(edge_index_raw.min()) >= 0 and int(edge_index_raw.max()) < n_nodes:
            edge_index = edge_index_raw
        elif edge_index_raw.numel() == 0:
            edge_index = edge_index_raw
        else:
            raise ValueError(
                "Failed to map collated edge_index to local node indices for "
                f"compound_idx={compound_idx}. x_span=({x_start}, {x_end}), "
                f"edge_span=({e_start}, {e_end}), "
                f"raw_min={int(edge_index_raw.min())}, raw_max={int(edge_index_raw.max())}."
            )

        edge_weight = None
        if hasattr(data_obj, "edge_weight") and data_obj.edge_weight is not None:
            edge_weight = data_obj.edge_weight[e_start:e_end]

        return x, edge_index, edge_weight, "collated_tuple"

    # Format 2: sequence of per-graph Data-like objects.
    if isinstance(graph_data, (list, tuple)) and graph_data:
        if compound_idx >= len(graph_data):
            raise IndexError(
                f"compound_idx {compound_idx} out of range for dataset of size {len(graph_data)}."
            )
        selected = graph_data[compound_idx]
        if not hasattr(selected, "x") or not hasattr(selected, "edge_index"):
            raise ValueError("Selected graph object is missing 'x' or 'edge_index'.")

        x = selected.x
        edge_index = selected.edge_index
        edge_weight = selected.edge_weight if hasattr(selected, "edge_weight") else None
        return x, edge_index, edge_weight, "graph_list"

    # Format 3: InMemoryDataset-like object supporting len() and indexing.
    if hasattr(graph_data, "__len__") and hasattr(graph_data, "__getitem__"):
        try:
            n_graphs = len(graph_data)
            if n_graphs > 0:
                if compound_idx >= n_graphs:
                    raise IndexError(
                        f"compound_idx {compound_idx} out of range for dataset of size {n_graphs}."
                    )
                selected = graph_data[compound_idx]
                if hasattr(selected, "x") and hasattr(selected, "edge_index"):
                    x = selected.x
                    edge_index = selected.edge_index
                    edge_weight = selected.edge_weight if hasattr(selected, "edge_weight") else None
                    return x, edge_index, edge_weight, "dataset_object"
        except TypeError:
            # Some Data-like objects define __len__ with non-dataset semantics.
            pass

    # Format 4: single Data-like object.
    if hasattr(graph_data, "x") and hasattr(graph_data, "edge_index"):
        if compound_idx != 0:
            raise IndexError(
                "Loaded graph_data contains a single graph, but compound_idx is "
                f"{compound_idx}. Use compound_idx=0 or provide a multi-graph dataset file."
            )

        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_weight = graph_data.edge_weight if hasattr(graph_data, "edge_weight") else None
        return x, edge_index, edge_weight, "single_graph"

    raise ValueError(
        "Unsupported graph_data format. Expected one of: "
        "(data, slices) tuple, list/tuple of Data objects, InMemoryDataset-like object, "
        "or single Data object with x and edge_index."
    )


def _extract_graph_smiles(graph_data: Any) -> Optional[list[str]]:
    """Best-effort extraction of per-graph SMILES metadata from graph_data."""
    if isinstance(graph_data, tuple) and len(graph_data) >= 3 and isinstance(graph_data[2], dict):
        smiles_list = graph_data[2].get("smiles_list")
        if isinstance(smiles_list, list):
            return [str(s) for s in smiles_list]

    if hasattr(graph_data, "smiles_list") and isinstance(graph_data.smiles_list, list):
        return [str(s) for s in graph_data.smiles_list]

    if isinstance(graph_data, (list, tuple)) and graph_data:
        collected: list[str] = []
        for item in graph_data:
            if hasattr(item, "smiles") and item.smiles is not None:
                collected.append(str(item.smiles))
            else:
                return None
        return collected if collected else None

    return None


def _load_split_smiles(split_file_path: str, split: Literal["train", "val", "test"]) -> Optional[list[str]]:
    """Load split-specific SMILES list from split .pkl/.joblib file when available."""
    split_obj = None
    try:
        with open(split_file_path, "rb") as f:
            split_obj = pickle.load(f)
    except Exception:
        try:
            split_obj = joblib.load(split_file_path)
        except Exception:
            split_obj = None

    if not isinstance(split_obj, dict):
        return None

    key = f"{split}_smiles"
    values = split_obj.get(key)
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return None


def _resolve_compound_idx(
    compound_idx: int,
    compound_smiles: Optional[str],
    graph_smiles: Optional[list[str]],
    split_smiles: Optional[list[str]],
) -> tuple[int, Optional[str], str]:
    """Resolve compound index from either explicit index or a SMILES query."""
    if compound_smiles is None:
        selected_smiles = None
        if graph_smiles is not None and 0 <= compound_idx < len(graph_smiles):
            selected_smiles = graph_smiles[compound_idx]
        elif split_smiles is not None and 0 <= compound_idx < len(split_smiles):
            selected_smiles = split_smiles[compound_idx]
        return compound_idx, selected_smiles, "index"

    sources: list[tuple[str, list[str]]] = []
    if graph_smiles is not None:
        sources.append(("graph_data", graph_smiles))
    if split_smiles is not None:
        sources.append(("split_file", split_smiles))

    for source_name, smiles_list in sources:
        matches = [i for i, smi in enumerate(smiles_list) if smi == compound_smiles]
        if matches:
            return matches[0], compound_smiles, f"smiles:{source_name}"

    raise ValueError(
        "compound_smiles was provided but not found in available SMILES metadata. "
        "Recreate processed graph datasets with prepare_gnn_dataset() to persist SMILES, "
        "or pass a valid compound_idx."
    )


def _build_graph_from_smiles(compound_smiles: str) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build a single-graph (x, edge_index, edge_weight) tuple from a SMILES string."""
    nx_graph = smiles_to_nx_graph(compound_smiles)
    pyg_data = nx_graph_to_pyg_data(nx_graph, label=0)
    if pyg_data is None or not hasattr(pyg_data, "x") or not hasattr(pyg_data, "edge_index"):
        raise ValueError(f"Could not build a valid graph from SMILES: {compound_smiles}")

    x = pyg_data.x
    edge_index = pyg_data.edge_index
    edge_weight = pyg_data.edge_weight if hasattr(pyg_data, "edge_weight") else None
    return x, edge_index, edge_weight


def _load_edgeshaper_model(
    model: Optional[Any],
    model_path: Optional[str],
    model_class_name: str = "GCN",
    node_features_dim: int = 4,
    hidden_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_layers: int = 4,
    custom_model_module: Optional[str] = None,
    custom_model_class_name: Optional[str] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a GNN model object or reconstruct it from a state dict checkpoint."""
    if model is None:
        if not model_path:
            raise ValueError("Provide either 'model' or 'model_path'.")
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        loaded_obj = torch.load(model_file, map_location=device, weights_only=False)
        model = loaded_obj

        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj

        if isinstance(loaded_obj, dict):
            checkpoint = loaded_obj if "state_dict" in loaded_obj else None
            state_dict = loaded_obj["state_dict"] if checkpoint is not None else loaded_obj

            checkpoint_model_class = None
            if checkpoint is not None:
                checkpoint_model_class = checkpoint.get("model_class_name") or checkpoint.get("model_class")
            resolved_name = checkpoint_model_class or model_class_name
            model_class, _ = _resolve_edgeshaper_model_class(
                model_class_name=resolved_name,
                custom_model_module=custom_model_module,
                custom_model_class_name=custom_model_class_name,
            )

            if not isinstance(state_dict, dict):
                raise ValueError("Invalid checkpoint format: expected a state dict dictionary.")

            inferred_hidden = hidden_channels
            if checkpoint is not None and checkpoint.get("hidden_channels") is not None:
                inferred_hidden = int(checkpoint["hidden_channels"])

            if inferred_hidden is None:
                if "lin.weight" in state_dict and hasattr(state_dict["lin.weight"], "shape"):
                    inferred_hidden = int(state_dict["lin.weight"].shape[1])
                else:
                    for k in (
                        "conv1.bias",
                        "conv1.lin.weight",
                        "conv1.lin_l.weight",
                        "conv1.att_src",
                        "convs.0.bias",
                        "convs.0.lin.weight",
                        "convs.0.lin_l.weight",
                        "convs.0.att_src",
                    ):
                        if k in state_dict and hasattr(state_dict[k], "shape"):
                            inferred_hidden = int(state_dict[k].shape[0])
                            break
            if inferred_hidden is None:
                raise ValueError(
                    "Could not infer hidden_channels from state_dict. "
                    "Pass hidden_channels explicitly."
                )

            inferred_classes = num_classes
            if checkpoint is not None and checkpoint.get("num_classes") is not None:
                inferred_classes = int(checkpoint["num_classes"])
            if inferred_classes is None and "lin.weight" in state_dict:
                inferred_classes = int(state_dict["lin.weight"].shape[0])
            if inferred_classes is None:
                raise ValueError(
                    "Could not infer num_classes from state_dict. "
                    "Pass num_classes explicitly."
                )

            inferred_node_features = node_features_dim
            if checkpoint is not None and checkpoint.get("node_features_dim") is not None:
                inferred_node_features = int(checkpoint["node_features_dim"])

            inferred_num_layers = num_layers
            if checkpoint is not None and checkpoint.get("num_layers") is not None:
                inferred_num_layers = int(checkpoint["num_layers"])
            else:
                conv_indices = {
                    int(k.split(".")[1])
                    for k in state_dict.keys()
                    if k.startswith("convs.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
                }
                if conv_indices:
                    inferred_num_layers = max(conv_indices) + 1

            model = model_class(
                node_features_dim=inferred_node_features,
                hidden_channels=inferred_hidden,
                num_classes=inferred_classes,
                num_layers=inferred_num_layers,
            )
            model.load_state_dict(state_dict)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "'model' must be a loaded torch.nn.Module instance with .eval(). "
            f"Got {type(model)!r}."
        )

    model = model.to(device)
    model.eval()
    return model


def _get_graph_count(graph_data: Any) -> int:
    """Best-effort graph count for supported serialized graph containers."""
    if isinstance(graph_data, tuple) and len(graph_data) >= 2 and isinstance(graph_data[1], dict):
        slices = graph_data[1]
        if "x" in slices:
            return int(slices["x"].numel() - 1)

    if isinstance(graph_data, (list, tuple)) and graph_data:
        return len(graph_data)

    if hasattr(graph_data, "__len__") and hasattr(graph_data, "__getitem__"):
        try:
            return len(graph_data)
        except TypeError:
            pass

    if hasattr(graph_data, "x") and hasattr(graph_data, "edge_index"):
        return 1

    raise ValueError(
        "Unsupported graph_data format. Expected a collated tuple, a list of graph objects, "
        "an InMemoryDataset-like object, or a single graph object."
    )


def _extract_graph_label(graph_data: Any, compound_idx: int) -> int:
    """Extract the class label for a graph at a given index."""
    if isinstance(graph_data, tuple) and len(graph_data) >= 2 and isinstance(graph_data[1], dict):
        data_obj, slices = graph_data[0], graph_data[1]
        if "y" not in slices:
            raise ValueError("Collated graph data is missing required slices for 'y'.")

        y_slices = slices["y"]
        n_graphs = int(y_slices.numel() - 1)
        if compound_idx >= n_graphs:
            raise IndexError(
                f"compound_idx {compound_idx} out of range for dataset of size {n_graphs}."
            )

        y_start = int(y_slices[compound_idx].item())
        y_end = int(y_slices[compound_idx + 1].item())
        y_raw = data_obj.y[y_start:y_end]
        if y_raw.numel() == 0:
            raise ValueError(f"No label data found for compound_idx={compound_idx}.")
        return int(y_raw.view(-1)[0].item())

    if isinstance(graph_data, (list, tuple)) and graph_data:
        if compound_idx >= len(graph_data):
            raise IndexError(
                f"compound_idx {compound_idx} out of range for dataset of size {len(graph_data)}."
            )
        selected = graph_data[compound_idx]
        if not hasattr(selected, "y"):
            raise ValueError("Selected graph object is missing 'y'.")
        return int(selected.y.view(-1)[0].item())

    if hasattr(graph_data, "__len__") and hasattr(graph_data, "__getitem__"):
        try:
            n_graphs = len(graph_data)
            if n_graphs > 0:
                if compound_idx >= n_graphs:
                    raise IndexError(
                        f"compound_idx {compound_idx} out of range for dataset of size {n_graphs}."
                    )
                selected = graph_data[compound_idx]
                if hasattr(selected, "y"):
                    return int(selected.y.view(-1)[0].item())
        except TypeError:
            pass

    if hasattr(graph_data, "y"):
        if compound_idx != 0:
            raise IndexError(
                "Loaded graph_data contains a single graph, but compound_idx is "
                f"{compound_idx}. Use compound_idx=0 or provide a multi-graph dataset file."
            )
        return int(graph_data.y.view(-1)[0].item())

    raise ValueError("Could not extract a class label from the provided graph_data.")


def _predict_gnn_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Run a GNN forward pass for a single graph and return logits."""
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        logits = model(x, edge_index, batch, edge_weight=edge_weight)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return logits


def select_compound_for_edgeshaper(
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    graph_data_path: str | None = None,
    split: Literal["train", "val", "test"] = "test",
    split_file_path: Optional[str] = None,
    model_class_name: str = "GCN",
    node_features_dim: int = 4,
    hidden_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_layers: int = 4,
    custom_model_module: Optional[str] = None,
    custom_model_class_name: Optional[str] = None,
    target_class: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Select a correctly predicted GNN compound for EdgeSHAPer analysis.

    LLM agent routing note: use this helper only when the downstream explainability
    workflow is EdgeSHAPer. It selects a valid graph-level test example that the
    model predicts correctly, so you can pass the returned graph to
    explain_gnn_with_edgeshaper(). Do not use it to replace the explicit
    compound_idx / compound_smiles workflow; that direct-input path remains
    available in the explainer.

    Args:
    model : torch.nn.Module, optional
        Loaded GNN model object. Takes precedence over model_path.
    model_path : str, optional
        Path to a serialized model or state_dict checkpoint.
    graph_data_path : str
        Path to a processed GNN graph dataset (.pt).
    split : str, optional
        Split name used for metadata fallback (default: "test").
    split_file_path : str, optional
        Optional split .pkl file for SMILES fallback metadata.
    model_class_name : str, optional
        Built-in model class label (GCN, GraphSAGE, GAT, GC_GNN, GINE, GIN)
        or a custom class label used with ``custom_model_module``.
    node_features_dim : int, optional
        Node feature dimension used when reconstructing from state_dict/checkpoint.
    hidden_channels : int, optional
        Hidden channels for reconstruction; inferred when available in checkpoint.
    num_classes : int, optional
        Number of output classes for reconstruction; inferred when available.
    num_layers : int, optional
        Number of GNN message-passing layers for reconstruction (default: 4).
        Checkpoint metadata overrides this value when present.
    custom_model_module : str, optional
        Import path (e.g., ``my_pkg.models``) or ``.py`` path for custom model definitions.
    custom_model_class_name : str, optional
        Class name inside ``custom_model_module``.
    target_class : int, optional
        If provided, restrict selection to correctly predicted compounds of this class.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    dict
        Graph-selection metadata with keys including compound_idx, compound_smiles,
        true_label, predicted_label, prediction_confidence, selection_source, and status.
    """
    logger = _get_session_logger()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        model = _load_edgeshaper_model(
            model=model,
            model_path=model_path,
            model_class_name=model_class_name,
            node_features_dim=node_features_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            custom_model_module=custom_model_module,
            custom_model_class_name=custom_model_class_name,
            device=device,
        )

        if not graph_data_path:
            raise ValueError("graph_data_path is required for EdgeSHAPer compound selection.")

        graph_file = Path(graph_data_path)
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph data not found: {graph_file}")

        graph_data = torch.load(graph_file, map_location=device, weights_only=False)
        graph_smiles = _extract_graph_smiles(graph_data)
        split_smiles = _load_split_smiles(split_file_path, split) if split_file_path else None
        n_graphs = _get_graph_count(graph_data)

        correct_candidates: list[dict[str, Any]] = []
        for idx in range(n_graphs):
            x, edge_index, edge_weight, _ = _extract_graph_components(graph_data, idx)
            true_label = _extract_graph_label(graph_data, idx)
            if target_class is not None and true_label != target_class:
                continue

            logits = _predict_gnn_logits(model, x, edge_index, edge_weight, device)
            predicted_label = int(logits.argmax(dim=1).item())
            if hasattr(torch, "softmax") and logits.shape[1] > 1:
                confidence = float(torch.softmax(logits, dim=1).max(dim=1).values.item())
            else:
                confidence = float(torch.sigmoid(logits).view(-1)[0].item())

            if predicted_label != true_label:
                continue

            smiles = None
            if graph_smiles is not None and idx < len(graph_smiles):
                smiles = graph_smiles[idx]
            elif split_smiles is not None and idx < len(split_smiles):
                smiles = split_smiles[idx]
            if smiles is None:
                smiles = f"compound_{idx}"

            correct_candidates.append(
                {
                    "compound_idx": int(idx),
                    "compound_smiles": smiles,
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "prediction_confidence": confidence,
                    "selection_source": "graph_data",
                }
            )

        if not correct_candidates:
            class_note = f" for class {target_class}" if target_class is not None else ""
            raise ValueError(
                f"No correctly predicted compounds found in the '{split}' graph set{class_note}."
            )

        selected = random.choice(correct_candidates)
        result = {
            **selected,
            "split": split,
            "total_candidates": len(correct_candidates),
            "status": "completed",
        }

        logger.log_event(
            "edgeshaper_compound_selected",
            split=split,
            total_candidates=len(correct_candidates),
            compound_idx=result["compound_idx"],
            compound_smiles=result["compound_smiles"],
            true_label=result["true_label"],
            predicted_label=result["predicted_label"],
            selection_source=result["selection_source"],
        )

        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        logger.log_event("edgeshaper_compound_selection_failed", error=error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "job_id": f"edgeshaper_select_failed_{int(np.random.random() * 1e9)}",
        }


def explain_gnn_with_edgeshaper(
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    graph_data_path: Optional[str] = None,
    split: Literal["train", "val", "test"] = "test",
    split_file_path: Optional[str] = None,
    model_class_name: str = "GCN",
    node_features_dim: int = 4,
    hidden_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_layers: int = 4,
    custom_model_module: Optional[str] = None,
    custom_model_class_name: Optional[str] = None,
    compound_idx: int = 0,
    compound_smiles: Optional[str] = None,
    allow_external_smiles: bool = True,
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

    Args:
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
        Built-in model class label (GCN, GraphSAGE, GAT, GC_GNN, GINE, GIN)
        or a custom class label used with ``custom_model_module`` (default: "GCN").
    node_features_dim : int, optional
        Node feature dimension for model reconstruction (default: 4).
    hidden_channels : int, optional
        Hidden channels for model reconstruction. If omitted, inferred from state_dict.
    num_classes : int, optional
        Number of output classes for model reconstruction. If omitted, inferred from state_dict.
    num_layers : int, optional
        Number of GNN message-passing layers for reconstruction (default: 4).
        Checkpoint metadata overrides this value when present.
    custom_model_module : str, optional
        Import path (e.g., ``my_pkg.models``) or ``.py`` path for custom model definitions.
    custom_model_class_name : str, optional
        Class name inside ``custom_model_module``.
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

    Returns:
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

    Examples:
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
        resolved_compound_idx = compound_idx
        resolved_compound_smiles = compound_smiles
        selection_source = "index"
        graph_data_format = "from_smiles"

        if graph_data_path is None and compound_smiles is not None and allow_external_smiles:
            x, edge_index, edge_weight = _build_graph_from_smiles(compound_smiles)
            resolved_compound_idx = -1
            selection_source = "smiles:external"
        else:
            if not graph_data_path:
                default_graph_path = Path("data") / "gnn_dataset" / "processed" / f"{split}_processed.pt"
                if default_graph_path.exists():
                    graph_data_path = str(default_graph_path)
                else:
                    processed_dir = Path("data") / "gnn_dataset" / "processed"
                    candidates = sorted(
                        processed_dir.glob(f"{split}_*_processed.pt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if candidates:
                        graph_data_path = str(candidates[0])
                    else:
                        raise ValueError(
                            "'graph_data_path' is required."
                            f" Could not find default path for split '{split}': {default_graph_path}"
                        )

        model = _load_edgeshaper_model(
            model=model,
            model_path=model_path,
            model_class_name=model_class_name,
            node_features_dim=node_features_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            custom_model_module=custom_model_module,
            custom_model_class_name=custom_model_class_name,
            device=device,
        )

        if selection_source != "smiles:external":
            # Load graph data
            graph_data_path = Path(graph_data_path)
            if not graph_data_path.exists():
                raise FileNotFoundError(f"Graph data not found: {graph_data_path}")

            graph_data = torch.load(graph_data_path, map_location=device, weights_only=False)

            graph_smiles = _extract_graph_smiles(graph_data)
            split_smiles = _load_split_smiles(split_file_path, split) if split_file_path else None

            try:
                resolved_compound_idx, resolved_compound_smiles, selection_source = _resolve_compound_idx(
                    compound_idx=compound_idx,
                    compound_smiles=compound_smiles,
                    graph_smiles=graph_smiles,
                    split_smiles=split_smiles,
                )
                x, edge_index, edge_weight, graph_data_format = _extract_graph_components(
                    graph_data=graph_data,
                    compound_idx=resolved_compound_idx,
                )
            except ValueError:
                if compound_smiles is not None and allow_external_smiles:
                    x, edge_index, edge_weight = _build_graph_from_smiles(compound_smiles)
                    graph_data_format = "from_smiles"
                    resolved_compound_idx = -1
                    resolved_compound_smiles = compound_smiles
                    selection_source = "smiles:external"
                else:
                    raise

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
            "graph_data_format": graph_data_format,
            "selection_source": selection_source,
            "phi_edges": [float(v) for v in phi_edges],
            "edge_index": edge_index.detach().cpu().tolist(),
            "pertinent_positive_set": pertinent_positive_set,
            "minimal_top_k_set": minimal_top_k_set,
            "infidelity": float(infidelity) if infidelity is not None else None,
            "fidelity": float(fidelity) if fidelity is not None else None,
            "trustworthiness": float(trustworthiness) if trustworthiness is not None else None,
            "num_edges": edge_index.shape[1],
            "original_pred_prob": float(original_pred_prob) if original_pred_prob is not None else None,
            "compound_idx": resolved_compound_idx,
            "compound_smiles": resolved_compound_smiles,
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
            compound_idx=resolved_compound_idx,
            compound_smiles=resolved_compound_smiles,
            selection_source=selection_source,
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
) -> Any:
    """Visualize EdgeSHAPer explanations as molecular heatmaps.

    Creates RDKit-based heatmap image(s) showing edge importance on the molecular
    structure. Returns a serializable summary plus saved image paths.

    Args:
    smiles : str
        SMILES string of the molecule to visualize
    phi_edges_json : str
        JSON string containing list of Shapley values (one per edge)
    edge_index_json : str
        JSON string containing edge indices as [source_nodes, target_nodes]
    save_results : bool, optional
        If True, saves PNG files to session directory (default: True)

    Returns:
    If a PNG is generated and saved:
        [MCPImage(...), summary_json] so the plot can render inline directly.

    Otherwise:
        dict with status/error and metadata fields.

    Examples:
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
        matched_directed_edges = 0
        for i in range(len(phi_edges)):
            phi_value = phi_edges[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()

            if (init_atom, end_atom) in rdkit_bonds:
                bond_idx = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_idx] += phi_value
                matched_directed_edges += 1

        matched_bonds = sum(1 for v in rdkit_bonds_phi if abs(v) > 0.0)

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
                "matched_directed_edges": matched_directed_edges,
                "matched_bonds": matched_bonds,
                "note": "A PNG visualization was generated; whether it appears inline depends on the MCP client.",
            }

            if num_edges > 0 and matched_directed_edges < max(2, int(0.5 * num_edges)):
                result["warning"] = (
                    "Low edge-to-bond mapping ratio detected. The provided SMILES likely does not match "
                    "the explained graph. Use compound_smiles returned by explain_gnn_with_edgeshaper."
                )

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
                result["rendered_inline"] = True

                # Return the image first, then the JSON metadata.
                return [MCPImage(data=viz_path.read_bytes(), format="png"), json.dumps(result, indent=2)]
            else:
                result["image_paths"] = []
                result["rendered_inline"] = False

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

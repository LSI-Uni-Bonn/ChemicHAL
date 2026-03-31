"""chemagent.explainability.molanchor_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for MolAnchor explainability analysis and visualization.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
explain_with_molanchor            — identify molecular anchors (fragments) critical for a single prediction
explain_batch_with_molanchor      — run analysis on all correctly predicted compounds of a given class
identify_recurrent_anchor_rules    — compute substructure & anchor occurrence metrics to identify robust rules
visualize_molanchor_anchors        — draw molecular structure with identified anchors highlighted
select_compound_for_xai           — randomly select a correctly predicted compound for analysis
get_molanchor_info                — reference information about MolAnchor parameters and methods

The MolAnchor methodology identifies which molecular fragments (substructures) are
critical for a model's prediction on a given compound. The visualization tool highlights
these anchors directly on the molecular structure for intuitive interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union
import random
import base64

import joblib
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from mcp.server.fastmcp import Image as MCPImage

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.explainability.MolAnchor.MolAnchor import MolecularAnchor
from chemagent.session_utils import get_session_logger as _get_session_logger
from chemagent.featurization.fingerprints import ECFP


def _parse_bool(value: Union[bool, str]) -> bool:
    """Coerce MCP string booleans ('true'/'false') to Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no", "")


def _persist_image_output(img: Any, img_path: Path) -> None:
    """Persist image output from MolAnchor regardless of concrete image type."""
    if hasattr(img, "save"):
        img.save(str(img_path))
        return

    data = getattr(img, "data", None)
    if isinstance(data, bytes):
        img_path.write_bytes(data)
        return

    if isinstance(data, str):
        if data.startswith("data:image") and "," in data:
            payload = data.split(",", 1)[1]
            img_path.write_bytes(base64.b64decode(payload))
            return
        img_path.write_text(data, encoding="utf-8")
        return

    if isinstance(img, (bytes, bytearray)):
        img_path.write_bytes(bytes(img))
        return

    raise TypeError(f"Unsupported image object type for saving: {type(img)!r}")




def _smiles_to_mol_for_matching(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to a molecule for substructure matching.
    
    This directly parses the SMILES and returns the molecule in a standard form
    suitable for substructure matching against other molecules.
    
    Parameters
    ----------
    smiles : str
        SMILES string to convert
    
    Returns
    -------
    Chem.Mol or None
        Molecule object for substructure matching, or None if invalid
    """
    try:
        # mol = Chem.MolFromSmiles(smiles)
        # if mol is None:
        #     return None
        
        # # Sanitize and canonicalize for consistent matching
        # smarts = Chem.MolToSmarts(mol)
        mol = Chem.MolFromSmarts(smiles)
        return mol
    except Exception:
        return None


def get_molanchor_info() -> dict[str, Any]:
    """Return reference information about MolAnchor parameters and methods.
    
    Call once before using explain_with_molanchor() to understand available options.
    
    Returns:
        dict with:
            - fragment_schemes: available fragmentation methods
            - representations: available molecular representations
            - default_parameters: recommended default parameter values
            - bit_info_requirement: how to provide bit information for ECFP
            - description: overview of MolAnchor methodology
    """
    return {
        "fragment_schemes": {
            "BRICS": "Break Retrosynthetically Interesting Chemical Substructures (default)"
        },
        "representations": {
            "ECFP": "Extended Connectivity Fingerprints (default, requires fingerprint bit info)",
            "graphs": "Graph-based representation (requires graph neural network setup)"
        },
        "default_parameters": {
            "fragment_scheme": "BRICS",
            "representation": "ECFP",
            "target_class": 1,
            "cutoff": 0.95,
            "allow_frag_combinations": True,
            "return_multiple_anchors": False,
            "acc_for_radius": False
        },
        "bit_info_requirement": {
            "automatic": "Bit information is automatically generated internally from the SMILES. "
                        "Just call explain_with_molanchor(smiles='CCO', model_path='model.pkl'). "
                        "The function regenerates the fingerprint for the query SMILES.",
            "optional_dataset_context": "For reference, you can optionally pass dataset_id, but it's not required for analysis.",
            "manual_override": "Advanced users can provide an explicit bit_info_path .pkl file to override auto-generation.",
            "note": "Bit information maps ECFP bits to atom environments (atom_idx, radius)."
        },
        "description": 
            "MolAnchor identifies molecular fragments (anchors) that are critical for "
            "a machine learning model's prediction by systematically probing fragment presence/absence. "
            "It supports ECFP fingerprints or graph representations.",
        "workflow_example": {
            "simple": "explain_with_molanchor(smiles='CCO', model_path='model.pkl')",
            "with_dataset_context": "explain_with_molanchor(smiles='CCO', model_path='model.pkl', dataset_id='O00329_P42336')",
            "with_custom_fingerprint_params": "explain_with_molanchor(smiles='CCO', model_path='model.pkl', n_bits=1024, radius=3)",
            "full_training_pipeline": {
                "step_1": "load_dataset('path/to/data.csv')",
                "step_2": "compute_features(dataset_id, method='ECFP', n_bits=2048)  # optional: for reference",
                "step_3": "train_model(split_file_path, algorithm='RFC', ...)  # train your model",
                "step_4": "explain_with_molanchor(smiles='CCO', model_path='model.pkl')  # bit info generated automatically"
            }
        },
        "publications": [
            "MolAnchor: A novel tool for identifying fragments critical for model predictions"
        ]
    }


def _explain_with_molanchor(
    smiles: str,
    model_path: str,
    fragment_scheme: str = "BRICS",
    representation: str = "ECFP",
    target_class: int = 1,
    cutoff: float = 0.95,
    allow_frag_combinations: bool = True,
    return_multiple_anchors: bool = False,
    acc_for_radius: bool = False,
    n_bits: int = 2048,
    radius: int = 2,
    bit_info_path: Optional[str] = None,
    original_fp_path: Optional[str] = None,
) -> tuple[dict[str, Any], Any]:
    """Run MolAnchor analysis and return (result_dict, mol_anchor) for internal use."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")

    bit_inf = None
    original_fp = None

    if representation == "ECFP":
        if bit_info_path is not None:
            try:
                bit_inf = joblib.load(bit_info_path)
            except Exception:
                pass
        if bit_inf is None:
            try:
                fps, bit_inf = ECFP([smiles], n_bits=n_bits, radius=radius, return_bit_info=True)
                original_fp = np.array(fps[0])
            except Exception as e:
                raise ValueError(
                    f"Failed to generate ECFP fingerprint for SMILES '{smiles}': {e}"
                )

    if original_fp_path is not None:
        try:
            original_fp = np.load(original_fp_path)
        except Exception:
            pass

    mol_anchor = MolecularAnchor(
        mol=mol,
        model_obj=model,
        target_class=target_class,
        fragment_scheme=fragment_scheme,
        representation=representation,
        bit_inf=bit_inf,
        original_fp=original_fp,
        acc_for_radius=acc_for_radius,
    )

    df_combinations = mol_anchor.predict_frag_combinations()
    anchors_df = mol_anchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=cutoff,
        allow_frag_combinations=allow_frag_combinations,
        return_multiple_anchors=return_multiple_anchors,
    )

    anchor_indices: list[int] = []
    anchor_smiles_list: list[str] = []
    precision = 0.0
    multiple_used = False

    if not anchors_df.empty:
        first_row = anchors_df.iloc[0]
        raw = first_row["anchor_smile"]
        anchor_smiles_list = [raw] if isinstance(raw, str) else list(raw)
        precision = float(first_row.get("precision", 0.0))
        multiple_used = bool(first_row.get("plural_rule", False))

        if first_row["anchor_mol"] not in ("no_anchor", "all_frags"):
            anchor_mols = (
                [first_row["anchor_mol"]]
                if not isinstance(first_row["anchor_mol"], list)
                else first_row["anchor_mol"]
            )
            anchor_indices = [
                i for i, frag_mol in enumerate(mol_anchor.mol_frags)
                if any(frag_mol.GetNumAtoms() == am.GetNumAtoms() for am in anchor_mols)
            ]

    result = {
        "smiles": smiles,
        "fragment_combinations": df_combinations.drop(
            columns=["Predictions"] if "Predictions" in df_combinations.columns else []
        ).to_dict("records")[:10],
        "identified_anchors": anchors_df.drop(
            columns=["mol", "anchor_mol"] if "mol" in anchors_df.columns else []
        ).to_dict("records"),
        "num_fragments": len(mol_anchor.mol_frags),
        "anchor_indices": anchor_indices,
        "anchor_smiles": anchor_smiles_list,
        "precision": precision,
        "multiple_anchors_used": multiple_used,
        "status": "completed",
    }
    return result, mol_anchor


def explain_with_molanchor(
    smiles: str,
    model_path: str,
    fragment_scheme: str = "BRICS",
    representation: str = "ECFP",
    target_class: int = 1,
    cutoff: float = 0.95,
    allow_frag_combinations: Union[bool, str] = True,
    return_multiple_anchors: Union[bool, str] = False,
    acc_for_radius: Union[bool, str] = False,
    n_bits: int = 2048,
    radius: int = 2,
    bit_info_path: Optional[str] = None,
    original_fp_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> list:
    """
    Identify molecular anchors (critical fragments) for a model prediction using MolAnchor,
    and automatically visualize the anchors highlighted on the compound structure.

    The image is rendered directly in the chat window (LM Studio). Metadata —
    including anchor SMILES, precision, fragment count — is returned as a JSON string
    alongside the image.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound to analyze.
    model_path : str
        Path to the trained model file (.pkl format).
    fragment_scheme : str, optional
        Fragmentation scheme. Currently supports "BRICS" (default).
    representation : str, optional
        Molecular representation: "ECFP" (default) or "graphs".
    target_class : int, optional
        Class label to identify anchors for (default 1).
    cutoff : float, optional
        Precision cutoff (0–1) for identifying anchors (default 0.95).
    allow_frag_combinations : bool, optional
        If True, search for fragment combinations if no single fragment anchors (default True).
    return_multiple_anchors : bool, optional
        If True, return all fragments meeting the cutoff; otherwise only the highest (default False).
    acc_for_radius : bool, optional
        Account for atom environments spanning outside fragments (default False).
    n_bits : int, optional
        ECFP fingerprint length (default 2048). Must match model training setup.
    radius : int, optional
        ECFP Morgan radius (default 2). Must match model training setup.
    bit_info_path : str, optional
        Path to pre-saved bit information (.pkl). Auto-generated if not provided.
    original_fp_path : str, optional
        Path to original fingerprint array (.npy). Rarely needed.
    output_path : str, optional
        Path to save the visualization image (.png).
        Defaults to ``session_dir/plots/molanchor_<session_id>.png``.

    Returns
    -------
    list
        [MCPImage, json_metadata_str] — fastmcp converts this to an ImageContent block
        (renders in LM Studio) plus a TextContent block with the analysis metadata.

    Raises
    ------
    ValueError
        If SMILES is invalid, model cannot be loaded, or no anchors are identified.

    Examples
    --------
    >>> explain_with_molanchor(smiles="CCO", model_path="path/to/model.pkl")
    """
    allow_frag_combinations = _parse_bool(allow_frag_combinations)
    return_multiple_anchors = _parse_bool(return_multiple_anchors)
    acc_for_radius = _parse_bool(acc_for_radius)

    logger = _get_session_logger()

    result, mol_anchor = _explain_with_molanchor(
        smiles=smiles,
        model_path=model_path,
        fragment_scheme=fragment_scheme,
        representation=representation,
        target_class=target_class,
        cutoff=cutoff,
        allow_frag_combinations=allow_frag_combinations,
        return_multiple_anchors=return_multiple_anchors,
        acc_for_radius=acc_for_radius,
        n_bits=n_bits,
        radius=radius,
        bit_info_path=bit_info_path,
        original_fp_path=original_fp_path,
    )

    anchor_indices = result["anchor_indices"]

    if not anchor_indices:
        # No anchors found — return just the metadata as text
        return [json.dumps(result, indent=2)]

    img = mol_anchor.map_anchor_to_cpd(anchor_indices)

    if output_path is None:
        img_path = logger.session_dir / "plots" / f"molanchor_{logger.session_id}.png"
    else:
        img_path = Path(output_path)

    img_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_image_output(img, img_path)

    result["image_path"] = str(img_path)
    mcp_image = MCPImage(path=img_path)
    return [mcp_image, json.dumps(result, indent=2)]


def select_compound_for_xai(
    split_file_path: str,
    model_path: str,
    target_class: int,
    split: str = "test",
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Randomly select a correctly predicted compound of a specified class for XAI analysis.
    
    This tool helps identify good test cases for explainability analysis (e.g., MolAnchor)
    by finding compounds that the model predicted correctly and belong to a specific class.
    
    Parameters
    ----------
    split_file_path : str
        Path to the split .pkl file (from split_dataset)
    model_path : str
        Path to the trained model file (.pkl format)
    target_class : int
        Class label to filter by (e.g., 0 or 1 for binary classification)
    split : str, optional
        Which split to sample from: "train", "val", or "test" (default: "test")
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Compound information, including:
        - smiles: SMILES string of the selected compound
        - index: index in the split
        - true_label: actual class label
        - predicted_label: model's predicted class
        - prediction_confidence: confidence of the prediction (max probability)
        - split: which split the compound came from
        - total_candidates: total number of correctly predicted compounds in that class
        - status: completion status
    
    Raises
    ------
    ValueError
        If no correctly predicted compounds found for the specified class,
        or if split file/model cannot be loaded
    """
    logger = _get_session_logger()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load split file
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path}: {e}")
    
    # Get features and labels for the specified split
    split_key_features = f"{split}_features"
    split_key_labels = f"{split}_labels"
    split_key_smiles = f"{split}_smiles" if f"{split}_smiles" in split_data else None
    
    if split_key_features not in split_data or split_key_labels not in split_data:
        available = [k for k in split_data.keys() if "features" in k or "labels" in k]
        raise ValueError(
            f"Split '{split}' not found in file. Available splits: {available}"
        )
    
    features = split_data[split_key_features]
    labels = split_data[split_key_labels]
    smiles_list = split_data.get(split_key_smiles, None)
    
    if smiles_list is None:
        smiles_list = [f"compound_{i}" for i in range(len(labels))]
    
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    # Get predictions
    try:
        predictions = model.predict(features)
        # Try to get prediction probabilities for confidence scores
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = np.ones(len(predictions))  # Fallback: all confidence = 1
    except Exception as e:
        raise ValueError(f"Failed to run model predictions: {e}")
    
    # Filter for correctly predicted compounds of the target class
    correct_mask = (predictions == labels) & (labels == target_class)
    correct_indices = np.where(correct_mask)[0]
    
    if len(correct_indices) == 0:
        raise ValueError(
            f"No correctly predicted compounds found for class {target_class} in '{split}' split. "
            f"Try a different class or split."
        )
    
    # Randomly select one
    selected_idx = np.random.choice(correct_indices)
    
    # Extract information about the selected compound
    smiles = smiles_list[selected_idx]
    true_label = int(labels[selected_idx])
    predicted_label = int(predictions[selected_idx])
    confidence = float(confidences[selected_idx])
    
    return {
        "smiles": smiles,
        "index": int(selected_idx),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "prediction_confidence": confidence,
        "split": split,
        "total_candidates": len(correct_indices),
        "status": "completed"
    }


def explain_batch_with_molanchor(
    split_file_path: str,
    model_path: str,
    target_class: int,
    split: str = "test",
    fragment_scheme: str = "BRICS",
    representation: str = "ECFP",
    cutoff: float = 0.95,
    allow_frag_combinations: bool = True,
    return_multiple_anchors: bool = False,
    acc_for_radius: bool = False,
    n_bits: int = 2048,
    radius: int = 2,
    bit_info_path: Optional[str] = None,
    original_fp_path: Optional[str] = None,
    max_compounds: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run MolAnchor analysis for all correctly predicted compounds of a given class.
    
    This tool analyzes which molecular fragments are consistently critical across multiple
    compounds for a model's predictions. It systematically identifies anchors for all
    correctly predicted test compounds belonging to a specified class, then aggregates
    the results to show common anchor patterns.
    
    **Use Case**: Understand what makes a certain class of compounds predictable by your model,
    or validate that your model captures consistent chemical logic for a class.
    
    Parameters
    ----------
    split_file_path : str
        Path to the split .pkl file (from split_dataset)
    model_path : str
        Path to the trained model file (.pkl format)
    target_class : int
        Class label to analyze (e.g., 0 or 1 for binary classification)
    split : str, optional
        Which split to analyze: "train", "val", or "test" (default: "test")
    fragment_scheme : str, optional
        Fragmentation scheme to use. Currently supports "BRICS" (default)
    representation : str, optional
        Molecular representation: "ECFP" (default) or "graphs"
    cutoff : float, optional
        Precision cutoff (0-1) for identifying anchors (default=0.95)
    allow_frag_combinations : bool, optional
        If True, search for combinations of fragments if no single atoms anchor (default=True)
    return_multiple_anchors : bool, optional
        If True, return all fragments meeting cutoff; if False, return highest precision (default=False)
    acc_for_radius : bool, optional
        Account for atom environments spanning outside fragments (default=False)
    n_bits : int, optional
        ECFP fingerprint length in bits (default=2048). Must match training fingerprints.
    radius : int, optional
        ECFP Morgan radius (default=2). Use 2 for ECFP4 (most common) or 3 for ECFP6.
    bit_info_path : str, optional
        Path to bit information dictionary (.pkl) for external ECFP data.
        Not typically needed — bit info generated automatically.
    original_fp_path : str, optional
        Path to original fingerprint array (.npy) (rarely used)
    max_compounds : int, optional
        Limit analysis to this many compounds (default None, analyze all).
        Useful for large datasets to speed up computation.
    
    Returns
    -------
    dict
        Aggregated batch analysis results containing:
        
        - split: which split was analyzed
        - target_class: the class analyzed
        - total_compounds: number of correctly predicted compounds of the target class
        - compounds_analyzed: actual number analyzed (may differ from total if max_compounds set)
        - detailed_results: list of individual explain_with_molanchor results for each compound
        - aggregate_statistics: summary statistics across all compounds:
            - mean_num_fragments: average number of fragments per molecule
            - mean_precision: average anchor precision
            - mean_coverage: average anchor coverage
            - compounds_with_anchors: count of compounds where anchors were identified
            - anchor_frequency: dict mapping anchor SMILES to count of times identified
            - most_common_anchors: top 5 anchors by frequency
        - status: completion status
    
    Raises
    ------
    ValueError
        If split file or model cannot be loaded, or no correctly predicted compounds found
    
    Examples
    --------
    Analyze all correctly predicted actives (class=1) to understand what makes them predictable:
    
    >>> batch_results = explain_batch_with_molanchor(
    ...     split_file_path="data/logs/session_xxx/splits/data_random_0.7_0.0_0.3.pkl",
    ...     model_path="data/logs/session_xxx/models/data_random_RFC.pkl",
    ...     target_class=1,  # analyze active compounds
    ...     split="test"
    ... )
    >>> batch_results["aggregate_statistics"]["most_common_anchors"]
    
    Limit to first 10 compounds to speed up analysis:
    
    >>> batch_results = explain_batch_with_molanchor(
    ...     split_file_path="...",
    ...     model_path="...",
    ...     target_class=1,
    ...     max_compounds=10
    ... )
    """
    logger = _get_session_logger()
    
    # Load split file
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path}: {e}")
    
    # Get features and labels for the specified split
    split_key_features = f"{split}_features"
    split_key_labels = f"{split}_labels"
    split_key_smiles = f"{split}_smiles" if f"{split}_smiles" in split_data else None
    
    if split_key_features not in split_data or split_key_labels not in split_data:
        available = [k for k in split_data.keys() if "features" in k or "labels" in k]
        raise ValueError(
            f"Split '{split}' not found in file. Available splits: {available}"
        )
    
    features = split_data[split_key_features]
    labels = split_data[split_key_labels]
    smiles_list = split_data.get(split_key_smiles, None)
    
    if smiles_list is None:
        smiles_list = [f"compound_{i}" for i in range(len(labels))]
    
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    # Get predictions
    try:
        predictions = model.predict(features)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = np.ones(len(predictions))
    except Exception as e:
        raise ValueError(f"Failed to run model predictions: {e}")
    
    # Filter for correctly predicted compounds of the target class
    correct_mask = (predictions == labels) & (labels == target_class)
    correct_indices = np.where(correct_mask)[0]
    
    if len(correct_indices) == 0:
        raise ValueError(
            f"No correctly predicted compounds found for class {target_class} in '{split}' split. "
            f"Try a different class or split."
        )
    
    # Limit to max_compounds if specified
    if max_compounds is not None and len(correct_indices) > max_compounds:
        correct_indices = np.random.choice(correct_indices, size=max_compounds, replace=False)
    
    # Run explain_with_molanchor for each compound
    detailed_results = []
    anchor_frequency = {}
    num_fragments_list = []
    precision_list = []
    compounds_with_anchors = 0
    
    for compound_idx in correct_indices:
        smiles = smiles_list[compound_idx]
        
        try:
            result, _ = _explain_with_molanchor(
                smiles=smiles,
                model_path=model_path,
                fragment_scheme=fragment_scheme,
                representation=representation,
                target_class=target_class,
                cutoff=cutoff,
                allow_frag_combinations=allow_frag_combinations,
                return_multiple_anchors=return_multiple_anchors,
                acc_for_radius=acc_for_radius,
                n_bits=n_bits,
                radius=radius,
                bit_info_path=bit_info_path,
                original_fp_path=original_fp_path,
            )

            # Store detailed result
            result["compound_index"] = int(compound_idx)
            result["true_label"] = int(labels[compound_idx])
            result["predicted_confidence"] = float(confidences[compound_idx])
            detailed_results.append(result)
            
            # Aggregate statistics
            num_fragments_list.append(result.get("num_fragments", 0))
            precision_list.append(result.get("precision", 0.0))
            
            # Track anchor frequency (whole rule as one unit, single- or multi-fragment)
            if result.get("anchor_smiles"):
                compounds_with_anchors += 1
                anchor_key = "||".join(result["anchor_smiles"])
                anchor_frequency[anchor_key] = anchor_frequency.get(anchor_key, 0) + 1
                    
        except Exception as e:
            # Log error but continue with other compounds
            detailed_results.append({
                "smiles": smiles,
                "compound_index": int(compound_idx),
                "status": "failed",
                "error": str(e)
            })
    
    # Sort anchors by frequency, get top 5
    most_common_anchors = sorted(
        anchor_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Compile aggregate statistics
    aggregate_statistics = {
        "mean_num_fragments": float(np.mean(num_fragments_list)) if num_fragments_list else 0.0,
        "mean_precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "compounds_with_anchors": compounds_with_anchors,
        "anchor_frequency": anchor_frequency,
        "most_common_anchors": [{"anchor": smile, "frequency": freq} for smile, freq in most_common_anchors]
    }
    
    return {
        "split": split,
        "target_class": target_class,
        "total_compounds": len(correct_indices),
        "compounds_analyzed": len(detailed_results),
        "detailed_results": detailed_results,
        "aggregate_statistics": aggregate_statistics,
        "status": "completed"
    }


def identify_recurrent_anchor_rules(
    split_file_path: str,
    model_path: str,
    target_class: int,
    split: str = "test",
    fragment_scheme: str = "BRICS",
    representation: str = "ECFP",
    cutoff: float = 0.95,
    allow_frag_combinations: Union[bool, str] = True,
    return_multiple_anchors: Union[bool, str] = False,
    acc_for_radius: Union[bool, str] = False,
    n_bits: int = 2048,
    radius: int = 2,
    bit_info_path: Optional[str] = None,
    original_fp_path: Optional[str] = None,
    top_n_anchors: Optional[int] = 3,
) -> list:
    """
    Run batch MolAnchor analysis and identify recurrent anchor rules in one step.

    This tool first runs MolAnchor on all correctly predicted compounds of the target class
    (via explain_batch_with_molanchor), then computes two key metrics for each identified
    anchor to determine which fragment rules are most robust and consistent:

    1. **Anchor Occurrence**: Fraction of ANALYZED compounds (those where model correctly
       predicted the target class) where this fragment was identified as an anchor.
       Measures how important the fragment is for the model's predictions.

    2. **Substructure Occurrence**: Fraction of ANALYZED compounds in the split that contain
       this fragment. Supplies context for anchor occurrence, anchor occurrence can never be higher than substructure occurrence.
        If substructure occurrence is equal to anchor occurrence, it means that whenever the fragment is present, it is an anchor (strong indicator).
        If substructure occurrence is much higher than anchor occurrence, it means the fragment is common but not always critical (weaker indicator).
        
    High-occurrence fragments represent consistent chemical logic your model uses for a given
    class. 

    Parameters
    ----------
    split_file_path : str
        Path to the split .pkl file (from split_dataset)
    model_path : str
        Path to the trained model file (.pkl format)
    target_class : int
        Class label to analyze (e.g., 0 or 1 for binary classification)
    split : str, optional
        Which split to analyze: "train", "val", or "test" (default: "test")
    fragment_scheme : str, optional
        Fragmentation scheme to use. Currently supports "BRICS" (default)
    representation : str, optional
        Molecular representation: "ECFP" (default) or "graphs"
    cutoff : float, optional
        Precision cutoff (0-1) for identifying anchors (default=0.95)
    allow_frag_combinations : bool, optional
        If True, search for combinations of fragments if no single fragment anchors (default=True)
    return_multiple_anchors : bool, optional
        If True, return all fragments meeting cutoff; if False, return highest precision (default=False)
    acc_for_radius : bool, optional
        Account for atom environments spanning outside fragments (default=False)
    n_bits : int, optional
        ECFP fingerprint length in bits (default=2048). Must match training fingerprints.
    radius : int, optional
        ECFP Morgan radius (default=2). Use 2 for ECFP4 (most common) or 3 for ECFP6.
    bit_info_path : str, optional
        Path to bit information dictionary (.pkl). Not typically needed — auto-generated.
    original_fp_path : str, optional
        Path to original fingerprint array (.npy) (rarely used)
    top_n_anchors : int, optional
        Maximum number of top anchors to return, sorted by anchor_occurrence then
        substructure_occurrence. Default: 3. Set to None to return all.

    Returns
    -------
    list
        Interleaved per-rule items followed by a summary JSON:
        [MCPImage_1, rule_1_json, MCPImage_2, rule_2_json, ..., summary_json]

        Each rule JSON contains:
        - rank: position in the ranked list (1 = most recurrent)
        - fragment: SMILES string (single-fragment) or list of SMILES (multi-fragment rule)
        - anchor_occurrence: fraction of analyzed compounds where this rule was identified
        - substructure_occurrence: fraction of ALL split compounds containing the fragment(s)
        - num_compounds_with_anchor: absolute count for anchor_occurrence
        - num_compounds_with_substructure: absolute count for substructure_occurrence
        - image_path: path to the saved highlight image (if visualization succeeded)

        The final summary JSON contains:
        - target_class, split, num_analyzed_compounds, total_compounds_in_split
        - total_unique_anchor_rules, top_n_rules_shown, status

    Raises
    ------
    ValueError
        If split file or model cannot be loaded, or no correctly predicted compounds found

    Examples
    --------
    >>> rules = identify_recurrent_anchor_rules(
    ...     split_file_path="session/splits/data.pkl",
    ...     model_path="session/models/model.pkl",
    ...     target_class=1,
    ...     top_n_anchors=3
    ... )
    >>> for rule in rules["recurrent_rules"]:
    ...     print(f"Fragment {rule['fragment']}: "
    ...           f"{rule['anchor_occurrence']:.1%} anchor, "
    ...           f"{rule['substructure_occurrence']:.1%} substructure")
    """
    allow_frag_combinations = _parse_bool(allow_frag_combinations)
    return_multiple_anchors = _parse_bool(return_multiple_anchors)
    acc_for_radius = _parse_bool(acc_for_radius)

    logger = _get_session_logger()

    # Run batch analysis internally
    batch_results = explain_batch_with_molanchor(
        split_file_path=split_file_path,
        model_path=model_path,
        target_class=target_class,
        split=split,
        fragment_scheme=fragment_scheme,
        representation=representation,
        cutoff=cutoff,
        allow_frag_combinations=allow_frag_combinations,
        return_multiple_anchors=return_multiple_anchors,
        acc_for_radius=acc_for_radius,
        n_bits=n_bits,
        radius=radius,
        bit_info_path=bit_info_path,
        original_fp_path=original_fp_path,
    )

    # Load split file to get all SMILES for substructure searching
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path}: {e}")

    split_key_smiles = f"{split}_smiles" if f"{split}_smiles" in split_data else None
    split_key_labels = f"{split}_labels"

    if split_key_labels not in split_data:
        available = [k for k in split_data.keys() if "labels" in k]
        raise ValueError(f"Split '{split}' not found in file. Available splits: {available}")

    smiles_list = split_data.get(split_key_smiles, None)
    if smiles_list is None:
        raise ValueError(f"No SMILES found in {split} split. Cannot perform substructure search.")

    # Extract analyzed compounds and build anchor→representative compound mapping
    analyzed_compounds = {}
    anchor_representative: dict[str, str] = {}  # anchor key -> first compound SMILES showing it
    for result in batch_results.get("detailed_results", []):
        if "compound_index" in result and "smiles" in result and result.get("status") == "completed":
            analyzed_compounds[result["compound_index"]] = result["smiles"]
            anchor_key = "||".join(result.get("anchor_smiles", []))
            if anchor_key and anchor_key not in anchor_representative:
                anchor_representative[anchor_key] = result["smiles"]

    num_analyzed_compounds = len(analyzed_compounds)
    if num_analyzed_compounds == 0:
        raise ValueError("No successfully analyzed compounds found in batch results")

    total_compounds_in_split = len(smiles_list)

    # Get all unique anchors from batch results
    anchor_frequency = batch_results.get("aggregate_statistics", {}).get("anchor_frequency", {})
    
    if not anchor_frequency:
        return {
            "recurrent_rules": [],
            "rule_details": [],
            "statistics": {
                "total_unique_anchors": 0,
            },
            "status": "no anchors found"
        }
    
    # Compute metrics for each anchor fragment
    recurrent_rules = []
    
    for anchor_key, anchor_count in anchor_frequency.items():
        # Split key back into individual fragment SMILES
        fragment_smiles_list = anchor_key.split("||")
        anchor_mols = [_smiles_to_mol_for_matching(smi) for smi in fragment_smiles_list]
        anchor_mols = [m for m in anchor_mols if m is not None]
        if not anchor_mols:
            continue

        # Anchor occurrence: fraction of ANALYZED compounds where this rule was identified
        anchor_occurrence = anchor_count / num_analyzed_compounds if num_analyzed_compounds > 0 else 0.0

        # Substructure occurrence: fraction of ALL compounds in split containing ALL rule fragments
        substructure_count = 0
        for compound_smiles in smiles_list:
            try:
                compound_mol = Chem.MolFromSmiles(compound_smiles)
                if compound_mol is not None and all(
                    compound_mol.HasSubstructMatch(am) for am in anchor_mols
                ):
                    substructure_count += 1
            except Exception:
                pass

        substructure_occurrence = substructure_count / num_analyzed_compounds if num_analyzed_compounds > 0 else 0.0

        rule_entry = {
            "fragment": fragment_smiles_list[0] if len(fragment_smiles_list) == 1 else fragment_smiles_list,
            "substructure_occurrence": float(substructure_occurrence),
            "anchor_occurrence": float(anchor_occurrence),
            "num_compounds_with_substructure": int(substructure_count),
            "num_compounds_with_anchor": int(anchor_count),
        }

        recurrent_rules.append(rule_entry)

    # Sort by anchor occurrence (primary) then substructure occurrence (secondary)
    recurrent_rules.sort(
        key=lambda x: (x["anchor_occurrence"], x["substructure_occurrence"]),
        reverse=True
    )

    final_rules = recurrent_rules
    
    # Limit to top N anchors if specified
    if top_n_anchors is not None and len(final_rules) > top_n_anchors:
        final_rules = final_rules[:top_n_anchors]
    
    # Generate one highlighted image per anchor rule; interleave with per-rule JSON
    plots_dir = logger.session_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_items = []  # alternating: MCPImage, rule_json_str, MCPImage, rule_json_str, ...
    for i, rule in enumerate(final_rules):
        rule["rank"] = i + 1
        frag = rule["fragment"]
        lookup_key = "||".join(frag) if isinstance(frag, list) else frag
        rep_smiles = anchor_representative.get(lookup_key)
        if rep_smiles:
            try:
                vis_result, mol_anchor = _explain_with_molanchor(
                    smiles=rep_smiles,
                    model_path=model_path,
                    fragment_scheme=fragment_scheme,
                    representation=representation,
                    target_class=target_class,
                    cutoff=cutoff,
                    allow_frag_combinations=allow_frag_combinations,
                    return_multiple_anchors=return_multiple_anchors,
                    acc_for_radius=acc_for_radius,
                    n_bits=n_bits,
                    radius=radius,
                    bit_info_path=bit_info_path,
                    original_fp_path=original_fp_path,
                )
                anchor_indices = vis_result.get("anchor_indices", [])
                if anchor_indices:
                    img = mol_anchor.map_anchor_to_cpd(anchor_indices)
                    img_path = plots_dir / f"anchor_rule_{i + 1}_{logger.session_id}.png"
                    img.save(str(img_path))
                    rule["image_path"] = str(img_path)
                    output_items.append(MCPImage(path=img_path))
            except Exception:
                pass
        output_items.append(json.dumps(rule, indent=2))

    summary = {
        "target_class": target_class,
        "split": split,
        "num_analyzed_compounds": num_analyzed_compounds,
        "total_compounds_in_split": total_compounds_in_split,
        "total_unique_anchor_rules": len(anchor_frequency),
        "top_n_rules_shown": len(final_rules),
        "status": "completed",
    }
    return output_items + [json.dumps(summary, indent=2)]




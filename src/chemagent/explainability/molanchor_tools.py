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
from typing import Any, Literal, Optional
import random

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image as PILImage
from mcp.server.fastmcp import Image as MCPImage
import io

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.explainability.MolAnchor.MolAnchor import MolecularAnchor
from chemagent.session_utils import get_session_logger as _get_session_logger
from chemagent.featurization.fingerprints import ECFP


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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Sanitize and canonicalize for consistent matching
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL)
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


def explain_with_molanchor(
    smiles: str,
    model_path: str,
    dataset_id: Optional[str] = None,
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
) -> dict[str, Any]:
    """
    Identify molecular anchors (critical fragments) for a model prediction using MolAnchor.
    
    This tool analyzes which molecular fragments are essential for a trained model's prediction
    on a given compound. It works by systematically testing fragment combinations and measuring
    their impact on the model's output.
    
    **Key Feature**: The function automatically generates the ECFP fingerprint and bit information
    for the given SMILES internally, making it self-contained and accessible. Just provide the
    SMILES and model path.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the compound to analyze
    model_path : str
        Path to the trained model file (.pkl format)
    dataset_id : str, optional
        Dataset ID for reference/logging. Not required for the analysis.
    fragment_scheme : str, optional
        Fragmentation scheme to use. Currently supports "BRICS" (default)
    representation : str, optional
        Molecular representation: "ECFP" (default) or "graphs"
    target_class : int, optional
        Class label to identify anchors for (default=1, relevant for classification)
    cutoff : float, optional
        Precision cutoff (0-1) for identifying anchors (default=0.95). Higher cutoff
        means fragments must be more selective to be called anchors
    allow_frag_combinations : bool, optional
        If True, search for combinations of fragments if no single atoms anchor the prediction (default=True)
    return_multiple_anchors : bool, optional
        If True, return all fragments meeting the cutoff; if False, return only the highest precision anchors (default=False)
    acc_for_radius : bool, optional
        Account for atom environments spanning outside fragments (default=False)
    n_bits : int, optional
        ECFP fingerprint length in bits (default=2048). Must match the fingerprints used during model training.
        Common values: 1024, 2048, 4096
    radius : int, optional
        ECFP Morgan radius (default=2). Use 2 for ECFP4 (most common) or 3 for ECFP6.
        Must match the radius used during model training.
    bit_info_path : str, optional
        Path to bit information dictionary (.pkl) for external ECFP data.
        **Not typically needed** — the function automatically regenerates bit info internally.
        Provide only if you want to use pre-saved bit information instead.
    original_fp_path : str, optional
        Path to original fingerprint array (.npy) (rarely used, only for advanced debugging)
    
    Returns
    -------
    dict
        Analysis results containing:
        - smiles: input SMILES
        - fragment_combinations: DataFrame showing all fragment combinations and their predictions
        - identified_anchors: DataFrame with identified anchor fragments and their metrics
        - num_fragments: number of fragments in the molecule
        - anchor_indices: indices of the identified anchors
        - anchor_smiles: SMILES of the anchor fragments
        - precision: precision of the identified anchors
        - coverage: coverage of the identified anchors
        - multiple_anchors_used: whether multiple anchor fragments were combined
        - status: completion status
    
    Raises
    ------
    ValueError
        If SMILES is invalid, model cannot be loaded, or representation is unsupported
    
    Examples
    --------
    The simplest usage — just provide SMILES and model path:
    
    >>> result = explain_with_molanchor(
    ...     smiles="CCO",
    ...     model_path="path/to/trained_model.pkl"
    ... )
    >>> result["anchor_smiles"]  # Get the identified critical fragments
    
    If your model used different ECFP parameters, specify them to match:
    
    >>> result = explain_with_molanchor(
    ...     smiles="CCO",
    ...     model_path="path/to/trained_model.pkl",
    ...     n_bits=1024,
    ...     radius=3
    ... )
    """
    logger = _get_session_logger()
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    # Load or generate bit info and original FP
    bit_inf = None
    original_fp = None
    
    if representation == "ECFP":
        # First, try to use provided bit_info_path
        if bit_info_path is not None:
            try:
                bit_inf = joblib.load(bit_info_path)
            except Exception as e:
                pass
        
        # If not provided, generate bit info from the SMILES using ECFP
        if bit_inf is None:
            try:
                fps, bit_inf = ECFP([smiles], n_bits=n_bits, radius=radius, return_bit_info=True)
                # Store the generated fingerprint as original_fp for reference
                original_fp = np.array(fps[0])
            except Exception as e:
                raise ValueError(
                    f"Failed to generate ECFP fingerprint for SMILES '{smiles}': {e}\n"
                    f"Please verify the SMILES is valid and matches the model's expected fingerprint dimension."
                )
    
    if original_fp_path is not None:
        try:
            original_fp = np.load(original_fp_path)
        except Exception as e:
            pass
    
    mol_anchor = MolecularAnchor(
        mol=mol,
        model_obj=model,
        target_class=target_class,
        fragment_scheme=fragment_scheme,
        representation=representation,
        bit_inf=bit_inf,
        original_fp=original_fp,
        acc_for_radius=acc_for_radius
    )
    
    # Get fragment combinations and predictions
    df_combinations = mol_anchor.predict_frag_combinations()
    
    # Identify anchors
    anchors_df = mol_anchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=cutoff,
        allow_frag_combinations=allow_frag_combinations,
        return_multiple_anchors=return_multiple_anchors
    )
    
    # Extract anchor information
    anchor_indices = []
    anchor_smiles_list = []
    precision = 0.0
    coverage = 0.0
    multiple_used = False
    
    if not anchors_df.empty:
        first_row = anchors_df.iloc[0]
        anchor_smiles_list = (
            [first_row["anchor_smile"]] 
            if isinstance(first_row["anchor_smile"], str) 
            else first_row["anchor_smile"]
        )
        precision = float(first_row.get("precision", 0.0))
        coverage = float(first_row.get("coverage", 0.0))
        multiple_used = bool(first_row.get("plural_rule", False))
        
        # Extract anchor indices if anchor_mol is valid
        if first_row["anchor_mol"] != "no_anchor" and first_row["anchor_mol"] != "all_frags":
            anchor_mols = (
                [first_row["anchor_mol"]]
                if not isinstance(first_row["anchor_mol"], list)
                else first_row["anchor_mol"]
            )
            anchor_indices = [
                i for i, frag_mol in enumerate(mol_anchor.mol_frags)
                if any(
                    frag_mol.GetNumAtoms() == am.GetNumAtoms()
                    for am in anchor_mols
                )
            ]
    
    final_anchor_smiles = anchor_smiles_list if isinstance(anchor_smiles_list, list) else [anchor_smiles_list]

    # Automatically visualize identified anchors on the compound structure
    visualization = None
    if final_anchor_smiles:
        try:
            visualization = visualize_molanchor_anchors(
                smiles=smiles,
                anchor_smiles_list=final_anchor_smiles,
            )
        except Exception as viz_err:
            visualization = {"status": "failed", "error": str(viz_err)}

    return {
        "smiles": smiles,
        "fragment_combinations": df_combinations.drop(columns=["Predictions"] if "Predictions" in df_combinations.columns else []).to_dict("records")[:10],  # Sample
        "identified_anchors": anchors_df.drop(columns=["mol", "anchor_mol"] if "mol" in anchors_df.columns else []).to_dict("records"),
        "num_fragments": len(mol_anchor.mol_frags),
        "anchor_indices": anchor_indices,
        "anchor_smiles": final_anchor_smiles,
        "precision": precision,
        "coverage": coverage,
        "multiple_anchors_used": multiple_used,
        "visualization": visualization,
        "status": "completed"
    }


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
    coverage_list = []
    compounds_with_anchors = 0
    
    for idx, compound_idx in enumerate(correct_indices):
        smiles = smiles_list[compound_idx]
        
        try:
            result = explain_with_molanchor(
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
            coverage_list.append(result.get("coverage", 0.0))
            
            # Track anchor frequency
            if result.get("anchor_smiles"):
                compounds_with_anchors += 1
                for anchor_smile in result["anchor_smiles"]:
                    anchor_frequency[anchor_smile] = anchor_frequency.get(anchor_smile, 0) + 1
                    
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
        "mean_coverage": float(np.mean(coverage_list)) if coverage_list else 0.0,
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
    allow_frag_combinations: bool = True,
    return_multiple_anchors: bool = False,
    acc_for_radius: bool = False,
    n_bits: int = 2048,
    radius: int = 2,
    bit_info_path: Optional[str] = None,
    original_fp_path: Optional[str] = None,
    max_compounds: Optional[int] = None,
    single_fragment_rules_only: bool = True,
    top_n_anchors: Optional[int] = 5,
) -> dict[str, Any]:
    """
    Run batch MolAnchor analysis and identify recurrent anchor rules in one step.

    This tool first runs MolAnchor on all correctly predicted compounds of the target class
    (via explain_batch_with_molanchor), then computes two key metrics for each identified
    anchor to determine which fragment rules are most robust and consistent:

    1. **Anchor Occurrence**: Fraction of ANALYZED compounds (those where model correctly
       predicted the target class) where this fragment was identified as an anchor.
       Measures how important the fragment is for the model's predictions.

    2. **Substructure Occurrence**: Fraction of ALL compounds in the split that contain
       this fragment. A population-wide statistic showing how prevalent the fragment is
       in the dataset, regardless of model predictions.

    High-occurrence fragments represent consistent chemical logic your model uses for a given
    class. Single-fragment rules (anchor_occurrence >= 0.8) are fragments that alone are
    strong indicators of the prediction class.

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
    max_compounds : int, optional
        Limit batch analysis to this many compounds (default None = analyze all).
    single_fragment_rules_only : bool, optional
        If True (default), return only single-fragment rules (anchor_occurrence >= 0.8).
        If False, return all identified rules.
    top_n_anchors : int, optional
        Maximum number of top anchors to return, sorted by anchor_occurrence then
        substructure_occurrence. Default: 5. Set to None to return all.

    Returns
    -------
    dict
        - recurrent_rules: list of fragment rules (filtered/ranked):
            - fragment: SMILES of the fragment
            - substructure_occurrence: fraction of ALL compounds in split containing this fragment
            - anchor_occurrence: fraction of ANALYZED compounds where identified as anchor
            - num_compounds_with_substructure: count of compounds containing fragment
            - num_compounds_with_anchor: count of analyzed compounds where identified as anchor
            - single_fragment_rule: bool, True if anchor_occurrence >= 0.8
        - rule_details: alias for recurrent_rules
        - statistics:
            - total_unique_anchors: unique fragments identified in batch analysis
            - total_single_fragment_rules: fragments with anchor_occurrence >= 0.8
            - filtered_rules_count: rules in final recurrent_rules list
        - batch_summary: summary statistics from the underlying batch analysis
        - num_analyzed_compounds: correctly predicted compounds analyzed with MolAnchor
        - total_compounds_in_split: total size of the dataset split
        - single_fragment_rules_only: echo of the filter parameter used
        - status: completion status

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
    ...     single_fragment_rules_only=True,
    ...     top_n_anchors=5
    ... )
    >>> for rule in rules["recurrent_rules"]:
    ...     print(f"Fragment {rule['fragment']}: "
    ...           f"{rule['anchor_occurrence']:.1%} anchor, "
    ...           f"{rule['substructure_occurrence']:.1%} substructure")
    """
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
        max_compounds=max_compounds,
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

    # Extract analyzed compounds from batch results
    analyzed_compounds = {}
    for result in batch_results.get("detailed_results", []):
        if "compound_index" in result and "smiles" in result and result.get("status") == "completed":
            analyzed_compounds[result["compound_index"]] = result["smiles"]

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
    
    for anchor_smiles, anchor_count in anchor_frequency.items():
        # Convert SMILES to molecule for direct substructure matching
        anchor_mol = _smiles_to_mol_for_matching(anchor_smiles)
        if anchor_mol is None:
            continue
        
        # Anchor occurrence: fraction of ANALYZED compounds where this was identified as anchor
        anchor_occurrence = anchor_count / num_analyzed_compounds if num_analyzed_compounds > 0 else 0.0
        
        # Substructure occurrence: count ALL compounds in split containing this fragment
        # This is a population-wide statistic showing how prevalent the fragment is
        substructure_count = 0
        for compound_smiles in smiles_list:
            try:
                compound_mol = Chem.MolFromSmiles(compound_smiles)
                if compound_mol is not None and compound_mol.HasSubstructMatch(anchor_mol):
                    substructure_count += 1
            except Exception:
                pass
        
        substructure_occurrence = substructure_count / total_compounds_in_split if total_compounds_in_split > 0 else 0.0
        
        rule_entry = {
            "fragment": anchor_smiles,
            "substructure_occurrence": float(substructure_occurrence),
            "anchor_occurrence": float(anchor_occurrence),
            "num_compounds_with_substructure": int(substructure_count),
            "num_compounds_with_anchor": int(anchor_count),
            "single_fragment_rule": anchor_occurrence >= 0.8  # Very high consistency
        }
        
        recurrent_rules.append(rule_entry)
    
    # Sort by anchor occurrence (primary) then substructure occurrence (secondary)
    recurrent_rules.sort(
        key=lambda x: (x["anchor_occurrence"], x["substructure_occurrence"]),
        reverse=True
    )
    
    # Count single-fragment rules (anchor_occurrence >= 0.8)
    single_fragment_rules = [r for r in recurrent_rules if r["single_fragment_rule"]]
    
    # Filter if requested
    if single_fragment_rules_only:
        final_rules = single_fragment_rules
    else:
        final_rules = recurrent_rules
    
    # Limit to top N anchors if specified
    if top_n_anchors is not None and len(final_rules) > top_n_anchors:
        final_rules = final_rules[:top_n_anchors]
    
    return {
        "recurrent_rules": final_rules,
        "rule_details": final_rules,  # Alias for clarity
        "statistics": {
            "total_unique_anchors": len(anchor_frequency),
            "total_single_fragment_rules": len(single_fragment_rules),
            "filtered_rules_count": len(final_rules),
        },
        "batch_summary": batch_results.get("aggregate_statistics", {}),
        "target_class": target_class,
        "split": split,
        "num_analyzed_compounds": num_analyzed_compounds,
        "total_compounds_in_split": total_compounds_in_split,
        "single_fragment_rules_only": single_fragment_rules_only,
        "top_n_anchors_limit": top_n_anchors,
        "status": "completed"
    }


def visualize_molanchor_anchors(
    smiles: str,
    anchor_smiles_list: list[str],
    output_path: Optional[str] = None,
    size: tuple[int, int] = (400, 400),
    highlight_color: tuple[int, int, int] = (255, 100, 100),
    include_atom_indices: bool = True,
) -> dict[str, Any]:
    """
    Visualize identified MolAnchor anchor fragments highlighted on the molecular structure.
    
    This tool creates a molecular structure image with identified anchor fragments
    highlighted in a distinct color. Useful for understanding which substructures
    your model considers critical for a prediction.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the compound to visualize
    anchor_smiles_list : list[str]
        List of anchor SMILES strings to highlight (from explain_with_molanchor results)
    output_path : str, optional
        Path to save the visualization image (.png format).
        If not provided, image is saved to session directory: `session_dir/plots/molanchor_<timestamp>.png`
    size : tuple[int, int], optional
        Image dimensions in pixels (width, height). Default: (400, 400)
    highlight_color : tuple[int, int, int], optional
        RGB color for highlighting anchor atoms. Default: red (255, 100, 100)
        Examples: (255, 100, 100) = red, (100, 255, 100) = green, (100, 100, 255) = blue
    include_atom_indices : bool, optional
        If True, show atom indices on the structure (default: True)
    
    Returns
    -------
    dict
        Visualization result containing:
        - image: MCP Image object that renders directly in LM Studio GUI
        - image_path: path to saved PNG file for reference/download
        - num_anchors: number of anchor fragments highlighted
        - num_atoms_highlighted: total atom indices highlighted
        - anchor_info: list of dicts with anchor SMILES and atom indices matched
        - status: completion status
    
    Raises
    ------
    ValueError
        If main SMILES or anchor SMILES are invalid
    
    Examples
    --------
    Visualize anchors from a single prediction (renders automatically in LM Studio):
    
    >>> result = explain_with_molanchor(
    ...     smiles="CCO",
    ...     model_path="model.pkl"
    ... )
    >>> viz = visualize_molanchor_anchors(
    ...     smiles="CCO",
    ...     anchor_smiles_list=result["anchor_smiles"]
    ... )
    # The image renders directly in the LM Studio chat window
    # viz['image'] contains the MCP Image object
    
    With custom highlighting and output path:
    
    >>> viz = visualize_molanchor_anchors(
    ...     smiles="CCO",
    ...     anchor_smiles_list=["CC", "CO"],
    ...     output_path="outputs/anchors.png",
    ...     highlight_color=(100, 200, 255)  # cyan
    ... )
    """
    logger = _get_session_logger()
    
    # Parse main molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for main compound: {smiles}")
    
    # Validate anchor SMILES and build list of atom indices to highlight
    all_highlight_indices = set()
    anchor_info_list = []
    
    for anchor_smile in anchor_smiles_list:
        # Convert SMILES to molecule for direct substructure matching
        anchor_mol = _smiles_to_mol_for_matching(anchor_smile)
        if anchor_mol is None:
            continue  # Skip invalid anchors
        
        # Find all substructure matches in the main molecule
        matches = mol.GetSubstructMatches(anchor_mol)
        if matches:
            for match in matches:
                all_highlight_indices.update(match)
            
            anchor_info_list.append({
                "anchor_smiles": anchor_smile,
                "num_matches": len(matches),
                "atom_indices": sorted(list(set().union(*(list(m) for m in matches))))
            })
    
    if not all_highlight_indices:
        raise ValueError(
            f"No anchor SMILES could be found as substructures in the main compound. "
            f"Main SMILES: {smiles}, Anchors: {anchor_smiles_list}"
        )
    
    # Draw molecule with highlighted atoms
    highlight_atom_list = list(all_highlight_indices)
    
    # Use custom highlight colors
    highlight_atom_map = {atom_idx: highlight_color for atom_idx in highlight_atom_list}
    
    # Draw the molecule
    img = Draw.MolToImage(
        mol,
        size=size,
        kekulize=True,
        includeAtomNumbers=include_atom_indices,
        highlightAtoms=highlight_atom_list,
        highlightAtomColors=highlight_atom_map,
    )
    
    # Determine output path
    if output_path is None:
        output_path = logger.session_dir / "plots" / f"molanchor_{logger.session_id}.png"
    else:
        output_path = Path(output_path)
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    img.save(str(output_path))
    
    # Create MCP Image object for rendering in LM Studio
    mcp_image = MCPImage(path=output_path)
    
    return {
        "image": mcp_image,
        "image_path": str(output_path),
        "num_anchors": len(anchor_info_list),
        "anchor_info": anchor_info_list,
        "compound_smiles": smiles,
        "status": "completed"
    }


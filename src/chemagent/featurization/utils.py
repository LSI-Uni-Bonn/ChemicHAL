"""
Molecular utility helpers — SMILES validation and RDKit molecule construction.

Usage
-----
    from chemagent.featurization.utils import get_mol_list, validate_smiles

    mols = get_mol_list(["CCO", "c1ccccc1"])
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
from rdkit import Chem


# SMILES → RDKit Mol
def get_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    """Convert a list of SMILES strings to RDKit molecule objects.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.

    Returns
    -------
    List[Chem.Mol]
        One ``Chem.Mol`` per input SMILES, in the same order.

    Raises
    ------
    ValueError
        If any SMILES string cannot be parsed by RDKit.
    """
    mol_obj_list = [Chem.MolFromSmiles(s) for s in smiles_list]

    invalid = [s for s, m in zip(smiles_list, mol_obj_list) if m is None]
    if invalid:
        raise ValueError(
            f"The following SMILES are invalid:\n" + "\n".join(invalid)
        )

    return mol_obj_list  # type: ignore[return-value]


def validate_smiles(smiles_list: List[str]) -> dict:
    """Check each SMILES string for validity without raising.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings to validate.

    Returns
    -------
    dict
        ``{"valid": [...], "invalid": [...], "n_valid": int, "n_invalid": int}``
    """
    valid, invalid = [], []
    for s in smiles_list:
        (valid if Chem.MolFromSmiles(s) is not None else invalid).append(s)
    return {
        "valid":    valid,
        "invalid":  invalid,
        "n_valid":  len(valid),
        "n_invalid": len(invalid),
    }


# Reproducibility helpers
def set_seeds(seed: int) -> None:
    """Set Python, NumPy, and hash-seed random states.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed: int) -> None:
    """Alias for :func:`set_seeds` — sets all random states for reproducibility."""
    set_seeds(seed)


# Directory helpers
def create_directory(path: Union[str, Path], verbose: bool = True) -> Path:
    """Create *path* (and any missing parents) if it does not already exist.

    Parameters
    ----------
    path:
        Target directory path.
    verbose:
        Print a message when the directory is created.

    Returns
    -------
    Path
        Resolved ``Path`` object for the directory.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created new directory '{path_obj}'")
    return path_obj

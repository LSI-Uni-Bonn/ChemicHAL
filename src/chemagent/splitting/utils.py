"""
Shared helpers for the chemagent.splitting package.

Covers:
  - random seed management
  - split-proportion validation
  - Murcko scaffold generation (RDKit)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# Seed helpers
def set_random_seed(seed: Optional[int]) -> None:
    """Set Python and NumPy random seeds.

    Args:
    seed:
        Integer seed, or ``None`` to leave the global state unchanged.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


# Proportion validation
def validate_split_sizes(
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    """Raise ``ValueError`` if split proportions are invalid.

    Args:
    train_size, val_size, test_size:
        Fractions that must be non-negative and sum to 1.0.

    Raises:
    ValueError
        If any size is negative, or the sum is not 1.0.
    """
    if train_size < 0 or val_size < 0 or test_size < 0:
        raise ValueError("Split sizes must be non-negative.")
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total:.4f}. "
            f"(train={train_size}, val={val_size}, test={test_size})"
        )



# Scaffold helper
def generate_murcko_scaffold(smiles: str) -> str:
    """Return the canonical SMILES of the Murcko scaffold for *smiles*.

    Args:
    smiles:
        Input SMILES string.

    Returns:
    str
        Canonical SMILES of the Murcko scaffold, or ``""`` if *smiles* is
        invalid or the molecule has no ring system.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

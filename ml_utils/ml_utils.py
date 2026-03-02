"""Machine learning utilities for molecular fingerprinting and data handling."""

import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def create_directory(path: Union[str, Path], verbose: bool = True) -> Path:
    """Create a directory if it doesn't exist.
    
    Args:
        path: Path to the directory to create
        verbose: Whether to print creation message
        
    Returns:
        Path object of the created/existing directory
    """
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created new directory '{path_obj}'")
    return path_obj


def get_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    """Convert SMILES strings to RDKit molecule objects.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of RDKit Mol objects
        
    Raises:
        ValueError: If any SMILES string is invalid
    """
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    if None in mol_obj_list:
        invalid_smiles = [
            smiles for smiles, mol_obj in zip(smiles_list, mol_obj_list)
            if mol_obj is None
        ]
        invalid_smiles_str = "\n".join(invalid_smiles)
        raise ValueError(
            f"The following SMILES are invalid:\n{invalid_smiles_str}"
        )
    
    return mol_obj_list



def ECFP4(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
    sparse: bool = False
) -> Union[List[NDArray[np.uint8]], List]:
    """Convert SMILES strings to ECFP (Morgan) fingerprints.
    
    Extended-Connectivity Fingerprints (ECFP) are circular fingerprints that encode
    molecular structure. ECFP4 typically uses radius=2.
    
    Args:
        smiles_list: List of SMILES strings to convert
        n_bits: Number of bits in the fingerprint (default: 2048)
        radius: ECFP fingerprint radius (default: 2 for ECFP4)
        sparse: If True, return sparse fingerprints; if False, return dense numpy arrays
        
    Returns:
        List of fingerprints (sparse RDKit objects or numpy arrays)
        
    Raises:
        ValueError: If any SMILES string is invalid
    """
    mols = get_mol_list(smiles_list)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=n_bits, radius=radius)
    
    if sparse:
        fps_sparse = fp_gen.GetSparseFingerprints(mols)
        return list(fps_sparse)
    else:
        return [fp_gen.GetFingerprintAsNumPy(mol) for mol in mols]
    

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed: int) -> None:
    """Set global random state for deterministic behavior.
    
    This ensures reproducibility across Python's random, numpy, and hash-based operations.
    
    Args:
        seed: Integer seed value for all random number generators
    """
    set_seeds(seed=seed)
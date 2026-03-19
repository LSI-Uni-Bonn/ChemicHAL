"""
Molecular fingerprint generators.

Each public UpperCase function in this module is auto-discovered by
``dataset_loader_mcp.featurize_dataset()`` via ``inspect.signature``.
Adding a new method here makes it immediately available as a featurizer —
no changes needed elsewhere.

Supported fingerprints
----------------------
ECFP   : Extended-Connectivity Fingerprints (Morgan algorithm, bit-vector)
MACCS  : MACCS 166-bit structural-key fingerprints

Usage
-----
    from chemagent.featurization.fingerprints import ECFP, MACCS

    fps  = ECFP(["CCO", "c1ccccc1"], n_bits=2048, radius=2)
    maccs = MACCS(["CCO", "c1ccccc1"])
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict

from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from .utils import get_mol_list



# ECFP (Morgan) fingerprints
def ECFP(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
    sparse: bool = False,
    return_bit_info: bool = False,
) -> List | Tuple[List, Dict]:
    """Generate ECFP (Morgan) bit-vector fingerprints from SMILES strings.

    ECFP radius mapping
    -------------------
    ECFP4 = radius=2 (default, most common)
    ECFP6 = radius=3

    Parameters
    ----------
    smiles_list:
        List of SMILES strings, e.g. ``["CCO", "CC(=O)O", "c1ccccc1"]``.
    n_bits:
        Fingerprint length in bits. Typical values: 1024, 2048 (default), 4096.
    radius:
        Morgan radius. Use 2 for ECFP4 (default) or 3 for ECFP6.
    sparse:
        If ``True`` return RDKit sparse fingerprint objects instead of NumPy
        arrays. Useful for memory-efficient downstream processing.
    return_bit_info:
        If ``True``, also return bit information (which bits correspond to which atom environments).
        Required for MolAnchor explainability. Only works with the first molecule in smiles_list.

    Returns
    -------
    List
        One fingerprint per molecule, each a ``list[int]`` of length *n_bits*
        (or a sparse fingerprint object when *sparse=True*).
    Tuple[List, Dict] (if return_bit_info=True)
        (fingerprints_list, bit_info_dict) where bit_info_dict maps bit indices 
        to atom environment tuples (atom_idx, radius).

    Raises
    ------
    ValueError
        If any SMILES string is invalid.
    """
    mols = get_mol_list(smiles_list)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=n_bits, radius=radius)

    if sparse:
        fps = list(fp_gen.GetSparseFingerprints(mols))
    else:
        fps = [fp_gen.GetFingerprintAsNumPy(mol).tolist() for mol in mols]

    if return_bit_info:
        # Generate bit information for the first molecule
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        _ = fp_gen.GetFingerprint(mols[0], additionalOutput=ao)
        bit_info = ao.GetBitInfoMap()
        return fps, bit_info
    
    return fps


# MACCS keys
def MACCS(smiles_list: List[str]) -> List[List[int]]:
    """Generate 166-bit MACCS structural-key fingerprints from SMILES strings.

    MACCS keys encode the presence (1) or absence (0) of 166 predefined
    structural fragments. They are fixed-length, interpretable, and well-suited
    for substructure-based similarity searches.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.

    Returns
    -------
    List[List[int]]
        One 166-element integer list per molecule.

    Raises
    ------
    ValueError
        If any SMILES string is invalid.
    """
    mols = get_mol_list(smiles_list)
    return [
        list(rdMolDescriptors.GetMACCSKeysFingerprint(mol).ToList())
        for mol in mols
    ]



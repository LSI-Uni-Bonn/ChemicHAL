"""
chemagent.featurization — molecular featurization utilities.

Sub-modules
-----------
utils         : SMILES validation, RDKit mol objects, reproducibility helpers
fingerprints  : ECFP, MACCS, RDKitFP, AtomPairFP, TopologicalTorsionFP

Any public UpperCase function added to ``fingerprints.py`` is automatically
available as a ``method`` in ``dataset_loader_mcp.featurize_dataset()`` and
appears in ``list_featurizers()`` — no changes needed elsewhere.

Usage
-----
    from chemagent.featurization import ECFP, MACCS, get_mol_list

    fps = ECFP(["CCO", "c1ccccc1"], n_bits=2048, radius=2)
"""

from .utils import get_mol_list, validate_smiles, set_seeds, set_global_determinism, create_directory
from .fingerprints import ECFP, MACCS, RDKitFP, AtomPairFP, TopologicalTorsionFP

__all__ = [
    # utils
    "get_mol_list",
    "validate_smiles",
    "set_seeds",
    "set_global_determinism",
    "create_directory",
    # fingerprints
    "ECFP",
    "MACCS",
    "RDKitFP",
    "AtomPairFP",
    "TopologicalTorsionFP",
]

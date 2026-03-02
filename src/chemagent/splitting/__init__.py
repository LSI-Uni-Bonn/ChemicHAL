"""
chemagent.splitting — dataset splitting utilities.

Sub-modules
-----------
utils          : seed management, proportion validation, Murcko scaffold generation
random_split   : random and stratified random splits (sklearn-backed)
scaffold_split : Murcko scaffold-based split (prevents data leakage)
statistics     : split statistics, class distribution, leakage checks

Usage
-----
    from chemagent.splitting import random_split, scaffold_split, get_split_statistics

    idx = random_split(n_samples=1000, train_size=0.8, val_size=0.1, test_size=0.1, seed=42)
    idx = scaffold_split(smiles, 0.8, 0.1, 0.1, seed=42, labels=y)
    stats = get_split_statistics(idx)
"""

from .random_split   import random_split
from .scaffold_split import scaffold_split
from .statistics     import get_split_statistics, class_distribution, check_leakage
from .utils          import validate_split_sizes, set_random_seed, generate_murcko_scaffold

__all__ = [
    "random_split",
    "scaffold_split",
    "get_split_statistics",
    "class_distribution",
    "check_leakage",
    "validate_split_sizes",
    "set_random_seed",
    "generate_murcko_scaffold",
]

"""
Murcko scaffold-based dataset splitting.

Scaffold splitting groups molecules by their Murcko scaffold and assigns
entire scaffold groups to train / val / test, preventing data leakage from
structurally similar compounds.

When class labels are provided, scaffold groups are assigned greedily to
keep per-class proportions balanced across splits (stratified scaffold split).

Usage
-----
    from chemagent.splitting.scaffold_split import scaffold_split

    indices = scaffold_split(
        smiles_list=smiles,
        train_size=0.8, val_size=0.1, test_size=0.1,
        seed=42,
    )
    # {"train": [...], "val": [...], "test": [...]}
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .utils import validate_split_sizes, set_random_seed, generate_murcko_scaffold


def scaffold_split(
    smiles_list: List[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: Optional[int] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, List[int]]:
    """Split a molecular dataset by Murcko scaffold to avoid data leakage.

    All molecules sharing the same Murcko scaffold are assigned to the
    same split.  When *labels* are supplied, groups are assigned greedily
    based on per-class deficits to keep class ratios balanced.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings (one per sample).
    train_size:
        Fraction for the training set (default 0.8).
    val_size:
        Fraction for the validation set (default 0.1). Use ``0.0`` to skip.
    test_size:
        Fraction for the test set (default 0.1).
    seed:
        Random seed applied to scaffold-group ordering.
    labels:
        Optional 1-D list of integer class labels for stratification.

    Returns
    -------
    Dict[str, List[int]]
        Keys ``"train"``, ``"val"``, ``"test"`` each containing a list of
        integer sample indices.

    Raises
    ------
    ValueError
        If sizes are invalid, any SMILES is unparseable, or *labels*
        length mismatches *smiles_list*.
    """
    validate_split_sizes(train_size, val_size, test_size)
    set_random_seed(seed)

    n_samples = len(smiles_list)

    if labels is not None and len(labels) != n_samples:
        raise ValueError(
            f"labels length ({len(labels)}) must match smiles_list length ({n_samples})"
        )

    # ------------------------------------------------------------------
    # Build scaffold → molecule-index groups
    # ------------------------------------------------------------------
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = generate_murcko_scaffold(smi)
        if not scaffold:
            raise ValueError(
                f"Could not generate Murcko scaffold for SMILES at index {idx}: {smi!r}"
            )
        scaffold_to_indices[scaffold].append(idx)

    scaffold_groups = list(scaffold_to_indices.values())

    train_indices: List[int] = []
    val_indices:   List[int] = []
    test_indices:  List[int] = []

    train_target = int(n_samples * train_size)
    val_target   = int(n_samples * val_size)

    # ------------------------------------------------------------------
    # Stratified scaffold split
    # ------------------------------------------------------------------
    if labels is not None:
        labels_arr = np.array(labels, dtype=int)
        classes    = np.unique(labels_arr)

        class_train_target = {
            c: int(np.sum(labels_arr == c) * train_size) for c in classes
        }
        class_val_target = {
            c: int(np.sum(labels_arr == c) * val_size) for c in classes
        }

        # Sort largest groups first, shuffle within equal-size ties
        scaffold_groups.sort(key=len, reverse=True)
        random.shuffle(scaffold_groups)
        scaffold_groups.sort(key=len, reverse=True)

        class_train_count = {c: 0 for c in classes}
        class_val_count   = {c: 0 for c in classes}

        for group in scaffold_groups:
            group_labels = labels_arr[group]
            dominant = int(np.bincount(group_labels).argmax())

            train_deficit = class_train_target[dominant] - class_train_count[dominant]
            val_deficit   = class_val_target[dominant]   - class_val_count[dominant]

            if train_deficit > 0:
                train_indices.extend(group)
                for lbl in group_labels:
                    class_train_count[int(lbl)] += 1
            elif val_deficit > 0:
                val_indices.extend(group)
                for lbl in group_labels:
                    class_val_count[int(lbl)] += 1
            else:
                test_indices.extend(group)

    # ------------------------------------------------------------------
    # Plain scaffold split (no stratification)
    # ------------------------------------------------------------------
    else:
        random.shuffle(scaffold_groups)
        for group in scaffold_groups:
            if len(train_indices) < train_target:
                train_indices.extend(group)
            elif len(val_indices) < val_target:
                val_indices.extend(group)
            else:
                test_indices.extend(group)

    return {
        "train": train_indices,
        "val":   val_indices,
        "test":  test_indices,
    }

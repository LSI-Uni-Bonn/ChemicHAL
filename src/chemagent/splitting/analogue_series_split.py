"""
Analogue-series dataset splitting.

Molecules are grouped by a pre-computed core / scaffold label and entire
groups are assigned to train or test, preventing data leakage between
structurally related compounds.

Unlike :mod:`chemagent.splitting.scaffold_split`, this module does **not**
compute scaffolds from SMILES — it expects core labels that have already
been assigned (e.g. from a matched-molecular-pair analysis or a prior
Murcko extraction step).

Usage
-----
    from chemagent.splitting.analogue_series_split import analogue_series_split

    indices = analogue_series_split(
        cores=df["core"].tolist(),
        test_size=0.3,
        seed=42,
    )
    # {"train": [...], "val": [], "test": [...]}
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np


def analogue_series_split(
    cores: List[str],
    test_size: float = 0.3,
    seed: int = 42,
    n_attempts: int = 1,
    n_cpds_tolerance: int = 5,
) -> Dict[str, List[int]]:
    """Split a dataset by analogue series (core/scaffold groups) into train / test.

    Molecules sharing the same core are kept together (no cross-split leakage).
    The function tries up to *n_attempts* random shuffles and returns the first
    split whose test-set size falls within *n_cpds_tolerance* of the target.
    If no shuffle meets the tolerance the split with the closest test size is
    returned instead.

    Parameters
    ----------
    cores:
        List of core/scaffold identifiers, one per sample (e.g. a Murcko
        scaffold SMILES or any group label).
    test_size:
        Fraction of samples to place in the test set (default 0.3).
    seed:
        Base random seed (default 42).
    n_attempts:
        Number of random shuffles to try before accepting the best result
        (default 1).
    n_cpds_tolerance:
        Maximum allowed deviation (in number of compounds) between the actual
        and target test-set size (default 5).

    Returns
    -------
    Dict[str, List[int]]
        Keys ``"train"``, ``"val"``, ``"test"`` each containing a list of
        integer sample indices.  ``"val"`` is always empty.
    """
    n_samples = len(cores)
    n_total_test = int(np.floor(test_size * n_samples))

    core_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, core in enumerate(cores):
        core_to_indices[core].append(idx)

    core_groups = list(core_to_indices.values())
    rng = np.random.RandomState(seed)

    best_train: List[int] = []
    best_test: List[int] = []
    best_deviation = float("inf")

    for _ in range(n_attempts):
        order = rng.permutation(len(core_groups))
        shuffled = [core_groups[i] for i in order]

        train_index: List[int] = []
        test_index: List[int] = []

        for group in shuffled:
            if len(test_index) + len(group) <= n_total_test:
                test_index.extend(group)
            else:
                train_index.extend(group)

        deviation = abs(len(test_index) - n_total_test)
        if deviation < best_deviation:
            best_deviation = deviation
            best_train = train_index
            best_test = test_index

        if deviation <= n_cpds_tolerance:
            break

    return {
        "train": best_train,
        "val":   [],
        "test":  best_test,
    }

"""
Random and stratified random dataset splitting.

Uses scikit-learn ``train_test_split`` for the underlying splitting logic
so edge cases (very small sets, deterministic behaviour) are handled robustly.

Usage
-----
    from chemagent.splitting.random_split import random_split

    indices = random_split(
        n_samples=1000,
        train_size=0.8, val_size=0.1, test_size=0.1,
        seed=42,
    )
    # {"train": [...], "val": [...], "test": [...]}

    # Stratified (preserves class proportions):
    indices = random_split(1000, 0.8, 0.1, 0.1, seed=42, labels=y, stratified=True)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split

from .utils import validate_split_sizes, set_random_seed


def random_split(
    n_samples: int,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: Optional[int] = None,
    labels: Optional[List[int]] = None,
    stratified: bool = False,
) -> Dict[str, List[int]]:
    """Randomly split *n_samples* indices into train / val / test sets.

    When *stratified* is ``True`` and *labels* are supplied, each class is
    split proportionally using sklearn's ``train_test_split``.

    Args:
    n_samples:
        Total number of samples.
    train_size:
        Fraction allocated to training (default 0.8).
    val_size:
        Fraction allocated to validation (default 0.1). Use ``0.0`` to
        skip validation entirely.
    test_size:
        Fraction allocated to testing (default 0.1).
    seed:
        Random seed for reproducibility.
    labels:
        Optional 1-D array-like of class labels. Required when
        *stratified* is ``True``.
    stratified:
        If ``True``, both the train/test and val/test splits are
        stratified to preserve class proportions. Requires *labels*.

    Returns:
    Dict[str, List[int]]
        Keys ``"train"``, ``"val"``, ``"test"`` — each maps to a list of
        integer sample indices.

    Raises:
    ValueError
        If sizes are invalid, *labels* length mismatches *n_samples*, or
        *stratified* is ``True`` but *labels* is ``None``.
    """
    validate_split_sizes(train_size, val_size, test_size)
    set_random_seed(seed)

    if stratified and labels is None:
        raise ValueError("stratified=True requires labels to be provided.")

    if labels is not None and len(labels) != n_samples:
        raise ValueError(
            f"labels length ({len(labels)}) must match n_samples ({n_samples})"
        )

    all_indices = list(range(n_samples))
    stratify_arr = list(labels) if (stratified and labels is not None) else None

    #split off test first
    if test_size > 0:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_arr,
        )
        stratify_tv = (
            [labels[i] for i in train_val_idx]
            if (stratified and labels is not None)
            else None
        )
    else:
        train_val_idx = all_indices
        test_idx = []
        stratify_tv = stratify_arr

    #split val from remaining
    if val_size > 0 and len(train_val_idx) > 0:
        # Relative val size within the train+val portion
        rel_val = val_size / (train_size + val_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=rel_val,
            random_state=seed,
            stratify=stratify_tv,
        )
    else:
        train_idx = train_val_idx
        val_idx = []

    return {
        "train": list(train_idx),
        "val":   list(val_idx),
        "test":  list(test_idx),
    }

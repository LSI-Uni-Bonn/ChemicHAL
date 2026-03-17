"""
Split statistics and diagnostics.

Usage
-----
    from chemagent.splitting.statistics import get_split_statistics, class_distribution

    stats = get_split_statistics({"train": [...], "val": [...], "test": [...]})
    dist  = class_distribution({"train": train_idx, "test": test_idx}, labels=y)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def get_split_statistics(
    split_dict: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """Return count and percentage for each partition in *split_dict*.

    Parameters
    ----------
    split_dict:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` (or any subset)
        mapping to lists of integer indices.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-partition ``{"count": int, "percentage": float}`` plus a
        ``"total"`` entry.
    """
    total = sum(len(v) for v in split_dict.values())
    stats: Dict[str, Dict[str, float]] = {}

    for name, indices in split_dict.items():
        count = len(indices)
        pct   = round(count / total * 100, 2) if total > 0 else 0.0
        stats[name] = {"count": count, "percentage": pct}

    stats["total"] = {"count": total, "percentage": 100.0}
    return stats


def class_distribution(
    split_dict: Dict[str, List[int]],
    labels: List[int],
) -> Dict[str, Dict[str, int]]:
    """Return per-class counts for each partition.

    Useful for verifying that stratification preserved class ratios.

    Parameters
    ----------
    split_dict:
        Dict mapping partition names to lists of integer indices.
    labels:
        Full label array (indexed by the values in *split_dict*).

    Returns
    -------
    Dict[str, Dict[str, int]]
        ``{partition: {class_label: count, ...}, ...}``
    """
    labels_arr = np.array(labels)
    classes    = sorted(np.unique(labels_arr).tolist())
    result: Dict[str, Dict[str, int]] = {}

    for name, indices in split_dict.items():
        if not indices:
            result[name] = {str(c): 0 for c in classes}
            continue
        subset = labels_arr[indices]
        result[name] = {str(c): int(np.sum(subset == c)) for c in classes}

    return result

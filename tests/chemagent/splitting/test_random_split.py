"""
Tests for chemagent.splitting.random_split.

Covers:
- Return keys are always "train", "val", "test"
- Indices are non-overlapping and cover every sample
- Split sizes are approximately correct
- val_size=0 produces an empty val split
- Reproducibility: same seed → same indices
- Different seeds → different indices
- Stratified split preserves class proportions
- Error on stratified=True without labels
- Error on labels length mismatch
"""

import numpy as np
import pytest

from chemagent.splitting.random_split import random_split


N = 200
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_indices(result: dict) -> set:
    return set(result["train"]) | set(result["val"]) | set(result["test"])


def _no_overlap(result: dict) -> bool:
    train = set(result["train"])
    val   = set(result["val"])
    test  = set(result["test"])
    return len(train & val) == 0 and len(train & test) == 0 and len(val & test) == 0


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestRandomSplitStructure:
    def test_keys(self):
        result = random_split(N, seed=SEED)
        assert set(result.keys()) == {"train", "val", "test"}

    def test_covers_all_indices(self):
        result = random_split(N, seed=SEED)
        assert _all_indices(result) == set(range(N))

    def test_no_overlap(self):
        result = random_split(N, seed=SEED)
        assert _no_overlap(result)

    def test_sizes_approx(self):
        result = random_split(N, train_size=0.8, val_size=0.1, test_size=0.1, seed=SEED)
        assert abs(len(result["train"]) - 160) <= 2
        assert abs(len(result["val"])   - 20)  <= 2
        assert abs(len(result["test"])  - 20)  <= 2

    def test_no_val(self):
        result = random_split(N, train_size=0.7, val_size=0.0, test_size=0.3, seed=SEED)
        assert result["val"] == []
        assert _all_indices(result) == set(range(N))

    def test_returns_lists(self):
        result = random_split(N, seed=SEED)
        assert isinstance(result["train"], list)
        assert isinstance(result["val"],   list)
        assert isinstance(result["test"],  list)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestRandomSplitReproducibility:
    def test_same_seed_same_result(self):
        r1 = random_split(N, seed=SEED)
        r2 = random_split(N, seed=SEED)
        assert r1["train"] == r2["train"]
        assert r1["val"]   == r2["val"]
        assert r1["test"]  == r2["test"]

    def test_different_seeds_differ(self):
        r1 = random_split(N, seed=0)
        r2 = random_split(N, seed=99)
        assert r1["train"] != r2["train"]


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

class TestRandomSplitStratified:
    @pytest.fixture(scope="class")
    def balanced_labels(self):
        return [i % 2 for i in range(N)]  # exactly 50 % each class

    def test_stratified_preserves_ratio(self, balanced_labels):
        result = random_split(
            N, train_size=0.8, val_size=0.1, test_size=0.1,
            seed=SEED, labels=balanced_labels, stratified=True,
        )
        for split in ("train", "val", "test"):
            labels_in_split = [balanced_labels[i] for i in result[split]]
            n_pos = sum(labels_in_split)
            ratio = n_pos / len(labels_in_split)
            assert abs(ratio - 0.5) < 0.1, f"{split} class ratio {ratio:.2f} deviates too much"

    def test_stratified_requires_labels(self):
        with pytest.raises(ValueError, match="labels"):
            random_split(N, stratified=True)

    def test_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            random_split(N, labels=[0, 1, 0])  # wrong length


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRandomSplitEdgeCases:
    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            random_split(N, train_size=0.5, val_size=0.1, test_size=0.1)  # sum != 1

    def test_small_dataset(self):
        result = random_split(10, train_size=0.8, val_size=0.0, test_size=0.2, seed=SEED)
        assert _all_indices(result) == set(range(10))
        assert _no_overlap(result)

"""
Tests for chemagent.splitting.utils.

Covers:
- validate_split_sizes: accepts valid proportions, rejects negatives and bad sums
- set_random_seed: produces reproducible NumPy and Python random draws
- generate_murcko_scaffold: returns correct scaffold for ring-containing SMILES,
  empty string for acyclic molecules, and empty string for invalid SMILES
"""

import random

import numpy as np
import pytest

from chemagent.splitting.utils import (
    generate_murcko_scaffold,
    set_random_seed,
    validate_split_sizes,
)


# ---------------------------------------------------------------------------
# validate_split_sizes
# ---------------------------------------------------------------------------

class TestValidateSplitSizes:
    def test_valid_three_way(self):
        validate_split_sizes(0.8, 0.1, 0.1)  # should not raise

    def test_valid_no_val(self):
        validate_split_sizes(0.7, 0.0, 0.3)  # should not raise

    def test_negative_train(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_split_sizes(-0.1, 0.5, 0.6)

    def test_negative_val(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_split_sizes(0.8, -0.1, 0.3)

    def test_negative_test(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_split_sizes(0.8, 0.1, -0.1)

    def test_sum_not_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_split_sizes(0.6, 0.1, 0.1)

    def test_sum_slightly_over(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_split_sizes(0.8, 0.15, 0.1)


# ---------------------------------------------------------------------------
# set_random_seed
# ---------------------------------------------------------------------------

class TestSetRandomSeed:
    def test_numpy_reproducible(self):
        set_random_seed(42)
        a = np.random.rand(5)
        set_random_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible(self):
        set_random_seed(7)
        a = [random.random() for _ in range(5)]
        set_random_seed(7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_none_does_not_raise(self):
        set_random_seed(None)  # should not raise


# ---------------------------------------------------------------------------
# generate_murcko_scaffold
# ---------------------------------------------------------------------------

class TestGenerateMurckoScaffold:
    def test_benzene_returns_itself(self):
        scaffold = generate_murcko_scaffold("c1ccccc1")
        assert scaffold != ""
        # benzene has no substituents — scaffold is benzene itself
        assert "c1ccccc1" in scaffold or scaffold == "c1ccccc1"

    def test_substituted_benzenes_share_scaffold(self):
        s1 = generate_murcko_scaffold("c1ccc(O)cc1")
        s2 = generate_murcko_scaffold("c1ccc(N)cc1")
        s3 = generate_murcko_scaffold("c1ccc(F)cc1")
        assert s1 == s2 == s3

    def test_acyclic_smiles_returns_empty(self):
        # Murcko scaffold of a pure chain is empty
        assert generate_murcko_scaffold("CCCC") == ""

    def test_invalid_smiles_returns_empty(self):
        assert generate_murcko_scaffold("not_a_smiles") == ""

    def test_different_ring_systems_differ(self):
        benzene_scaffold = generate_murcko_scaffold("c1ccccc1C")
        pyridine_scaffold = generate_murcko_scaffold("c1ccncc1C")
        assert benzene_scaffold != pyridine_scaffold

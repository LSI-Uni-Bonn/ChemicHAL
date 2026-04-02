"""
Tests for chemagent.splitting.analogue_series_split.

Covers:
- Return keys are always "train", "val", "test"
- "val" is always empty
- Indices are non-overlapping and cover every sample
- Molecules sharing the same core always land in the same partition
- Test-set size is approximately correct (within tolerance)
- Reproducibility: same seed → same result
- Different seeds can produce different splits
- n_attempts > 1 improves tolerance adherence
- Single-core dataset: everything goes to train (test target unreachable)
"""

import pytest

from chemagent.splitting.analogue_series_split import analogue_series_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 30 molecules spread across 6 analogue series (5 per core)
CORES_BALANCED = (
    ["core_A"] * 5
    + ["core_B"] * 5
    + ["core_C"] * 5
    + ["core_D"] * 5
    + ["core_E"] * 5
    + ["core_F"] * 5
)  # 30 total

# Unequal series sizes
CORES_UNEQUAL = (
    ["core_A"] * 10
    + ["core_B"] * 4
    + ["core_C"] * 6
    + ["core_D"] * 2
    + ["core_E"] * 8
)  # 30 total

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_indices(result: dict) -> set:
    return set(result["train"]) | set(result["val"]) | set(result["test"])


def _no_overlap(result: dict) -> bool:
    train = set(result["train"])
    test  = set(result["test"])
    return len(train & test) == 0


def _core_of(idx: int, cores: list) -> str:
    return cores[idx]


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestAnalogueSeriesSplitStructure:
    def test_keys(self):
        result = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert set(result.keys()) == {"train", "val", "test"}

    def test_val_always_empty(self):
        result = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert result["val"] == []

    def test_covers_all_indices(self):
        result = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert _all_indices(result) == set(range(len(CORES_BALANCED)))

    def test_no_overlap(self):
        result = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert _no_overlap(result)

    def test_returns_lists(self):
        result = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert isinstance(result["train"], list)
        assert isinstance(result["val"],   list)
        assert isinstance(result["test"],  list)


# ---------------------------------------------------------------------------
# Core grouping integrity
# ---------------------------------------------------------------------------

class TestAnalogueSeriesGrouping:
    def test_same_core_same_split(self):
        """Every molecule from the same core must be in the same partition."""
        result = analogue_series_split(CORES_BALANCED, seed=SEED)

        train_cores = {_core_of(i, CORES_BALANCED) for i in result["train"]}
        test_cores  = {_core_of(i, CORES_BALANCED) for i in result["test"]}

        # No core should appear in both train and test
        assert train_cores.isdisjoint(test_cores), (
            f"Cores appear in both splits: {train_cores & test_cores}"
        )

    def test_same_core_same_split_unequal(self):
        result = analogue_series_split(CORES_UNEQUAL, seed=SEED)

        train_cores = {_core_of(i, CORES_UNEQUAL) for i in result["train"]}
        test_cores  = {_core_of(i, CORES_UNEQUAL) for i in result["test"]}

        assert train_cores.isdisjoint(test_cores)


# ---------------------------------------------------------------------------
# Test-set size
# ---------------------------------------------------------------------------

class TestAnalogueSeriesTestSize:
    def test_test_size_approx(self):
        result = analogue_series_split(
            CORES_BALANCED, test_size=0.3, seed=SEED,
            n_attempts=10, n_cpds_tolerance=5,
        )
        n_total = len(CORES_BALANCED)
        target  = int(0.3 * n_total)  # 9
        assert abs(len(result["test"]) - target) <= 5

    def test_more_attempts_meets_tolerance(self):
        """With enough attempts the tolerance should be met for balanced data."""
        result = analogue_series_split(
            CORES_BALANCED, test_size=0.3, seed=SEED,
            n_attempts=20, n_cpds_tolerance=5,
        )
        n_total = len(CORES_BALANCED)
        target  = int(0.3 * n_total)
        assert abs(len(result["test"]) - target) <= 5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestAnalogueSeriesReproducibility:
    def test_same_seed_same_result(self):
        r1 = analogue_series_split(CORES_BALANCED, seed=SEED)
        r2 = analogue_series_split(CORES_BALANCED, seed=SEED)
        assert r1["train"] == r2["train"]
        assert r1["test"]  == r2["test"]

    def test_different_seeds_differ(self):
        results = [
            tuple(sorted(analogue_series_split(CORES_BALANCED, seed=s)["test"]))
            for s in range(10)
        ]
        assert len(set(results)) > 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAnalogueSeriesEdgeCases:
    def test_single_core_all_train(self):
        """When only one core exists the test target can't be filled; train gets all."""
        cores = ["core_A"] * 20
        result = analogue_series_split(cores, test_size=0.3, seed=SEED)
        # The only group is larger than the test target, so it goes to train
        assert set(result["train"]) == set(range(20))
        assert result["test"] == []

    def test_two_cores_split(self):
        cores = ["core_A"] * 15 + ["core_B"] * 15
        result = analogue_series_split(cores, test_size=0.5, seed=SEED)
        assert _all_indices(result) == set(range(30))
        assert _no_overlap(result)
        # Each partition should have exactly one core
        train_cores = {cores[i] for i in result["train"]}
        test_cores  = {cores[i] for i in result["test"]}
        assert len(train_cores) == 1
        assert len(test_cores)  == 1
        assert train_cores != test_cores

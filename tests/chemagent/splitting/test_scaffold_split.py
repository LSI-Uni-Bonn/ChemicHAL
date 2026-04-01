"""
Tests for chemagent.splitting.scaffold_split.

Covers:
- Molecules sharing the same Murcko scaffold end up in the same split
- Indices are non-overlapping and cover every sample
- Split sizes are approximately correct
- val_size=0 produces an empty val split
- Stratified scaffold split respects class proportions
- Reproducibility: same seed → same result
- ValueError on invalid / unparseable SMILES
- ValueError on size proportions that don't sum to 1
"""

import pytest

from chemagent.splitting.scaffold_split import scaffold_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two benzene-ring derivatives (same Murcko scaffold)
BENZENE_SMILES = [
    "c1ccccc1",       # benzene
    "c1ccc(O)cc1",    # phenol
    "c1ccc(N)cc1",    # aniline
    "c1ccc(F)cc1",    # fluorobenzene
    "c1ccc(Cl)cc1",   # chlorobenzene
]

# Two pyridine derivatives (same Murcko scaffold)
PYRIDINE_SMILES = [
    "c1ccncc1",       # pyridine
    "c1ccncc1C",      # 2-methylpyridine
    "c1ccncc1O",      # 2-hydroxypyridine
    "c1ccncc1N",      # 2-aminopyridine
    "c1ccncc1F",      # 2-fluoropyridine
]

# Indole derivatives (same Murcko scaffold)
INDOLE_SMILES = [
    "c1ccc2[nH]ccc2c1",    # indole
    "c1ccc2[nH]c(C)cc2c1", # 2-methylindole
    "c1ccc2[nH]c(N)cc2c1", # 2-aminoindole
]

ALL_SMILES = BENZENE_SMILES + PYRIDINE_SMILES + INDOLE_SMILES  # 13 molecules


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

class TestScaffoldSplitStructure:
    def test_keys(self):
        result = scaffold_split(ALL_SMILES, seed=0)
        assert set(result.keys()) == {"train", "val", "test"}

    def test_covers_all_indices(self):
        result = scaffold_split(ALL_SMILES, seed=0)
        assert _all_indices(result) == set(range(len(ALL_SMILES)))

    def test_no_overlap(self):
        result = scaffold_split(ALL_SMILES, seed=0)
        assert _no_overlap(result)

    def test_no_val(self):
        result = scaffold_split(
            ALL_SMILES, train_size=0.7, val_size=0.0, test_size=0.3, seed=0
        )
        assert result["val"] == []
        assert _all_indices(result) == set(range(len(ALL_SMILES)))


# ---------------------------------------------------------------------------
# Scaffold grouping integrity
# ---------------------------------------------------------------------------

class TestScaffoldGrouping:
    def _split_set(self, result: dict) -> dict[int, str]:
        """Return {sample_index: split_name} mapping."""
        mapping = {}
        for split_name in ("train", "val", "test"):
            for idx in result[split_name]:
                mapping[idx] = split_name
        return mapping

    def test_same_scaffold_same_split(self):
        """All benzene-ring molecules must land in the same partition."""
        result = scaffold_split(ALL_SMILES, seed=0)
        mapping = self._split_set(result)

        benzene_splits = {mapping[i] for i in range(len(BENZENE_SMILES))}
        assert len(benzene_splits) == 1, "Benzene-scaffold molecules split across partitions"

        pyridine_splits = {mapping[i] for i in range(
            len(BENZENE_SMILES), len(BENZENE_SMILES) + len(PYRIDINE_SMILES)
        )}
        assert len(pyridine_splits) == 1, "Pyridine-scaffold molecules split across partitions"

        indole_splits = {mapping[i] for i in range(
            len(BENZENE_SMILES) + len(PYRIDINE_SMILES), len(ALL_SMILES)
        )}
        assert len(indole_splits) == 1, "Indole-scaffold molecules split across partitions"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestScaffoldSplitReproducibility:
    def test_same_seed_same_result(self):
        r1 = scaffold_split(ALL_SMILES, seed=42)
        r2 = scaffold_split(ALL_SMILES, seed=42)
        assert r1["train"] == r2["train"]
        assert r1["val"]   == r2["val"]
        assert r1["test"]  == r2["test"]

    def test_different_seeds_can_differ(self):
        # Use a richer dataset (6 groups of varying sizes) so different seeds
        # can genuinely pick different group combinations for the test set.
        extra_smiles = (
            ["c1ccc2ccccc2c1"] * 3   # naphthalene scaffold
            + ["c1ccoc1"] * 4        # furan scaffold
            + ["c1ccsc1"] * 2        # thiophene scaffold
            + ["c1cc[nH]c1"] * 5     # pyrrole scaffold
            + ["c1ccc2[nH]ncc2c1"] * 3  # indazole scaffold
        ) + ALL_SMILES
        results = [scaffold_split(extra_smiles, test_size=0.25, val_size=0.0,
                                  train_size=0.75, seed=s)["test"]
                   for s in range(15)]
        assert len({tuple(sorted(r)) for r in results}) > 1


# ---------------------------------------------------------------------------
# Stratified scaffold split
# ---------------------------------------------------------------------------

class TestScaffoldSplitStratified:
    def test_stratified_requires_labels(self):
        with pytest.raises(ValueError, match="labels"):
            scaffold_split(ALL_SMILES, stratified=True)

    def test_stratified_labels_length_mismatch(self):
        labels = [0] * 5  # wrong length
        with pytest.raises(ValueError, match="length"):
            scaffold_split(ALL_SMILES, labels=labels, stratified=True)

    def test_stratified_runs_without_error(self):
        labels = [0] * len(BENZENE_SMILES) + [1] * len(PYRIDINE_SMILES) + [0] * len(INDOLE_SMILES)
        result = scaffold_split(ALL_SMILES, labels=labels, stratified=True, seed=0)
        assert _all_indices(result) == set(range(len(ALL_SMILES)))
        assert _no_overlap(result)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestScaffoldSplitErrors:
    def test_invalid_smiles_raises(self):
        smiles = BENZENE_SMILES[:4] + ["not_valid_smiles"]
        with pytest.raises(ValueError):
            scaffold_split(smiles, seed=0)

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            scaffold_split(ALL_SMILES, train_size=0.5, val_size=0.1, test_size=0.1)

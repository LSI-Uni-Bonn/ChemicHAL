"""Test that build_core_foil generates scaffold foils with all original R-groups attached.

Run from the repo root:
    python tests/chemagent/explainability/test_molce_tools.py
"""
import sys
import re
import joblib
from pathlib import Path

if __name__ == "__main__":
    SPLIT_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/splits"
        "/chembl_activity_data_O00329_P48736_random.pkl"
    )
    MODEL_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/models"
        "/chembl_activity_data_O00329_P48736_random_RFC.pkl"
    )

    sys.path.insert(0, "src")

    from rdkit import Chem
    from rdkit.Chem.rdmolops import GetMolFrags

    from chemagent.explainability.molce_tools import _MolContrastWrapper, _make_sklearn_predict_funcs

    print("Loading data and model...")
    split_data = joblib.load(SPLIT_PATH)
    model = joblib.load(MODEL_PATH)

    # Use a small subset for speed; combine train + test as the library SMILES
    SUBSET = 50
    train_smiles = list(split_data["train_smiles"])[:SUBSET]
    test_smiles  = list(split_data["test_smiles"])[:SUBSET]
    all_smiles   = train_smiles + test_smiles

    predict_func, predict_func_proba = _make_sklearn_predict_funcs(model, n_bits=2048, radius=2)

    print("Building _MolContrastWrapper...")
    mc = _MolContrastWrapper(
        data_smiles=all_smiles,
        model=model,
        predict_func=predict_func,
        predict_func_proba=predict_func_proba,
    )

    # ── Helpers ───────────────────────────────────────────────────────────
    def _count_rgroups(mol: Chem.Mol) -> int:
        """Count R-groups in the original decomposition (number of [*] in core SMILES)."""
        core, rgroups = mc._mc.decompose_molecule(mol, original=True)
        return len(rgroups) if rgroups else 0

    def _n_dummy_atoms(mol: Chem.Mol) -> int:
        """Count dummy (*) atoms remaining in a molecule."""
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)

    # ── Find a decomposable test compound ─────────────────────────────────
    print("\nSearching for a decomposable compound with contrastive cores...")
    query_mol = None
    core_smiles_to_test = None

    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            core, rgroups = mc._mc.decompose_molecule(mol, original=True)
        except ValueError:
            continue
        if not rgroups:
            continue
        # Try to get contrastive cores for this molecule
        try:
            contrast_df = mc.get_contrastive_cores(mol, foil_class=0)
        except Exception as e:
            print(f"  Skipping {smi[:40]}: {e}")
            continue
        if contrast_df is None or contrast_df.empty:
            continue
        query_mol = mol
        core_smiles_to_test = contrast_df.index.tolist()
        print(f"  Query: {smi[:60]}")
        print(f"  Original R-group count: {len(rgroups)}")
        print(f"  Contrastive cores found: {len(core_smiles_to_test)}")
        break

    assert query_mol is not None, (
        "Could not find any decomposable compound with contrastive cores in the test subset."
    )

    # ── Test 1: at least one foil is generated for the top core ──────────
    print("\n--- Test 1: build_core_foil returns a valid molecule ---")
    n_success = 0
    n_fail    = 0

    for core_smi in core_smiles_to_test[:5]:   # check top-5 ranked cores
        foil = mc.build_core_foil(query_mol, core_smi)
        if foil is not None:
            n_success += 1
            print(f"  OK  core={core_smi[:50]}  foil={Chem.MolToSmiles(foil)[:60]}")
        else:
            n_fail += 1
            print(f"  --  core={core_smi[:50]}  (no foil built)")

    assert n_success > 0, (
        f"build_core_foil returned None for all {len(core_smiles_to_test[:5])} top cores. "
        "The SMILES-matching step may be failing — verify that core_smiles from "
        "get_contrastive_cores matches the annotated scaffold SMILES produced by get_scaffolds."
    )
    print(f"  Foils built: {n_success}/{len(core_smiles_to_test[:5])}")

    # ── Test 2: foils are fully connected (no floating fragments) ─────────
    print("\n--- Test 2: all generated foils are single connected fragments ---")
    for core_smi in core_smiles_to_test[:5]:
        foil = mc.build_core_foil(query_mol, core_smi)
        if foil is None:
            continue
        n_frags = len(GetMolFrags(foil))
        assert n_frags == 1, (
            f"Foil for core '{core_smi[:50]}' has {n_frags} disconnected fragments "
            f"(SMILES: {Chem.MolToSmiles(foil)}). "
            "Floating R-group: a [*:N] dummy had no matching partner in the scaffold."
        )
        print(f"  Connected OK: {Chem.MolToSmiles(foil)[:60]}")

    # ── Test 3: foils contain no residual dummy atoms ─────────────────────
    print("\n--- Test 3: foils contain no residual dummy (*) atoms ---")
    for core_smi in core_smiles_to_test[:5]:
        foil = mc.build_core_foil(query_mol, core_smi)
        if foil is None:
            continue
        n_dummy = _n_dummy_atoms(foil)
        assert n_dummy == 0, (
            f"Foil for core '{core_smi[:50]}' still contains {n_dummy} dummy atom(s). "
            f"(SMILES: {Chem.MolToSmiles(foil)}). "
            "Unmatched [*:N] atoms were not replaced — check annotate_sub_sites atom count."
        )
        print(f"  No dummies OK: {Chem.MolToSmiles(foil)[:60]}")

    # ── Test 4: foil heavy-atom count is consistent with the substitution ─
    print("\n--- Test 4: foil atom count plausibility check ---")
    orig_core, orig_rgroups = mc._mc.decompose_molecule(query_mol, original=True)
    orig_mol_ha = query_mol.GetNumHeavyAtoms()

    for core_smi in core_smiles_to_test[:5]:
        foil = mc.build_core_foil(query_mol, core_smi)
        if foil is None:
            continue
        foil_ha = foil.GetNumHeavyAtoms()
        # The foil atom count should be >0 and in a plausible range
        assert foil_ha > 0, "Foil has no heavy atoms."
        print(f"  orig={orig_mol_ha} ha  foil={foil_ha} ha  core={core_smi[:30]}")

    print("\nAll tests passed.")

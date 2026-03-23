"""Quick test for generate_counterfactuals and CFGenerator (multi-substituent)."""
import sys
import joblib
from pathlib import Path

# Windows multiprocessing (used by ccrlib) requires the __main__ guard.
if __name__ == "__main__":
    SPLIT_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/splits"
        "/chembl_activity_data_O00329_P48736_random.pkl"
    )
    MODEL_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/models"
        "/chembl_activity_data_O00329_P48736_random_RFC.pkl"
    )

    print("Loading split file...")
    split_data = joblib.load(SPLIT_PATH)

    SUBSET = 50
    patched_path = Path("data/logs/test_cf_split_small.pkl")
    patched = {k: v[:SUBSET] if hasattr(v, "__len__") else v for k, v in split_data.items()}
    joblib.dump(patched, patched_path)
    print(f"Patched split: train={len(patched['train_smiles'])}  test={len(patched['test_smiles'])}")

    sys.path.insert(0, "src")
    from chemagent.explainability.counterfactual_tools import generate_counterfactuals

    # ── Test 1: single substituents (default) ─────────────────────────────
    print("\n--- Test 1: single substituents ---")
    result = generate_counterfactuals(
        split_file_path=str(patched_path),
        model_path=str(MODEL_PATH),
        target_class=1,
        max_cfs_per_compound=5,
        seed=42,
    )
    print(f"Status:              {result['status']}")
    print(f"Unique SMILES used:  {result['num_total_smiles']}")
    print(f"CFs generated:       {result['num_cfs_found']}")
    print(f"Compounds with CFs:  {result['num_compounds_with_cfs']}")
    for smi, compound in list(result["results_by_compound"].items())[:2]:
        print(f"\n  Query: {smi[:60]}")
        for cf in compound["counterfactuals"][:2]:
            print(f"    CF: {cf['cf_smiles'][:60]}  class={cf['predicted_class']}")

    # ── Test 2: multiple substituents with sample limit ────────────────────
    print("\n--- Test 2: use_multiple_substituents=True, max_double_sub_samples=100 ---")
    from chemagent.explainability.Counterfactuals.CF_generator_v2 import CFGenerator

    model = joblib.load(MODEL_PATH)
    all_smiles = list(patched["train_smiles"]) + list(patched["test_smiles"])

    cf_gen = CFGenerator(
        smiles_lst=all_smiles,
        model_obj=model,
        use_multiple_substituents=True,
        max_double_sub_samples=100,
        random_seed=42,
        num_classes=2,
    )
    cf_df = cf_gen.find_cfs()
    print(f"CFs found (multi-sub): {len(cf_df)}")
    if not cf_df.empty:
        print(cf_df[["CF_smiles", "core_smiles", "Predicted"]].head(3).to_string(index=False))

    print("\nAll tests passed.")

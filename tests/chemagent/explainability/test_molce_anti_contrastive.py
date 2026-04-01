"""Quick test for MolCE anti-contrastive analysis.

Run from repo root:
    python tests/chemagent/explainability/test_molce_anti_contrastive.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, "src")

MODEL_PATH = (
    "data/logs/session_alamens_20260318_110929_6a0d6a/models"
    "/chembl_activity_data_O00329_P48736_random_RFC.pkl"
)
SPLIT_PATH = (
    "data/logs/session_alamens_20260318_110929_6a0d6a/splits"
    "/chembl_activity_data_O00329_P48736_random.pkl"
)
OUTPUT_BASE = "plots/molce_anti_contrastive_test"

import joblib
from rdkit import Chem
from chemagent.explainability.molce_tools import (
    _make_sklearn_predict_funcs,
    _MolContrastWrapper,
    explain_with_molce,
)

# ── Pick a decomposable test compound ────────────────────────────────────────
print("Loading split data...")
split_data = joblib.load(SPLIT_PATH)
model = joblib.load(MODEL_PATH)
predict_func, predict_func_proba = _make_sklearn_predict_funcs(model, n_bits=2048, radius=2)

all_smiles = list(split_data.get("train_smiles", [])) + list(split_data.get("test_smiles", []))
mc = _MolContrastWrapper(
    data_smiles=all_smiles,
    model=model,
    predict_func=predict_func,
    predict_func_proba=predict_func_proba,
)

print("Searching for a suitable query compound...")
query_smiles = None
for smi in split_data.get("test_smiles", []):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    try:
        core, rgroups = mc._mc.decompose_molecule(mol, original=True)
    except ValueError:
        continue
    if not rgroups:
        continue
    try:
        rdf = mc.get_contrastive_rgroups(mol, foil_class=0)
        # Need at least some negative scores for anti-contrastive to be interesting
        if (rdf["contrast"] < 0).any():
            query_smiles = smi
            print(f"  Found: {smi[:70]}")
            break
    except Exception:
        continue

assert query_smiles is not None, "No suitable compound found."

# ── Run explain_with_molce with include_anti_contrastive=True ─────────────
print("\nRunning explain_with_molce with include_anti_contrastive=True ...")
results = explain_with_molce(
    smiles=query_smiles,
    model_path=MODEL_PATH,
    split_file_path=SPLIT_PATH,
    foil_class=0,
    output_path=OUTPUT_BASE,
    include_anti_contrastive=True,
)

# Last element is the JSON metadata string
metadata = json.loads(results[-1])
print("\n=== Metadata ===")
print(json.dumps(metadata, indent=2))

# Verify anti-contrastive keys are present
assert "anti_contrastive_rgroups" in metadata, "anti_contrastive_rgroups missing from output"
assert "anti_contrastive_scaffolds" in metadata, "anti_contrastive_scaffolds missing from output"

print("\n=== Contrastive R-groups ===")
for r in metadata["contrastive_rgroups"]:
    print(f"  Rank {r['rank']}: {r['r_group_smiles']}  score={r['contrast_score']:.3f}")

print("\n=== Anti-contrastive R-groups ===")
for r in metadata["anti_contrastive_rgroups"]:
    print(f"  Anti-{r['rank']}: {r['r_group_smiles']}  score={r['contrast_score']:.3f}")

print("\n=== Contrastive scaffolds ===")
for r in metadata["contrastive_scaffolds"]:
    print(f"  Rank {r['rank']}: {r['core_smiles'][:60]}  score={r['contrast_score']:.3f}")

print("\n=== Anti-contrastive scaffolds ===")
for r in metadata["anti_contrastive_scaffolds"]:
    print(f"  Anti-{r['rank']}: {r['core_smiles'][:60]}  score={r['contrast_score']:.3f}")

print(f"\nImages saved to:")
print(f"  R-groups:  {metadata.get('image_path_rgroups')}")
print(f"  Scaffolds: {metadata.get('image_path_scaffolds')}")

print("\nAll assertions passed.")

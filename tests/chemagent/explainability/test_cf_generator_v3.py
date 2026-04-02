"""Test CFGenerator v3 and visualize_counterfactuals for three class-1 compounds.

Loads a model and split file, predicts the test set, selects three correctly
predicted class-1 compounds, runs counterfactual generation for each, and saves
the visualisation images to plots/.

Run from the repo root:
    python tests/chemagent/explainability/test_cf_generator_v3.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    # ── Paths ────────────────────────────────────────────────────────────────
    SPLIT_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/splits"
        "/chembl_activity_data_O00329_P48736_random.pkl"
    )
    MODEL_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/models"
        "/chembl_activity_data_O00329_P48736_random_RFC.pkl"
    )
    PLOTS_DIR = Path("plots")
    PLOTS_DIR.mkdir(exist_ok=True)

    N_BITS  = 2048
    RADIUS  = 2
    TARGET_CLASS = 1
    N_COMPOUNDS  = 3   # how many class-1 compounds to analyse

    sys.path.insert(0, "src")

    # ── Imports ──────────────────────────────────────────────────────────────
    import joblib
    import numpy as np
    from rdkit import Chem

    from chemagent.featurization.fingerprints import ECFP
    from chemagent.explainability.Counterfactuals.CF_generator_v3 import CFGenerator
    from chemagent.explainability.counterfactual_tools import visualize_counterfactuals

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading model and split file...")
    model      = joblib.load(MODEL_PATH)
    split_data = joblib.load(SPLIT_PATH)

    test_smiles = list(split_data["test_smiles"])
    test_labels = list(split_data["test_labels"])

    print(f"  Test set size : {len(test_smiles)}")
    print(f"  Model classes : {model.classes_}")

    # ── Predict test set ─────────────────────────────────────────────────────
    print("\nPredicting test set...")
    test_fps   = np.array(ECFP(test_smiles, n_bits=N_BITS, radius=RADIUS))
    pred_class = model.predict(test_fps)

    correct_mask = np.array(pred_class) == np.array(test_labels)
    accuracy = correct_mask.mean()
    print(f"  Test accuracy : {accuracy:.3f}")

    # Select correctly predicted class-1 compounds
    query_smiles_list = [
        smi
        for smi, true_lbl, pred_lbl, correct in zip(
            test_smiles, test_labels, pred_class, correct_mask
        )
        if true_lbl == TARGET_CLASS and pred_lbl == TARGET_CLASS and correct
    ]

    if len(query_smiles_list) < N_COMPOUNDS:
        raise RuntimeError(
            f"Only {len(query_smiles_list)} correctly predicted class-{TARGET_CLASS} "
            f"compounds found in the test set; need at least {N_COMPOUNDS}."
        )

    selected = query_smiles_list[:N_COMPOUNDS]
    print(f"\nSelected {N_COMPOUNDS} correctly predicted class-{TARGET_CLASS} compounds.")

    # ── Build shared R-group library ─────────────────────────────────────────
    all_smiles    = list(split_data["train_smiles"]) + list(split_data["test_smiles"])
    unique_smiles = list(dict.fromkeys(s for s in all_smiles if s))
    print(f"Unique SMILES for R-group library: {len(unique_smiles)}")

    # ── Counterfactual generation and visualisation ──────────────────────────
    for idx, query_smi in enumerate(selected, start=1):
        print(f"\n{'='*60}")
        print(f"Compound {idx}/{N_COMPOUNDS}: {query_smi[:70]}")
        print(f"{'='*60}")

        # --- Generate counterfactuals ---
        print("  Building CFGenerator...")
        try:
            gen = CFGenerator(
                query_smiles=query_smi,
                model_obj=model,
                data_smiles=unique_smiles,
                n_bits=N_BITS,
                radius=RADIUS,
            )
        except ValueError as e:
            print(f"  SKIP — decomposition failed: {e}")
            continue

        print(
            f"  Query class    : {gen.query_class}  "
            f"probas: {[round(p, 3) for p in gen.query_probas]}"
        )
        print(
            f"  R-group library: {len(gen._mc.external_rgroups)} entries "
            f"(dataset + static union)"
        )

        print("  Running find_cfs()...")
        cf_df = gen.find_cfs(max_counterfactuals=20)

        print(f"  Candidates tested  : {gen.n_candidates_tested}")
        print(f"  Counterfactuals    : {len(cf_df)}")

        if cf_df.empty:
            print("  No counterfactuals found — skipping visualisation.")
            continue

        # Breakdown by change type
        counts = cf_df["change_type"].value_counts()
        for ctype, n in counts.items():
            print(f"    {ctype:<20}: {n}")

        # Top 3 by Tanimoto similarity
        print("\n  Top 3 counterfactuals (by Tanimoto similarity):")
        for _, row in cf_df.head(3).iterrows():
            proba_cols = [c for c in cf_df.columns if c.startswith("proba_class_")]
            probas_str = "  ".join(
                f"class {c.split('_')[-1]}: {row[c]:.3f}" for c in proba_cols
            )
            print(
                f"    sim={row['tanimoto_similarity']:.3f}  "
                f"pred={row['predicted_class']}  "
                f"type={row['change_type']:<20}  "
                f"{probas_str}"
            )

        # --- Visualise ---
        n_classes   = len(gen.query_probas)
        proba_cols  = [f"proba_class_{i}" for i in range(n_classes)]
        cf_result   = {
            "query_smiles"      : query_smi,
            "predicted_class"   : gen.query_class,
            "probabilities"     : gen.query_probas,
            "n_candidates_tested": gen.n_candidates_tested,
            "num_counterfactuals": len(cf_df),
            "status"            : "completed",
            "counterfactuals"   : [
                {
                    "cf_smiles"          : row["cf_smiles"],
                    "predicted_class"    : int(row["predicted_class"]),
                    "probabilities"      : [float(row[c]) for c in proba_cols],
                    "tanimoto_similarity": float(row["tanimoto_similarity"]),
                    "change_type"        : row["change_type"],
                }
                for _, row in cf_df.iterrows()
            ],
        }

        out_path = PLOTS_DIR / f"cf_compound_{idx}.png"
        print(f"\n  Saving visualisation -> {out_path}")
        try:
            visualize_counterfactuals(
                cf_result   = cf_result,
                output_path = str(out_path),
                top_n       = 3,
            )
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Visualisation failed: {e}")

    print("\nAll done.")

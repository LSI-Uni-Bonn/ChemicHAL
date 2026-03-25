"""Tests for explain_batch_with_molanchor and identify_recurrent_anchor_rules."""
import sys
import json
import joblib
from pathlib import Path

# Windows multiprocessing requires the __main__ guard.
if __name__ == "__main__":
    SPLIT_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/splits"
        "/chembl_activity_data_O00329_P48736_random.pkl"
    )
    MODEL_PATH = Path(
        "data/logs/session_alamens_20260318_110929_6a0d6a/models"
        "/chembl_activity_data_O00329_P48736_random_RFC.pkl"
    )

    # Patch split to a small subset so the test runs fast
    print("Loading split file...")
    split_data = joblib.load(SPLIT_PATH)
    SUBSET = 50
    patched_path = Path("data/logs/test_anchor_split_small.pkl")
    patched = {k: v[:SUBSET] if hasattr(v, "__len__") else v for k, v in split_data.items()}
    joblib.dump(patched, patched_path)
    print(f"Patched split: train={len(patched['train_smiles'])}  test={len(patched['test_smiles'])}")

    sys.path.insert(0, "src")
    from chemagent.explainability.molanchor_tools import (
        explain_batch_with_molanchor,
        identify_recurrent_anchor_rules,
    )

    # ── Test 1: anchor rules counted as whole units ───────────────────────
    print("\n--- Test 1: anchor rules counted as whole units (single + multi-fragment) ---")
    batch = explain_batch_with_molanchor(
        split_file_path=str(patched_path),
        model_path=str(MODEL_PATH),
        target_class=1,
        split="test",
        max_compounds=20,
        allow_frag_combinations=True,
    )
    assert batch["status"] == "completed"
    freq = batch["aggregate_statistics"]["anchor_frequency"]

    # Each key is a "||"-joined rule string; each compound contributes exactly one entry
    assert len(freq) <= batch["compounds_analyzed"], (
        "Cannot have more unique rules than analyzed compounds"
    )
    # Each key should be a string (hashable, JSON-safe)
    for key in freq:
        assert isinstance(key, str), f"anchor_frequency key should be str, got {type(key)}"
    # Multi-fragment keys contain "||", single-fragment keys do not
    single_keys = [k for k in freq if "||" not in k]
    multi_keys  = [k for k in freq if "||" in k]
    print(f"  Total unique anchor rules:   {len(freq)}")
    print(f"  Single-fragment rules:       {len(single_keys)}")
    print(f"  Multi-fragment rules:        {len(multi_keys)}")
    print("  Key format OK")

    # ── Test 2: identify_recurrent_anchor_rules — structure checks ────────
    print("\n--- Test 2: identify_recurrent_anchor_rules structure ---")
    output = identify_recurrent_anchor_rules(
        split_file_path=str(patched_path),
        model_path=str(MODEL_PATH),
        target_class=1,
        split="test",
        top_n_anchors=3,
        allow_frag_combinations=True,
    )

    # Returns [*images, json_str] on success, or a plain dict when no anchors found
    if isinstance(output, dict):
        result = output
    else:
        assert isinstance(output, list) and len(output) >= 1
        result = json.loads(output[-1])

    assert "status" in result
    print(f"  Status: {result['status']}")

    if result["status"] == "completed":
        assert "recurrent_rules" in result
        assert "statistics" in result
        assert len(result["recurrent_rules"]) <= 3, "top_n_anchors=3 not respected"

        print(f"  Analyzed compounds:    {result['num_analyzed_compounds']}")
        print(f"  Unique anchors found:  {result['statistics']['total_unique_anchors']}")
        print(f"  Rules returned:        {len(result['recurrent_rules'])}")

        for rule in result["recurrent_rules"]:
            assert "fragment" in rule
            assert 0.0 <= rule["substructure_occurrence"] <= 1.0
            assert 0.0 <= rule["anchor_occurrence"] <= 1.0
            frag = rule["fragment"]
            frag_display = frag if isinstance(frag, str) else " + ".join(frag)
            print(
                f"    {frag_display[:50]:50s} "
                f"anchor={rule['anchor_occurrence']:.1%}  "
                f"subst={rule['substructure_occurrence']:.1%}"
            )

        # Rules should be sorted by anchor_occurrence descending
        occs = [r["anchor_occurrence"] for r in result["recurrent_rules"]]
        assert occs == sorted(occs, reverse=True), "Rules not sorted by anchor_occurrence desc"
        print("  Sort order OK")
    else:
        assert result["status"] == "no anchors found"
        print("  No anchors found (valid for this subset size)")

    # ── Test 3: top_n_anchors=None returns all anchors ────────────────────
    print("\n--- Test 3: top_n_anchors=None returns all anchors ---")
    output_all = identify_recurrent_anchor_rules(
        split_file_path=str(patched_path),
        model_path=str(MODEL_PATH),
        target_class=1,
        split="test",
        top_n_anchors=None,
        allow_frag_combinations=True,
    )
    if isinstance(output_all, dict):
        result_all = output_all
    else:
        result_all = json.loads(output_all[-1])

    if result_all["status"] == "completed":
        n_rules = len(result_all["recurrent_rules"])
        n_unique = result_all["statistics"]["total_unique_anchors"]
        assert n_rules == n_unique, (
            f"Expected all {n_unique} anchors when top_n_anchors=None, got {n_rules}"
        )
        print(f"  All {n_rules} anchors returned (matches total_unique_anchors)")
    else:
        print("  No anchors found — top_n_anchors=None check skipped")

    # ── Test 4: allow_frag_combinations=False produces only single-fragment keys ─
    print("\n--- Test 4: allow_frag_combinations=False (no combination rules) ---")
    batch_no_combo = explain_batch_with_molanchor(
        split_file_path=str(patched_path),
        model_path=str(MODEL_PATH),
        target_class=1,
        split="test",
        max_compounds=20,
        allow_frag_combinations=False,
    )
    assert batch_no_combo["status"] == "completed"
    for key in batch_no_combo["aggregate_statistics"]["anchor_frequency"]:
        assert "||" not in key, f"Unexpected multi-fragment key '{key}' with allow_frag_combinations=False"
    print(f"  Anchor entries: {len(batch_no_combo['aggregate_statistics']['anchor_frequency'])}")
    print("  No multi-fragment keys present")

    print("\nAll tests passed.")

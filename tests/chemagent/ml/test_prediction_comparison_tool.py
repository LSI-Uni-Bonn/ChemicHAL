"""Tests for compare_exported_predictions MCP helper."""

from pathlib import Path

import pandas as pd

from chemagent.ml.ml_model_tools import compare_exported_predictions


def _write_csv(tmp_path: Path, name: str, frame: pd.DataFrame) -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


def _get_pair(result: dict, model_a: str, model_b: str) -> dict:
    for pair in result["pairwise_comparison"]:
        if {pair.get("model_a"), pair.get("model_b")} == {model_a, model_b}:
            return pair
    raise AssertionError(f"Pair ({model_a}, {model_b}) not found")


def test_compare_exported_predictions_classification_summary_and_rows(tmp_path):
    model_a = pd.DataFrame(
        {
            "cid": [101, 102, 103],
            "smiles": ["CCO", "CCC", "CCN"],
            "true_label": [1, 0, 1],
            "predicted_label": [1, 0, 1],
            "prob_class_0": [0.1, 0.8, 0.2],
            "prob_class_1": [0.9, 0.2, 0.8],
        }
    )
    model_b = pd.DataFrame(
        {
            "cid": [101, 102, 103],
            "smiles": ["CCO", "CCC", "CCN"],
            "true_label": [1, 0, 1],
            "predicted_label": [1, 1, 0],
            "prob_class_0": [0.2, 0.3, 0.7],
            "prob_class_1": [0.8, 0.7, 0.3],
        }
    )

    path_a = _write_csv(tmp_path, "rfc_predictions.csv", model_a)
    path_b = _write_csv(tmp_path, "svc_predictions.csv", model_b)

    result = compare_exported_predictions(
        prediction_paths=[path_a, path_b],
        model_names=["RFC", "SVC"],
        task="classification",
        match_on="cid",
        include_rows="all",
        include_probabilities=True,
        max_compounds=20,
    )

    assert result["task"] == "classification"
    assert result["matching"]["method"] == "cid"
    assert result["n_compounds_compared"] == 3

    summary = result["agreement_summary"]
    assert summary["unanimous_count"] == 1
    assert summary["disagreement_count"] == 2

    pair = _get_pair(result, "RFC", "SVC")
    assert pair["n_overlap"] == 3
    assert pair["n_disagree"] == 2
    assert pair["agreement_rate"] == 1 / 3

    rows_by_cid = {int(row["cid"]): row for row in result["compounds"]}
    assert rows_by_cid[101]["unanimous"] is True
    assert rows_by_cid[102]["unanimous"] is False
    assert rows_by_cid[102]["predictions_by_model"] == {"RFC": 0, "SVC": 1}
    assert "probabilities_by_model" in rows_by_cid[102]


def test_compare_exported_predictions_auto_row_index_fallback(tmp_path):
    model_a = pd.DataFrame(
        {
            "true_label": [1, 0, 1],
            "predicted_label": [1, 0, 1],
        }
    )
    model_b = pd.DataFrame(
        {
            "true_label": [1, 0],
            "predicted_label": [0, 0],
        }
    )

    path_a = _write_csv(tmp_path, "model_a_predictions.csv", model_a)
    path_b = _write_csv(tmp_path, "model_b_predictions.csv", model_b)

    result = compare_exported_predictions(
        prediction_paths=[path_a, path_b],
        model_names=["A", "B"],
        task="auto",
        match_on="auto",
        include_rows="all",
    )

    assert result["task"] == "classification"
    assert result["matching"]["method"] == "row_index"
    assert result["n_compounds_compared"] == 2
    assert {row["row_index"] for row in result["compounds"]} == {0, 1}


def test_compare_exported_predictions_regression_tolerance(tmp_path):
    model_a = pd.DataFrame(
        {
            "cid": [1, 2, 3],
            "true_label": [0.10, 0.30, 0.20],
            "predicted_value": [0.12, 0.28, 0.19],
        }
    )
    model_b = pd.DataFrame(
        {
            "cid": [1, 2, 3],
            "true_label": [0.10, 0.30, 0.20],
            "predicted_value": [0.11, 0.55, 0.25],
        }
    )

    path_a = _write_csv(tmp_path, "rfr_a_predictions.csv", model_a)
    path_b = _write_csv(tmp_path, "rfr_b_predictions.csv", model_b)

    result = compare_exported_predictions(
        prediction_paths=[path_a, path_b],
        model_names=["RFR_A", "RFR_B"],
        task="auto",
        match_on="cid",
        include_rows="disagreements",
        regression_tolerance=0.10,
        max_compounds=20,
    )

    assert result["task"] == "regression"
    assert result["n_compounds_compared"] == 3

    summary = result["agreement_summary"]
    assert summary["disagreement_count"] == 1

    assert len(result["compounds"]) == 1
    row = result["compounds"][0]
    assert int(row["cid"]) == 2
    assert row["prediction_spread"] > 0.10

    pair = _get_pair(result, "RFR_A", "RFR_B")
    assert pair["n_overlap"] == 3
    assert pair["mean_abs_difference"] > 0.0

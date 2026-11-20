#!/usr/bin/env python

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import torch

import explain_session_edgeshaper as workflow
from chemagent.explainability.edgeshaper_tools import (
    explain_gnn_with_edgeshaper,
    visualize_edgeshaper_results,
)
from chemagent.ml.gnn_models import GCN
from chemagent.ml.gnn_training import train_gnn_model


# Edit these values before running the script.
SESSION_DIR = "data/logs/session_andmastro_20261023_111226_a049de"
OUTPUT_DIR = "scripts/results_run_5"
SMILES_1 = "CNC(=O)c1cccc(OCCCCN2CCN(c3cccc(Cl)c3Cl)CC2)c1C"
SMILES_2 = "C#CCN(C)CC(=C)c1ccc(F)c(C)c1"
MODEL_PATH = None
TRAIN_MODEL = True  # True: train from SPLIT_PATH; False: load the saved GCN checkpoint.
SPLIT_PATH = "splits/chembl_activity_data_P14416_binary_random.pkl"
SMILES_CSV = None
SMILES_COLUMN = "smiles"
DEVICE = None
BATCH_SIZE = 32
EDGE_SHAP_M = 100
TRAIN_EPOCHS = 100
TRAIN_LR = 0.001
TRAIN_BATCH_SIZE = 32
TRAIN_HIDDEN_CHANNELS = 64
TRAIN_NUM_LAYERS = 4


def main() -> int:
    session_dir = Path(SESSION_DIR).expanduser()
    if not session_dir.is_absolute():
        session_dir = workflow.REPO_ROOT / session_dir
    session_dir = session_dir.resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory not found: {session_dir}")

    output_dir = Path(OUTPUT_DIR).expanduser()
    if not output_dir.is_absolute():
        output_dir = workflow.REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if BATCH_SIZE <= 0 or EDGE_SHAP_M <= 0:
        raise SystemExit("BATCH_SIZE and EDGE_SHAP_M must be positive integers.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    split_path = workflow._find_split(session_dir, SPLIT_PATH)
    split_payload = joblib.load(split_path)
    if not isinstance(split_payload, dict):
        raise SystemExit(f"Split file must contain a dictionary: {split_path}")

    test_smiles, test_labels = workflow._get_test_split(
        session_dir,
        split_payload,
        SMILES_CSV,
        SMILES_COLUMN,
    )
    test_graphs, valid_labels, skipped_graphs = workflow._make_graphs(
        test_smiles,
        test_labels,
    )

    device = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    if TRAIN_MODEL:
        if "train_smiles" in split_payload and "test_smiles" in split_payload:
            training_smiles: list[str] = []
        else:
            if SMILES_CSV is None:
                csv_candidates = sorted((session_dir / "datasets").glob("*.csv"))
                smiles_csv_path = workflow._choose_file(
                    csv_candidates,
                    "dataset CSV with SMILES for training",
                )
            else:
                smiles_csv_path = workflow._session_path(session_dir, SMILES_CSV)
            if smiles_csv_path is None or not smiles_csv_path.is_file():
                raise FileNotFoundError(f"Training SMILES CSV not found: {smiles_csv_path}")
            training_smiles = workflow._load_full_smiles(smiles_csv_path, SMILES_COLUMN)

        model_path = output_dir / f"gnn_GCN_best_{run_id}.pt"
        training_result = train_gnn_model(
            split_file_path=str(split_path),
            smiles_list=training_smiles,
            model_class=GCN,
            model_save_path=str(model_path),
            hidden_channels=TRAIN_HIDDEN_CHANNELS,
            num_layers=TRAIN_NUM_LAYERS,
            epochs=TRAIN_EPOCHS,
            lr=TRAIN_LR,
            batch_size=TRAIN_BATCH_SIZE,
            device=str(device),
        )
        if not model_path.is_file():
            raise RuntimeError(
                "Training finished without saving a best GCN checkpoint. "
                f"Training result: {training_result}"
            )
        model = workflow._load_edgeshaper_model(
            model=None,
            model_path=str(model_path),
            model_class_name="GCN",
            device=str(device),
        )
    else:
        model_path = workflow._find_model(session_dir, MODEL_PATH)
        model = workflow._load_edgeshaper_model(
            model=None,
            model_path=str(model_path),
            model_class_name="GCN",
            device=str(device),
        )
    if type(model).__name__ != "GCN":
        raise SystemExit(
            f"Expected a GCN checkpoint, but loaded {type(model).__name__}: {model_path}"
        )

    evaluation = workflow._evaluate_model(
        model,
        test_graphs,
        valid_labels,
        device,
        BATCH_SIZE,
    )
    phi_edges_by_compound: dict[str, Any] = {}
    image_paths: list[Path] = []

    for index, compound_smiles in enumerate((SMILES_1, SMILES_2), start=1):
        predicted_class, _ = workflow._predict_smiles(
            model,
            compound_smiles,
            device,
        )
        explanation = explain_gnn_with_edgeshaper(
            model=model,
            compound_smiles=compound_smiles,
            model_class_name="GCN",
            target_class=predicted_class,
            M=EDGE_SHAP_M,
            batch_size=min(BATCH_SIZE, EDGE_SHAP_M),
            device=str(device),
            save_results=False,
        )
        if explanation.get("status") != "completed":
            raise RuntimeError(
                f"EdgeSHAPer failed for compound {index}: {explanation.get('error')}"
            )
        phi_edges_by_compound[f"compound_{index}"] = {
            "phi_edges": explanation["phi_edges"],
            "fidelity": explanation.get("fidelity"),
            "infidelity": explanation.get("infidelity"),
            "trustworthiness": explanation.get("trustworthiness"),
            "pertinent_positive_set": explanation.get("pertinent_positive_set"),
            "minimal_top_k_set": explanation.get("minimal_top_k_set"),
        }

        image_path = output_dir / f"edgeshaper_{run_id}_compound_{index}.png"
        visualized = visualize_edgeshaper_results(
            smiles=compound_smiles,
            phi_edges_json=json.dumps(explanation["phi_edges"]),
            edge_index_json=json.dumps(explanation["edge_index"]),
            save_results=True,
        )
        if not isinstance(visualized, list) or len(visualized) < 2:
            raise RuntimeError(f"Could not visualize compound {index}: {visualized}")

        visualization_summary = json.loads(visualized[1])
        generated_paths = visualization_summary.get("image_paths", [])
        if not generated_paths:
            raise RuntimeError(f"Visualizer returned no image path for compound {index}.")
        generated_image = Path(generated_paths[0])
        if not generated_image.is_file():
            raise FileNotFoundError(f"Visualizer image not found: {generated_image}")
        shutil.move(str(generated_image), str(image_path))
        image_paths.append(image_path)

    metric_keys = ("Accuracy", "BA", "MCC", "Dataset size", "FP", "FN", "TP", "TN")
    test_metrics = {key: evaluation[key] for key in metric_keys if key in evaluation}
    metrics_path = output_dir / f"edgeshaper_{run_id}_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    phi_edges_path = output_dir / f"edgeshaper_{run_id}_phi_edges.json"
    phi_edges_path.write_text(
        json.dumps(phi_edges_by_compound, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(test_metrics, indent=2))
    if TRAIN_MODEL:
        print(f"\nSaved best trained GCN model: {model_path}")
    print(f"\nSaved test metrics: {metrics_path}")
    print(f"Saved phi_edges: {phi_edges_path}")
    for image_path in image_paths:
        print(f"Saved visualization: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI script to quickly test GNN training on a split dataset.

Example:
    uv run python scripts/test_gnn.py \
        --split splits/my_split.pkl \
        --csv data/datasets/chembl_activity_data_O00329_P42336.csv \
        --model GCN \
        --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from repository root without editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.ml.gnn_models import GAT, GCN, GC_GNN, GIN, GINE, GraphSAGE
from chemagent.ml.gnn_training import train_gnn_model

MODEL_MAP = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "GC_GNN": GC_GNN,
    "GINE": GINE,
    "GIN": GIN,
}


def _read_smiles(csv_path: Path, smiles_column: str) -> list[str]:
    smiles: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_column not in (reader.fieldnames or []):
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Column '{smiles_column}' not found in {csv_path}. "
                f"Available columns: {available}"
            )
        for row in reader:
            value = row.get(smiles_column, "")
            if value:
                smiles.append(value)
    if not smiles:
        raise ValueError(f"No SMILES values found in column '{smiles_column}'.")
    return smiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test GNN training with a split file.")
    parser.add_argument(
        "--split",
        required=True,
        help="Path to split .pkl file containing train/test labels and smiles or indices.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with SMILES (used for index-based splits).",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Column name for SMILES in CSV (default: smiles).",
    )
    parser.add_argument(
        "--model",
        default="GCN",
        choices=sorted(MODEL_MAP.keys()),
        help="GNN architecture to train (default: GCN).",
    )
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cpu, cuda). Default: auto-detect.",
    )
    parser.add_argument(
        "--output-model",
        default="models/gnn_test_model.pt",
        help="Path for saving the best model (default: models/gnn_test_model.pt).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    split_path = Path(args.split)
    csv_path = Path(args.csv)
    output_model = Path(args.output_model)

    if not split_path.exists():
        print(f"Error: split file not found: {split_path}", file=sys.stderr)
        return 1
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    output_model.parent.mkdir(parents=True, exist_ok=True)

    try:
        smiles_list = _read_smiles(csv_path, args.smiles_column)
        result = train_gnn_model(
            split_file_path=str(split_path),
            smiles_list=smiles_list,
            model_class=MODEL_MAP[args.model],
            model_save_path=str(output_model),
            hidden_channels=args.hidden_channels,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

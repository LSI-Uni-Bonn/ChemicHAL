"""chemagent.explainability.molce_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for MolCE contrastive molecular explanations.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
explain_with_molce               — contrastive R-group attribution for a single compound
visualize_molce_foils            — draw a compound and its top contrastive foil molecules as a grid
identify_recurrent_molce_rules   — global MolCE: aggregate contrastive R-group rules across a class

MolCE (Molecular Contrastive Explanations) explains *why* a model predicts class A
rather than class B (the foil class) by systematically substituting R-groups from a
dataset library and measuring how much each substitution shifts the prediction toward
the foil class.  High contrastive score → the current R-group strongly distinguishes
the predicted class from the foil.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from mcp.server.fastmcp import Image as MCPImage

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.featurization.fingerprints import ECFP
from chemagent.session_utils import get_session_logger as _get_session_logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_sklearn_predict_funcs(model, n_bits: int, radius: int):
    """Return (predict_func, predict_func_proba) adapters for an sklearn model.

    Both adapters accept the calling convention used inside MolContrast:

        predict_func(model, mol, singular=False)
        predict_func_proba(model=model, mol=mol, singular=False)

    where *mol* is either a single :class:`rdkit.Chem.Mol` (when
    ``singular=True``) or a list of molecules.
    """

    def _featurize(mol_or_mols, singular: bool) -> np.ndarray:
        if singular:
            smiles_list = [Chem.MolToSmiles(mol_or_mols)]
        else:
            smiles_list = [Chem.MolToSmiles(m) for m in mol_or_mols]
        fps = ECFP(smiles_list, n_bits=n_bits, radius=radius)
        return np.asarray(fps)

    def predict_func(model, mol, singular: bool = False):
        fps = _featurize(mol, singular)
        preds = model.predict(fps)
        return int(preds[0]) if singular else preds

    def predict_func_proba(model, mol, singular: bool = False):
        fps = _featurize(mol, singular)
        probas = model.predict_proba(fps)
        return probas[0] if singular else probas

    return predict_func, predict_func_proba


class _MolContrastWrapper:
    """Wrapper around MolContrast with controlled scaffold dict loading.

    ``MolContrast.__init__`` unconditionally calls ``get_scaffold_dict()``,
    which reads a ``core_dict_generic.pkl`` file from the working directory.
    This wrapper overrides that method so the dict is only loaded when a
    valid ``core_dict_path`` is explicitly provided, avoiding a hard crash
    when scaffold-based analysis is not needed.

    ``supports_cores`` is True when a scaffold dict was successfully loaded.
    """

    def __init__(
        self,
        data_smiles: List[str],
        model,
        predict_func,
        predict_func_proba,
        core_dict_path: Optional[str] = None,
    ):
        import pickle
        from chemagent.explainability.MolCE.MolCE import MolContrast

        _core_dict: dict = {}
        if core_dict_path is not None:
            try:
                with open(core_dict_path, "rb") as fh:
                    _core_dict = pickle.load(fh)
            except Exception as e:
                raise ValueError(
                    f"Failed to load scaffold dict from {core_dict_path!r}: {e}"
                )

        class _SafeMolContrast(MolContrast):
            def get_scaffold_dict(self_inner):  # noqa: N805
                return _core_dict

        self._mc = _SafeMolContrast(
            data_smiles=data_smiles,
            model=model,
            predict_func=predict_func,
            predict_func_proba=predict_func_proba,
        )
        self.supports_cores: bool = bool(_core_dict)

    def get_contrastive_rgroups(self, mol, foil_class: int, random_order: bool = False):
        return self._mc.get_contrastive_rgroups(mol, foil_class, random_order)

    def get_contrastive_cores(
        self, mol, foil_class: int, similarity_threshold: Optional[float] = None
    ):
        return self._mc.get_contrastive_cores(
            mol, foil_class, similarity_threshold=similarity_threshold
        )


# ---------------------------------------------------------------------------
# Public MCP tool functions
# ---------------------------------------------------------------------------

def explain_with_molce(
    smiles: str,
    model_path: str,
    split_file_path: str,
    foil_class: int = 0,
    n_bits: int = 2048,
    radius: int = 2,
    top_n: int = 10,
    core_dict_path: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
) -> list:
    """Run MolCE contrastive attribution for a single compound (R-groups and optionally scaffolds).

    MolCE asks: *"Why does the model predict class A rather than class B?"*

    **R-group analysis** (always run): replaces each R-group in the query
    compound with R-groups extracted from the dataset library and measures how
    much each substitution shifts the probability toward the foil class.

    **Scaffold analysis** (optional, requires ``core_dict_path``): swaps the
    Murcko scaffold of the query with similar scaffolds from a prebuilt library
    (``core_dict_generic.pkl``) while keeping the original R-groups, measuring
    how much each core swap shifts the prediction.

    A high **contrast score** means the current R-group / scaffold strongly
    separates the predicted class from the foil class.  A negative score means
    the substitution would help the compound look more like the foil class.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound to explain.
    model_path : str
        Path to the trained sklearn model (.pkl).
    split_file_path : str
        Path to the split .pkl file (from split_dataset).  Used to build the
        R-group library from all unique SMILES in train + test.
    foil_class : int, optional
        The *contrast target* class — the class we ask "why not?" about.
        Typically the opposite of the compound's predicted class (default 0).
    n_bits : int, optional
        ECFP fingerprint length (default 2048).  Must match model training setup.
    radius : int, optional
        ECFP Morgan radius (default 2).  Must match model training setup.
    top_n : int, optional
        Number of top-ranked R-group / scaffold substitutions to visualize
        (default 10).
    core_dict_path : str, optional
        Path to the ``core_dict_generic.pkl`` scaffold library required for
        scaffold contrastive analysis.  When omitted only R-group attribution
        is performed.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for external scaffolds: keeps only those
        whose atom count differs from the original core by at most
        ``(1 - similarity_threshold) * 100 %``.  Only used when
        ``core_dict_path`` is provided.
    output_path : str, optional
        Base path for output images (.png).  Two images are saved when scaffold
        analysis is active: ``<base>_rgroups.png`` and ``<base>_scaffolds.png``.
        Defaults to ``session_dir/plots/molce_<session_id>``.

    Returns
    -------
    list
        One or two MCPImage objects followed by a JSON metadata string.

        JSON fields:
        - smiles, predicted_class, foil_class
        - contrastive_rgroups: top R-group attributions (rank, r_group_smiles,
          r_group_site, contrast_score)
        - contrastive_scaffolds: top scaffold attributions (rank, core_smiles,
          contrast_score) — empty list if scaffold analysis was not run
        - num_rgroups_evaluated, num_scaffolds_evaluated
        - image_path_rgroups, image_path_scaffolds
        - status: "completed"

    Raises
    ------
    ValueError
        If SMILES is invalid, model/split file cannot be loaded, or the
        molecule cannot be decomposed into a Murcko scaffold + R-groups.

    Examples
    --------
    >>> result = explain_with_molce(
    ...     smiles="c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ...     foil_class=0,
    ...     core_dict_path="data/core_dict_generic.pkl",
    ... )
    """
    logger = _get_session_logger()

    # ---- Validate query molecule ----
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # ---- Load model ----
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path!r}: {e}")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support predict_proba.  MolCE requires probability "
            "estimates — use a probabilistic classifier (e.g. RFC, GBC, SVC with "
            "probability=True)."
        )

    # ---- Collect SMILES from split file ----
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path!r}: {e}")

    all_smiles: list[str] = []
    for sp in ("train", "test"):
        key = f"{sp}_smiles"
        if key in split_data:
            all_smiles.extend(split_data[key])

    seen: set[str] = set()
    unique_smiles: list[str] = []
    for s in all_smiles:
        if s and s not in seen:
            seen.add(s)
            unique_smiles.append(s)

    if not unique_smiles:
        raise ValueError("No SMILES found in train or test splits of the split file.")

    # ---- Build predict function adapters ----
    predict_func, predict_func_proba = _make_sklearn_predict_funcs(model, n_bits, radius)

    # ---- Instantiate MolContrast ----
    try:
        mc = _MolContrastWrapper(
            data_smiles=unique_smiles,
            model=model,
            predict_func=predict_func,
            predict_func_proba=predict_func_proba,
            core_dict_path=core_dict_path,
        )
    except Exception as e:
        raise ValueError(f"Failed to initialise MolContrast: {e}")

    # ---- Predicted class for the query ----
    predicted_class = predict_func(model, mol, singular=True)

    # ---- Base output path ----
    base_path = (
        logger.session_dir / "plots" / f"molce_{logger.session_id}"
        if output_path is None
        else Path(output_path)
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)

    output_images: list = []

    # ==== R-group attribution ====
    try:
        rgroup_df = mc.get_contrastive_rgroups(mol, foil_class=foil_class)
    except ValueError as e:
        raise ValueError(
            f"MolCE decomposition failed for SMILES {smiles!r}: {e}.  "
            "The molecule must have a decomposable Murcko scaffold with at least "
            "one R-group attachment point."
        )
    except Exception as e:
        raise ValueError(f"MolCE R-group attribution failed: {e}")

    rgroup_df = rgroup_df.reset_index()
    rgroup_records = []
    for rank, (_, row) in enumerate(rgroup_df.head(top_n).iterrows(), start=1):
        rgroup_records.append({
            "rank": rank,
            "r_group_smiles": str(row["R-group"]),
            "r_group_site": int(row["R_group_site"]),
            "contrast_score": float(row["contrast"]),
        })

    # Draw R-group foil grid
    img_path_rgroups: Optional[Path] = None
    foil_mols: list[Chem.Mol] = []
    foil_legends: list[str] = []
    for rec in rgroup_records:
        foil_mol = _try_build_foil(mol, rec["r_group_smiles"], rec["r_group_site"])
        if foil_mol is not None:
            foil_mols.append(foil_mol)
            foil_legends.append(
                f"Rank {rec['rank']} | site {rec['r_group_site']}\n"
                f"contrast={rec['contrast_score']:.3f}"
            )

    try:
        img_path_rgroups = base_path.parent / f"{base_path.name}_rgroups.png"
        grid_img = Draw.MolsToGridImage(
            [mol] + foil_mols,
            molsPerRow=min(4, 1 + len(foil_mols)),
            subImgSize=(300, 300),
            legends=[f"Query\npred={predicted_class}"] + foil_legends,
        )
        grid_img.save(str(img_path_rgroups))
        output_images.append(MCPImage(path=img_path_rgroups))
    except Exception:
        img_path_rgroups = None

    # ==== Scaffold attribution (optional) ====
    scaffold_records: list[dict] = []
    img_path_scaffolds: Optional[Path] = None
    num_scaffolds_evaluated = 0

    if mc.supports_cores:
        try:
            core_df = mc.get_contrastive_cores(
                mol, foil_class=foil_class, similarity_threshold=similarity_threshold
            )
            num_scaffolds_evaluated = len(core_df)
            core_df = core_df.reset_index()
            for rank, (_, row) in enumerate(core_df.head(top_n).iterrows(), start=1):
                scaffold_records.append({
                    "rank": rank,
                    "core_smiles": str(row["index"] if "index" in row.index else row.iloc[0]),
                    "contrast_score": float(row["contrast"]),
                })

            # Draw core foil grid — reconstruct full foil molecules
            core_foil_mols: list[Chem.Mol] = []
            core_foil_legends: list[str] = []
            for rec in scaffold_records:
                foil_mol = _try_build_core_foil(mol, rec["core_smiles"])
                if foil_mol is not None:
                    core_foil_mols.append(foil_mol)
                    core_foil_legends.append(
                        f"Rank {rec['rank']}\ncontrast={rec['contrast_score']:.3f}"
                    )

            try:
                img_path_scaffolds = base_path.parent / f"{base_path.name}_scaffolds.png"
                grid_img_cores = Draw.MolsToGridImage(
                    [mol] + core_foil_mols,
                    molsPerRow=min(4, 1 + len(core_foil_mols)),
                    subImgSize=(300, 300),
                    legends=[f"Query\npred={predicted_class}"] + core_foil_legends,
                )
                grid_img_cores.save(str(img_path_scaffolds))
                output_images.append(MCPImage(path=img_path_scaffolds))
            except Exception:
                img_path_scaffolds = None

        except Exception:
            pass  # scaffold analysis failure is non-fatal

    metadata: dict[str, Any] = {
        "smiles": smiles,
        "predicted_class": int(predicted_class),
        "foil_class": foil_class,
        "contrastive_rgroups": rgroup_records,
        "contrastive_scaffolds": scaffold_records,
        "num_rgroups_evaluated": len(rgroup_df),
        "num_scaffolds_evaluated": num_scaffolds_evaluated,
        "image_path_rgroups": str(img_path_rgroups) if img_path_rgroups else None,
        "image_path_scaffolds": str(img_path_scaffolds) if img_path_scaffolds else None,
        "status": "completed",
    }

    return output_images + [json.dumps(metadata, indent=2)]


def visualize_molce_foils(
    query_smiles: str,
    foil_smiles_list: List[str],
    contrast_scores: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    mol_size: tuple[int, int] = (300, 300),
    mols_per_row: int = 4,
) -> list:
    """Draw the query compound and MolCE foil molecules as a molecule grid image.

    Use this tool when you already have foil SMILES from a previous
    ``explain_with_molce`` call and want a standalone visualization, or when
    you want to customize the grid layout.

    Parameters
    ----------
    query_smiles : str
        SMILES of the original query compound.
    foil_smiles_list : list[str]
        SMILES of the foil compounds to display (structurally similar molecules
        with different R-groups that shift the prediction toward the foil class).
    contrast_scores : list[float], optional
        Contrast scores to display as legends (one per foil SMILES).
    output_path : str, optional
        Path to save the image (.png).  Defaults to the session plots directory.
    mol_size : tuple[int, int], optional
        Pixel size of each molecule cell (default (300, 300)).
    mols_per_row : int, optional
        Molecules per row in the grid (default 4).

    Returns
    -------
    list
        [MCPImage, json_metadata_str]

    Raises
    ------
    ValueError
        If the query SMILES is invalid.

    Examples
    --------
    >>> result = explain_with_molce(
    ...     smiles="CCO",
    ...     model_path="model.pkl",
    ...     split_file_path="split.pkl",
    ...     foil_class=0,
    ... )
    >>> import json
    >>> meta = json.loads(result[-1])
    >>> viz = visualize_molce_foils(
    ...     query_smiles=meta["smiles"],
    ...     foil_smiles_list=[r["r_group_smiles"] for r in meta["contrastive_rgroups"]],
    ...     contrast_scores=[r["contrast_score"] for r in meta["contrastive_rgroups"]],
    ... )
    """
    logger = _get_session_logger()

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles!r}")

    foil_mols = []
    foil_legends = []
    for i, smi in enumerate(foil_smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            foil_mols.append(m)
            score_label = (
                f"  c={contrast_scores[i]:.3f}" if contrast_scores and i < len(contrast_scores) else ""
            )
            foil_legends.append(f"Foil {i + 1}{score_label}")

    all_mols = [query_mol] + foil_mols
    all_legends = ["Query (original)"] + foil_legends

    img = Draw.MolsToGridImage(
        all_mols,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=all_legends,
    )

    if output_path is None:
        img_path = logger.session_dir / "plots" / f"molce_foils_{logger.session_id}.png"
    else:
        img_path = Path(output_path)

    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(img_path))

    mcp_image = MCPImage(path=img_path)
    metadata = {
        "image_path": str(img_path),
        "num_foils": len(foil_mols),
        "status": "completed",
    }
    return [mcp_image, json.dumps(metadata, indent=2)]


def identify_recurrent_molce_rules(
    split_file_path: str,
    model_path: str,
    fact_class: int,
    foil_class: int,
    split: str = "test",
    n_bits: int = 2048,
    radius: int = 2,
    top_n: int = 10,
    min_occurrences: int = 2,
    max_compounds: Optional[int] = None,
    core_dict_path: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
) -> list:
    """Global MolCE analysis: aggregate contrastive R-group (and optionally scaffold) rules.

    Runs ``get_contrastive_rgroups`` on every correctly predicted compound of
    *fact_class* and aggregates scores by R-group SMILES:

        mean_contrast = groupby(r_group_smiles).contrast.mean()

    When ``core_dict_path`` is provided, also runs ``get_contrastive_cores`` per
    compound and aggregates scaffold-level contrastive scores the same way.

    Motifs with a high mean contrast are the structural features that most
    consistently distinguish *fact_class* from *foil_class* across the
    population — the dataset-level chemical logic the model has learned.

    Parameters
    ----------
    split_file_path : str
        Path to the split .pkl file (from split_dataset).
    model_path : str
        Path to the trained sklearn model (.pkl).
    fact_class : int
        The class whose compounds are analyzed (e.g. 1 for actives).
    foil_class : int
        The contrast target class — "why fact_class and not foil_class?"
        Must differ from *fact_class*.
    split : str, optional
        Which split to analyze: "train", "val", or "test" (default "test").
    n_bits : int, optional
        ECFP fingerprint length (default 2048).
    radius : int, optional
        ECFP Morgan radius (default 2).
    top_n : int, optional
        Number of top rules to return and visualize for each attribution type
        (default 10).
    min_occurrences : int, optional
        Minimum number of compounds in which a motif must appear to be included
        (default 2).  Filters out idiosyncratic motifs seen in only one compound.
    max_compounds : int, optional
        Cap the number of compounds analyzed (default None = all).
    core_dict_path : str, optional
        Path to ``core_dict_generic.pkl``.  When provided, scaffold contrastive
        analysis is also performed and a second bar chart is added to the output.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for external scaffolds (only used when
        ``core_dict_path`` is provided).
    output_path : str, optional
        Path to save the bar chart image (.png).
        Defaults to ``session_dir/plots/molce_global_<session_id>.png``.
        When scaffold analysis is active a second image is saved at
        ``<base>_scaffolds.png``.

    Returns
    -------
    list
        One or two MCPImage objects followed by a JSON metadata string.

        JSON fields:
        - fact_class, foil_class, split
        - compounds_analyzed, compounds_failed
        - total_rgroup_evaluations, total_scaffold_evaluations
        - rgroup_rules: list of dicts (r_group_smiles, mean_contrast,
          std_contrast, occurrences, r_group_site_most_common)
        - scaffold_rules: list of dicts (core_smiles, mean_contrast,
          std_contrast, occurrences) — empty list if scaffold analysis skipped
        - status: "completed"

    Raises
    ------
    ValueError
        If fact_class == foil_class, or split file/model cannot be loaded,
        or no correctly predicted compounds are found.

    Examples
    --------
    >>> rules = identify_recurrent_molce_rules(
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     fact_class=1,
    ...     foil_class=0,
    ...     top_n=10,
    ...     core_dict_path="data/core_dict_generic.pkl",
    ... )
    """
    logger = _get_session_logger()

    if fact_class == foil_class:
        raise ValueError("fact_class and foil_class must differ.")

    # ---- Load model ----
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path!r}: {e}")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support predict_proba. MolCE requires probability "
            "estimates — use a probabilistic classifier (e.g. RFC, GBC)."
        )

    # ---- Load split file ----
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path!r}: {e}")

    split_key_features = f"{split}_features"
    split_key_labels = f"{split}_labels"
    split_key_smiles = f"{split}_smiles"

    if split_key_features not in split_data or split_key_labels not in split_data:
        available = [k for k in split_data if "features" in k or "labels" in k]
        raise ValueError(f"Split '{split}' not found. Available: {available}")

    features = split_data[split_key_features]
    labels = split_data[split_key_labels]
    smiles_list = split_data.get(split_key_smiles, [f"compound_{i}" for i in range(len(labels))])

    # ---- Collect all unique SMILES for the R-group library ----
    all_smiles: list[str] = []
    for s in ("train", "test"):
        key = f"{s}_smiles"
        if key in split_data:
            all_smiles.extend(split_data[key])
    seen: set[str] = set()
    unique_smiles: list[str] = []
    for s in all_smiles:
        if s and s not in seen:
            seen.add(s)
            unique_smiles.append(s)

    # ---- Find correctly predicted compounds of fact_class ----
    predictions = model.predict(features)
    correct_mask = (predictions == labels) & (labels == fact_class)
    correct_indices = np.where(correct_mask)[0]

    if len(correct_indices) == 0:
        raise ValueError(
            f"No correctly predicted compounds found for class {fact_class} "
            f"in '{split}' split."
        )

    if max_compounds is not None and len(correct_indices) > max_compounds:
        rng = np.random.default_rng(seed=42)
        correct_indices = rng.choice(correct_indices, size=max_compounds, replace=False)

    # ---- Build predict function adapters ----
    predict_func, predict_func_proba = _make_sklearn_predict_funcs(model, n_bits, radius)

    # ---- Instantiate MolContrast once (shared across all compounds) ----
    try:
        mc = _MolContrastWrapper(
            data_smiles=unique_smiles,
            model=model,
            predict_func=predict_func,
            predict_func_proba=predict_func_proba,
            core_dict_path=core_dict_path,
        )
    except Exception as e:
        raise ValueError(f"Failed to initialise MolContrast: {e}")

    # ---- Per-compound attribution ----
    all_rgroup_rows: list[pd.DataFrame] = []
    all_scaffold_rows: list[pd.DataFrame] = []
    compounds_analyzed = 0
    compounds_failed = 0

    for compound_idx in correct_indices:
        smi = smiles_list[compound_idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            compounds_failed += 1
            continue

        success = False
        try:
            df_r = mc.get_contrastive_rgroups(mol, foil_class=foil_class)
            df_r = df_r.reset_index()
            df_r["compound_smiles"] = smi
            all_rgroup_rows.append(df_r)
            success = True
        except Exception:
            pass

        if mc.supports_cores:
            try:
                df_c = mc.get_contrastive_cores(
                    mol, foil_class=foil_class, similarity_threshold=similarity_threshold
                )
                df_c = df_c.reset_index().rename(columns={"index": "core_smiles"})
                df_c["compound_smiles"] = smi
                all_scaffold_rows.append(df_c)
            except Exception:
                pass

        if success:
            compounds_analyzed += 1
        else:
            compounds_failed += 1

    if not all_rgroup_rows:
        raise ValueError(
            "No compounds could be decomposed into Murcko scaffold + R-groups. "
            "Ensure the dataset contains drug-like molecules with clear ring systems."
        )

    # ---- Aggregate R-groups ----
    combined_r = pd.concat(all_rgroup_rows, ignore_index=True)
    total_rgroup_evals = len(combined_r)

    grouped_r = (
        combined_r.groupby("R-group")
        .agg(
            mean_contrast=("contrast", "mean"),
            std_contrast=("contrast", "std"),
            occurrences=("contrast", "count"),
            r_group_site_most_common=("R_group_site", lambda x: int(x.mode().iloc[0])),
        )
        .reset_index()
        .rename(columns={"R-group": "r_group_smiles"})
    )
    grouped_r = (
        grouped_r[grouped_r["occurrences"] >= min_occurrences]
        .sort_values("mean_contrast", ascending=False)
        .reset_index(drop=True)
    )
    top_rgroups = grouped_r.head(top_n)

    # ---- Aggregate scaffolds (if available) ----
    top_scaffolds = pd.DataFrame()
    total_scaffold_evals = 0

    if all_scaffold_rows:
        combined_c = pd.concat(all_scaffold_rows, ignore_index=True)
        total_scaffold_evals = len(combined_c)

        grouped_c = (
            combined_c.groupby("core_smiles")
            .agg(
                mean_contrast=("contrast", "mean"),
                std_contrast=("contrast", "std"),
                occurrences=("contrast", "count"),
            )
            .reset_index()
        )
        grouped_c = (
            grouped_c[grouped_c["occurrences"] >= min_occurrences]
            .sort_values("mean_contrast", ascending=False)
            .reset_index(drop=True)
        )
        top_scaffolds = grouped_c.head(top_n)

    # ---- Visualization ----
    if output_path is None:
        img_path = logger.session_dir / "plots" / f"molce_global_{logger.session_id}.png"
    else:
        img_path = Path(output_path)
    img_path.parent.mkdir(parents=True, exist_ok=True)

    output_images: list = []
    n_panels = 2 if not top_scaffolds.empty else 1
    title_suffix = f"({compounds_analyzed} compounds, {split} split)"

    try:
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(8 * n_panels, max(4, max(len(top_rgroups), len(top_scaffolds) if not top_scaffolds.empty else 0) * 0.55)),
            squeeze=False,
        )

        def _draw_bar(ax, data, label_col, title):
            colors = ["#d94f3d" if v >= 0 else "#4878d0" for v in data["mean_contrast"]]
            y_pos = range(len(data))
            tick_labels = [
                f"{str(row[label_col])[:28]}{'…' if len(str(row[label_col])) > 28 else ''}  (n={int(row['occurrences'])})"
                for _, row in data.iterrows()
            ]
            ax.barh(
                list(y_pos),
                data["mean_contrast"].tolist(),
                xerr=data["std_contrast"].fillna(0).tolist(),
                color=colors,
                edgecolor="white",
                capsize=3,
                height=0.6,
            )
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Mean Contrastive Score", fontsize=10)
            ax.set_title(f"{title}\nclass {fact_class} vs. {foil_class} — {title_suffix}", fontsize=10)
            ax.invert_yaxis()

        _draw_bar(axes[0][0], top_rgroups, "r_group_smiles", "R-group rules")
        if not top_scaffolds.empty:
            _draw_bar(axes[0][1], top_scaffolds, "core_smiles", "Scaffold rules")

        plt.tight_layout()
        fig.savefig(str(img_path), dpi=150)
        plt.close(fig)
        output_images.append(MCPImage(path=img_path))
    except Exception:
        plt.close("all")

    # ---- Serialize results ----
    def _serialize(df: pd.DataFrame) -> list[dict]:
        out = df.fillna(0).to_dict("records")
        for r in out:
            r["mean_contrast"] = float(r["mean_contrast"])
            r["std_contrast"] = float(r["std_contrast"])
            r["occurrences"] = int(r["occurrences"])
        return out

    rgroup_rules_out = _serialize(top_rgroups)
    for r in rgroup_rules_out:
        r["r_group_site_most_common"] = int(r.get("r_group_site_most_common", 0))

    scaffold_rules_out = _serialize(top_scaffolds) if not top_scaffolds.empty else []

    metadata: dict[str, Any] = {
        "fact_class": fact_class,
        "foil_class": foil_class,
        "split": split,
        "compounds_analyzed": compounds_analyzed,
        "compounds_failed": compounds_failed,
        "total_rgroup_evaluations": total_rgroup_evals,
        "total_scaffold_evaluations": total_scaffold_evals,
        "rgroup_rules": rgroup_rules_out,
        "scaffold_rules": scaffold_rules_out,
        "image_path": str(img_path) if output_images else None,
        "status": "completed",
    }

    return output_images + [json.dumps(metadata, indent=2)]


# ---------------------------------------------------------------------------
# Private helpers — attempt to reconstruct foil molecules for visualization
# ---------------------------------------------------------------------------

def _try_build_core_foil(query_mol: Chem.Mol, new_core_smiles: str) -> Optional[Chem.Mol]:
    """Attempt to build a foil molecule by swapping the scaffold of *query_mol*.

    Keeps the original R-groups and attaches them to the new core.  Used for
    visualization only — returns ``None`` on any failure.
    """
    try:
        import re
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.Chem import rdRGroupDecomposition

        # Decompose query into core + R-groups
        core = MurckoScaffold.GetScaffoldForMol(query_mol)
        rgd, _ = rdRGroupDecomposition.RGroupDecompose([core], [query_mol], asRows=False)
        if len(rgd) != 1:
            return None

        rgd.pop("Core")
        original_rgroups = []
        for i in range(len(rgd)):
            smi = Chem.MolToSmiles(rgd[f"R{i + 1}"][0])
            smi = re.sub(r":\d+", "", smi)
            if smi.count("*") == 1:
                original_rgroups.append(Chem.MolFromSmiles(smi))
            else:
                original_rgroups.append(None)

        # Parse the new core (may contain [*:N] attachment markers)
        new_core = Chem.MolFromSmiles(new_core_smiles)
        if new_core is None:
            return None

        # Attach original R-groups to the new core
        tm = Chem.RWMol(new_core)
        for i, r in enumerate(original_rgroups):
            if r is not None:
                subbed = re.sub(r"\*", f"[*:{i + 1}]", Chem.MolToSmiles(r))
                r_re = Chem.MolFromSmiles(subbed)
                if r_re is not None:
                    tm.InsertMol(r_re)

        prod = Chem.molzip(tm)
        if prod is None:
            return None

        du = Chem.MolFromSmiles("*")
        prod = AllChem.ReplaceSubstructs(prod, du, Chem.MolFromSmiles("[H]"), True)[0]
        prod = Chem.RemoveHs(prod)
        prod.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(prod)
        return prod

    except Exception:
        return None


def _try_build_foil(
    query_mol: Chem.Mol,
    new_rgroup_smiles: str,
    site: int,
) -> Optional[Chem.Mol]:
    """Attempt to build a foil molecule by swapping an R-group on *query_mol*.

    This is a best-effort reconstruction used solely for visualization.  If
    the molecule cannot be rebuilt cleanly the function returns ``None`` and
    the query R-group slot is simply omitted from the image grid.
    """
    try:
        import re
        import copy
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.Chem import rdRGroupDecomposition

        core = MurckoScaffold.GetScaffoldForMol(query_mol)
        rgd, _ = rdRGroupDecomposition.RGroupDecompose([core], [query_mol], asRows=False)
        if len(rgd) != 1:
            return None

        core_mol = rgd.pop("Core")[0]
        original_rgroups = []
        for i in range(len(rgd)):
            smi = Chem.MolToSmiles(rgd[f"R{i + 1}"][0])
            smi = re.sub(r":\d+", "", smi)
            if smi.count("*") == 1:
                original_rgroups.append(Chem.MolFromSmiles(smi))
            else:
                original_rgroups.append(None)

        # Replace the R-group at the requested site (1-indexed)
        idx = site - 1
        if idx < 0 or idx >= len(original_rgroups):
            return None

        new_r = Chem.MolFromSmiles(new_rgroup_smiles)
        if new_r is None:
            return None

        modified = copy.deepcopy(original_rgroups)
        modified[idx] = new_r

        # Rebuild the molecule
        tm = Chem.RWMol(core_mol)
        for i, r in enumerate(modified):
            if r is not None:
                subbed = re.sub(r"\*", f"[*:{i + 1}]", Chem.MolToSmiles(r))
                r_re = Chem.MolFromSmiles(subbed)
                if r_re is not None:
                    tm.InsertMol(r_re)

        prod = Chem.molzip(tm)
        if prod is None:
            return None

        du = Chem.MolFromSmiles("*")
        prod = AllChem.ReplaceSubstructs(prod, du, Chem.MolFromSmiles("[H]"), True)[0]
        prod = Chem.RemoveHs(prod)
        prod.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(prod)
        return prod

    except Exception:
        return None

"""chemagent.explainability.counterfactual_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for counterfactual molecular generation.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
generate_counterfactuals         — generate counterfactuals for a single query compound
visualize_counterfactuals        — draw query + top CFs ranked by Tanimoto similarity
get_most_confident_counterfactual — draw query + top CFs ranked by model confidence

Counterfactuals are structurally similar molecules that the model predicts
differently from the query compound.  They are generated via three perturbation
strategies (implemented in ``CF_generator_v3``):

1. **Substituent change** — original Murcko scaffold, R-groups swapped from the
   dataset library.
2. **Core change** — original R-groups, Murcko scaffold swapped with similar
   scaffolds from the bundled library.
3. **Combination** — alternative scaffold *and* one R-group swapped simultaneously.

Each candidate is re-predicted; those whose class changes are returned as CFs,
ranked by Tanimoto similarity to the original.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import joblib
from rdkit import Chem
from mcp.server.fastmcp import Image as MCPImage

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.session_utils import get_session_logger as _get_session_logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_unique_smiles(split_data: dict) -> List[str]:
    all_smiles: list[str] = []
    for sp in ("train", "test"):
        key = f"{sp}_smiles"
        if key in split_data:
            all_smiles.extend(split_data[key])
    seen: set[str] = set()
    unique: list[str] = []
    for s in all_smiles:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _mol_to_pil(
    mol: "Chem.Mol",
    size: tuple[int, int] = (350, 300),
    original_cpd=None,
    ref_mol=None,
):
    """Render a molecule to a PIL Image.

    Parameters
    ----------
    original_cpd : Chem.Mol, optional
        When provided, atoms that differ from the MCS with this molecule are
        highlighted (changed-atom highlighting).
    ref_mol : Chem.Mol, optional
        When provided, the 2D layout of *mol* is aligned to *ref_mol* via
        ``GenerateDepictionMatching2DStructure`` before drawing, so shared
        substructure stays in the same position across the grid — same
        approach used by ``_show_mol_as_pil`` in ``molce_tools.py``.

    Notes
    -----
    MCS-based highlight atom indices are computed **before** calling
    ``PrepareMolForDrawing`` because kekulisation (performed inside that
    function) can break substructure matching against aromatic query mols.
    Alignment is similarly done before ``PrepareMolForDrawing`` so that the
    assigned 2D coordinates are not discarded.
    """
    try:
        from rdkit.Chem.Draw import MolDraw2DCairo
        from rdkit.Chem import rdFMCS, rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image as PILImage

        mol = Chem.RWMol(mol)

        # -- 1. Compute highlight atoms before any coordinate/kekulization step --
        atoms_to_highlight = []
        if original_cpd is not None:
            mcs = rdFMCS.FindMCS([mol, original_cpd])
            mcs_smarts = Chem.MolFromSmarts(mcs.smartsString)
            all_idx = {atom.GetIdx() for atom in mol.GetAtoms()}
            atoms_to_highlight = list(all_idx - set(mol.GetSubstructMatch(mcs_smarts)))

        # -- 2. Align 2D layout to reference before PrepareMolForDrawing --
        # Only compute ref_mol coords if it has none yet; the caller should
        # pre-compute them once so all CFs share the same reference frame.
        if ref_mol is not None:
            rdDepictor.SetPreferCoordGen(True)
            if not ref_mol.GetNumConformers():
                rdDepictor.Compute2DCoords(ref_mol)
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    mol, ref_mol, acceptFailure=False
                )
            except Exception:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    mol, ref_mol, acceptFailure=True
                )

        # -- 3. Finalise drawing geometry --
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)

        d2d = MolDraw2DCairo(size[0], size[1])
        dopts = d2d.drawOptions()
        dopts.useBWAtomPalette()
        dopts.prepareMolsBeforeDrawing = True
        dopts.clearBackground = True

        d2d.DrawMolecule(mol, highlightAtoms=atoms_to_highlight)
        d2d.FinishDrawing()
        return PILImage.open(BytesIO(d2d.GetDrawingText()))
    except Exception:
        return None


def _build_cf_panel(
    mol_img,
    probas: List[float],
    title: str,
    n_classes: int,
    ax_mol,
    ax_bar,
):
    """Fill one (ax_mol, ax_bar) column of the CF figure."""
    import numpy as np

    # Molecule image
    ax_mol.axis("off")
    ax_mol.set_title(title, fontsize=8, pad=3, wrap=True)
    if mol_img is not None:
        ax_mol.imshow(np.array(mol_img))

    # Probability bar chart
    _PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_classes)]
    class_labels = [f"Class {i}" for i in range(n_classes)]
    bars = ax_bar.barh(class_labels, probas, color=colors, edgecolor="white", height=0.55)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Probability", fontsize=7)
    ax_bar.tick_params(axis="both", labelsize=7)
    ax_bar.spines[["top", "right"]].set_visible(False)
    for bar, p in zip(bars, probas):
        ax_bar.text(
            min(p + 0.03, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}",
            va="center",
            ha="left",
            fontsize=7,
        )


# ---------------------------------------------------------------------------
# Public MCP tool functions
# ---------------------------------------------------------------------------

def generate_counterfactuals(
    query_smiles: str,
    model_path: str,
    split_file_path: str,
    n_bits: int = 2048,
    radius: int = 2,
    core_dict_path: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    max_counterfactuals: int = 20,
    max_multisite_rgroup_combos: int = 500,
    max_rgroups_for_enumeration: int = 1000,
    max_combination_cores: int = 10,
) -> dict[str, Any]:
    """Generate counterfactual molecules for a single query compound.

    Uses MolCE-style Murcko scaffold decomposition to build four kinds of
    hypothetical analogues:

    1. **Substituent (single)** — original core, one R-group position swapped
       at a time from the combined dataset + static R-group library.
    2. **Substituent (multi)** — original core, two or more R-group positions
       swapped simultaneously (randomly sampled, capped by
       ``max_multisite_rgroup_combos``).
    3. **Core change** — original R-groups, Murcko scaffold swapped with
       similar scaffolds from the bundled library.
    4. **Combination** — alternative scaffold *and* one R-group swapped
       simultaneously.

    Candidates whose predicted class differs from the query are returned as
    counterfactuals, ranked by Tanimoto similarity (highest first).  Tanimoto
    is computed only for confirmed counterfactuals, not the full candidate pool.

    Parameters
    ----------
    query_smiles : str
        SMILES of the compound to analyse.
    model_path : str
        Path to the trained sklearn model (.pkl).
    split_file_path : str
        Path to the split .pkl file (from split_dataset). All unique SMILES in
        train + test are used to build the R-group library.
    n_bits : int, optional
        ECFP fingerprint length (default 2048). Must match model training setup.
    radius : int, optional
        ECFP Morgan radius (default 2). Must match model training setup.
    core_dict_path : str, optional
        Path to a custom ``core_dict_generic.pkl`` scaffold library. When
        omitted the bundled MolCE library is used.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for alternative scaffolds.
    max_counterfactuals : int, optional
        Maximum number of CFs to return (default 20). Candidates are sorted by
        Tanimoto similarity before trimming.
    max_multisite_rgroup_combos : int, optional
        Maximum number of randomly sampled multi-site R-group combinations
        (default 200). Set to 0 to disable multi-site generation.
    max_rgroups_for_enumeration : int, optional
        Maximum R-groups sampled from the full library for candidate generation
        (default 500). Controls the speed/coverage trade-off: O(n × n_positions)
        molzip calls per step. Set to 0 to use the full library.
    max_combination_cores : int, optional
        Maximum alternative scaffold cores tried in the combination step
        (default 5). Prevents combinatorial explosion when many scaffold matches
        are found.

    Returns
    -------
    dict
        - query_smiles: the input SMILES
        - predicted_class: model prediction for the query
        - probabilities: list of per-class probabilities for the query
        - n_candidates_tested: number of unique hypothetical compounds evaluated
        - counterfactuals: list of CF dicts sorted by tanimoto_similarity desc:
            - cf_smiles: SMILES of the counterfactual
            - predicted_class: model prediction for the CF
            - probabilities: list of per-class probabilities for the CF
            - tanimoto_similarity: Tanimoto similarity to the original
            - change_type: "substituent", "substituent_multi", "core", or
              "combination"
        - num_counterfactuals: total number of CFs found
        - status: "completed", "no_counterfactuals_found", or
          "decomposition_failed"

    Raises
    ------
    ValueError
        If the SMILES is invalid or model/split file cannot be loaded.

    Examples
    --------
    >>> result = generate_counterfactuals(
    ...     query_smiles="c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ... )
    >>> for cf in result["counterfactuals"][:3]:
    ...     print(cf["cf_smiles"], "class", cf["predicted_class"],
    ...           "sim", round(cf["tanimoto_similarity"], 2))
    """
    _get_session_logger()

    # Validate query
    if Chem.MolFromSmiles(query_smiles) is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles!r}")

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path!r}: {e}")

    # Collect dataset SMILES for R-group library
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path!r}: {e}")

    unique_smiles = _collect_unique_smiles(split_data)
    if not unique_smiles:
        raise ValueError("No SMILES found in train or test splits of the split file.")

    # Run generator
    from chemagent.explainability.Counterfactuals.CF_generator_v3 import CFGenerator

    try:
        gen = CFGenerator(
            query_smiles=query_smiles,
            model_obj=model,
            data_smiles=unique_smiles,
            n_bits=n_bits,
            radius=radius,
            core_dict_path=core_dict_path,
            similarity_threshold=similarity_threshold,
            max_multisite_rgroup_combos=max_multisite_rgroup_combos,
            max_rgroups_for_enumeration=max_rgroups_for_enumeration,
            max_combination_cores=max_combination_cores,
        )
    except ValueError as e:
        # Decomposition failed before any candidates were generated
        return {
            "query_smiles": query_smiles,
            "predicted_class": None,
            "probabilities": [],
            "n_candidates_tested": 0,
            "counterfactuals": [],
            "num_counterfactuals": 0,
            "status": "decomposition_failed",
        }

    n_classes = len(gen.query_probas)

    try:
        cf_df = gen.find_cfs(max_counterfactuals=max_counterfactuals)
    except ValueError as e:
        return {
            "query_smiles": query_smiles,
            "predicted_class": gen.query_class,
            "probabilities": gen.query_probas,
            "n_candidates_tested": gen.n_candidates_tested,
            "counterfactuals": [],
            "num_counterfactuals": 0,
            "status": "decomposition_failed",
        }

    if cf_df.empty:
        return {
            "query_smiles": query_smiles,
            "predicted_class": gen.query_class,
            "probabilities": gen.query_probas,
            "n_candidates_tested": gen.n_candidates_tested,
            "counterfactuals": [],
            "num_counterfactuals": 0,
            "status": "no_counterfactuals_found",
        }

    # Build output list
    counterfactuals = []
    for _, row in cf_df.iterrows():
        probas = [float(row[f"proba_class_{i}"]) for i in range(n_classes)]
        counterfactuals.append({
            "cf_smiles": str(row["cf_smiles"]),
            "predicted_class": int(row["predicted_class"]),
            "probabilities": probas,
            "tanimoto_similarity": float(row["tanimoto_similarity"]),
            "change_type": str(row["change_type"]),
        })

    return {
        "query_smiles": query_smiles,
        "predicted_class": gen.query_class,
        "probabilities": gen.query_probas,
        "n_candidates_tested": gen.n_candidates_tested,
        "counterfactuals": counterfactuals,
        "num_counterfactuals": len(counterfactuals),
        "status": "completed",
    }


def get_most_confident_counterfactual(
    cf_result: dict[str, Any],
    output_path: Optional[str] = None,
    mol_size: tuple[int, int] = (350, 300),
    top_n: int = 3,
) -> list:
    """Return the top most-confident counterfactuals with a visualisation.

    Ranks counterfactuals by the model's confidence in their predicted class
    (i.e. the probability assigned to the CF's own predicted class — the one
    that differs from the query's class).  The single best CF is the primary
    result; the figure shows the query + up to ``top_n`` CFs sorted the same
    way.

    Parameters
    ----------
    cf_result : dict
        Output of ``generate_counterfactuals``.
    output_path : str, optional
        Path to save the PNG.  Defaults to the session plots directory.
    mol_size : tuple[int, int], optional
        Pixel size used to render each molecule (default (350, 300)).
    top_n : int, optional
        Number of CFs to display in the figure (default 3).

    Returns
    -------
    list
        ``[result_dict, MCPImage, json_metadata_str]`` where *result_dict*
        contains:

        - query_smiles: original query SMILES
        - query_predicted_class: model prediction for the query
        - query_probabilities: per-class probabilities for the query
        - most_confident_cf: the CF with highest confidence, with an extra
          ``confidence`` field (probability for its predicted class)
        - top_confident_cfs: list of up to *top_n* CFs sorted by confidence
        - rank_by_similarity: 1-based rank of the best CF in the
          similarity-sorted ``cf_result["counterfactuals"]`` list
        - status: "found" or "no_counterfactuals"

    Examples
    --------
    >>> result = generate_counterfactuals(
    ...     query_smiles="c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    ...     model_path="...",
    ...     split_file_path="...",
    ... )
    >>> out = get_most_confident_counterfactual(result)
    >>> best = out[0]["most_confident_cf"]
    >>> print(best["cf_smiles"], best["confidence"])
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    logger = _get_session_logger()

    cfs = cf_result.get("counterfactuals", [])
    query_smiles: str = cf_result.get("query_smiles", "")
    query_probas: List[float] = cf_result.get("probabilities", [])
    query_class: int = cf_result.get("predicted_class", -1)

    no_cf_result = {
        "query_smiles": query_smiles,
        "query_predicted_class": query_class,
        "query_probabilities": query_probas,
        "most_confident_cf": None,
        "top_confident_cfs": [],
        "rank_by_similarity": None,
        "status": "no_counterfactuals",
    }

    if not cfs:
        metadata = {"image_path": None, "num_counterfactuals_shown": 0, "status": "no_counterfactuals"}
        return [no_cf_result, None, json.dumps(metadata, indent=2)]

    # --- Rank by confidence (probability for the CF's own predicted class) ---
    def _confidence(cf: dict) -> float:
        pred_cls = cf["predicted_class"]
        probas = cf["probabilities"]
        return probas[pred_cls] if pred_cls < len(probas) else max(probas)

    sorted_cfs = sorted(cfs, key=_confidence, reverse=True)
    top_cfs = sorted_cfs[:top_n]

    best_cf = top_cfs[0]
    best_cf_out = {**best_cf, "confidence": _confidence(best_cf)}
    top_cfs_out = [{**cf, "confidence": _confidence(cf)} for cf in top_cfs]

    # 1-based rank of best CF in the original similarity-sorted list
    rank = next(
        (i + 1 for i, cf in enumerate(cfs) if cf["cf_smiles"] == best_cf["cf_smiles"]),
        None,
    )

    result = {
        "query_smiles": query_smiles,
        "query_predicted_class": query_class,
        "query_probabilities": query_probas,
        "most_confident_cf": best_cf_out,
        "top_confident_cfs": top_cfs_out,
        "rank_by_similarity": rank,
        "status": "found",
    }

    # --- Visualisation ---
    query_mol = Chem.MolFromSmiles(query_smiles)
    n_classes = len(query_probas)
    n_cols = 1 + len(top_cfs)

    # Pre-compute query 2D coords once so every CF aligns to the same frame.
    if query_mol is not None:
        from rdkit.Chem import rdDepictor as _rdDepictor
        _rdDepictor.SetPreferCoordGen(True)
        _rdDepictor.Compute2DCoords(query_mol)

    fig = plt.figure(figsize=(n_cols * 4.2, 7.5))
    gs = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        height_ratios=[3, 1],
        hspace=0.45,
        wspace=0.25,
    )

    # Query column
    query_img = _mol_to_pil(query_mol, size=mol_size) if query_mol else None
    ax_mol = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    _build_cf_panel(
        mol_img=query_img,
        probas=query_probas,
        title=f"Original\nPredicted class: {query_class}",
        n_classes=n_classes,
        ax_mol=ax_mol,
        ax_bar=ax_bar,
    )
    for spine in ax_mol.spines.values():
        spine.set_edgecolor("#4878CF")
        spine.set_linewidth(2)
        spine.set_visible(True)

    # CF columns — ordered by confidence
    for col, cf in enumerate(top_cfs, start=1):
        cf_mol = Chem.MolFromSmiles(cf["cf_smiles"])
        cf_img = (
            _mol_to_pil(cf_mol, size=mol_size, original_cpd=query_mol, ref_mol=query_mol)
            if cf_mol else None
        )
        sim = cf["tanimoto_similarity"]
        ctype = cf["change_type"]
        title = (
            f"CF #{col}  ({ctype})\n"
            f"Predicted class: {cf['predicted_class']} | Similarity: {sim:.2f}"
        )
        ax_mol = fig.add_subplot(gs[0, col])
        ax_bar = fig.add_subplot(gs[1, col])
        _build_cf_panel(
            mol_img=cf_img,
            probas=cf["probabilities"],
            title=title,
            n_classes=n_classes,
            ax_mol=ax_mol,
            ax_bar=ax_bar,
        )
        for spine in ax_mol.spines.values():
            spine.set_edgecolor("#D65F5F")
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig.suptitle(
        f"Most Confident Counterfactuals\n"
        f"{query_smiles[:80]}{'...' if len(query_smiles) > 80 else ''}",
        fontsize=10,
        y=1.01,
    )

    if output_path is None:
        out = logger.session_dir / "plots" / f"confident_cfs_{logger.session_id}.png"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "image_path": str(out),
        "num_counterfactuals_shown": len(top_cfs),
        "status": "completed",
    }
    return [result, MCPImage(path=out), json.dumps(metadata, indent=2)]


def visualize_counterfactuals(
    cf_result: dict[str, Any],
    output_path: Optional[str] = None,
    mol_size: tuple[int, int] = (350, 300),
    top_n: int = 3,
) -> list:
    """Visualize the query compound and its top counterfactuals.

    Renders a figure with one column per compound (query + top ``top_n`` CFs).
    Each column shows the molecule structure (with changed atoms highlighted)
    and a horizontal probability bar chart for all output classes.
    Counterfactuals are ordered by Tanimoto similarity (highest first, as
    returned by ``generate_counterfactuals``).

    Parameters
    ----------
    cf_result : dict
        Output of ``generate_counterfactuals``.
    output_path : str, optional
        Path to save the PNG.  Defaults to session plots directory.
    mol_size : tuple[int, int], optional
        Pixel size used to render each molecule (default (350, 300)).
    top_n : int, optional
        Number of counterfactuals to display (default 3).

    Returns
    -------
    list
        ``[MCPImage, json_metadata_str]``

    Raises
    ------
    ValueError
        If the query SMILES in cf_result is invalid.

    Examples
    --------
    >>> result = generate_counterfactuals(
    ...     query_smiles="c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ... )
    >>> viz = visualize_counterfactuals(result)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    logger = _get_session_logger()

    query_smiles = cf_result.get("query_smiles", "")
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES in cf_result: {query_smiles!r}")

    # Pre-compute query 2D coords once so every CF aligns to the same frame.
    from rdkit.Chem import rdDepictor as _rdDepictor
    _rdDepictor.SetPreferCoordGen(True)
    _rdDepictor.Compute2DCoords(query_mol)

    query_probas: List[float] = cf_result.get("probabilities", [])
    query_class: int = cf_result.get("predicted_class", -1)
    n_classes = len(query_probas)
    cfs = cf_result.get("counterfactuals", [])[:top_n]

    n_cols = 1 + len(cfs)

    fig = plt.figure(figsize=(n_cols * 4.2, 7.5))
    gs = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        height_ratios=[3, 1],
        hspace=0.45,
        wspace=0.25,
    )

    # --- Query column ---
    query_img = _mol_to_pil(query_mol, size=mol_size)
    ax_mol = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    _build_cf_panel(
        mol_img=query_img,
        probas=query_probas,
        title=f"Original\nPredicted class: {query_class}",
        n_classes=n_classes,
        ax_mol=ax_mol,
        ax_bar=ax_bar,
    )
    # Blue border on original
    for spine in ax_mol.spines.values():
        spine.set_edgecolor("#4878CF")
        spine.set_linewidth(2)
        spine.set_visible(True)

    # --- CF columns ---
    for col, cf in enumerate(cfs, start=1):
        cf_mol = Chem.MolFromSmiles(cf["cf_smiles"])
        cf_img = (
            _mol_to_pil(cf_mol, size=mol_size, original_cpd=query_mol, ref_mol=query_mol)
            if cf_mol else None
        )
        cf_class = cf["predicted_class"]
        sim = cf["tanimoto_similarity"]
        ctype = cf["change_type"]

        title = (
            f"CF #{col}  ({ctype})\n"
            f"Predicted class: {cf_class} | Similarity: {sim:.2f}"
        )
        ax_mol = fig.add_subplot(gs[0, col])
        ax_bar = fig.add_subplot(gs[1, col])
        _build_cf_panel(
            mol_img=cf_img,
            probas=cf["probabilities"],
            title=title,
            n_classes=n_classes,
            ax_mol=ax_mol,
            ax_bar=ax_bar,
        )
        # Orange border on CFs
        for spine in ax_mol.spines.values():
            spine.set_edgecolor("#D65F5F")
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig.suptitle(
        f"Counterfactual Analysis\n{query_smiles[:80]}{'…' if len(query_smiles) > 80 else ''}",
        fontsize=10,
        y=1.01,
    )

    # Save
    if output_path is None:
        out = logger.session_dir / "plots" / f"counterfactuals_{logger.session_id}.png"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "image_path": str(out),
        "num_counterfactuals_shown": len(cfs),
        "status": "completed",
    }
    return [MCPImage(path=out), json.dumps(metadata, indent=2)]

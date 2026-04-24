"""chemagent.explainability.molce_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for MolCE contrastive molecular explanations.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
explain_with_molce               — contrastive R-group + scaffold attribution for a single compound
identify_recurrent_molce_rules   — global MolCE: aggregate contrastive R-group + scaffold rules across a class

MolCE (Molecular Contrastive Explanations) explains *why* a model predicts class A
rather than class B (the foil class) by systematically substituting R-groups from a
dataset library and measuring how much each substitution shifts the prediction toward
the foil class.  High contrastive score → the current R-group strongly distinguishes
the predicted class from the foil.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

import PIL
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from mcp.server.fastmcp import Image as MCPImage

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.featurization.fingerprints import ECFP
from chemagent.session_utils import get_session_logger as _get_session_logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _show_mol_as_pil(
    mol: "Chem.Mol",
    legend: str = "",
    highlightAtoms: Optional[list] = None,
    corestructure: Optional["Chem.Mol"] = None,
    original_cpd: Optional["Chem.Mol"] = None,
    substructure_smiles: Optional[str] = None,
    rotate: float = 0,
    alt_dummy: bool = False,
    highlight_subs: bool = False,
    ref_mol: Optional["Chem.Mol"] = None,
    size: tuple = (400, 350),
) -> Optional["PIL.Image.Image"]:
    """Render *mol* to a PIL Image using RDKit Cairo drawing with optional atom highlighting.

    Adapted from the ``show_mol`` helper used in MolCE notebooks.

    - ``corestructure`` + ``highlight_subs=False``: highlights core atoms.
    - ``corestructure`` + ``highlight_subs=True``: highlights substituent atoms (non-core).
    - ``original_cpd``: highlights atoms that differ from the MCS with the original.
    - ``alt_dummy=True``: renders dummy atoms as attachment-point wedges.
    - ``ref_mol``: align *mol* 2D coordinates to match this reference molecule
      before drawing (shared substructure used for alignment).
    """
    try:
        from rdkit.Chem.Draw import MolDraw2DCairo
        from rdkit.Chem import AllChem, rdFMCS
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image as PILImage
        from io import BytesIO

        # --- Compute highlight atoms on raw mol BEFORE PrepareMolForDrawing ---
        # PrepareMolForDrawing kekulizes aromaticity which breaks MCS/substructure
        # matching against aromatic reference molecules. Atom indices are preserved
        # (no atoms added when addChiralHs=False), so precomputed indices stay valid.
        mol = Chem.RWMol(mol)
        atoms_to_highlight = list(highlightAtoms) if highlightAtoms else []

        if corestructure is not None:
            du = Chem.MolFromSmiles("*")
            core = AllChem.ReplaceSubstructs(corestructure, du, Chem.MolFromSmiles("[H]"), True)[0]
            core = Chem.RemoveHs(core)
            core.UpdatePropertyCache(strict=True)
            Chem.SanitizeMol(core)
            core_match = set(mol.GetSubstructMatch(Chem.MolFromSmiles(Chem.MolToSmiles(core))))
            all_idx = {atom.GetIdx() for atom in mol.GetAtoms()}
            if not highlight_subs:
                atoms_to_highlight = list(core_match)
            else:
                atoms_to_highlight = list(all_idx - core_match)

        if original_cpd is not None:
            mcs = rdFMCS.FindMCS([mol, original_cpd])
            mcs_smarts = Chem.MolFromSmarts(mcs.smartsString)
            all_idx = {atom.GetIdx() for atom in mol.GetAtoms()}
            atoms_to_highlight = list(all_idx - set(mol.GetSubstructMatch(mcs_smarts)))

        if substructure_smiles is not None:
            atoms_to_highlight = list(mol.GetSubstructMatch(Chem.MolFromSmarts(substructure_smiles)))

        # --- 2D coordinate alignment and drawing preparation ---
        if ref_mol is not None:
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(ref_mol)
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    mol, ref_mol, acceptFailure=False
                )
            except Exception:
                rdDepictor.GenerateDepictionMatching2DStructure(
                    mol, ref_mol, acceptFailure=True
                )
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)

        d2d = MolDraw2DCairo(size[0], size[1])
        dopts = d2d.drawOptions()
        dopts.useBWAtomPalette()
        dopts.prepareMolsBeforeDrawing = False
        dopts.dummiesAreAttachments = alt_dummy
        dopts.rotate = rotate
        dopts.clearBackground = True
        if legend:
            n_lines = legend.count("\n") + 1
            dopts.legendFontSize = 14
            dopts.legendFraction = min(0.05 * n_lines + 0.12, 0.40)

        d2d.DrawMolecule(mol, legend=legend, highlightAtoms=atoms_to_highlight)
        d2d.FinishDrawing()
        return PILImage.open(BytesIO(d2d.GetDrawingText()))
    except Exception:
        return None


def _make_mol_grid(images: list, cols: int = 4) -> "PIL.Image.Image":
    """Stitch a flat list of PIL Images into a grid with *cols* columns."""
    from PIL import Image as PILImage

    valid = [img for img in images if img is not None]
    if not valid:
        raise ValueError("No images to assemble into grid.")
    w, h = valid[0].size
    rows = (len(valid) + cols - 1) // cols
    grid = PILImage.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(valid):
        grid.paste(img.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return grid


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
    This wrapper overrides that method to load from the bundled dict by
    default, or from a custom path when provided.
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

        _BUNDLED_CORE_DICT = Path(__file__).parent / "MolCE" / "core_dict_generic.pkl"
        resolved_path = core_dict_path if core_dict_path is not None else str(_BUNDLED_CORE_DICT)

        try:
            with open(resolved_path, "rb") as fh:
                _core_dict: dict = pickle.load(fh)
        except Exception as e:
            raise ValueError(
                f"Failed to load scaffold dict from {resolved_path!r}: {e}"
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

    def get_contrastive_rgroups(self, mol, foil_class: int, random_order: bool = False):
        return self._mc.get_contrastive_rgroups(mol, foil_class, random_order)

    def get_contrastive_cores(
        self, mol, foil_class: int, similarity_threshold: Optional[float] = None
    ):
        from rdkit.Chem import AllChem

        df = self._mc.get_contrastive_cores(
            mol, foil_class, similarity_threshold=similarity_threshold
        )

        # Remove any row whose scaffold is identical to the original core so
        # the original core is never returned as a contrastive explanation.
        try:
            original_core, _ = self._mc.decompose_molecule(mol, original=True)
            du = Chem.MolFromSmiles("*")
            orig_stripped = AllChem.ReplaceSubstructs(
                original_core, du, Chem.MolFromSmiles("[H]"), True
            )[0]
            orig_stripped = Chem.RemoveHs(orig_stripped)
            Chem.SanitizeMol(orig_stripped)
            orig_smi = Chem.MolToSmiles(orig_stripped)

            def _stripped_smi(core_smi: str) -> str:
                m = Chem.MolFromSmiles(core_smi)
                if m is None:
                    return ""
                m = AllChem.ReplaceSubstructs(m, du, Chem.MolFromSmiles("[H]"), True)[0]
                m = Chem.RemoveHs(m)
                Chem.SanitizeMol(m)
                return Chem.MolToSmiles(m)

            mask = [_stripped_smi(smi) != orig_smi for smi in df.index]
            df = df[mask]
        except Exception:
            pass

        return df

    def build_core_foil(
        self,
        mol: "Chem.Mol",
        core_smiles: str,
        similarity_threshold: Optional[float] = None,
    ) -> "tuple[Optional[Chem.Mol], Optional[Chem.Mol]]":
        """Build a foil by attaching the original R-groups onto a new core scaffold.

        Mirrors the approach used in the MolCE notebook:

            scaffolds = CE.get_scaffolds(original_core, similarity_threshold)
            gen = CE.ext_core_rgroup_enumeration(original_r, scaffolds)
            for prod, core in zip(gen, scaffolds):
                if Chem.MolToSmiles(core) == target_core_smiles:
                    visualise(prod, corestructure=core, highlight_subs=True)

        Returns (foil_mol, annotated_core) so the caller can pass the
        annotated core (which carries ``[*:N]`` dummy atoms) directly to
        ``_show_mol_as_pil`` as ``corestructure``, matching the notebook style.
        Returns (None, None) on failure.
        """
        try:
            original_core, rgroups = self._mc.decompose_molecule(mol, original=True)
            if not rgroups:
                return None, None

            scaffolds = self._mc.get_scaffolds(original_core, similarity_threshold)
            if not scaffolds:
                return None, None

            gen = self._mc.ext_core_rgroup_enumeration(rgroups, scaffolds)
            for prod, core in zip(gen, scaffolds):
                if Chem.MolToSmiles(core) == core_smiles:
                    return prod, core

            return None, None
        except Exception:
            return None, None


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
    core_dict_path: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
    include_anti_contrastive: bool = False,
    gnn_model_class_name: Optional[str] = None,
    gnn_hidden_channels: int = 64,
    gnn_num_classes: int = 2,
) -> list:
    """Run MolCE contrastive attribution for a single compound (R-groups and scaffolds).

    MolCE asks: *"Why does the model predict class A rather than class B?"*

    **R-group analysis**: replaces each R-group in the query compound with
    R-groups from the dataset library and measures how much each substitution
    shifts the probability toward the foil class.

    **Scaffold analysis**: swaps the Murcko scaffold with similar scaffolds
    from the bundled library, keeping original R-groups, measuring how much
    each core swap shifts the prediction.

    Returns the **top 3** most contrastive substituents and scaffolds with
    their contrast scores and molecule-grid visualizations inline.

    A high **contrast score** means the current R-group / scaffold strongly
    separates the predicted class from the foil class.

    For GNN models, set ``gnn_model_class_name`` (e.g. ``'GCN'``) and point
    ``model_path`` to the ``.pt`` checkpoint. The ``n_bits`` / ``radius``
    parameters are ignored in GNN mode.

    Args:
    smiles : str
        SMILES string of the compound to explain.
    model_path : str
        Path to the trained model (.pkl for sklearn, .pt for GNN).
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
    core_dict_path : str, optional
        Path to a custom ``core_dict_generic.pkl`` scaffold library.  When
        omitted the bundled library is used.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for external scaffolds: keeps only those
        whose atom count differs from the original core by at most
        ``(1 - similarity_threshold) * 100 %``.
    output_path : str, optional
        Base path for output images (.png).  Two images are saved:
        ``<base>_rgroups.png`` and ``<base>_scaffolds.png``.
        Defaults to ``session_dir/plots/molce_foil<foil_class>_<smi_hash>_<session_id>``.
    include_anti_contrastive : bool, optional
        When True, also return the top-3 anti-contrastive substituents and
        scaffolds — those that *reinforce* the predicted class (negative
        contrastive scores).  They are appended to the image grids and
        reported under ``anti_contrastive_rgroups`` / ``anti_contrastive_scaffolds``
        in the JSON output.  Default False.
    gnn_model_class_name : str, optional
        GNN architecture name: one of GCN | GraphSAGE | GAT | GC_GNN | GIN.
        Auto-detected from .pt checkpoint metadata when omitted.
    gnn_hidden_channels : int, optional
        Hidden dimension of the GNN (default 64). Must match training config.
    gnn_num_classes : int, optional
        Number of output classes of the GNN (default 2). Must match training config.

    Returns:
    list
        Two MCPImage objects (R-group grid, scaffold grid) followed by a JSON
        metadata string.

        JSON fields:
        - smiles, predicted_class, foil_class
        - contrastive_rgroups: top-3 R-group attributions (rank, r_group_smiles,
          r_group_site, contrast_score) — positive scores, shift toward foil class
        - anti_contrastive_rgroups: top-3 R-groups that *reinforce* the predicted
          class (rank, r_group_smiles, r_group_site, contrast_score) — negative scores
          (only present when include_anti_contrastive=True)
        - contrastive_scaffolds: top-3 scaffold attributions (rank, core_smiles,
          contrast_score) — positive scores, shift toward foil class
        - anti_contrastive_scaffolds: top-3 scaffolds that reinforce the predicted
          class (rank, core_smiles, contrast_score) — negative scores
          (only present when include_anti_contrastive=True)
        - num_rgroups_evaluated, num_scaffolds_evaluated
        - image_path_rgroups, image_path_scaffolds
        - status: "completed"

    Raises:
    ValueError
        If SMILES is invalid, model/split file cannot be loaded, or the
        molecule cannot be decomposed into a Murcko scaffold + R-groups.

    Examples:
    >>> result = explain_with_molce(
    ...     smiles="c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ...     foil_class=0,
    ... )
    """
    logger = _get_session_logger()

    # ---- Validate query molecule ----
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # ---- Auto-detect GNN checkpoints from .pt metadata ----
    from chemagent.explainability.gnn_compat import infer_gnn_params
    gnn_model_class_name, gnn_hidden_channels, gnn_num_classes = infer_gnn_params(
        model_path, gnn_model_class_name, gnn_hidden_channels, gnn_num_classes,
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

    # ---- Load model & build predict function adapters ----
    if gnn_model_class_name is not None:
        from chemagent.explainability.gnn_compat import (
            load_chemagent_gnn, make_gnn_molce_predict_funcs,
        )
        try:
            gnn = load_chemagent_gnn(
                model_path, gnn_model_class_name,
                gnn_hidden_channels, gnn_num_classes,
            )
        except Exception as e:
            raise ValueError(f"Failed to load GNN model from {model_path!r}: {e}")
        model = gnn  # passed to MolContrastWrapper (ignored by GNN predict funcs)
        predict_func, predict_func_proba = make_gnn_molce_predict_funcs(gnn)
    elif str(model_path).endswith(".pt"):
        raise ValueError(
            f"Model path {model_path!r} is a .pt file but no GNN architecture "
            "could be inferred from checkpoint metadata. Supply "
            "gnn_model_class_name explicitly (e.g. 'GCN', 'GraphSAGE')."
        )
    else:
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
    # Qualify by foil_class + SMILES hash so repeat calls with different
    # foil_class / SMILES don't overwrite each other in the same session.
    smi_h = hashlib.md5(smiles.encode("utf-8")).hexdigest()[:8]
    base_path = (
        logger.session_dir / "plots"
        / f"molce_foil{foil_class}_{smi_h}_{logger.session_id}"
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
    for rank, (_, row) in enumerate(rgroup_df.head(3).iterrows(), start=1):
        rgroup_records.append({
            "rank": rank,
            "r_group_smiles": str(row["R-group"]),
            "r_group_site": int(row["R_group_site"]),
            "contrast_score": float(row["contrast"]),
        })

    # Anti-contrastive: substituents that reinforce the predicted class (negative scores)
    anti_rgroup_records = []
    if include_anti_contrastive:
        for rank, (_, row) in enumerate(rgroup_df[rgroup_df["contrast"] < 0].tail(3).iloc[::-1].iterrows(), start=1):
            anti_rgroup_records.append({
                "rank": rank,
                "r_group_smiles": str(row["R-group"]),
                "r_group_site": int(row["R_group_site"]),
                "contrast_score": float(row["contrast"]),
            })

    # Pre-render query molecule (reused in both grids)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    original_core = MurckoScaffold.GetScaffoldForMol(mol)
    query_pil = _show_mol_as_pil(mol, legend=f"Original\npredicted_class={predicted_class}")

    # ==== R-group foil grid — same core, contrastive substituents highlighted ====
    img_path_rgroups: Optional[Path] = None
    rgroup_foil_pils = []
    for rec in rgroup_records:
        foil_mol = _try_build_foil(mol, rec["r_group_smiles"], rec["r_group_site"])
        if foil_mol is not None:
            pil = _show_mol_as_pil(
                foil_mol,
                legend=f"Rank {rec['rank']} | site {rec['r_group_site']}\ncontrast={rec['contrast_score']:.3f}",
                original_cpd=mol,
            )
            if pil is not None:
                rgroup_foil_pils.append(pil)

    # Anti-contrastive R-group foils (negative scores — reinforce predicted class)
    anti_rgroup_foil_pils = []
    for rec in anti_rgroup_records:
        foil_mol = _try_build_foil(mol, rec["r_group_smiles"], rec["r_group_site"])
        if foil_mol is not None:
            pil = _show_mol_as_pil(
                foil_mol,
                legend=f"Anti-{rec['rank']} | site {rec['r_group_site']}\ncontrast={rec['contrast_score']:.3f}",
                original_cpd=mol,
            )
            if pil is not None:
                anti_rgroup_foil_pils.append(pil)

    try:
        all_rgroup_pils = (
            ([query_pil] if query_pil is not None else [])
            + rgroup_foil_pils
            + anti_rgroup_foil_pils
        )
        if all_rgroup_pils:
            img_path_rgroups = base_path.parent / f"{base_path.name}_rgroups.png"
            _make_mol_grid(all_rgroup_pils, cols=len(all_rgroup_pils)).save(str(img_path_rgroups))
            output_images.append(MCPImage(path=img_path_rgroups))
    except Exception:
        img_path_rgroups = None

    # ==== Scaffold attribution — contrastive cores highlighted ====
    scaffold_records: list[dict] = []
    anti_scaffold_records: list[dict] = []
    img_path_scaffolds: Optional[Path] = None
    num_scaffolds_evaluated = 0

    try:
        core_df = mc.get_contrastive_cores(
            mol, foil_class=foil_class, similarity_threshold=similarity_threshold
        )
        num_scaffolds_evaluated = len(core_df)
        core_df = core_df.reset_index()
        for rank, (_, row) in enumerate(core_df.head(3).iterrows(), start=1):
            scaffold_records.append({
                "rank": rank,
                "core_smiles": str(row["index"] if "index" in row.index else row.iloc[0]),
                "contrast_score": float(row["contrast"]),
            })

        # Anti-contrastive scaffolds (negative scores — reinforce predicted class)
        if include_anti_contrastive:
            for rank, (_, row) in enumerate(
                core_df[core_df["contrast"] < 0].tail(3).iloc[::-1].iterrows(), start=1
            ):
                anti_scaffold_records.append({
                    "rank": rank,
                    "core_smiles": str(row["index"] if "index" in row.index else row.iloc[0]),
                    "contrast_score": float(row["contrast"]),
                })

        # Draw core foil grid — original, extracted core, then foils with substituents highlighted
        original_core_pil = _show_mol_as_pil(original_core, legend="Extracted core")

        core_foil_pils = []
        for rec in scaffold_records:
            foil_mol, annotated_core = mc.build_core_foil(
                mol, rec["core_smiles"], similarity_threshold=similarity_threshold
            )
            if foil_mol is not None:
                pil = _show_mol_as_pil(
                    foil_mol,
                    legend=f"Rank {rec['rank']}\ncontrast={rec['contrast_score']:.3f}",
                    corestructure=annotated_core,
                    highlight_subs=True,
                    alt_dummy=True,
                    ref_mol=mol,
                )
                if pil is not None:
                    core_foil_pils.append(pil)

        # Anti-contrastive scaffold foils
        anti_core_foil_pils = []
        for rec in anti_scaffold_records:
            foil_mol, annotated_core = mc.build_core_foil(
                mol, rec["core_smiles"], similarity_threshold=similarity_threshold
            )
            if foil_mol is not None:
                pil = _show_mol_as_pil(
                    foil_mol,
                    legend=f"Anti-{rec['rank']}\ncontrast={rec['contrast_score']:.3f}",
                    corestructure=annotated_core,
                    highlight_subs=True,
                    alt_dummy=True,
                    ref_mol=mol,
                )
                if pil is not None:
                    anti_core_foil_pils.append(pil)

        try:
            all_core_pils = (
                ([query_pil] if query_pil is not None else [])
                + ([original_core_pil] if original_core_pil is not None else [])
                + core_foil_pils
                + anti_core_foil_pils
            )
            if all_core_pils:
                img_path_scaffolds = base_path.parent / f"{base_path.name}_scaffolds.png"
                _make_mol_grid(all_core_pils, cols=len(all_core_pils)).save(str(img_path_scaffolds))
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
        **({"anti_contrastive_rgroups": anti_rgroup_records} if include_anti_contrastive else {}),
        "contrastive_scaffolds": scaffold_records,
        **({"anti_contrastive_scaffolds": anti_scaffold_records} if include_anti_contrastive else {}),
        "num_rgroups_evaluated": len(rgroup_df),
        "num_scaffolds_evaluated": num_scaffolds_evaluated,
        "image_path_rgroups": str(img_path_rgroups) if img_path_rgroups else None,
        "image_path_scaffolds": str(img_path_scaffolds) if img_path_scaffolds else None,
        "status": "completed",
    }

    return output_images + [json.dumps(metadata, indent=2)]



def identify_recurrent_molce_rules(
    split_file_path: str,
    model_path: str,
    fact_class: int,
    foil_class: int,
    split: str = "test",
    n_bits: int = 2048,
    radius: int = 2,
    min_occurrences: int = 2,
    min_core_occurrences: int = 3,
    max_compounds: Optional[int] = None,
    core_dict_path: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
    include_anti_contrastive: bool = False,
    gnn_model_class_name: Optional[str] = None,
    gnn_hidden_channels: int = 64,
    gnn_num_classes: int = 2,
) -> list:
    """Global MolCE analysis: aggregate the top-3 contrastive R-group and scaffold rules.

    Runs ``get_contrastive_rgroups`` and ``get_contrastive_cores`` on every
    correctly predicted compound of *fact_class* and aggregates scores by
    R-group / scaffold SMILES:

        mean_contrast = groupby(smiles).contrast.mean()

    Returns the **top 3** most recurrent contrastive substituents and scaffolds
    with their mean contrast scores and molecule-grid visualizations inline.

    Motifs with a high mean contrast are the structural features that most
    consistently distinguish *fact_class* from *foil_class* across the
    population — the dataset-level chemical logic the model has learned.

    Args:
    split_file_path : str
        Path to the split .pkl file (from split_dataset).
    model_path : str
        Path to the trained model (.pkl for sklearn, .pt for GNN).
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
    min_occurrences : int, optional
        Minimum number of compounds in which an R-group must appear to be
        included (default 2).  Filters out idiosyncratic substituents.
    min_core_occurrences : int, optional
        Minimum number of compounds in which a scaffold core must appear to be
        included (default 3).  Higher threshold keeps only well-supported cores.
    max_compounds : int, optional
        Cap the number of compounds analyzed (default None = all).
    core_dict_path : str, optional
        Path to a custom ``core_dict_generic.pkl``.  When omitted the bundled
        library is used.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for external scaffolds.
    output_path : str, optional
        Base path for output images (.png).  Two images are saved:
        ``<base>_rgroups.png`` and ``<base>_scaffolds.png``.
        Defaults to ``session_dir/plots/molce_global_fact<fact_class>_foil<foil_class>_<session_id>``.
    include_anti_contrastive : bool, optional
        When True, also return the top-3 anti-contrastive substituents and
        scaffolds — those that *reinforce* the fact class (negative mean
        contrastive scores).  They are appended to the image grids and
        reported under ``anti_rgroup_rules`` / ``anti_scaffold_rules`` in
        the JSON output.  Default False.
    gnn_model_class_name : str, optional
        GNN architecture name: one of GCN | GraphSAGE | GAT | GC_GNN | GIN.
        Auto-detected from .pt checkpoint metadata when omitted.
    gnn_hidden_channels : int, optional
        Hidden dimension of the GNN (default 64). Must match training config.
    gnn_num_classes : int, optional
        Number of output classes of the GNN (default 2). Must match training config.

    Returns:
    list
        Two MCPImage objects (R-group grid, scaffold grid) followed by a JSON
        metadata string.

        JSON fields:
        - fact_class, foil_class, split
        - compounds_analyzed, compounds_failed
        - total_rgroup_evaluations, total_scaffold_evaluations
        - rgroup_rules: top-3 dicts (r_group_smiles, mean_contrast,
          std_contrast, occurrences, r_group_site_most_common) — positive scores
        - anti_rgroup_rules: top-3 R-groups reinforcing the fact class (same
          fields, negative mean_contrast) — only present when
          include_anti_contrastive=True
        - scaffold_rules: top-3 dicts (core_smiles, mean_contrast,
          std_contrast, occurrences) — positive scores
        - anti_scaffold_rules: top-3 scaffolds reinforcing the fact class (same
          fields, negative mean_contrast) — only present when
          include_anti_contrastive=True
        - image_path_rgroups, image_path_scaffolds
        - status: "completed"

    Raises:
    ValueError
        If fact_class == foil_class, or split file/model cannot be loaded,
        or no correctly predicted compounds are found.

    Examples:
    >>> rules = identify_recurrent_molce_rules(
    ...     split_file_path="data/logs/session_xxx/splits/data_random.pkl",
    ...     model_path="data/logs/session_xxx/models/data_RFC.pkl",
    ...     fact_class=1,
    ...     foil_class=0,
    ... )
    """
    logger = _get_session_logger()

    if fact_class == foil_class:
        raise ValueError("fact_class and foil_class must differ.")

    # ---- Auto-detect GNN checkpoints from .pt metadata ----
    from chemagent.explainability.gnn_compat import infer_gnn_params
    gnn_model_class_name, gnn_hidden_channels, gnn_num_classes = infer_gnn_params(
        model_path, gnn_model_class_name, gnn_hidden_channels, gnn_num_classes,
    )

    # ---- Load split file ----
    try:
        split_data = joblib.load(split_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load split file from {split_file_path!r}: {e}")

    if f"{split}_labels" not in split_data:
        available = [k for k in split_data if "features" in k or "labels" in k]
        raise ValueError(f"Split '{split}' not found. Available: {available}")

    labels = split_data[f"{split}_labels"]
    smiles_list = split_data.get(f"{split}_smiles", [f"compound_{i}" for i in range(len(labels))])

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

    # ---- Load model, get predictions, build predict adapters ----
    if gnn_model_class_name is not None:
        from chemagent.explainability.gnn_compat import (
            load_chemagent_gnn, infer_from_mols, make_gnn_molce_predict_funcs,
        )
        try:
            gnn = load_chemagent_gnn(
                model_path, gnn_model_class_name,
                gnn_hidden_channels, gnn_num_classes,
            )
        except Exception as e:
            raise ValueError(f"Failed to load GNN model from {model_path!r}: {e}")
        model = gnn
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [mols[i] for i in valid_idx]
        _preds, _ = infer_from_mols(gnn, valid_mols)
        predictions = np.full(len(labels), -1, dtype=int)
        for arr_i, orig_i in enumerate(valid_idx):
            predictions[orig_i] = _preds[arr_i]
        predict_func, predict_func_proba = make_gnn_molce_predict_funcs(gnn)
    elif str(model_path).endswith(".pt"):
        raise ValueError(
            f"Model path {model_path!r} is a .pt file but no GNN architecture "
            "could be inferred from checkpoint metadata. Supply "
            "gnn_model_class_name explicitly (e.g. 'GCN', 'GraphSAGE')."
        )
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path!r}: {e}")
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba. MolCE requires probability "
                "estimates — use a probabilistic classifier (e.g. RFC, GBC)."
            )
        features = split_data[f"{split}_features"]
        predictions = model.predict(features)
        predict_func, predict_func_proba = _make_sklearn_predict_funcs(model, n_bits, radius)

    # ---- Find correctly predicted compounds of fact_class ----
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
    top_rgroups = grouped_r.head(3)
    anti_top_rgroups = (
        grouped_r[grouped_r["mean_contrast"] < 0].tail(3).iloc[::-1].reset_index(drop=True)
        if include_anti_contrastive else pd.DataFrame()
    )

    # ---- Aggregate scaffolds ----
    top_scaffolds = pd.DataFrame()
    anti_top_scaffolds = pd.DataFrame()
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
            grouped_c[grouped_c["occurrences"] >= min_core_occurrences]
            .sort_values("mean_contrast", ascending=False)
            .reset_index(drop=True)
        )
        top_scaffolds = grouped_c.head(3)
        anti_top_scaffolds = (
            grouped_c[grouped_c["mean_contrast"] < 0].tail(3).iloc[::-1].reset_index(drop=True)
            if include_anti_contrastive else pd.DataFrame()
        )

    # ---- Base output path ----
    # Qualify by fact_class + foil_class so repeat calls with different
    # class pairings don't overwrite each other in the same session.
    if output_path is None:
        base_path = (
            logger.session_dir / "plots"
            / f"molce_global_fact{fact_class}_foil{foil_class}_{logger.session_id}"
        )
    else:
        base_path = Path(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    output_images: list = []
    img_path_rgroups: Optional[Path] = None
    img_path_scaffolds: Optional[Path] = None

    # ---- R-group molecule grid ----
    def _mol_from_fragment(smi: str) -> Optional[Chem.Mol]:
        """Parse a fragment SMILES, replacing attachment points with H for rendering."""
        import re
        cleaned = re.sub(r"\[\*:\d+\]|\*", "[H]", smi)
        return Chem.MolFromSmiles(cleaned)

    rgroup_pils = []
    for rank, (_, row) in enumerate(top_rgroups.iterrows(), start=1):
        m = _mol_from_fragment(str(row["r_group_smiles"]))
        if m is not None:
            pil = _show_mol_as_pil(
                m,
                legend=f"Rank {rank} substituent\n\nmean contrast: {float(row['mean_contrast']):.3f}  (n={int(row['occurrences'])})",
            )
            if pil is not None:
                rgroup_pils.append(pil)

    # Anti-contrastive R-group grid entries
    for rank, (_, row) in enumerate(anti_top_rgroups.iterrows(), start=1):
        m = _mol_from_fragment(str(row["r_group_smiles"]))
        if m is not None:
            pil = _show_mol_as_pil(
                m,
                legend=f"Anti-{rank} substituent\n\nmean contrast: {float(row['mean_contrast']):.3f}  (n={int(row['occurrences'])})",
            )
            if pil is not None:
                rgroup_pils.append(pil)

    try:
        if rgroup_pils:
            img_path_rgroups = base_path.parent / f"{base_path.name}_rgroups.png"
            _make_mol_grid(rgroup_pils, cols=len(rgroup_pils)).save(str(img_path_rgroups))
            output_images.append(MCPImage(path=img_path_rgroups))
    except Exception:
        img_path_rgroups = None

    # ---- Scaffold molecule grid ----
    scaffold_pils = []
    if not top_scaffolds.empty:
        for rank, (_, row) in enumerate(top_scaffolds.iterrows(), start=1):
            m = _mol_from_fragment(str(row["core_smiles"]))
            if m is not None:
                pil = _show_mol_as_pil(
                    m,
                    legend=f"Rank {rank} core\nmean contrast: {float(row['mean_contrast']):.3f}  (n={int(row['occurrences'])})",
                )
                if pil is not None:
                    scaffold_pils.append(pil)

    if not anti_top_scaffolds.empty:
        for rank, (_, row) in enumerate(anti_top_scaffolds.iterrows(), start=1):
            m = _mol_from_fragment(str(row["core_smiles"]))
            if m is not None:
                pil = _show_mol_as_pil(
                    m,
                    legend=f"Anti-{rank} core\nmean contrast: {float(row['mean_contrast']):.3f}  (n={int(row['occurrences'])})",
                )
                if pil is not None:
                    scaffold_pils.append(pil)

    try:
        if scaffold_pils:
            img_path_scaffolds = base_path.parent / f"{base_path.name}_scaffolds.png"
            _make_mol_grid(scaffold_pils, cols=len(scaffold_pils)).save(str(img_path_scaffolds))
            output_images.append(MCPImage(path=img_path_scaffolds))
    except Exception:
        img_path_scaffolds = None

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

    anti_rgroup_rules_out = _serialize(anti_top_rgroups)
    for r in anti_rgroup_rules_out:
        r["r_group_site_most_common"] = int(r.get("r_group_site_most_common", 0))

    scaffold_rules_out = _serialize(top_scaffolds) if not top_scaffolds.empty else []
    anti_scaffold_rules_out = _serialize(anti_top_scaffolds) if not anti_top_scaffolds.empty else []

    metadata: dict[str, Any] = {
        "fact_class": fact_class,
        "foil_class": foil_class,
        "split": split,
        "compounds_analyzed": compounds_analyzed,
        "compounds_failed": compounds_failed,
        "total_rgroup_evaluations": total_rgroup_evals,
        "total_scaffold_evaluations": total_scaffold_evals,
        "rgroup_rules": rgroup_rules_out,
        **({"anti_rgroup_rules": anti_rgroup_rules_out} if include_anti_contrastive else {}),
        "scaffold_rules": scaffold_rules_out,
        **({"anti_scaffold_rules": anti_scaffold_rules_out} if include_anti_contrastive else {}),
        "image_path_rgroups": str(img_path_rgroups) if img_path_rgroups else None,
        "image_path_scaffolds": str(img_path_scaffolds) if img_path_scaffolds else None,
        "status": "completed",
    }

    return output_images + [json.dumps(metadata, indent=2)]


# ---------------------------------------------------------------------------
# Private helpers — attempt to reconstruct foil molecules for visualization
# ---------------------------------------------------------------------------

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
        if len(rgd) < 2:
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

"""chemagent.explainability.Counterfactuals.CF_generator_v2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core-based counterfactual (CF) generator using CCR fragmentation.

Updated from the original cf_generator_reduced:
- Proper relative imports for the bundled ccrlib package
- Sklearn-compatible model interface (model.predict / model.predict_proba)
- Internal ECFP featurization via chemagent.featurization.fingerprints
- Relative path for the default R-group substituent library
- Standard Python naming conventions (CFGenerator)
- Backward-compatible alias: cf_generator_reduced = CFGenerator
"""
from __future__ import annotations

import logging
import random
import sys

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, PandasTools
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")

# --- path setup -----------------------------------------------------------
# src/ must be on sys.path for chemagent imports
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The bundled ccrlib lives alongside this file (Counterfactuals/ccrlib/)
_CF_DIR = Path(__file__).resolve().parent
if str(_CF_DIR) not in sys.path:
    sys.path.insert(0, str(_CF_DIR))

from ccrlib import run_ccr  # noqa: E402

# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_DEFAULT_R_GROUP_PATH = _CF_DIR / "lst_r_group_all_revised.pkl"


class CFGenerator:
    """Generate counterfactual molecules using CCR core extraction and R-group substitution.

    Given a list of input molecules, the generator:

    1. Extracts molecular cores using CCR (Computational Chemistry R-group) analysis.
    2. Substitutes attachment points with a library of R-groups.
    3. Runs the trained model on each analog and filters for prediction class changes.

    Parameters
    ----------
    smiles_lst : list[str]
        SMILES of the input molecules (typically just the query compound).
    model_obj : sklearn estimator
        Trained sklearn-compatible model with ``.predict()`` and ``.predict_proba()``
        methods.  The model expects ECFP fingerprint arrays as input.
    n_bits : int, optional
        ECFP fingerprint length (default 2048).  Must match the model's training setup.
    radius : int, optional
        ECFP Morgan radius (default 2).  Must match the model's training setup.
    random_seed : int, optional
        Random seed for reproducibility (default 2023).
    custom_ccr_param_grid : dict, optional
        Custom CCR fragmentation parameters.  If None, sensible defaults are used.
    custom_r_group : list[str], optional
        Custom list of R-group SMILES.  If None, the bundled default library is used.
    base_class : int, optional
        Predicted class of the input molecules.  CFs are analogs that change this
        prediction (default 1).
    use_multiple_substituents : bool, optional
        If True, also try pairs of substituents on cores with two attachment points
        (default False).  Warning: significantly increases computation time.
    max_double_sub_samples : int, optional
        Maximum number of unique substituent *pairs* to try when
        ``use_multiple_substituents=True`` (default 500).  Pairs are drawn via
        random sampling without replacement from the full combinatorial space, so
        every sampled pair uses a different combination of two R-groups.  Has no
        effect when ``use_multiple_substituents=False``.
    return_all_conformations : bool, optional
        If True, keep all reaction products for each substitution (default False).
    num_classes : int, optional
        Number of model output classes (default 3 for ternary classification).
    """

    def __init__(
        self,
        smiles_lst: List[str],
        model_obj,
        n_bits: int = 2048,
        radius: int = 2,
        random_seed: int = 2023,
        custom_ccr_param_grid: Optional[dict] = None,
        custom_r_group: Optional[List[str]] = None,
        base_class: int = 1,
        use_multiple_substituents: bool = False,
        max_double_sub_samples: int = 1000,
        return_all_conformations: bool = False,
        num_classes: int = 3,
    ):
        self.smiles_lst = smiles_lst
        self.model = model_obj
        self.n_bits = n_bits
        self.radius = radius
        self.seed = random_seed
        self.base_class = base_class
        self.custom_ccr_param_grid = custom_ccr_param_grid
        self.custom_r_group = custom_r_group
        self.use_multiple_substituents = use_multiple_substituents
        self.max_double_sub_samples = max_double_sub_samples
        self.return_all_conformations = return_all_conformations
        self.num_classes = num_classes

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.ccr_hparameters = self._ccr_hyperparameters()
        self.suc_df = self._run_ccr_analysis()
        self.subs_lst_smiles, self.sub_lst_mols = self._get_substituents()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ccr_hyperparameters(self) -> dict:
        if self.custom_ccr_param_grid is not None:
            return self.custom_ccr_param_grid
        return {
            "cut_type": "synthesizable",
            "max_cuts": 5,
            "min_rel_core_size": 0.666,
            "max_frag_size": 13,
            "max_time": 300,
            "mol_filter": lambda x: Descriptors.MolWt(x) <= 1000,
        }

    def _run_ccr_analysis(
        self,
        hparameters: Optional[dict] = None,
        data_set: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if hparameters is None:
            hparameters = self.ccr_hparameters
        smiles_source = data_set if data_set is not None else self.smiles_lst
        suppl = [Chem.MolFromSmiles(s) for s in smiles_source]

        result = run_ccr(
            suppl,
            hparameters["cut_type"],
            hparameters["max_cuts"],
            hparameters["min_rel_core_size"],
            hparameters["max_frag_size"],
            hparameters["max_time"],
            hparameters["mol_filter"],
            basename=None,
        )

        suc_df = pd.DataFrame(result.unique.get_fragmented_compound_list())
        if suc_df.empty or "core" not in suc_df.columns:
            return suc_df  # No shared cores found; find_cfs will return empty DataFrame
        PandasTools.AddMoleculeColumnToFrame(suc_df, "core", "core_mol")
        PandasTools.AddMoleculeColumnToFrame(suc_df, "substituents", "subst_mol")
        return suc_df

    def _get_substituents(self):
        if self.custom_r_group is not None:
            subs_smiles = list(set(self.custom_r_group))
        else:
            subs_smiles = pd.read_pickle(_DEFAULT_R_GROUP_PATH).SMILES.tolist()
        sub_mols = [Chem.MolFromSmiles(s) for s in subs_smiles if Chem.MolFromSmiles(s) is not None]
        return subs_smiles, sub_mols

    def _featurize(self, mols: List) -> np.ndarray:
        """Convert RDKit mol objects to ECFP fingerprint array for model input."""
        from chemagent.featurization.fingerprints import ECFP
        smiles = [Chem.MolToSmiles(m) for m in mols]
        fps = ECFP(smiles, n_bits=self.n_bits, radius=self.radius)
        return np.array(fps)

    def _protonate_clean_mol(self, mol):
        du = Chem.MolFromSmiles("*")
        mol = AllChem.ReplaceSubstructs(mol, du, Chem.MolFromSmiles("[H]"), True)[0]
        mol = Chem.RemoveHs(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
        return mol

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_samples(self):
        """Generator yielding per-core DataFrames of substituted molecules."""
        if self.suc_df.empty or "core" not in self.suc_df.columns:
            return  # No cores found; yield nothing

        rxn = AllChem.ReactionFromSmarts("[*:1][#0].[#0][*:2]>>[*:1]-[*:2]")

        for core_str in self.suc_df.core.unique().tolist():
            CF_lst: List = []
            sub_lst_used: List = []

            core_mol_series = list(self.suc_df.loc[self.suc_df["core"] == core_str].core_mol)
            if not core_mol_series:
                continue
            core_mol = core_mol_series[0]
            core_list = [core_mol]

            # Single substituent replacement
            for R_group_mol in self.sub_lst_mols:
                core_sub = rxn.RunReactants((core_mol, R_group_mol))
                if not core_sub:
                    continue
                if self.return_all_conformations:
                    for struc in core_sub:
                        try:
                            interim = self._protonate_clean_mol(struc[0])
                            CF_lst.append(interim)
                            sub_lst_used.append([R_group_mol])
                        except Exception:
                            pass
                else:
                    try:
                        interim = self._protonate_clean_mol(core_sub[0][0])
                        CF_lst.append(interim)
                        sub_lst_used.append([R_group_mol])
                    except Exception:
                        pass

            # Double substituent replacement (optional, limited for diversity)
            if self.use_multiple_substituents and "[*:1]" in core_str and "[*:2]" in core_str:
                n_subs = len(self.sub_lst_mols)
                n_pairs = n_subs * n_subs
                sample_size = min(self.max_double_sub_samples, n_pairs)
                # Sample unique flat indices then convert to (i, j) pairs — guarantees
                # diverse substituent combinations within the budget.
                flat_indices = random.sample(range(n_pairs), sample_size)
                sampled_pairs = [(self.sub_lst_mols[k // n_subs], self.sub_lst_mols[k % n_subs])
                                 for k in flat_indices]
                for sub1_mol, sub2_mol in sampled_pairs:
                    core_sub_1 = rxn.RunReactants((core_list[0], sub1_mol))
                    if not core_sub_1:
                        continue
                    if self.return_all_conformations:
                        for struc in core_sub_1:
                            try:
                                interim = self._protonate_clean_mol(struc[0])
                                core_sub_2 = rxn.RunReactants((interim, sub2_mol))
                                if not core_sub_2:
                                    CF_lst.append(interim)
                                    sub_lst_used.append([sub1_mol])
                                else:
                                    for struc_2 in core_sub_2:
                                        try:
                                            interim2 = self._protonate_clean_mol(struc_2[0])
                                            CF_lst.append(interim2)
                                            sub_lst_used.append([sub1_mol, sub2_mol])
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    else:
                        try:
                            idx = random.randint(0, len(core_sub_1) - 1)
                            core_sub_2 = rxn.RunReactants((core_sub_1[idx][0], sub2_mol))
                            if core_sub_2:
                                idx2 = random.randint(0, len(core_sub_2) - 1)
                                interim = self._protonate_clean_mol(core_sub_2[idx2][0])
                            else:
                                interim = self._protonate_clean_mol(core_sub_1[idx][0])
                            CF_lst.append(interim)
                            sub_lst_used.append([sub1_mol, sub2_mol])
                        except Exception:
                            pass

            if not CF_lst:
                continue

            CF_smile_lst = [Chem.MolToSmiles(cf) for cf in CF_lst]
            mol_weight_cfs = [ExactMolWt(mol) for mol in CF_lst]
            MW_core = ExactMolWt(core_mol)

            cf_df_core = pd.DataFrame({
                "CF_mol": CF_lst,
                "CF_smiles": CF_smile_lst,
                "sub_mols": sub_lst_used,
                "CF_MW": mol_weight_cfs,
            })
            cf_df_core["core_smiles"] = core_str

            core_mol_clean = self._protonate_clean_mol(core_mol)
            fps_core = self._featurize([core_mol_clean])
            predictions_core = self.model.predict(fps_core)

            cf_df_core["core_mol"] = core_mol_clean
            cf_df_core["core_MW"] = MW_core
            cf_df_core["change_MW"] = cf_df_core.CF_MW - cf_df_core.core_MW
            cf_df_core["core_predicted"] = int(predictions_core[0])

            if hasattr(self.model, "predict_proba"):
                proba_core = self.model.predict_proba(fps_core)
                if self.num_classes > 2:
                    for num in range(self.num_classes):
                        cf_df_core[f"core_proba_class_{num}"] = float(proba_core[0][num])
                else:
                    cf_df_core["core_proba_class_1"] = float(proba_core[0][1])

            cf_df_core = cf_df_core.drop_duplicates(subset="CF_smiles").reset_index(drop=True)
            yield cf_df_core

    def find_cfs(self) -> pd.DataFrame:
        """Generate and filter counterfactual molecules.

        Returns
        -------
        pd.DataFrame
            Analogs that change the model's prediction away from ``base_class``.
            Returns an empty DataFrame if no counterfactuals are found.
        """
        all_dfs = list(self._generate_samples())
        if not all_dfs:
            return pd.DataFrame()

        cf_df = pd.concat(all_dfs).reset_index(drop=True)

        fps_cfs = self._featurize(cf_df.CF_mol.tolist())
        predictions_cfs = self.model.predict(fps_cfs)

        predictions_df = pd.DataFrame({"Predicted": predictions_cfs})

        if hasattr(self.model, "predict_proba"):
            proba_cfs = self.model.predict_proba(fps_cfs)
            if self.num_classes == 2:
                proba_df = pd.DataFrame({
                    "proba_class_0": proba_cfs[:, 0],
                    "proba_class_1": proba_cfs[:, 1],
                })
            else:
                proba_df = pd.DataFrame(
                    proba_cfs,
                    columns=[f"proba_class_{i}" for i in range(self.num_classes)],
                )
        else:
            proba_df = pd.DataFrame()

        cf_df = pd.concat([cf_df, predictions_df, proba_df], axis=1).reset_index(drop=True)

        actual_cf_df = cf_df.loc[
            (cf_df["Predicted"] != cf_df["core_predicted"]) &
            (cf_df["core_predicted"] == self.base_class)
        ].reset_index(drop=True)

        logger.info("Total CFs found: %d", len(actual_cf_df))
        return actual_cf_df


# Backward-compatible alias
cf_generator_reduced = CFGenerator

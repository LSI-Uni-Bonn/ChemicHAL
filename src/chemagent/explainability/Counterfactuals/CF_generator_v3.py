"""chemagent.explainability.Counterfactuals.CF_generator_v3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MolCE-based counterfactual (CF) generator for a single query compound.

Four perturbation strategies are used to build hypothetical analogues:

1. **Substituent (single)** — original core, one R-group position swapped at
   a time from a random sample of the combined dataset + static R-group library.
2. **Substituent (multi)** — original core, two or more R-group positions
   swapped simultaneously (randomly sampled, capped by
   ``max_multisite_rgroup_combos``).
3. **Core change** — original R-groups, Murcko scaffold swapped with similar
   scaffolds from the bundled library.
4. **Combination** — alternative scaffold *and* one R-group swapped
   simultaneously (capped by ``max_combination_cores``).

Performance notes
-----------------
``get_order`` in MolCE iterates over the *entire* external R-group library for
every attachment-point position, producing O(n_rgroups × n_positions) molzip
calls.  With the combined dataset + static library this can easily exceed
10 000 entries, making generation and subsequent RFC prediction very slow.

``max_rgroups_for_enumeration`` caps the library used during enumeration to a
random sample (default 500).  The full library is still built for richness, but
only the sample is used to generate candidates.

``max_combination_cores`` limits how many alternative scaffolds are tried in the
combination step (default 5), preventing combinatorial explosion when the
scaffold dictionary returns many matches.

All MolCE "seen product before" / "failed product" console prints are silenced
by overriding the relevant methods in the internal ``_SafeMolContrast`` subclass.

Tanimoto similarity to the query is computed *only* for confirmed counterfactuals
(after the class-change filter), not for the full candidate pool.
"""
from __future__ import annotations

import copy
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_BUNDLED_CORE_DICT = Path(__file__).resolve().parents[1] / "MolCE" / "core_dict_generic.pkl"
_STATIC_R_GROUP_PATH = Path(__file__).resolve().parent / "lst_r_group_all_revised.pkl"

logger = logging.getLogger(__name__)


class CFGenerator:
    """Generate counterfactuals for a single query compound using MolCE decomposition.

    Parameters
    ----------
    query_smiles : str
        SMILES of the compound to analyse.
    model_obj : sklearn estimator
        Trained model with ``.predict()`` and ``.predict_proba()`` methods.
        The model expects ECFP fingerprint arrays.
    data_smiles : list[str]
        Dataset SMILES used to build the R-group library (typically all unique
        SMILES from train + test splits).  The library is automatically
        extended with R-groups from the bundled ``lst_r_group_all_revised.pkl``
        static library; duplicates are removed.
    n_bits : int, optional
        ECFP fingerprint length (default 2048).  Must match model training setup.
    radius : int, optional
        ECFP Morgan radius (default 2).  Must match model training setup.
    core_dict_path : str, optional
        Path to a custom ``core_dict_generic.pkl`` scaffold library.  When
        omitted the bundled MolCE library is used.
    similarity_threshold : float, optional
        Size-similarity filter (0–1) for alternative scaffolds: only scaffolds
        whose atom count differs from the original by at most
        ``(1 - similarity_threshold) × 100 %`` are considered.
    max_rgroups_for_enumeration : int, optional
        Maximum number of R-groups sampled from the full library for candidate
        generation (default 500).  Larger values explore more chemical space at
        the cost of speed: O(n × n_positions) molzip calls per step.
        Set to 0 to use the full library.
    max_combination_cores : int, optional
        Maximum number of alternative scaffold cores tried in the combination
        step (default 5).  Prevents combinatorial explosion when many scaffold
        matches are found.
    max_multisite_rgroup_combos : int, optional
        Maximum number of randomly sampled multi-site R-group combinations to
        try (default 200).  Set to 0 to disable multi-site generation.
    random_seed : int, optional
        Seed for all random sampling (default 42).

    Attributes (set after ``find_cfs()``)
    --------------------------------------
    n_candidates_tested : int
        Number of unique hypothetical compounds evaluated by the model during
        the last ``find_cfs()`` call.
    """

    def __init__(
        self,
        query_smiles: str,
        model_obj,
        data_smiles: List[str],
        n_bits: int = 2048,
        radius: int = 2,
        core_dict_path: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        max_rgroups_for_enumeration: int = 1000,
        max_combination_cores: int = 10,
        max_multisite_rgroup_combos: int = 500,
        random_seed: int = 42,
    ):
        self.query_smiles = query_smiles
        self.model = model_obj
        self.data_smiles = list(data_smiles)
        self.n_bits = n_bits
        self.radius = radius
        self.similarity_threshold = similarity_threshold
        self.max_rgroups_for_enumeration = max_rgroups_for_enumeration
        self.max_combination_cores = max_combination_cores
        self.max_multisite_rgroup_combos = max_multisite_rgroup_combos
        self.random_seed = random_seed
        self.n_candidates_tested: int = 0

        self.query_mol = Chem.MolFromSmiles(query_smiles)
        if self.query_mol is None:
            raise ValueError(f"Invalid query SMILES: {query_smiles!r}")

        self._mc = self._build_mol_contrast(core_dict_path)
        self._augment_rgroups()

        fps = self._featurize([self.query_mol])
        self.query_class = int(self.model.predict(fps)[0])
        self.query_probas: List[float] = self.model.predict_proba(fps)[0].tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _featurize(self, mols: List[Chem.Mol]) -> np.ndarray:
        from chemagent.featurization.fingerprints import ECFP
        smiles = [Chem.MolToSmiles(m) for m in mols]
        fps = ECFP(smiles, n_bits=self.n_bits, radius=self.radius)
        return np.array(fps)

    def _build_mol_contrast(self, core_dict_path: Optional[str]):
        """Instantiate MolContrast with a controlled scaffold-dict load.

        The ``_SafeMolContrast`` subclass additionally overrides
        ``core_ext_rgroup_enumeration`` and ``generate_ext_r_foils`` to silence
        all console prints ("failed product", "seen product before") that appear
        in MolCE's original implementation.
        """
        import pickle
        from chemagent.explainability.MolCE.MolCE import MolContrast

        resolved = core_dict_path if core_dict_path is not None else str(_BUNDLED_CORE_DICT)
        try:
            with open(resolved, "rb") as fh:
                _core_dict: dict = pickle.load(fh)
        except Exception as e:
            raise ValueError(f"Failed to load scaffold dict from {resolved!r}: {e}")

        n_bits, radius, model = self.n_bits, self.radius, self.model

        class _SafeMolContrast(MolContrast):

            def get_scaffold_dict(self_inner):  # noqa: N805
                return _core_dict

            def core_ext_rgroup_enumeration(self_inner, core, order):  # noqa: N805
                """Silent version: skip failed products instead of printing."""
                for tpl in order:
                    try:
                        tm = Chem.RWMol(core)
                        for i, r in enumerate(tpl):
                            if r is not None:
                                subbed_str = re.sub(
                                    r"\*", f"[*:{i + 1}]", Chem.MolToSmiles(r)
                                )
                                r_re = Chem.MolFromSmiles(subbed_str)
                                if r_re is None:
                                    raise ValueError("invalid R-group SMILES")
                                tm.InsertMol(r_re)
                        prod = Chem.molzip(tm)
                        if prod is None:
                            continue
                        if None in tpl:
                            du = Chem.MolFromSmiles("*")
                            prod = AllChem.ReplaceSubstructs(
                                prod, du, Chem.MolFromSmiles("[H]"), True
                            )[0]
                            prod = Chem.RemoveHs(prod)
                        prod.UpdatePropertyCache(strict=True)
                        Chem.SanitizeMol(prod)
                        yield prod
                    except Exception:
                        continue

            def generate_ext_r_foils(self_inner, core, order, external_rgroups_used):  # noqa: N805
                """Silent version: drop duplicates without printing."""
                products, rgroups, seen_local = [], [], set()
                gen = self_inner.core_ext_rgroup_enumeration(core, order)
                for prod, rgroup in zip(gen, external_rgroups_used):
                    smi = Chem.MolToSmiles(prod)
                    if smi not in seen_local:
                        products.append(prod)
                        rgroups.append(rgroup)
                        seen_local.add(smi)
                return products, [Chem.MolToSmiles(r) for r in rgroups]

        def _featurize_inner(mol_or_mols, singular: bool) -> np.ndarray:
            from chemagent.featurization.fingerprints import ECFP
            smiles_list = (
                [Chem.MolToSmiles(mol_or_mols)] if singular
                else [Chem.MolToSmiles(m) for m in mol_or_mols]
            )
            fps = ECFP(smiles_list, n_bits=n_bits, radius=radius)
            return np.asarray(fps)

        def predict_func(m, mol, singular: bool = False):
            fps = _featurize_inner(mol, singular)
            preds = m.predict(fps)
            return int(preds[0]) if singular else preds

        def predict_func_proba(m=None, mol=None, singular: bool = False):
            fps = _featurize_inner(mol, singular)
            probas = m.predict_proba(fps)
            return probas[0] if singular else probas

        return _SafeMolContrast(
            data_smiles=self.data_smiles,
            model=model,
            predict_func=predict_func,
            predict_func_proba=predict_func_proba,
        )

    def _augment_rgroups(self) -> None:
        """Extend ``_mc.external_rgroups`` with R-groups from the static library.

        ``MolContrast.decompose_dataset`` already populated ``external_rgroups``
        from the dataset SMILES.  This method adds any R-groups present in
        ``lst_r_group_all_revised.pkl`` that are not yet in the library, giving
        a richer substitution pool without duplicates.
        """
        try:
            static_df = pd.read_pickle(str(_STATIC_R_GROUP_PATH))
            static_smiles: List[str] = static_df["SMILES"].tolist()
        except Exception as exc:
            logger.warning("Could not load static R-group library: %s", exc)
            return

        existing_smi = {
            Chem.MolToSmiles(m)
            for m in self._mc.external_rgroups
            if m is not None
        }

        added = 0
        for smi in static_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in existing_smi:
                self._mc.external_rgroups.append(mol)
                existing_smi.add(canon)
                added += 1

        logger.debug(
            "R-group library: %d from dataset + %d from static library = %d total",
            len(self._mc.external_rgroups) - added,
            added,
            len(self._mc.external_rgroups),
        )

    def _get_enumeration_rgroups(self) -> List[Chem.Mol]:
        """Return the R-group subset used for enumeration.

        Returns a random sample of ``max_rgroups_for_enumeration`` entries from
        the full library.  If the library is smaller than the cap, the full
        library is returned.  Set ``max_rgroups_for_enumeration=0`` to always
        use the full library.
        """
        full = self._mc.external_rgroups
        if self.max_rgroups_for_enumeration <= 0 or len(full) <= self.max_rgroups_for_enumeration:
            return full
        rng = random.Random(self.random_seed)
        return rng.sample(full, self.max_rgroups_for_enumeration)

    def _build_order(
        self, core: Chem.Mol, rgroups: List[Chem.Mol], ext_rgroups: List[Chem.Mol]
    ) -> List[List[Optional[Chem.Mol]]]:
        """Build a ``get_order``-style list using a pre-sampled R-group subset.

        Equivalent to ``MolContrast.get_order`` but uses ``ext_rgroups``
        directly instead of ``self.external_rgroups``, so the caller controls
        the sample size.
        """
        n_positions = Chem.MolToSmiles(core).count("*")
        order = []
        for i in range(n_positions):
            for ext in ext_rgroups:
                modified = copy.deepcopy(rgroups)
                modified[i] = ext
                order.append(modified)
        return order

    def _enumerate_rfoils(
        self,
        core: Chem.Mol,
        order: List[List[Optional[Chem.Mol]]],
        seen: set,
    ) -> List[Chem.Mol]:
        """Run ``core_ext_rgroup_enumeration`` and deduplicate against *seen*.

        The ``seen`` set is shared across all generation steps so cross-type
        duplicates are also eliminated without extra passes.
        """
        products: List[Chem.Mol] = []
        for mol in self._mc.core_ext_rgroup_enumeration(core, order):
            # core_ext_rgroup_enumeration already silences failures; mols here
            # are valid and sanitized.
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if smi and smi not in seen:
                seen.add(smi)
                products.append(mol)
        return products

    def _sample_multisite_orders(
        self,
        rgroups: List[Chem.Mol],
        ext_rgroups: List[Chem.Mol],
    ) -> List[List[Optional[Chem.Mol]]]:
        """Sample random multi-site R-group combinations.

        Each sample replaces 2 or more R-group positions simultaneously with
        R-groups drawn from ``ext_rgroups``.
        """
        n_positions = len(rgroups)
        if n_positions < 2 or self.max_multisite_rgroup_combos <= 0:
            return []

        n_ext = len(ext_rgroups)
        if n_ext == 0:
            return []

        rng = random.Random(self.random_seed)
        sampled_keys: set = set()
        orders: List[List[Optional[Chem.Mol]]] = []
        max_attempts = self.max_multisite_rgroup_combos * 10

        for _ in range(max_attempts):
            if len(orders) >= self.max_multisite_rgroup_combos:
                break
            n_replace = rng.randint(2, n_positions)
            positions = tuple(sorted(rng.sample(range(n_positions), n_replace)))
            rgroup_indices = tuple(rng.randrange(n_ext) for _ in positions)
            key = (positions, rgroup_indices)
            if key in sampled_keys:
                continue
            sampled_keys.add(key)
            modified = list(rgroups)
            for pos, ridx in zip(positions, rgroup_indices):
                modified[pos] = ext_rgroups[ridx]
            orders.append(modified)

        return orders

    def _generate_candidates(self) -> List[Tuple[Chem.Mol, str]]:
        """Return deduplicated (mol, change_type) tuples for all perturbation types."""
        try:
            core, rgroups = self._mc.decompose_molecule(self.query_mol, original=True)
        except ValueError as e:
            raise ValueError(f"Query molecule cannot be decomposed: {e}")

        # Sample R-group subset once — shared by all substituent/combination steps.
        ext_rgroups = self._get_enumeration_rgroups()

        # Global seen-set, seeded with the query, shared across all steps.
        query_canon = Chem.MolToSmiles(self.query_mol)
        seen: set[str] = {query_canon}

        candidates: List[Tuple[Chem.Mol, str]] = []

        # 1. Substituent (single) — original core, one R-group at a time
        try:
            order = self._build_order(core, rgroups, ext_rgroups)
            for mol in self._enumerate_rfoils(core, order, seen):
                candidates.append((mol, "substituent"))
            logger.debug("Substituent (single): %d candidates", sum(1 for _, t in candidates if t == "substituent"))
        except Exception as exc:
            logger.debug("Single-substituent generation failed: %s", exc)

        # 2. Substituent (multi) — original core, 2+ positions simultaneously
        try:
            multi_order = self._sample_multisite_orders(rgroups, ext_rgroups)
            for mol in self._enumerate_rfoils(core, multi_order, seen):
                candidates.append((mol, "substituent_multi"))
            logger.debug("Substituent (multi): %d candidates", sum(1 for _, t in candidates if t == "substituent_multi"))
        except Exception as exc:
            logger.debug("Multi-substituent generation failed: %s", exc)

        # 3. Core change — original R-groups, swap Murcko scaffold
        try:
            core_prods, _ = self._mc.generate_ext_core_foils(
                core, rgroups, similarity_threshold=self.similarity_threshold
            )
            for mol in core_prods:
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    continue
                if smi and smi not in seen:
                    seen.add(smi)
                    candidates.append((mol, "core"))
            logger.debug("Core change: %d candidates", sum(1 for _, t in candidates if t == "core"))
        except Exception as exc:
            logger.debug("Core generation failed: %s", exc)

        # 4. Combination — alternative core + one R-group swap
        #    Capped at max_combination_cores to avoid combinatorial explosion.
        try:
            alt_cores = self._mc.get_scaffolds(core, self.similarity_threshold)
            for alt_core in alt_cores[: self.max_combination_cores]:
                try:
                    combo_order = self._build_order(alt_core, rgroups, ext_rgroups)
                    for mol in self._enumerate_rfoils(alt_core, combo_order, seen):
                        candidates.append((mol, "combination"))
                except Exception:
                    continue
            logger.debug("Combination: %d candidates", sum(1 for _, t in candidates if t == "combination"))
        except Exception as exc:
            logger.debug("Combination generation failed: %s", exc)

        logger.info(
            "Generated %d unique candidates  (R-group pool used: %d / %d total)",
            len(candidates),
            len(ext_rgroups),
            len(self._mc.external_rgroups),
        )
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_cfs(self, max_counterfactuals: int = 20) -> pd.DataFrame:
        """Generate, predict, filter and return counterfactuals.

        After this method returns, ``self.n_candidates_tested`` is set to the
        number of unique hypothetical compounds evaluated by the model.

        Candidates where the model prediction class differs from the query are
        kept as counterfactuals.  Tanimoto similarity to the query is computed
        only for confirmed counterfactuals (not the full candidate pool), then
        results are sorted by similarity (highest first).

        Parameters
        ----------
        max_counterfactuals : int, optional
            Maximum number of CFs to return (default 20).

        Returns
        -------
        pd.DataFrame
            Columns:

            - ``cf_smiles``            — canonical SMILES of the counterfactual
            - ``predicted_class``      — model-predicted class for the CF
            - ``proba_class_<i>``      — per-class probabilities
            - ``tanimoto_similarity``  — Tanimoto similarity to the query
            - ``change_type``          — ``"substituent"``, ``"substituent_multi"``,
              ``"core"``, or ``"combination"``
            - ``query_smiles``         — original query SMILES
            - ``query_class``          — original predicted class

            Returns an empty DataFrame if no counterfactuals are found.
        """
        candidates = self._generate_candidates()
        self.n_candidates_tested = len(candidates)

        if not candidates:
            return pd.DataFrame()

        cand_mols = [m for m, _ in candidates]
        change_types = [t for _, t in candidates]

        # Batch predict all candidates
        fps = self._featurize(cand_mols)
        pred_classes = self.model.predict(fps)
        pred_probas = self.model.predict_proba(fps)
        n_classes = pred_probas.shape[1]

        # Filter to counterfactuals — Tanimoto computed only for these
        cf_mols: List[Chem.Mol] = []
        cf_ctypes: List[str] = []
        cf_pred_classes: List[int] = []
        cf_probas: List[np.ndarray] = []

        for mol, ctype, pred_cls, proba in zip(
            cand_mols, change_types, pred_classes, pred_probas
        ):
            if int(pred_cls) != self.query_class:
                cf_mols.append(mol)
                cf_ctypes.append(ctype)
                cf_pred_classes.append(int(pred_cls))
                cf_probas.append(proba)

        if not cf_mols:
            logger.info(
                "CFGenerator: 0 counterfactuals from %d candidates tested",
                self.n_candidates_tested,
            )
            return pd.DataFrame()

        # Tanimoto similarity — only for confirmed counterfactuals
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_query = fp_gen.GetFingerprint(self.query_mol)
        fp_cfs = fp_gen.GetFingerprints(cf_mols)
        similarities = DataStructs.BulkTanimotoSimilarity(fp_query, list(fp_cfs))

        rows = []
        for mol, ctype, pred_cls, proba, sim in zip(
            cf_mols, cf_ctypes, cf_pred_classes, cf_probas, similarities
        ):
            row: dict = {
                "cf_smiles": Chem.MolToSmiles(mol),
                "predicted_class": pred_cls,
                "tanimoto_similarity": float(sim),
                "change_type": ctype,
                "query_smiles": self.query_smiles,
                "query_class": self.query_class,
            }
            for i in range(n_classes):
                row[f"proba_class_{i}"] = float(proba[i])
            rows.append(row)

        cf_df = (
            pd.DataFrame(rows)
            .sort_values("tanimoto_similarity", ascending=False)
            .head(max_counterfactuals)
            .reset_index(drop=True)
        )

        logger.info(
            "CFGenerator: %d counterfactuals from %d candidates tested",
            len(cf_df),
            self.n_candidates_tested,
        )
        return cf_df

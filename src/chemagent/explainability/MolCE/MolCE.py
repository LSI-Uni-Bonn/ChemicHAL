from typing import *
import pandas as pd
import pickle
import numpy as np
import re
import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdRGroupDecomposition
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import copy
from .MolCE_utils import *

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


class MolContrast:

    def __init__(self,
                 data_smiles: List[str],
                 model,
                 predict_func: Callable,
                 predict_func_proba: Callable,
                 ):
        """
        Initialize the MolContrast class.

        Args:
            data_smiles (List[str]): List of SMILES strings from the dataset.
            model (object): Model for prediction.
            predict_func (Callable): Function to predict the class of a molecule.
            predict_func_proba (Callable): Function to predict probabilities for a molecule.
        """

        self.model = model
        self.predict_func = predict_func
        self.predict_func_proba = predict_func_proba
        self.external_rgroups = self.decompose_dataset(data_smiles)
        self.scaffold_dict = self.get_scaffold_dict()

    def decompose_molecule(self,
                           mol: Chem.Mol,
                           original: bool = False) -> Tuple[Chem.Mol, List[Chem.Mol]]:
        """
        Decompose a molecule into Murcko scaffold and R-groups.

        Args:
            Mol[Chem.Mol]: Molecule to be decomposed.

        Returns:
           Tuple[[Chem.Mol], [List[Chem.Mol]]]: Murcko scaffold and list of R-groups
        """

        core = MurckoScaffold.GetScaffoldForMol(mol)
        rgd, fails = rdRGroupDecomposition.RGroupDecompose([core], [mol], asRows=False)

        if original is True and len(rgd) == 1:
            raise ValueError("molecule cannot be decomposed")

        elif len(rgd) == 1:
            return None, None

        else:
            core = rgd.pop("Core")[0]

            rgroups = []

            for i in range(len(rgd)):
                smile = Chem.MolToSmiles(rgd[f'R{i+1}'][0])
                smile = re.sub(r":\d+", "", smile)
                if smile.count('*') == 1:
                    rgroups.append(Chem.MolFromSmiles(smile))

            return core, rgroups

    def decompose_dataset(self, data_smiles: List[str]) -> List[Chem.Mol]:
        """
        Decompose a dataset of SMILES into unique R-groups.

        Args:
            data_smiles(List[str]): list of molecule smiles to be decomposed into
            R-groups

        Returns:
            List[Chem.Mol]: Extracted unique R-groups.
        """

        unique_rgroups = {}

        for smile in data_smiles:
            mol = Chem.MolFromSmiles(smile)
            core, rgroups = self.decompose_molecule(mol)

            if core is None or rgroups is None:
                continue

            for r in rgroups:
                r_smi = Chem.MolToSmiles(r)
                if r_smi not in unique_rgroups:
                    unique_rgroups[r_smi] = r

        return list(unique_rgroups.values())

    def get_scaffold_dict(self) -> dict:
        """
        Load the scaffold dictionary from a pickle file.

        Returns:
            dict: Dictionary mapping generic cores to scaffold lists.
        """

        with open('core_dict_generic.pkl', 'rb') as f:
            core_dict = pickle.load(f)

        return core_dict

    def get_order(self,  core: Chem.Mol, original_rgoups: List[Chem.Mol], random_order: bool = False) -> tuple[List[Chem.Mol], List[Chem.Mol]]:
        """
        Generate all combinations of R-groups with one replacement from external R-groups.

        Args:
            core (Chem.Mol): The core structure of instance.
            original_rgoups (List[Chem.Mol]): Original R- of instance.
            random_order (bool): Whether to shuffle the order of generated samples.

        Returns:
            Tuple[List[List[Chem.Mol]], List[Chem.Mol]]: Combinations of R-groups with core and used external R-groups.
        """

        order = []
        external_rgroups_used = []
        for i in range(Chem.MolToSmiles(core).count("*")):
            for ext in self.external_rgroups:
                # Create a copy of the original R-groups
                modified_rgroups = copy.deepcopy(original_rgoups)
                # Replace the ith R-group with an external one
                modified_rgroups[i] = ext
                external_rgroups_used.append(ext)
                # Add the combination to the list
                order.append(modified_rgroups)

        if random_order:
            order = list(order)
            random.shuffle(order)

        return order, external_rgroups_used

    def core_ext_rgroup_enumeration(self, core: Chem.Mol, order: List[List[Chem.Mol]]) -> Generator[Chem.Mol, None, None]:

        """
        Generate molecules by combining a core with ordered R-groups.

        Args:
            core(Chem.Mol): original core for r_group enumeration.
            order (List[List[Chem.Mol]]): List of R-group combination for enumeration.

        Yields:
            Chem.Mol: Generated molecule with  R-group replacement.
        """

        for tpl in order:
            tm = Chem.RWMol(core)
            for i, r in enumerate(tpl):
                if r is not None:
                    subbed_str = re.sub(r"\*", f"[*:{i + 1}]", Chem.MolToSmiles(r))
                    r_re = Chem.MolFromSmiles(subbed_str)
                    tm.InsertMol(r_re)

            prod = Chem.molzip(tm)

            if None in tpl:
                du = Chem.MolFromSmiles('*')
                prod = AllChem.ReplaceSubstructs(prod, du, Chem.MolFromSmiles('[H]'), True)[0]
                prod = Chem.RemoveHs(prod)

            prod.UpdatePropertyCache(strict=True)
            Chem.SanitizeMol(prod)

            if prod is not None:
                # and finally yield the product molecule
                yield prod

            else:
                print("failed product")

    def generate_ext_r_foils(self, core: Chem.Mol, order: List[List[Optional[Chem.Mol]]],
                             external_rgroups_used: List[Chem.Mol]) -> Tuple[List[Chem.Mol], List[str]]:
        """
        Generate foils by substituting external R-groups into the core molecule.

        Args:
            core (Chem.Mol): The core molecule.
            order (List[List[Optional[Chem.Mol]]]): A nested list representing
            the order in which R-groups should be substituted.
            external_rgroups_used (List[Chem.Mol]): A list of external R-groups used for substitution.

        Returns:
            Tuple[List[Chem.Mol], List[str]]: Generated molecules and the corresponding external R-groups as SMILES.
        """
        #  get unique products
        products = []
        rgroups = []
        seen = set()

        original_core = core
        generator = self.core_ext_rgroup_enumeration(original_core, order)
        for prod, rgroup in zip(generator, external_rgroups_used):
            if prod is not None:
                smi = Chem.MolToSmiles(prod)
                if smi not in seen:
                    products.append(prod)
                    rgroups.append(rgroup)
                    seen.add(smi)
                else:
                    print("seen product before")

        assert len(products) == len(rgroups)

        return products, [Chem.MolToSmiles(r) for r in rgroups]

    def calculate_contrastive_behaviour(self, fact_class: int, foil_class: int, og_pred: np.ndarray, pred: np.ndarray) -> float:
        """
        Calculate the contrastive behavior measure between two predictions.

        Args:
            fact_class (int): The factual class.
            foil_class (int): The foil class.
            og_pred (np.ndarray): Prediction probabilities for the original molecule.
            pred (np.ndarray): Prediction probabilities for the modified molecule.

        Returns:
            float: Contrastive behavior measure.
        """

        assert fact_class != foil_class
        # Compute normalized probabilities for fact and foil
        p_norm = og_pred[fact_class] / (og_pred[fact_class] + og_pred[foil_class])
        q_norm = pred[fact_class] / (pred[fact_class] + pred[foil_class])

        # Compute the contrastive behavior measure
        return p_norm - q_norm

    def get_contrastive_rgroups(self, mol: Chem.Mol, foil_class: int, random_order: bool = False) -> pd.DataFrame:
        """
        Calculate attribution scores for R-groups.

        Returns:
            pd.DataFrame: DataFrame containing foil R-groups and their contrastive scores.
        """

        original_core, original_rgoups = self.decompose_molecule(mol, original=True)
        order, external_rgroups_used = self.get_order(original_core, original_rgoups, random_order=random_order)

        # generate the foils
        prods, rgroups = self.generate_ext_r_foils(original_core, order, external_rgroups_used)

        # prediction of the original molecule
        og_pred = self.predict_func_proba(model=self.model, mol=mol, singular=True)

        # prediction of foils
        preds_proba = self.predict_func_proba(model=self.model, mol=prods)

        fact_class = self.predict_func(self.model, mol, singular=True)

        # attribute value
        rgroup_attribution = {r: [] for r in rgroups}
        for rgroup, prediction in zip(rgroups, preds_proba):
            contr_beh = self.calculate_contrastive_behaviour(fact_class, foil_class, og_pred, prediction)
            rgroup_attribution.setdefault(rgroup, []).append(contr_beh)

        expanded_data = [(rgroup, idx+1, val) for rgroup, vals in rgroup_attribution.items() for idx, val in enumerate(vals)]
        contrast_df = pd.DataFrame(expanded_data, columns=["R-group", "R_group_site", "contrast"]).set_index("R-group")

        return contrast_df.sort_values(by="contrast", ascending=False)

    def get_scaffolds(self, core: Chem.Mol, similarity_threshold: Optional[float] = None) -> List[Chem.Mol]:
        """
        Retrieve similar scaffolds based on the given core and similarity threshold.
        """
        original_core = core
        n_subs = Chem.MolToSmiles(original_core).count("*")
        generic_core = MurckoScaffold.MakeScaffoldGeneric(original_core)
        # make representation more generic
        generic_core = MurckoScaffold.GetScaffoldForMol(generic_core)
        # make it even more generic
        generic_core = get_reduced_skeleton(generic_core)

        generic_smi = Chem.MolToSmiles(generic_core)

        if generic_smi in self.scaffold_dict.keys():
            scaffolds = self.scaffold_dict[generic_smi]

        else:
            raise ValueError("No cores match generic scaffold")

        mol_scaffolds = [Chem.MolFromSmiles(scaffold) for scaffold in scaffolds]
        annotated_scaffolds = annotate_sub_sites(core, mol_scaffolds, n_subs)

        if len(annotated_scaffolds) == 0:
            raise ValueError("No suitable alternative core was identified")

        if similarity_threshold:
            similar_cores = []
            for scaffold in annotated_scaffolds:
                if np.abs(original_core.GetNumAtoms() - scaffold.GetNumAtoms()) / original_core.GetNumAtoms() <= 1 - similarity_threshold:
                    # mcs_result = rdFMCS.FindMCS([og_core, scaffold], completeRingsOnly=True, ringMatchesRingOnly=True)
                    # common_atoms = mcs_result.numAtoms

                    # if common_atoms >= similarity_threshold * og_core.GetNumAtoms():
                    similar_cores.append(scaffold)

            if len(similar_cores) == 0:
                raise ValueError("No similar cores found")

            else:
                return similar_cores

        else:
            return annotated_scaffolds

    def ext_core_rgroup_enumeration(self, original_rgoups: List[Chem.Mol], cores: List[Chem.Mol]) -> Generator[Chem.Mol, None, None]:
        """
        Generate molecules by substituting the original core with
        external cores. Keeping original R-groups.

        Args:
            cores (List[Chem.Mol]): List of external cores.

        Yields:
            Chem.Mol: Generated molecule with external core and
            original R-groups.
        """

        du = Chem.MolFromSmiles('*')

        # loop over each combination
        for core in cores:
            tm = Chem.RWMol(core)
            for i, r in enumerate(original_rgoups):
                if r is not None:
                    subbed_str = re.sub(r"\*", f"[*:{i + 1}]", Chem.MolToSmiles(r))
                    r_re = Chem.MolFromSmiles(subbed_str)
                    tm.InsertMol(r_re)

            prod = Chem.molzip(tm)
            if prod:
                prod = AllChem.ReplaceSubstructs(prod, du, Chem.MolFromSmiles('[H]'), True)[0]
                prod = Chem.RemoveHs(prod)
                prod.UpdatePropertyCache(strict=True)
                Chem.SanitizeMol(prod)
                #  yield the product molecule
                yield prod

    def calculate_sims(self, instance: Chem.Mol, comparisons: List[Chem.Mol]) -> List[float]:
        """
        Calculate Tanimoto similarity between an instance and a list of comparison molecules.
        """
        mfpgen_sim = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_comparisons = mfpgen_sim.GetFingerprints(comparisons)
        fp_instance = mfpgen_sim.GetFingerprint(instance)

        return DataStructs.BulkTanimotoSimilarity(fp_instance, fp_comparisons)

    def generate_ext_core_foils(self, core: Chem.Mol, original_rgoups: List[Chem.Mol], similarity_threshold: Optional[float] = None) -> Tuple[List[Chem.Mol], List[str], List[float]]:

        """
        Generate foils by substituting external cores to original R-groups.

        Args:
            core (Chem.Mol): The core of original instance.
            original_rgoups (List[Chem.Mol]): List of original R-groups.
            similarity_threshold (Optional[float]):Size similarity threshold for external scaffolds.

        Returns:
           Tuple[List[Chem.Mol], List[str], List[float]]: Generated molecules, corresponding external cores, and similarity scores (if return_sims is True).

        """
        #  get unique products
        products = []
        cores = []
        seen = set()

        original_core = core
        cores_to_use = self.get_scaffolds(original_core, similarity_threshold)

        ext_core_foil_generator = self.ext_core_rgroup_enumeration(original_rgoups, cores=cores_to_use)

        for prod, core in zip(ext_core_foil_generator, cores_to_use):
            if prod is not None:
                smi = Chem.MolToSmiles(prod)
                if smi not in seen:
                    products.append(prod)
                    cores.append(core)
                    seen.add(smi)

        assert len(products) == len(cores)


        return products, [Chem.MolToSmiles(core) for core in cores]

    def get_contrastive_cores(self, mol, foil_class: int, similarity_threshold: Optional[float] = None, return_sims: bool = False) -> pd.DataFrame:
        """
        Calculate the contrastive attribution for external cores.

        Args:
            mol: The input molecule for decomposition and prediction.
            foil_class (int): The class label for the foil.
            similarity_threshold (Optional[float]):Size similarity threshold for external scaffolds.
            return_sims (bool, optional): Whether to return core similarity scores.

        Returns:
            pd.DataFrame: DataFrame containing foil cores and their contrastive scores.
        """

        original_core, original_rgoups = self.decompose_molecule(mol, original=True)

        # generate the foils
        prods, cores = self.generate_ext_core_foils(original_core, original_rgoups, similarity_threshold=similarity_threshold)

        # prediction of the original molecule
        og_pred = self.predict_func_proba(model=self.model, mol=mol, singular=True)

        # prediction of foils
        preds_proba = self.predict_func_proba(model=self.model, mol=prods)

        fact_class = self.predict_func(self.model, mol, singular=True)

        # attribute value
        core_attribution = {c: 0 for c in cores}
        for core, prediction in zip(cores, preds_proba):
            contr_beh = self.calculate_contrastive_behaviour(fact_class, foil_class, og_pred, prediction)
            core_attribution[core] += contr_beh

        contrast_df = pd.DataFrame.from_dict(core_attribution, orient="index", columns=["contrast"]).sort_values(by="contrast", ascending=False)

        if return_sims:
            sims = self.calculate_sims(instance=mol, comparisons=prods)
            core_sim_mapping = {core: sim for core, sim in zip(cores, sims)}
            contrast_df["similarity"] = contrast_df.index.map(core_sim_mapping)

        return contrast_df.sort_values(by="contrast", ascending=False)

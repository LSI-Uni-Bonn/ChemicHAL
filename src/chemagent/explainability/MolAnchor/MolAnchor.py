import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import rdkit
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from .utils_anchor import *


class MolecularAnchor:
    """
    A class to identify molecular fragments ("anchors") that are critical for a machine learning model's prediction.

    """

    def __init__(self, mol, model_obj, target_class=1, fragment_scheme="BRICS",
                 representation="ECFP", bit_inf=None, original_fp=None, graph_func=None, graph_predict=None, acc_for_radius=False):
        """
        Initialize the MolAnchor object.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object.
            model_obj: The prediction model (must implement `.predict()` method).
            target_class (int, optional): Class label to identify anchors for (default is 1).
            fragment_scheme (str, optional): Fragmentation scheme ("BRICS" or other; currently only "BRICS" supported).
            representation (str, optional): Molecular representation ("ECFP" or "graphs").
            bit_inf (dict, optional): Bit information for ECFP (maps bits to atom environments).
            original_fp (np.ndarray, optional): Original fingerprint array (for ECFP).
            graph_func (callable, optional): Function to convert molecules to graphs (for graph representation).
            graph_predict (callable, optional): Function to load graphs and predicted (for graph representation).
            acc_for_radius (bool, optional): Whether to correct for atom environments spanning outside fragments.
        """
        self.mol = mol
        self.fragment_scheme = fragment_scheme
        self.model = model_obj
        self.target_class = target_class
        self.representation = representation
        self.acc_for_radius = acc_for_radius
        self.mol_frags, self.mol_atom_ids = self.get_fragments()  # Get fragments and their atom IDs


        if representation == "ECFP":
            self.bit_inf = bit_inf
            self.original_fp = original_fp
            self.bit_dict = self.get_bit_dict()  # Generate bit dictionary for ECFP

        if representation == "graphs":
            self.frag_dict = self.get_frag_dict()  # Generate fragment dictionary for graphs
            self.graph_func = graph_func if graph_func is not None else default_mol_to_nx
            self.graph_predict = graph_predict if graph_predict is not None else default_graph_predict

    def get_fragments(self):
        """
        Fragment the molecule based on the chosen scheme.

        Returns:
            mol_frags (list of rdkit.Chem.Mol): List of fragment molecules.
            mol_atom_ids (list of tuples): List of tuples with atom indices for each fragment.
        """
        # Fragment according to BRICS bonds
        if self.fragment_scheme == "BRICS":
            fragmented_mols = Chem.FragmentOnBRICSBonds(self.mol)
        # Retrieve all fragments as mol and atoms indexes
        mol_atom_ids = []
        mol_frags = AllChem.GetMolFrags(fragmented_mols,
                                        asMols=True,
                                        sanitizeFrags=False,
                                        fragsMolAtomMapping=mol_atom_ids)

        # Check that only atoms that are present in Mol object are returned
        num_atoms = self.mol.GetNumAtoms() - 1
        mol_atom_ids = [tuple(item for item in tup if item <= num_atoms) for tup in mol_atom_ids]

        return mol_frags, mol_atom_ids

    def get_bit_dict(self):
        """
        Map molecular fragments to fingerprint bits (for ECFP representation).

        Returns:
            bit_dict (dict): Dictionary mapping fragments to sets of bits.
        """
        # Initialize an empty dictionary to store the matching bits for each fragment
        bit_dict = {}

        # Loop through each fragment in mol_atom_ids with its index
        for i, frag in enumerate(self.mol_atom_ids):
            # Initialize an empty list to store matching bits for the current fragment
            matching_bits = []

            # Loop through each atom_id in the current fragment
            for atom_id in frag:
                # Loop through each bit and its corresponding paths in bitinf dictionary
                for bit, paths in self.bit_inf.items():

                    # Check if there are multiple paths for the current bit
                    if len(paths) > 1:
                        # Extract the first atom in each path
                        atoms_path = [t[0] for t in paths]
                        # Check if the current atom_id is in the atoms_path list
                        if atom_id in atoms_path:
                            matching_bits.append(bit)

                    else:
                        # If there's only one path, extract the single atom_id
                        atoms_path = paths[0][0]

                        # Check if the current atom_id matches the atom in the single path
                        if atom_id == atoms_path:
                            # If true, append the current bit to matching_bits list
                            matching_bits.append(bit)

            # After processing all atom_ids in the current fragment, add the set of matching bits
            # to the bit_dict with a key indicating the fragment index
            bit_dict[f"frag_{i}"] = set(matching_bits)

        return bit_dict

    def get_frag_dict(self):
        """
        Generate a dictionary of fragments.

        Returns:
            frag_dict (dict): Dictionary mapping fragment names to sets of atom indices.
        """
        frag_dict = {}
        for i, frag in enumerate(self.mol_atom_ids):
            frag_dict[f"frag_{i}"] = set(frag)  # Store atom indices for each fragment

        return frag_dict

    def check_bit_env(self, bit_spec_inf, atoms_present):
        """
        Function to identify and remove bits that might contain atoms that are meant to be excluded as a part of its
        larger environment

        Parameters:
            bit_spec_inf (tuple): (atom_idx, radius) describing an atom environment.
            atoms_present (set): Atom indices present in the fragment.

        Returns:
            set: Set of bits to turn off.
        """
        atom_env = Chem.FindAtomEnvironmentOfRadiusN(mol=self.mol, radius=bit_spec_inf[1], rootedAtAtom=bit_spec_inf[0])

        # Initialize an empty set to store the atom indices
        atoms_in_radius = set()

        # Loop over the bond indices to collect the involved atom indices
        for bond_idx in atom_env:
            bond = self.mol.GetBondWithIdx(bond_idx)
            atoms_in_radius.add(bond.GetBeginAtomIdx())
            atoms_in_radius.add(bond.GetEndAtomIdx())

        if any(item not in atoms_present for item in atoms_in_radius):
            return True

        else:
            return False

    def get_bits_to_turn_off_env(self, bit_turn_off, atoms_present):
        """
        Function to identify and remove bits that might contain atoms that are meant to be excluded as a part of its
        larger environment

        Parameters:
            bit_turn_off (set): Bits initially flagged for exclusion.
            atoms_present (set): Atom indices present in the fragment combination.

        Returns:
            set: Additional bits to turn off.
        """
        # find all bits currently on after corrections
        on_bits = set(np.where(self.original_fp == 1)[0]) - bit_turn_off
        # list of all bit that include atoms that are excluded from current combination
        bits_turn_off_env = []
        for bit in on_bits:
            bit_spec_inf = self.bit_inf[bit]

            if len(bit_spec_inf) == 1:
                if self.check_bit_env(bit_spec_inf[0], atoms_present) is True:
                    bits_turn_off_env.append(bit)

            elif len(bit_spec_inf) > 1:
                for sub_bit in bit_spec_inf:
                    if self.check_bit_env(sub_bit, atoms_present) is True:
                        bits_turn_off_env.append(bit)

        return set(bits_turn_off_env)

    def predict_frag_combinations(self):
        """
        Predict the outcome for each fragment combination.

        Returns:
            pd.DataFrame: DataFrame with combinations and predictions.
        """

        if self.representation == "graphs":
            # Generate all fragment combinations
            all_combinations = generate_combinations(self.frag_dict)

            frag_graphs = []

            base_graph = self.graph_func(self.mol)

            # Infer edge feature structure
            example_edge = next(iter(base_graph.edges(data=True)), None)
            if example_edge is not None:
                _, _, attr_dict = example_edge
                zero_edge_attrs = {
                    key: (0. if not isinstance(value, list) else [0. for _ in value])
                    for key, value in attr_dict.items()
                }
            else:
                zero_edge_attrs = {}

            for combination in all_combinations:
                fragments_to_exclude = set(self.frag_dict.keys()) - set(combination)
                atoms_exclude = atoms_to_exclude(self.mol_atom_ids, fragments_to_exclude, self.frag_dict)

                # Work on a copy to avoid modifying the base graph
                graph_copy = base_graph.copy()
                graph_copy.remove_nodes_from(atoms_exclude)

                # Add self-loop edges with zeroed attributes for isolated nodes
                for node in list(graph_copy.nodes):
                    if graph_copy.degree[node] == 0:
                        graph_copy.add_edge(node, node, **zero_edge_attrs)

                frag_graphs.append(graph_copy)

            predictions = self.graph_predict(self.model, frag_graphs)

        elif self.representation == "ECFP":
            all_combinations = generate_combinations(self.bit_dict)
            modified_fps = []
            # Iterate through all combinations and modify the fingerprint
            for combination in all_combinations:
                # Fragments to exclude are those not in the current combination
                fragments_to_exclude = set(self.bit_dict.keys()) - set(combination)
                bits_to_turn_off = get_bits_to_turn_off(fragments_to_exclude, self.bit_dict)

                atoms_present = get_union_of_values(self.mol_atom_ids, set(combination), self.bit_dict)
                bit_keep_on = bit_not_turn_off(bits_to_turn_off, self.bit_inf, atoms_present)

                # Account for present atoms are in bits that would be turned off (due to repeated molecular envs)
                bits_to_turn_off_corr_1 = bits_to_turn_off - bit_keep_on

                if self.acc_for_radius is True:
                    # Account for excluded atoms that are in the env of bits that would be kept on (depends on radius)
                    bits_to_turn_off_corr_2 = bits_to_turn_off_corr_1.union(self.get_bits_to_turn_off_env(bits_to_turn_off_corr_1, atoms_present))
                    modified_fp = modify_fingerprint(self.original_fp, bits_to_turn_off_corr_2)
                    assert np.all(modified_fp[list(bits_to_turn_off_corr_2)] == 0)

                else:
                    modified_fp = modify_fingerprint(self.original_fp, bits_to_turn_off_corr_1)
                    assert np.all(modified_fp[list(bits_to_turn_off_corr_1)] == 0)

                modified_fps.append(modified_fp.tolist())

            predictions = self.model.predict(np.array(modified_fps))

        else:
            raise ValueError('Representation is not supported')

        # Extract unique fragment names from the combinations
        fragment_names = sorted({frag for combo in all_combinations for frag in combo})
        df_anchors = pd.DataFrame(0, index=range(len(all_combinations)), columns=fragment_names)
        # Populate the DataFrame with ones where the fragment is present in the combination
        for i, combo in enumerate(all_combinations):
            df_anchors.loc[i, combo] = 1
        # Add a column for the combination
        df_anchors['Combination'] = all_combinations
        # Reorder columns to have 'Combination' as the first column
        df_anchors = df_anchors[['Combination'] + fragment_names]

        df_anchors["Predictions"] = predictions

        return df_anchors

    def identify_anchor_in_combination(self, fragments, df_anchors, cutoff=0.95):
        """
        Identify anchors in combinations of fragments that meet the precision cutoff.

        Parameters:
            fragments (list): List of fragment names.
            df_anchors (pd.DataFrame): DataFrame containing fragment presence and predictions.
            cutoff (float): Precision cutoff for identifying anchors.

        Returns:
            anchors (list): List of fragment combinations acting as anchors.
            precision_dict (dict): Dictionary of fragment combinations and their precision.
            coverage_dict (dict): Dictionary of fragment combinations and their coverage.
        """
        precision_dict = {}
        coverage_dict = {}
        for r in range(2, len(fragments) + 1):
            for combo in itertools.combinations(fragments, r):
                combo_mask = df_anchors[list(combo)].all(axis=1)
                frag_present = df_anchors.loc[combo_mask]

                precision = round(len(frag_present.loc[frag_present["Predictions"] == self.target_class]) / len(frag_present), 2)
                coverage = round(len(frag_present.loc[frag_present["Predictions"] == self.target_class]) / len(df_anchors), 2)

                if precision >= cutoff:
                    precision_dict[combo] = precision
                    coverage_dict[combo] = coverage
                    anchors = list(combo)

                    return anchors, precision_dict, coverage_dict
        return set(), precision_dict, coverage_dict

    def identify_anchors(self, df_anchors, cutoff=0.95, allow_frag_combinations=False, return_multiple_anchors=False):
        """
        Identify anchoring fragments that meet the precision cutoff.

        Parameters:
            df_anchors (pd.DataFrame): DataFrame containing fragment presence and predictions.
            cutoff (float): Precision cutoff for identifying anchors.
            allow_frag_combinations (bool): Whether to search for combinations if no single anchors are found.
            return_multiple_anchors (bool): Whether to return multiple anchors if multiple fragments anchor the prediction

        Returns:
            anchor_df_cpd (pd.DataFrame): DataFrame with identified anchors and their details.
        """

        # variable to identify whether anchors act in combination
        multiple_used = False
        # all unique fragments
        if self.representation == "graphs":
            fragments = self.frag_dict.keys()
        elif self.representation == "ECFP":
            fragments = self.bit_dict.keys()
        # to document anchors
        anchors = set(fragments)
        coverage_dict = {}
        precision_dict = {}
        # remove fragments that do not anchor the prediction
        for frag in fragments:
            frag_present = df_anchors.loc[df_anchors[f"{frag}"] == 1]
            precision = round(len(frag_present.loc[frag_present["Predictions"] == self.target_class]) / len(frag_present), 2)
            coverage = round(len(frag_present.loc[frag_present["Predictions"] == self.target_class]) / len(df_anchors), 2)

            if precision >= cutoff:
                coverage_dict[frag] = coverage
                precision_dict[frag] = precision

            else:
                anchors.discard(frag)

        # if all fragments can constitute a anchor
        if len(anchors) == len(fragments):
            print("All fragments anchor")
            return pd.DataFrame([{"smile": Chem.MolToSmiles(self.mol),
                                                  "mol": self.mol,
                                                  "anchor_mol": "all_frags",
                                                  "anchor_smile": "all_frags",
                                                  "precision": 0,
                                                  "coverage": 0,
                                                  "plural_rule": multiple_used}])

        # if no single fragments can anchor the prediction an iterative search can explore fragments in combination
        if len(anchors) == 0 and allow_frag_combinations is True:
            anchors, precision_dict, coverage_dict = self.identify_anchor_in_combination(fragments, df_anchors, cutoff=cutoff)
            multiple_used = True

        # if there are multiple fragments that act as anchor, return the one with the highest precision
        if len(anchors) > 1 and multiple_used is False and return_multiple_anchors is False:
            frag_max_prec = max(precision_dict, key=precision_dict.get)

            cov_val = coverage_dict[frag_max_prec]
            coverage_dict.clear()
            coverage_dict[frag_max_prec] = cov_val

            prec_val = precision_dict[frag_max_prec]
            precision_dict.clear()
            precision_dict[frag_max_prec] = prec_val

            anchors = set([frag_max_prec])

        print(f"anchors identified = {anchors},",
              f"num fragments = {len(fragments)},",
              f"plural_rule = {multiple_used},",
              f"Precision = { [v for v in precision_dict.values()]},",
              f"Coverage = {[v for v in coverage_dict.values()]}")

        # indices of anchoring fragments
        always_present_indices = [int(frag.split('_')[1]) for frag in anchors]

        # mol representation of anchoring fragments
        anchor_frags_mol = [self.mol_frags[i] for i in always_present_indices]
        # smile representation of anchoring fragments
        if len(anchor_frags_mol) != 0:
            if len(anchor_frags_mol) == 1:
                anchor_smiles = delete_numbers_next_to_asterisk(Chem.MolToSmiles(anchor_frags_mol[0]))

            elif len(anchor_frags_mol) > 1:
                anchor_smiles = [delete_numbers_next_to_asterisk(Chem.MolToSmiles(mol)) for mol in anchor_frags_mol]

        else:
            anchor_smiles = 0

        if len(anchors) == 0:
            anchor_df_cpd = pd.DataFrame([{"smile": Chem.MolToSmiles(self.mol),
                                           "mol": self.mol,
                                           "anchor_mol": "no_anchor",
                                           "anchor_smile": "no_anchor",
                                           "precision": 0,
                                           "coverage": 0,
                                           "plural_rule": multiple_used}])

        elif len(anchors) == 1:
            anchor_df_cpd = pd.DataFrame({"smile": Chem.MolToSmiles(self.mol),
                                          "mol": self.mol,
                                          "anchor_mol": anchor_frags_mol,
                                          "anchor_smile": anchor_smiles,
                                          "precision": precision_dict[next(iter(anchors))],
                                          "coverage": coverage_dict[next(iter(anchors))],
                                          "plural_rule": multiple_used})

        elif len(anchors) > 1:
            if multiple_used is False:
                anchor_df_cpd = pd.DataFrame()
                for anc_i, anc_mol, anc_smile in zip(anchors, anchor_frags_mol, anchor_smiles):
                    anchor_df_cpd = pd.concat([anchor_df_cpd,
                                               pd.DataFrame([{"smile": Chem.MolToSmiles(self.mol),
                                                              "mol": self.mol,
                                                              "anchor_mol": anc_mol,
                                                              "anchor_smile": anc_smile,
                                                              "precision": precision_dict[anc_i],
                                                              "coverage": coverage_dict[anc_i],
                                                              "plural_rule": multiple_used}])])

            elif multiple_used is True:
                anchor_df_cpd = pd.DataFrame([{"smile": Chem.MolToSmiles(self.mol),
                                               "mol": self.mol,
                                               "anchor_mol": anchor_frags_mol,
                                               "anchor_smile": anchor_smiles,
                                               "precision": precision_dict[tuple(anchors)],
                                               "coverage": coverage_dict[tuple(anchors)],
                                               "plural_rule": multiple_used}])

        return anchor_df_cpd

    def map_anchor_to_cpd(self, frag_ids):
        """
        Map the anchor fragments to the original compound.

        Parameters:
            frag_ids (list): List of fragment IDs to map.

        Returns:
            Image: Molecule image with highlighted fragments.
        """

        if not frag_ids:
            raise ValueError("No anchors were identified.")

        if len(frag_ids) == 1:
            atoms_to_highlight = self.mol_atom_ids[frag_ids[0]]

        elif len(frag_ids) > 1:
            atoms_to_highlight = list(itertools.chain.from_iterable([self.mol_atom_ids[i] for i in frag_ids]))

        fig = Draw.MolsToGridImage([self.mol], highlightAtomLists=[atoms_to_highlight])

        return fig

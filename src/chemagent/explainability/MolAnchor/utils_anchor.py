import torch
from torch.utils.data import Dataset
from itertools import combinations
import random as _random
import re
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import rdkit
from rdkit import Chem
import numpy as np


def get_bits_to_turn_off(fragments_to_exclude, bit_dict):
    """
    Get the union of bit positions for the fragments to exclude.

    Parameters:
        fragments_to_exclude (set): Set of fragments to exclude.
        bit_dict (dict): Dictionary of bits for each fragment.

    Returns:
        set: Set of bits to turn off.
    """

    bits_to_turn_off = set()
    for frag in fragments_to_exclude:
        bits_to_turn_off.update(bit_dict[frag])  # Collect bits to turn off from excluded fragments

    return bits_to_turn_off



def modify_fingerprint(original_fingerprint, bits_to_turn_off):
    """
    Modify the fingerprint by turning off specified bits.

    Parameters:
        original_fingerprint (np.array): The original fingerprint.
        bits_to_turn_off (set): Set of bit indices to turn off.

    Returns:
        np.array: Modified fingerprint.
    """

    bit_fp = np.zeros_like(original_fingerprint)
    bit_fp[list(bits_to_turn_off)] = 1
    modified_fingerprint = original_fingerprint - bit_fp  # Turn off specified bits

    return modified_fingerprint


def bit_not_turn_off(bits_to_turn_off, bitinf, atoms_present):
    """
    Determine bits that should not be turned off because they are present in one of the fragments in combination.

    Parameters:
        bits_to_turn_off (set): Set of bits to turn off.
        bitinf (dict): Bit information.
        atoms_present (set): Set of atoms present in the fragment.

    Returns:
        set: Set of bits to keep on.
    """
    bits_to_keep_on = []
    for bit in bits_to_turn_off:

        bit_off = bitinf[bit]
        if len(bit_off) > 1:
            # Extract the first atom in each path
            atoms_path = [t[0] for t in bit_off]
            if any(atom in atoms_path for atom in atoms_present):
                bits_to_keep_on.append(bit)  # Keep bits on if their atoms are present in one of the bits

    return set(bits_to_keep_on)


def generate_combinations(bit_dict, max_sampled_combinations=4095):
    """
    Generate fragment combinations for anchor analysis.

    Uses full enumeration when the molecule has 12 or fewer fragments.
    For molecules with more than 12 fragments, uses stratified sampling
    to keep the total number of combinations manageable while ensuring
    every fragment appears across multiple combination sizes.

    Parameters:
        bit_dict (dict): Dictionary of bits/atoms for each fragment.
        max_sampled_combinations (int): Maximum number of combinations when
            sampling is used (default 4095, matching the full-enumeration cap
            at n=12 fragments so coverage doesn't drop abruptly at the boundary).

    Returns:
        list: List of fragment combinations (tuples of fragment keys).
    """

    fragment_keys = list(bit_dict.keys())
    n = len(fragment_keys)

    # Full enumeration for ≤12 fragments (up to 4095 combinations)
    if n <= 12:
        all_combinations = []
        for r in range(1, n + 1):
            all_combinations.extend(combinations(fragment_keys, r))
        return all_combinations

    # Stratified sampling for >12 fragments
    sampled = []
    seen = set()

    def _add(combos):
        for c in combos:
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                sampled.append(c)

    # Always include: singles (size 1), full molecule (size n),
    # and leave-one-out (size n-1) — essential for anchor identification
    _add(combinations(fragment_keys, 1))
    _add(combinations(fragment_keys, n))
    _add(combinations(fragment_keys, n - 1))

    remaining_budget = max_sampled_combinations - len(sampled)
    if remaining_budget <= 0:
        return sampled

    # Distribute remaining budget across intermediate sizes (2..n-2)
    middle_sizes = list(range(2, n - 1))
    per_size_budget = max(1, remaining_budget // len(middle_sizes))

    for r in middle_sizes:
        size_combos = list(combinations(fragment_keys, r))
        if len(size_combos) <= per_size_budget:
            _add(size_combos)
        else:
            _add(_random.sample(size_combos, per_size_budget))

    return sampled


def get_union_of_values(fragments_list, fragments_to_include, bit_dict):
    """
    Get the union of bit positions for the fragments to include.

    Parameters:
        fragments_list (list): List of fragments.
        fragments_to_include (set): Set of fragments to include.
        bit_dict (dict): Dictionary of bits for each fragment.

    Returns:
        set: Union of bit positions.
    """
    union_of_values = set()
    for frag_name, frag_values in zip(list(bit_dict.keys()), fragments_list):
        if frag_name in fragments_to_include:
            union_of_values.update(frag_values)  # Collect bit positions from included fragments
    return union_of_values


def atoms_to_exclude(fragments_list, fragments_to_exclude, bit_dict):
    """
    Get the atoms to exclude based on the fragments to exclude.

    Parameters:
        fragments_list (list): List of fragments.
        fragments_to_exclude (set): Set of fragments to exclude.
        bit_dict (dict): Dictionary of bits for each fragment.

    Returns:
        set: Set of atoms to exclude.
    """
    atoms_to_exclude = set()
    for frag_name, frag_values in zip(list(bit_dict.keys()), fragments_list):
        if frag_name in fragments_to_exclude:
            atoms_to_exclude.update(frag_values)  # Collect atoms to exclude from excluded fragments

    return atoms_to_exclude


def delete_numbers_next_to_asterisk(text):
    """
    Remove numbers next to asterisks in the given text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The processed text.
    """
    # Define a regular expression pattern to match any number next to an asterisk within square brackets
    pattern = r'\[(\d+)\*\]'

    # Use re.sub() to replace all occurrences of the pattern with just the square brackets
    result = re.sub(pattern, '[*]', text)

    return result


class GraphDataset(Dataset):
    def __init__(self, x, y=None, masks=None):
        # x should be a list of PyTorch Geometric Data objects
        self.x = x
        if y is None:
            y = [0] * len(x)
        self.y = torch.tensor(y)

        self.masks = masks if masks is not None else [None] * len(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.masks[idx]
        

def sigmoid(number: float):
    """ numerically semi-stable sigmoid function to map charge between 0 and 1 """
    return 1.0 / (1.0 + float(np.exp(-number)))
    

def default_mol_to_nx(mol):
    """
    Default function to convert a molecule to a NetworkX graph.

    Parameters:
        mol (rdkit.Chem.Mol): The input molecule.

    Returns:
        graph (networkx.Graph): The resulting NetworkX graph.
    """

    symbols = [
        "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
    ]

    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

    G = nx.Graph()

    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.

        hybridization_atom = [0.] * len(hybridizations)
        hybridization_atom[hybridizations.index(
            atom.GetHybridization())] = 1

        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=symbol,
                   atomic_weight=sigmoid(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetSymbol())),
                   n_valence=float(atom.GetTotalValence()),
                   n_hydrogens=float(atom.GetTotalNumHs()),
                   hybridization=hybridization_atom
                   )

    for bond in mol.GetBonds():

        bond_type_atom =bond.GetBondType()
        single = 1. if bond_type_atom == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type_atom == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type_atom == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type_atom == Chem.rdchem.BondType.AROMATIC else 0.

        conjugation = [0.] * 2
        if bond.GetIsConjugated():
            conjugation[0] = 1.
        else:
            conjugation[1] = 1.

        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   single=single,
                   double=double,
                   triple=triple,
                   aromatic=aromatic,
                   bond_conjugation=conjugation
                   )
    return G

def default_graph_predict(model, frag_graphs):
    """
    Predicts the output based on a list of graph fragments using the provided model.

    This function converts the given list of NetworkX graph fragments (`frag_graphs`)
    into a format suitable for the model.

    Args:
        model: The model that will be used to make predictions.
        frag_graphs (list of networkx.Graph): A list of NetworkX graphs, where each graph represents
                                              a fragment of a molecule. These graphs should contain
                                              node and edge attributes required by the model.

    Returns:
        numpy.ndarray: A 1D array of integer predictions, rounded from the model's output.
    """
    # Convert graphs into model input
    frag_x = [from_networkx(graph,
                            group_node_attrs=["atomic_num", "is_aromatic", "atomic_weight",
                                              "n_valence", "n_hydrogens", "atom_symbol", "hybridization"],
                            group_edge_attrs=["single", "double", "triple", "aromatic", "bond_conjugation"])
              for graph in frag_graphs]

    anchor_data = GraphDataset(frag_x)

    predictions = model.predict(anchor_data).numpy().round().astype(int).flatten()

    return predictions
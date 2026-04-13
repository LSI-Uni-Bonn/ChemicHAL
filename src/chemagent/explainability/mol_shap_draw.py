"""
chemagent.explainability.mol_shap_draw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities for mapping SHAP values onto molecular structures.

The pipeline is:

1. Compute ECFP bit information for a SMILES string with
   :func:`get_ecfp_morgan_generator_bit_info`.
2. Map per-bit SHAP values to per-atom weights with
   :func:`shap_to_atom_weight`.
3. Render the atom-weight heatmap with
   :func:`get_atom_wise_weight_map`.
"""

from __future__ import annotations

import io
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from rdkit import Chem, Geometry
from rdkit.Chem import Draw  # type: ignore[attr-defined]
from rdkit.Chem import rdDepictor, rdFingerprintGenerator

DEFAULT_COLORMAP = 'coolwarm' # Blue for negative SHAP values, red for positive, white for zero


def get_ecfp_morgan_generator_bit_info(smiles: str, radius: int = 2, n_bits: int = 2048) -> Dict:
    """Compute ECFP bit information for a SMILES string.

    Args:
    smiles:
        SMILES string of the molecule.
    radius:
        Morgan fingerprint radius (default 2 → ECFP4).
    n_bits:
        Size of the bit vector (default 2048).

    Returns:
    dict
        Bit-info map ``{bit_index: [(center_atom, radius), ...]}``,
        as returned by ``rdFingerprintGenerator.AdditionalOutput.GetBitInfoMap()``.
    """
    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)

    # Generate Morgan fingerprint with bit information
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()

    # Create Morgan fingerprint generator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = fpgen.GetFingerprint(mol, additionalOutput=ao)

    return ao.GetBitInfoMap()


def bit_to_atom_mapping(mol: Chem.Mol, dict_bit_info: dict) -> Dict[int, List[List[int]]]:
    """Map fingerprint bits to the atom indices that encode them.

    Args:
    mol:
        RDKit molecule object.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.

    Returns:
    dict
        ``{bit: [[atom_idx, ...], ...]}`` — each inner list is the set of
        atom indices that contribute to one match of that bit.
    """
    envs = {}
    for bit, matches in dict_bit_info.items():
        atoms = []
        for central_atom, radius in matches:
            if radius == 0:
                env_atoms = [central_atom]
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, central_atom, useHs=True)
                amap = {}
                _ = Chem.PathToSubmol(mol, env, atomMap=amap)
                env_atoms = list(amap)
            atoms.append(env_atoms)
        envs[bit] = atoms
    return envs


def assign_prediction_importance(bit_dict: Dict[int, List[List[int]]], weights: np.ndarray) -> Dict[int, float]:
    """Distribute per-bit SHAP weights uniformly over contributing atoms.

    Each bit's SHAP value is split equally across all atoms that encode it
    (across all environment matches and all atoms within each match).

    Args:
    bit_dict:
        Bit-to-atom mapping for one molecule, as returned by
        :func:`bit_to_atom_mapping`.
    weights:
        1-D array of per-bit SHAP values, shape ``(n_features,)``.

    Returns:
    dict
        ``{atom_idx: cumulative_shap_contribution}``.
    """
    atom_contribution = defaultdict(float)
    for bit, atom_env_list in bit_dict.items(): 
        n_matches = len(atom_env_list)
        for atom_set in atom_env_list:
            for atom in atom_set:
                atom_contribution[atom] += weights[bit] / (len(atom_set) * n_matches)
    return atom_contribution


def shap_to_atom_weight(mol: Chem.Mol, dict_bit_info: dict, shapley_values: np.ndarray) -> List[float]:
    """Compute a per-atom SHAP weight for every atom in *mol*.

    Each bit's SHAP value is distributed uniformly over all atoms that
    contribute to that bit.  The final atom weight is the sum of all such
    contributions.

    Args:
    mol:
        RDKit molecule object.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.
    shapley_values:
        1-D array of per-bit SHAP values, shape ``(n_features,)``.

    Returns:
    list of float
        Per-atom SHAP weights, length ``mol.GetNumAtoms()``.
    """
    bit_atom_env_dict = bit_to_atom_mapping(mol, dict_bit_info)
    atom_weight_dict = assign_prediction_importance(bit_atom_env_dict, shapley_values)
    atom_weight_list = [atom_weight_dict[a_idx] for a_idx in range(mol.GetNumAtoms())]
    return atom_weight_list
    

def get_atom_wise_weight_map(
    mol: Chem.Mol,
    weights: List[float],
    mol_size: tuple = (500, 500),
    cmap: Union[str, Any] = DEFAULT_COLORMAP,
    return_png: bool = True,
) -> Union[Image.Image, Draw.MolDraw2D]:
    """Render a Gaussian-smoothed atom-weight heatmap on a molecule image.

    Args:
    mol:
        RDKit molecule object to draw.
    weights:
        Per-atom weight list, as returned by :func:`shap_to_atom_weight`.
    mol_size:
        ``(width, height)`` in pixels for the output image.
    cmap:
        Matplotlib colormap name or object (default ``'coolwarm'``).
    return_png:
        If ``True`` (default), return a :class:`PIL.Image.Image`.
        If ``False``, return the raw :class:`rdkit.Chem.Draw.MolDraw2D` object.

    Returns:
    PIL.Image.Image or Draw.MolDraw2D
        Molecule image with atom-importance heatmap overlay.
    """
    draw2d = Draw.MolDraw2DCairo(*mol_size)
    draw2d.drawOptions().fixedScale = 0
    draw2d.drawOptions().useBWAtomPalette()

    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    
    if mol.GetNumBonds() > 0:
        bond = mol.GetBondWithIdx(0)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) -
                       mol.GetConformer().GetAtomPosition(idx2)).Length()
    else:
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0) -
                       mol.GetConformer().GetAtomPosition(1)).Length()
    sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x, p.y))

    draw2d.ClearDrawing()
    ps = Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.1#2.0

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    clrs = [tuple(cmap.get_under()), (1, 1, 1), tuple(cmap.get_over())]
    ps.setColourMap(clrs)

    contourLines = 10
    Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)

    if return_png:
        return convert_draw2d_to_png(draw2d)

    return draw2d


def convert_draw2d_to_png(draw2d: Draw.MolDraw2D) -> Image.Image:
    """Convert a ``MolDraw2D`` drawing to a :class:`PIL.Image.Image`.

    Args:
    draw2d:
        A finalised :class:`rdkit.Chem.Draw.MolDraw2D` object
        (``FinishDrawing()`` must have been called beforehand).

    Returns:
    PIL.Image.Image
        PNG image decoded from the drawing's byte buffer.
    """
    return Image.open(io.BytesIO(draw2d.GetDrawingText()))

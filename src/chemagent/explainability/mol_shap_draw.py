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


def get_most_important_bits_by_contribution(
    shapley_values: np.ndarray,
    top_k: int = 10,
) -> Dict[str, List[Dict[str, float]]]:
    """Return top positive and negative ECFP bits by SHAP contribution.

    Args:
    shapley_values:
        1-D array of per-bit SHAP values for a single sample.
    top_k:
        Number of bits to return per sign (positive and negative).

    Returns:
    dict
        ``{"positive": [{"bit": i, "contribution": v}, ...],
           "negative": [{"bit": j, "contribution": w}, ...]}``.
        Positive entries are sorted descending by contribution; negative entries
        are sorted ascending (most negative first).
    """
    values = np.asarray(shapley_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"Expected 1D SHAP values for a single sample, got shape {values.shape}."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")

    pos_idx = np.flatnonzero(values > 0)
    neg_idx = np.flatnonzero(values < 0)

    pos_sorted = pos_idx[np.argsort(values[pos_idx])[::-1]][:top_k]
    neg_sorted = neg_idx[np.argsort(values[neg_idx])][:top_k]

    return {
        "positive": [
            {"bit": int(i), "contribution": float(values[i])} for i in pos_sorted
        ],
        "negative": [
            {"bit": int(i), "contribution": float(values[i])} for i in neg_sorted
        ],
    }


def get_top_k_bit_environments_with_contribution(
    mol: Chem.Mol,
    dict_bit_info: dict,
    shapley_values: np.ndarray,
    top_k: int = 10,
    ranking: str = "absolute",
) -> List[Dict[str, Any]]:
    """Return top-k ECFP bit environments and their SHAP contributions.

    Args:
    mol:
        RDKit molecule object for which bit environments were computed.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.
    shapley_values:
        1-D array of per-bit SHAP values for one sample.
    top_k:
        Number of top bits to return.
    ranking:
        Ranking strategy: ``"absolute"`` (default), ``"positive"``, or
        ``"negative"``.

    Returns:
    list of dict
        One entry per selected bit with keys:
        ``bit``, ``contribution``, and ``environments``.
        Each environment contains:
        ``center_atom``, ``radius``, ``atom_indices``, ``bond_indices``, and
        ``environment_smiles``.
        Environments are deduplicated using both
        ``(atom_indices, bond_indices)`` and ``(radius, environment_smiles)``
        so repeated symmetric fragments are not duplicated in outputs.
    """
    values = np.asarray(shapley_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"Expected 1D SHAP values for a single sample, got shape {values.shape}."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")
    if ranking not in {"absolute", "positive", "negative"}:
        raise ValueError(
            f"ranking must be one of ['absolute', 'positive', 'negative'], got {ranking!r}."
        )

    available_bits = np.array(sorted(dict_bit_info.keys()), dtype=int)
    if available_bits.size == 0:
        return []
    if int(np.max(available_bits)) >= values.shape[0]:
        raise ValueError(
            "SHAP vector length is smaller than the largest bit index in dict_bit_info."
        )

    bit_values = values[available_bits]
    if ranking == "positive":
        mask = bit_values > 0
        ranked_bits = available_bits[mask]
        ranked_vals = bit_values[mask]
        order = np.argsort(ranked_vals)[::-1]
    elif ranking == "negative":
        mask = bit_values < 0
        ranked_bits = available_bits[mask]
        ranked_vals = bit_values[mask]
        order = np.argsort(ranked_vals)
    else:
        ranked_bits = available_bits
        ranked_vals = bit_values
        order = np.argsort(np.abs(ranked_vals))[::-1]

    ranked_bits = ranked_bits[order][:top_k]
    results: List[Dict[str, Any]] = []

    for bit in ranked_bits:
        matches = list(dict_bit_info[int(bit)])
        if not matches:
            continue
        env_entries = []
        seen_env_keys = set()
        seen_env_info_keys = set()
        for center_atom, radius in matches:
            if radius == 0:
                atom_indices = [int(center_atom)]
                bond_indices: List[int] = []
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(
                    mol,
                    int(radius),
                    int(center_atom),
                    useHs=True,
                )
                amap = {}
                _ = Chem.PathToSubmol(mol, env, atomMap=amap)
                atom_indices = sorted(int(a) for a in amap)
                bond_indices = sorted(int(b) for b in env)

            env_key = (tuple(atom_indices), tuple(bond_indices))
            if env_key in seen_env_keys:
                continue
            seen_env_keys.add(env_key)

            env_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=atom_indices,
                bondsToUse=bond_indices,
                canonical=True,
            )

            env_info_key = (int(radius), env_smiles)
            if env_info_key in seen_env_info_keys:
                continue
            seen_env_info_keys.add(env_info_key)

            env_entries.append(
                {
                    "center_atom": int(center_atom),
                    "radius": int(radius),
                    "atom_indices": atom_indices,
                    "bond_indices": bond_indices,
                    "environment_smiles": env_smiles,
                }
            )

        results.append(
            {
                "bit": int(bit),
                "contribution": float(values[int(bit)]),
                "environments": env_entries,
            }
        )

    return results


def _grid_image_to_pil_or_original(grid_image: Any) -> Any:
    """Convert RDKit grid image outputs to PIL when possible."""
    if isinstance(grid_image, bytes):
        return Image.open(io.BytesIO(grid_image)).copy()
    if isinstance(grid_image, Image.Image):
        return grid_image

    data_attr = getattr(grid_image, "data", None)
    if isinstance(data_attr, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(data_attr))).copy()

    return grid_image


def render_top_k_bit_environments_image(
    mol: Chem.Mol,
    dict_bit_info: dict,
    shapley_values: np.ndarray,
    top_k: int = 10,
    ranking: str = "absolute",
    max_environments: Optional[int] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple = (260, 210),
) -> Any:
    """Render top-k bit environments as a labeled molecule grid image.

    Args:
    mol:
        RDKit molecule object.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.
    shapley_values:
        1-D array of per-bit SHAP values for one sample.
    top_k:
        Number of top bits to consider.
    ranking:
        Ranking strategy: ``"absolute"`` (default), ``"positive"``, or
        ``"negative"``.
    max_environments:
        Optional limit on how many environments are drawn in total.
    mols_per_row:
        Number of molecules per row in the grid image.
    sub_img_size:
        Size of each sub-image in pixels.

    Returns:
    Any
        Grid image containing environment molecules with SHAP labels.
        Typically a :class:`PIL.Image.Image`; in some notebook backends RDKit
        may return an IPython display image object.
    """
    if max_environments is not None and max_environments <= 0:
        raise ValueError(f"max_environments must be > 0 when provided, got {max_environments}.")
    if mols_per_row <= 0:
        raise ValueError(f"mols_per_row must be > 0, got {mols_per_row}.")

    top_env = get_top_k_bit_environments_with_contribution(
        mol=mol,
        dict_bit_info=dict_bit_info,
        shapley_values=shapley_values,
        top_k=top_k,
        ranking=ranking,
    )

    env_mols: List[Chem.Mol] = []
    legends: List[str] = []

    for item in top_env:
        for env in item["environments"]:
            bond_indices = [int(b) for b in env.get("bond_indices", [])]
            atom_indices = [int(a) for a in env.get("atom_indices", [])]

            if bond_indices:
                env_mol = Chem.PathToSubmol(mol, bond_indices)
            elif atom_indices:
                atom_fragment_smiles = Chem.MolFragmentToSmiles(
                    mol,
                    atomsToUse=atom_indices,
                    canonical=True,
                    kekuleSmiles=True,
                )
                env_mol = Chem.MolFromSmiles(atom_fragment_smiles)
            else:
                env_mol = None

            if env_mol is None:
                continue

            env_mols.append(env_mol)
            legends.append(
                "\n".join(
                    [
                        f"bit {item['bit']} | SHAP {item['contribution']:+.4f}",
                        f"r={env['radius']}, center={env['center_atom']}",
                    ]
                )
            )

            if max_environments is not None and len(env_mols) >= max_environments:
                break
        if max_environments is not None and len(env_mols) >= max_environments:
            break

    if not env_mols:
        raise ValueError("No valid bit environments available to render.")

    grid_image = Draw.MolsToGridImage(
        env_mols,
        legends=legends,
        molsPerRow=int(mols_per_row),
        subImgSize=tuple(sub_img_size),
        useSVG=False,
        returnPNG=True,
    )

    return _grid_image_to_pil_or_original(grid_image)


def render_top_positive_negative_bit_environments_images(
    mol: Chem.Mol,
    dict_bit_info: dict,
    shapley_values: np.ndarray,
    top_k: int = 10,
    max_environments_per_sign: Optional[int] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple = (260, 210),
) -> Dict[str, Optional[Any]]:
    """Render fragment environment grids split into positive and negative SHAP.

    Args:
    mol:
        RDKit molecule object.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.
    shapley_values:
        1-D array of per-bit SHAP values for one sample.
    top_k:
        Number of top bits per sign to consider.
    max_environments_per_sign:
        Optional cap for number of rendered environments per sign.
    mols_per_row:
        Number of molecules per row in each sign-specific grid.
    sub_img_size:
        Size of each sub-image in pixels.

    Returns:
    dict
        ``{"positive": image_or_none, "negative": image_or_none}``.
    """
    values = np.asarray(shapley_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"Expected 1D SHAP values for a single sample, got shape {values.shape}."
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")

    images: Dict[str, Optional[Any]] = {"positive": None, "negative": None}

    if np.any(values > 0):
        images["positive"] = render_top_k_bit_environments_image(
            mol=mol,
            dict_bit_info=dict_bit_info,
            shapley_values=values,
            top_k=top_k,
            ranking="positive",
            max_environments=max_environments_per_sign,
            mols_per_row=mols_per_row,
            sub_img_size=sub_img_size,
        )

    if np.any(values < 0):
        images["negative"] = render_top_k_bit_environments_image(
            mol=mol,
            dict_bit_info=dict_bit_info,
            shapley_values=values,
            top_k=top_k,
            ranking="negative",
            max_environments=max_environments_per_sign,
            mols_per_row=mols_per_row,
            sub_img_size=sub_img_size,
        )

    return images


def render_top_k_parent_molecule_environment_highlights(
    mol: Chem.Mol,
    dict_bit_info: dict,
    shapley_values: np.ndarray,
    top_k: int = 10,
    ranking: str = "absolute",
    max_environments: Optional[int] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple = (320, 240),
) -> Any:
    """Render top-k environments as highlights on the full parent molecule.

    Each tile in the returned grid is the full parent molecule with one
    environment highlighted.

    Args:
    mol:
        RDKit molecule object.
    dict_bit_info:
        Bit-info map for *mol*, as returned by
        :func:`get_ecfp_morgan_generator_bit_info`.
    shapley_values:
        1-D array of per-bit SHAP values for one sample.
    top_k:
        Number of top bits to consider.
    ranking:
        Ranking strategy: ``"absolute"`` (default), ``"positive"``, or
        ``"negative"``.
    max_environments:
        Optional cap for number of highlighted environments in total.
    mols_per_row:
        Number of molecules per row in the grid image.
    sub_img_size:
        Size of each sub-image in pixels.

    Returns:
    Any
        Grid image, typically :class:`PIL.Image.Image`.
    """
    if max_environments is not None and max_environments <= 0:
        raise ValueError(f"max_environments must be > 0 when provided, got {max_environments}.")
    if mols_per_row <= 0:
        raise ValueError(f"mols_per_row must be > 0, got {mols_per_row}.")

    top_env = get_top_k_bit_environments_with_contribution(
        mol=mol,
        dict_bit_info=dict_bit_info,
        shapley_values=shapley_values,
        top_k=top_k,
        ranking=ranking,
    )

    parent_mols: List[Chem.Mol] = []
    legends: List[str] = []
    highlight_atom_lists: List[List[int]] = []
    highlight_bond_lists: List[List[int]] = []

    for item in top_env:
        for env in item["environments"]:
            atom_indices = [int(a) for a in env.get("atom_indices", [])]
            bond_indices = [int(b) for b in env.get("bond_indices", [])]
            if not atom_indices and not bond_indices:
                continue

            parent_mols.append(Chem.Mol(mol))
            highlight_atom_lists.append(atom_indices)
            highlight_bond_lists.append(bond_indices)
            legends.append(
                "\n".join(
                    [
                        f"bit {item['bit']} | SHAP {item['contribution']:+.4f}",
                        #f"r={env['radius']}, center={env['center_atom']}",
                    ]
                )
            )

            if max_environments is not None and len(parent_mols) >= max_environments:
                break
        if max_environments is not None and len(parent_mols) >= max_environments:
            break

    if not parent_mols:
        raise ValueError("No valid bit environments available to render on parent molecule.")

    grid_image = Draw.MolsToGridImage(
        parent_mols,
        legends=legends,
        molsPerRow=int(mols_per_row),
        subImgSize=tuple(sub_img_size),
        highlightAtomLists=highlight_atom_lists,
        highlightBondLists=highlight_bond_lists,
        useSVG=False,
        returnPNG=True,
    )

    return _grid_image_to_pil_or_original(grid_image)


def plot_bit_contribution_summary(
    bit_summary: Dict[str, List[Dict[str, float]]],
    title: str = "Top SHAP Bit Contributions",
    figsize: tuple = (15, 6),
    positive_color: str = "#d62728",
    negative_color: str = "#1f77b4",
) -> Any:
    """Plot top positive and negative SHAP bit contributions as a bar chart.

    Args:
    bit_summary:
        Output dictionary from :func:`get_most_important_bits_by_contribution`.
    title:
        Chart title.
    figsize:
        Matplotlib figure size.
    positive_color:
        Bar color for positive contributions.
    negative_color:
        Bar color for negative contributions.

    Returns:
    matplotlib.axes.Axes
        Axes containing the horizontal contribution bar plot.
    """
    positive = bit_summary.get("positive", [])
    negative = bit_summary.get("negative", [])

    records = []
    for item in negative + positive:
        bit = int(item["bit"])
        contribution = float(item["contribution"])
        records.append((f"{bit}", contribution))

    if not records:
        raise ValueError("No bit contributions found to plot.")

    labels = [r[0] for r in records]
    values = np.asarray([r[1] for r in records], dtype=float)
    order = np.argsort(values)

    labels_sorted = [labels[i] for i in order]
    values_sorted = values[order]
    colors = [negative_color if v < 0 else positive_color for v in values_sorted]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels_sorted, values_sorted, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("Fingerprint bit")
    ax.set_title(title)

    max_abs = float(np.max(np.abs(values_sorted)))
    text_offset = 0.02 * max_abs if max_abs > 0 else 0.01
    for i, value in enumerate(values_sorted):
        x = value + text_offset if value >= 0 else value - text_offset
        ha = "left" if value >= 0 else "right"
        ax.text(x, i, f"{value:.4f}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    return ax


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
    mol_size: tuple = (1200, 1200),
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
    dopts = draw2d.drawOptions()
    dopts.fixedScale = 0
    dopts.useBWAtomPalette()
    dopts.bondLineWidth = 2

    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    if mol.GetNumBonds() > 0:
        bond = mol.GetBondWithIdx(0)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        sigma = 0.22 * (mol.GetConformer().GetAtomPosition(idx1) -
                        mol.GetConformer().GetAtomPosition(idx2)).Length()
    else:
        sigma = 0.22 * (mol.GetConformer().GetAtomPosition(0) -
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
    ps.gridResolution = 0.04
    ps.extraGridPadding = 0.4

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    clrs = [tuple(cmap.get_under()), (1, 1, 1), tuple(cmap.get_over())]
    ps.setColourMap(clrs)

    contourLines = 20
    Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
    dopts.clearBackground = False
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

"""Unit tests for molecule SHAP drawing helpers."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image
from rdkit import Chem

from chemagent.explainability.mol_shap_draw import (
    get_ecfp_morgan_generator_bit_info,
    get_most_important_bits_by_contribution,
    get_top_k_bit_environments_with_contribution,
    render_top_k_bit_environments_image,
    render_top_positive_negative_bit_environments_images,
    render_top_k_parent_molecule_environment_highlights,
    plot_bit_contribution_summary,
)


def test_get_most_important_bits_by_contribution_returns_top_positive_and_negative():
    shap_values = np.array([0.2, -0.9, 0.8, -0.1, 0.5, -1.2], dtype=float)

    out = get_most_important_bits_by_contribution(shap_values, top_k=2)

    assert [x["bit"] for x in out["positive"]] == [2, 4]
    assert [x["contribution"] for x in out["positive"]] == pytest.approx([0.8, 0.5])
    assert [x["bit"] for x in out["negative"]] == [5, 1]
    assert [x["contribution"] for x in out["negative"]] == pytest.approx([-1.2, -0.9])


def test_get_most_important_bits_by_contribution_handles_top_k_larger_than_available():
    shap_values = np.array([0.0, 0.3, -0.4], dtype=float)

    out = get_most_important_bits_by_contribution(shap_values, top_k=10)

    assert len(out["positive"]) == 1
    assert out["positive"][0]["bit"] == 1
    assert out["positive"][0]["contribution"] == pytest.approx(0.3)

    assert len(out["negative"]) == 1
    assert out["negative"][0]["bit"] == 2
    assert out["negative"][0]["contribution"] == pytest.approx(-0.4)


def test_get_most_important_bits_by_contribution_validates_inputs():
    with pytest.raises(ValueError, match="Expected 1D SHAP values"):
        get_most_important_bits_by_contribution(np.zeros((2, 3), dtype=float), top_k=3)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        get_most_important_bits_by_contribution(np.zeros(8, dtype=float), top_k=0)


def test_plot_bit_contribution_summary_builds_bar_plot():
    summary = {
        "positive": [
            {"bit": 10, "contribution": 0.6},
            {"bit": 21, "contribution": 0.2},
        ],
        "negative": [
            {"bit": 7, "contribution": -0.4},
            {"bit": 3, "contribution": -0.1},
        ],
    }

    ax = plot_bit_contribution_summary(summary, title="Test plot")

    assert ax.get_title() == "Test plot"
    assert len(ax.patches) == 4
    plt.close(ax.figure)


def test_plot_bit_contribution_summary_validates_non_empty_input():
    with pytest.raises(ValueError, match="No bit contributions found to plot"):
        plot_bit_contribution_summary({"positive": [], "negative": []})


def test_get_top_k_bit_environments_with_contribution_returns_structured_environments():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    available_bits = sorted(bit_info.keys())
    shap_values[available_bits[0]] = 0.9
    if len(available_bits) > 1:
        shap_values[available_bits[1]] = -0.8

    out = get_top_k_bit_environments_with_contribution(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        ranking="absolute",
    )

    assert len(out) == 1
    assert out[0]["bit"] == available_bits[0]
    assert out[0]["contribution"] == pytest.approx(0.9)
    assert len(out[0]["environments"]) >= 1
    first_env = out[0]["environments"][0]
    assert "center_atom" in first_env
    assert "radius" in first_env
    assert "atom_indices" in first_env
    assert "bond_indices" in first_env
    assert "environment_smiles" in first_env


def test_get_top_k_bit_environments_with_contribution_positive_and_negative_ranking():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    bits = sorted(bit_info.keys())
    shap_values[bits[0]] = 0.2
    shap_values[bits[1]] = 0.7
    shap_values[bits[2]] = -0.6

    out_pos = get_top_k_bit_environments_with_contribution(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        ranking="positive",
    )
    out_neg = get_top_k_bit_environments_with_contribution(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        ranking="negative",
    )

    assert out_pos[0]["bit"] == bits[1]
    assert out_pos[0]["contribution"] == pytest.approx(0.7)
    assert out_neg[0]["bit"] == bits[2]
    assert out_neg[0]["contribution"] == pytest.approx(-0.6)


def test_get_top_k_bit_environments_with_contribution_deduplicates_by_atom_and_bond_indices():
    mol = Chem.MolFromSmiles("CCO")
    bit_info = {1: [(0, 0), (0, 0), (2, 0)]}
    shap_values = np.zeros(16, dtype=float)
    shap_values[1] = 0.9

    out = get_top_k_bit_environments_with_contribution(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        ranking="absolute",
    )

    assert len(out) == 1
    assert out[0]["bit"] == 1
    assert len(out[0]["environments"]) == 2
    assert out[0]["environments"][0]["atom_indices"] == [0]
    assert out[0]["environments"][1]["atom_indices"] == [2]
    assert out[0]["environments"][0]["bond_indices"] == []
    assert out[0]["environments"][1]["bond_indices"] == []


def test_get_top_k_bit_environments_with_contribution_deduplicates_same_information():
    mol = Chem.MolFromSmiles("CCO")
    bit_info = {1: [(0, 0), (1, 0)]}
    shap_values = np.zeros(16, dtype=float)
    shap_values[1] = 0.9

    out = get_top_k_bit_environments_with_contribution(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        ranking="absolute",
    )

    assert len(out) == 1
    assert out[0]["bit"] == 1
    assert len(out[0]["environments"]) == 1
    assert out[0]["environments"][0]["radius"] == 0
    assert out[0]["environments"][0]["environment_smiles"] == "C"


def test_get_top_k_bit_environments_with_contribution_validates_inputs():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    with pytest.raises(ValueError, match="Expected 1D SHAP values"):
        get_top_k_bit_environments_with_contribution(
            mol,
            bit_info,
            np.zeros((2, 3), dtype=float),
            top_k=3,
        )

    with pytest.raises(ValueError, match="top_k must be > 0"):
        get_top_k_bit_environments_with_contribution(
            mol,
            bit_info,
            np.zeros(128, dtype=float),
            top_k=0,
        )

    with pytest.raises(ValueError, match="ranking must be one of"):
        get_top_k_bit_environments_with_contribution(
            mol,
            bit_info,
            np.zeros(128, dtype=float),
            ranking="foo",
        )


def test_render_top_k_bit_environments_image_returns_pil_image():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    first_bit = sorted(bit_info.keys())[0]
    shap_values[first_bit] = 0.7

    img = render_top_k_bit_environments_image(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        max_environments=1,
        mols_per_row=1,
        sub_img_size=(220, 180),
    )

    assert isinstance(img, Image.Image)
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_render_top_k_bit_environments_image_validates_max_environments():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    with pytest.raises(ValueError, match="max_environments must be > 0"):
        render_top_k_bit_environments_image(
            mol,
            bit_info,
            np.zeros(128, dtype=float),
            max_environments=0,
        )


def test_render_top_positive_negative_bit_environments_images_splits_signs():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    bits = sorted(bit_info.keys())
    shap_values[bits[0]] = 0.8
    shap_values[bits[1]] = -0.6

    images = render_top_positive_negative_bit_environments_images(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        max_environments_per_sign=1,
        mols_per_row=1,
        sub_img_size=(220, 180),
    )

    assert set(images.keys()) == {"positive", "negative"}
    assert isinstance(images["positive"], Image.Image)
    assert isinstance(images["negative"], Image.Image)


def test_render_top_positive_negative_bit_environments_images_handles_missing_sign():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    shap_values[sorted(bit_info.keys())[0]] = 0.8

    images = render_top_positive_negative_bit_environments_images(
        mol,
        bit_info,
        shap_values,
        top_k=1,
    )

    assert isinstance(images["positive"], Image.Image)
    assert images["negative"] is None


def test_render_top_k_parent_molecule_environment_highlights_returns_pil_image():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=2, n_bits=128)

    shap_values = np.zeros(128, dtype=float)
    first_bit = sorted(bit_info.keys())[0]
    shap_values[first_bit] = 0.9

    image = render_top_k_parent_molecule_environment_highlights(
        mol,
        bit_info,
        shap_values,
        top_k=1,
        max_environments=1,
        mols_per_row=1,
        sub_img_size=(260, 220),
    )

    assert isinstance(image, Image.Image)
    assert image.size[0] > 0
    assert image.size[1] > 0


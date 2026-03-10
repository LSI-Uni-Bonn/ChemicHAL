"""
chemagent.explainability
~~~~~~~~~~~~~~~~~~~~~~~~
SHAP-based explainability tools for trained selectivity prediction models.

Public API
----------
* :class:`SHAPExplainer`  — wraps a trained sklearn model, auto-selects the
  right SHAP explainer (TreeExplainer for RF, KernelExplainer for SVM).
"""

from .shap_explainer import SHAPExplainer
from .mol_shap_draw import (
    get_ecfp_morgan_generator_bit_info,
    shap_to_atom_weight,
    get_atom_wise_weight_map,
)

__all__ = [
    "SHAPExplainer",
    "get_ecfp_morgan_generator_bit_info",
    "shap_to_atom_weight",
    "get_atom_wise_weight_map",
]

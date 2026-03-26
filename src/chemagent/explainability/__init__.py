"""
chemagent.explainability
~~~~~~~~~~~~~~~~~~~~~~~~
Explainability tools for trained selectivity prediction models.

Public API
----------
* :class:`SHAPExplainer`  — wraps a trained sklearn model, auto-selects the
  right SHAP explainer (TreeExplainer for RF, KernelExplainer for SVM).
* :class:`MolecularAnchor`  — identifies molecular fragments (anchors) critical
  for model predictions.
"""

from .shap_explainer import SHAPExplainer
from .mol_shap_draw import (
    get_ecfp_morgan_generator_bit_info,
    shap_to_atom_weight,
    get_atom_wise_weight_map,
)


def __getattr__(name: str):
    # Lazily import optional heavy modules to keep lightweight SHAP-only imports clean.
    if name == "MolecularAnchor":
        from .MolAnchor import MolecularAnchor

        return MolecularAnchor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SHAPExplainer",
    "get_ecfp_morgan_generator_bit_info",
    "shap_to_atom_weight",
    "get_atom_wise_weight_map",
    "MolecularAnchor",
]


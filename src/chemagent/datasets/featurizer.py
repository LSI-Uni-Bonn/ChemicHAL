"""
Dataset featurization helpers.

Bridges a loaded ``pd.DataFrame`` and the ``chemagent.featurization`` package:
  - ``featurize_df`` — compute fingerprints from a SMILES column server-side
  - ``build_processed_entry`` — assemble the processed-dataset dict stored in
    the MCP server's ``_processed_datasets`` registry
  - ``prepare_from_external_features`` — pair a DataFrame with an externally
    computed feature matrix

Usage
-----
    from chemagent.datasets.featurizer import featurize_df, build_processed_entry

    features, labels = featurize_df(df, method="ECFP", n_bits=2048, radius=2)
    processed = build_processed_entry(df, features, label_col="class_label")
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import chemagent.featurization as _feat


# Featurizer discovery
def available_featurizers() -> Dict[str, Any]:
    """Return all public UpperCase callables from ``chemagent.featurization``.

    Returns
    -------
    dict
        ``{method_name: callable}``
    """
    return {
        name: fn
        for name, fn in vars(_feat).items()
        if not name.startswith("_") and callable(fn) and name[0].isupper()
    }


def list_featurizers() -> Dict[str, Any]:
    """Return name, parameters, and one-line description for every featurizer.

    Returns
    -------
    dict
        ``{method_name: {"parameters": {...}, "description": str}}``
    """
    result: Dict[str, Any] = {}
    for name, fn in available_featurizers().items():
        sig = inspect.signature(fn)
        params = {
            k: str(v.default) if v.default is not inspect.Parameter.empty else "<required>"
            for k, v in sig.parameters.items()
            if k != "smiles_list"
        }
        doc_first = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        result[name] = {"parameters": params, "description": doc_first}
    return result


# Feature computation
def featurize_df(
    df: pd.DataFrame,
    method: str = "ECFP",
    n_bits: int = 2048,
    radius: int = 2,
    smiles_col: Optional[str] = None,
    return_bit_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict]]:
    """Compute a fingerprint feature matrix from a DataFrame's SMILES column.

    Parameters
    ----------
    df:
        DataFrame with a SMILES column (configured via ``df.attrs["smiles_col"]``
        or passed explicitly with *smiles_col*).
    method:
        Featurizer name — must match a public UpperCase function in
        ``chemagent.featurization.fingerprints``.
    n_bits:
        Passed to the featurizer if its signature accepts it (default 2048).
    radius:
        Passed to the featurizer if its signature accepts it (default 2).
    smiles_col:
        Override the SMILES column name (falls back to ``df.attrs["smiles_col"]``).
    return_bit_info:
        If True and method is "ECFP", also return bit information dictionary.
        Bit info maps fingerprint bit indices to atom environments.

    Returns
    -------
    np.ndarray
        2-D feature matrix, shape ``(n_samples, n_bits)``.
    tuple (if return_bit_info=True)
        (features, bit_info_dict) where bit_info_dict is the bit information for ECFP,
        or None for other methods.

    Raises
    ------
    ValueError
        If *method* is not registered or no SMILES column is found.
    """
    col = smiles_col or df.attrs.get("smiles_col", "smiles")
    if not col or col not in df.columns:
        raise ValueError(
            f"No SMILES column found (smiles_col={col!r}). "
            "Set smiles_col when calling load_dataset()."
        )

    featurizers = available_featurizers()
    if method not in featurizers:
        raise ValueError(
            f"Unknown featurizer {method!r}. "
            f"Available: {sorted(featurizers.keys())}"
        )

    fn  = featurizers[method]
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in {"n_bits": n_bits, "radius": radius}.items()
              if k in sig.parameters}

    # For ECFP with return_bit_info, also request bit information
    if return_bit_info and method == "ECFP" and "return_bit_info" in sig.parameters:
        result = fn(df[col].tolist(), return_bit_info=True, **kwargs)
        if isinstance(result, tuple):
            fps, bit_info = result
            return np.array(fps), bit_info
    
    features = fn(df[col].tolist(), **kwargs)
    
    if return_bit_info:
        return np.array(features), None
    
    return np.array(features)


# Processed-entry builder
def build_processed_entry(
    df: pd.DataFrame,
    features: np.ndarray,
    label_col: Optional[str] = None,
    smiles_col: Optional[str] = None,
    id_col: Optional[str] = None,
    core_col: Optional[str] = None,
    bit_info: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the processed-dataset dict stored in ``_processed_datasets``.

    Parameters
    ----------
    df:
        Source DataFrame (column config read from ``df.attrs`` when not
        overridden by explicit arguments).
    features:
        2-D feature array, shape ``(n_samples, n_features)``.
    label_col, smiles_col, id_col:
        Override column names; fall back to ``df.attrs`` values.
    core_col:
        Column containing analogue-series core / scaffold identifiers.
        When provided, the ``"core"`` key is added to the processed dict so
        that ``split_processed`` can use ``split_type="analogue_series"``.
    bit_info:
        Optional bit information dictionary for ECFP fingerprints.
        Maps bit indices to atom environment tuples (for explainability tools like MolAnchor).

    Returns
    -------
    dict
        Keys: ``features``, ``labels``, ``label_column``, and optionally
        ``smiles``, ``cid``, ``core``, and ``bit_info``.
    """
    lc = label_col  or df.attrs.get("label_col",  "class_label")
    sc = smiles_col or df.attrs.get("smiles_col", None)
    ic = id_col     or df.attrs.get("id_col",     None)
    cc = core_col   or df.attrs.get("core_col",   None)

    entry: Dict[str, Any] = {
        "features":     features,
        "labels":       np.array(df[lc].values),
        "label_column": lc,
    }
    if sc and sc in df.columns:
        entry["smiles"] = df[sc].values
    if ic and ic in df.columns:
        entry["cid"] = df[ic].values
    if cc and cc in df.columns:
        entry["core"] = df[cc].values
    if bit_info is not None:
        entry["bit_info"] = bit_info
    return entry


# External feature injection
def prepare_from_external_features(
    df: pd.DataFrame,
    features: List[List[float]],
    label_col: Optional[str] = None,
    smiles_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a processed entry from an externally computed feature matrix.

    Used when featurization was performed outside the server (e.g. via the
    mol-featurization MCP server).

    Parameters
    ----------
    df:
        Source DataFrame.
    features:
        2-D list of floats, shape ``(n_samples, n_features)``.
    label_col, smiles_col, id_col:
        Override column names; fall back to ``df.attrs``.

    Returns
    -------
    dict
        Same structure as :func:`build_processed_entry`.

    Raises
    ------
    ValueError
        If ``len(features) != len(df)``.
    """
    arr = np.array(features)
    if len(arr) != len(df):
        raise ValueError(
            f"Features length ({len(arr)}) does not match "
            f"dataset length ({len(df)})."
        )
    return build_processed_entry(
        df=df,
        features=arr,
        label_col=label_col,
        smiles_col=smiles_col,
        id_col=id_col,
    )

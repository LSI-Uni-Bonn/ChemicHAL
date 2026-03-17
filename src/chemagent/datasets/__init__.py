"""
chemagent.datasets — dataset handling utilities for ML pipelines.

Sub-modules
-----------
loader       : CSV loading, path resolution, label statistics
featurizer   : fingerprint computation, processed-entry builder, external feature injection
splitter     : train/val/test splitting, split serialisation
io           : split file I/O, ML-ready payload builder, dataset info

Usage
-----
    from chemagent.datasets import load_csv, featurize_df, split_processed
    from chemagent.datasets import load_split_file, get_ml_ready_data
"""

from .loader      import load_csv, resolve_path, list_csv_files, workspace_root
from .featurizer  import (
    featurize_df,
    build_processed_entry,
    prepare_from_external_features,
    available_featurizers,
    list_featurizers,
)
from .splitter    import split_processed, save_split
from .io          import load_split_file, get_ml_ready_data, get_dataset_info

__all__ = [
    # loader
    "load_csv",
    "resolve_path",
    "list_csv_files",
    "workspace_root",
    # featurizer
    "featurize_df",
    "build_processed_entry",
    "prepare_from_external_features",
    "available_featurizers",
    "list_featurizers",
    # splitter
    "split_processed",
    "save_split",
    # io
    "load_split_file",
    "get_ml_ready_data",
    "get_dataset_info",
]

"""Dataset container for features and labels with additional instance tracking."""

from typing import Any, Optional, Union
import numpy as np
from numpy.typing import NDArray


class Dataset:
    """Container for ML features and labels with flexible instance management.
    
    Attributes:
        features: Feature array
        labels: Label array
        _add_instances: Set of additional instance names
    """
    
    def __init__(self, features: NDArray[Any], labels: NDArray[Any]):
        self.features = features
        self.labels = labels
        self._add_instances: set[str] = set()

    def add_instance(self, name: str, values: NDArray[Any]) -> None:
        """Add a named instance array to the dataset.
        
        Args:
            name: Name for the instance
            values: Array of values to associate with the name
        """
        self._add_instances.add(name)
        self.__dict__[name] = values

    @property
    def columns(self) -> dict[str, NDArray[Any]]:
        """Return all data columns excluding private attributes.
        
        Returns:
            Dictionary mapping column names to arrays
        """
        data_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        data_dict['features'] = self.features
        data_dict['labels'] = self.labels
        return data_dict

    def __len__(self) -> int:
        """Return the number of instances in the dataset."""
        return self.labels.shape[0]

    def __iter__(self):
        """Iterate over dataset instances."""
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx: Union[int, slice, NDArray[np.integer]]) -> Union[dict[str, Any], 'Dataset']:
        """Get instance(s) by index.
        
        Args:
            idx: Integer index, slice, or array of indices
            
        Returns:
            Dictionary for single index, Dataset for multiple indices
        """
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        subset = Dataset(self.features[idx], self.labels[idx])
        for addt_instance in self._add_instances:
            subset.add_instance(addt_instance, self.__dict__[addt_instance][idx])

        return subset

    def shuffle_labels(self, seed: int = 1) -> None:
        """Shuffle labels in-place with reproducible seed.
        
        Args:
            seed: Random seed for reproducibility (default: 1)
        """
        rng = np.random.RandomState(seed)
        rng.shuffle(self.labels)

    def shuffle_instances(self, instance_names: Optional[list[str]] = None, seed: int = 1) -> None:
        """Shuffle specific instances in-place with reproducible seed.
        
        Args:
            instance_names: List of instance names to shuffle
            seed: Random seed for reproducibility (default: 1)
            
        Raises:
            ValueError: If instance_names is None or empty
        """
        if instance_names is None:
            raise ValueError('No instance names provided to shuffle')
        
        rng = np.random.RandomState(seed)
        for instance_name in instance_names:
            if instance_name in self._add_instances:
                rng.shuffle(self.__dict__[instance_name])

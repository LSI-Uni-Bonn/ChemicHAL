import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class _TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device


class _BaseDNN:
    def __init__(
        self,
        hidden_size: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        weight_decay: float = 0.0,
        random_seed: int = 42,
        use_cuda: bool = False,
        verbose: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_cuda = use_cuda
        self.verbose = verbose

        self._state: _TrainState | None = None
        self.n_features_in_: int | None = None

    # training pipeline that may call set_params/get_params.
    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "use_cuda": self.use_cuda,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter '{key}' for {self.__class__.__name__}")
            setattr(self, key, value)
        return self

    def _device(self) -> torch.device:
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _as_feature_array(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {arr.shape}")
        return arr

    def _fit_loop(self, X: np.ndarray, y: np.ndarray, output_size: int, criterion: nn.Module) -> None:
        _set_seed(self.random_seed)
        device = self._device()
        self.n_features_in_ = int(X.shape[1])

        model = _MLP(
            input_size=self.n_features_in_,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            output_size=output_size,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = self._build_target_tensor(y)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * xb.shape[0]

            if self.verbose:
                avg_loss = running_loss / max(int(X_tensor.shape[0]), 1)
                print(f"Epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

        model.eval()
        self._state = _TrainState(model=model, optimizer=optimizer, criterion=criterion, device=device)

    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Model is not fitted")
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self._state.device)
            logits = self._state.model(xb)
            out = logits.detach().cpu().numpy()
        return out

    def _build_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def get_torch_model(self) -> nn.Module:
        """Return the fitted torch model in eval mode for explainability hooks."""
        if self._state is None:
            raise RuntimeError("Model is not fitted")
        self._state.model.eval()
        return self._state.model

    def get_torch_device(self) -> torch.device:
        """Return the torch device used during fitting."""
        if self._state is None:
            raise RuntimeError("Model is not fitted")
        return self._state.device

    def as_torch_tensor(self, X) -> torch.Tensor:
        """Convert features to float32 torch tensor on the model device."""
        X_arr = self._as_feature_array(X)
        return torch.as_tensor(X_arr, dtype=torch.float32, device=self.get_torch_device())


class DNNClassifier(_BaseDNN):
    def __init__(self, class_weight: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.class_weight = class_weight
        self.classes_: np.ndarray | None = None
        self._is_binary: bool = True
        self._class_to_index: dict[int, int] = {}

    def get_params(self, deep: bool = True) -> dict:
        params = super().get_params(deep=deep)
        params["class_weight"] = self.class_weight
        return params

    def fit(self, X, y):
        X_arr = self._as_feature_array(X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("Expected 1D labels for classification")

        self.classes_ = np.unique(y_arr)
        self._is_binary = len(self.classes_) == 2

        if self._is_binary:
            y01 = (y_arr == self.classes_[1]).astype(np.float32)
            if self.class_weight == "balanced":
                n_pos = float(np.sum(y01 == 1.0))
                n_neg = float(np.sum(y01 == 0.0))
                if n_pos == 0 or n_neg == 0:
                    pos_weight = torch.tensor(1.0, dtype=torch.float32)
                else:
                    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
            self._fit_loop(X_arr, y01, output_size=1, criterion=criterion)
            return self

        self._class_to_index = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_index[int(c)] for c in y_arr], dtype=np.int64)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(np.float32)
            counts[counts == 0] = 1.0
            weights = len(y_idx) / (len(self.classes_) * counts)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        else:
            criterion = nn.CrossEntropyLoss()
        self._fit_loop(X_arr, y_idx, output_size=len(self.classes_), criterion=criterion)
        return self

    def _build_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        if self._is_binary:
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            return torch.as_tensor(y_arr, dtype=torch.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        return torch.as_tensor(y_arr, dtype=torch.long)

    def predict_proba(self, X) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted")

        X_arr = self._as_feature_array(X)
        logits = self._forward_numpy(X_arr)

        if self._is_binary:
            p1 = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            p0 = 1.0 - p1
            return np.column_stack((p0, p1))

        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted")

        proba = self.predict_proba(X)
        if self._is_binary:
            idx = (proba[:, 1] >= 0.5).astype(int)
            return self.classes_[idx]

        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class DNNRegressor(_BaseDNN):
    def fit(self, X, y):
        X_arr = self._as_feature_array(X)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim != 1:
            raise ValueError("Expected 1D targets for regression")

        criterion = nn.MSELoss()
        self._fit_loop(X_arr, y_arr, output_size=1, criterion=criterion)
        return self

    def _build_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return torch.as_tensor(y_arr, dtype=torch.float32)

    def predict(self, X) -> np.ndarray:
        X_arr = self._as_feature_array(X)
        pred = self._forward_numpy(X_arr).reshape(-1)
        return pred.astype(np.float64)
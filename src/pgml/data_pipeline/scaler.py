from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class BaseTorchScaler(nn.Module):
    """
    Abstract base class for PyTorch-native scalers.
    Supports grouped scaling (e.g., frequency-dependent scaling).
    """

    def __init__(self, dim: int = 0, feature_names: Optional[List[str]] = None):
        super().__init__()
        self.dim = dim
        self.feature_names = feature_names

    def forward(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        """
        PyTorch's required forward pass. Aliases to transform.
        """
        return self.transform(x, group_key)

    def fit(self, x: torch.Tensor, group_key: str = "default") -> "BaseTorchScaler":
        raise NotImplementedError

    def transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        self.fit(x, group_key)
        return self.transform(x, group_key)

    def load_from_stats(self, stats_dict: Dict[str, Dict], group_key: str = "default"):
        raise NotImplementedError


class TorchStandardScaler(BaseTorchScaler):
    def __init__(self, dim: int = 0, feature_names: Optional[List[str]] = None, epsilon: float = 1e-8):
        super().__init__(dim=dim, feature_names=feature_names)
        self.epsilon = epsilon
        # Dictionaries to hold references to dynamically registered buffers
        self.group_means: Dict[str, torch.Tensor] = {}
        self.group_stds: Dict[str, torch.Tensor] = {}

    def fit(self, x: torch.Tensor, group_key: str = "default") -> "TorchStandardScaler":
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, unbiased=False, keepdim=True)

        self._register_state(group_key, mean, std)
        return self

    def _register_state(self, group_key: str, mean: torch.Tensor, std: torch.Tensor):
        # Register as PyTorch buffer so it gets saved in state_dict and moves to target device
        self.register_buffer(f"mean_{group_key}", mean)
        self.register_buffer(f"std_{group_key}", std)

        self.group_means[group_key] = getattr(self, f"mean_{group_key}")
        self.group_stds[group_key] = getattr(self, f"std_{group_key}")

    def transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        if group_key not in self.group_means:
            raise ValueError(f"Scaler not fitted for group: {group_key}")

        mean = self.group_means[group_key]
        std = self.group_stds[group_key]
        return (x - mean) / (std + self.epsilon)

    def inverse_transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        mean = self.group_means[group_key]
        std = self.group_stds[group_key]
        return (x * (std + self.epsilon)) + mean

    def load_from_stats(self, stats_dict: Dict[str, Dict[str, float]], group_key: str = "default"):
        if not self.feature_names:
            raise ValueError("feature_names must be provided to map out-of-core statistics.")

        # Ensure dimensions match PyTorch keepdim behavior based on self.dim
        shape = [1] * (self.dim + 1)
        shape[self.dim] = len(self.feature_names)

        mean_vec = [stats_dict[feat]["mean"] for feat in self.feature_names]
        std_vec = [stats_dict[feat]["std"] for feat in self.feature_names]

        mean_tensor = torch.tensor(mean_vec, dtype=torch.float32).view(*shape)
        std_tensor = torch.tensor(std_vec, dtype=torch.float32).view(*shape)

        self._register_state(group_key, mean_tensor, std_tensor)


class TorchMinMaxScaler(BaseTorchScaler):
    def __init__(
            self,
            feature_range: Tuple[float, float] = (0.0, 1.0),
            dim: int = 0,
            feature_names: Optional[List[str]] = None
    ):
        super().__init__(dim=dim, feature_names=feature_names)
        self.feature_range = feature_range
        self.scale_min, self.scale_max = feature_range

        self.group_mins: Dict[str, torch.Tensor] = {}
        self.group_maxs: Dict[str, torch.Tensor] = {}

    def fit(self, x: torch.Tensor, group_key: str = "default") -> "TorchMinMaxScaler":
        # PyTorch max/min over a dimension return a tuple of (values, indices)
        x_min = x.min(dim=self.dim, keepdim=True)[0]
        x_max = x.max(dim=self.dim, keepdim=True)[0]

        self._register_state(group_key, x_min, x_max)
        return self

    def _register_state(self, group_key: str, x_min: torch.Tensor, x_max: torch.Tensor):
        self.register_buffer(f"min_{group_key}", x_min)
        self.register_buffer(f"max_{group_key}", x_max)

        self.group_mins[group_key] = getattr(self, f"min_{group_key}")
        self.group_maxs[group_key] = getattr(self, f"max_{group_key}")

    def transform(self, x: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        if group_key not in self.group_mins:
            raise ValueError(f"Scaler not fitted for group: {group_key}")

        x_min = self.group_mins[group_key]
        x_max = self.group_maxs[group_key]

        # Avoid division by zero
        range_diff = x_max - x_min
        range_diff = torch.where(range_diff == 0, torch.ones_like(range_diff), range_diff)

        x_std = (x - x_min) / range_diff
        return x_std * (self.scale_max - self.scale_min) + self.scale_min

    def inverse_transform(self, x_scaled: torch.Tensor, group_key: str = "default") -> torch.Tensor:
        x_min = self.group_mins[group_key]
        x_max = self.group_maxs[group_key]

        range_diff = x_max - x_min
        range_diff = torch.where(range_diff == 0, torch.ones_like(range_diff), range_diff)

        x_std = (x_scaled - self.scale_min) / (self.scale_max - self.scale_min)
        return (x_std * range_diff) + x_min

    def load_from_stats(self, stats_dict: Dict[str, Dict[str, float]], group_key: str = "default"):
        if not self.feature_names:
            raise ValueError("feature_names must be provided to map out-of-core statistics.")

        shape = [1] * (self.dim + 1)
        shape[self.dim] = len(self.feature_names)

        min_vec = [stats_dict[feat]["min"] for feat in self.feature_names]
        max_vec = [stats_dict[feat]["max"] for feat in self.feature_names]

        min_tensor = torch.tensor(min_vec, dtype=torch.float32).view(*shape)
        max_tensor = torch.tensor(max_vec, dtype=torch.float32).view(*shape)

        self._register_state(group_key, min_tensor, max_tensor)
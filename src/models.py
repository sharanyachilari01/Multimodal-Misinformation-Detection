import numpy as np
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    Small PyTorch MLP:
    Linear(in_features, hidden_dim) -> ReLU -> Dropout(dropout)
    -> Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout(dropout)
    -> Linear(hidden_dim, 1)
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch_size,) logits
        return self.net(x).squeeze(-1)


def build_features(model_name: str, T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Build input features for baseline models:
    - M1: T (text-only, 512-d)
    - M2: V (image-only, 512-d)
    - M3: [T, V] (concat fusion, 1024-d)
    - M4: [T, V, T*V, |T-V|, cosine(T, V)] (interaction fusion, 2049-d)
    """
    if model_name == "M1":
        return T.astype(np.float32)
    elif model_name == "M2":
        return V.astype(np.float32)
    elif model_name == "M3":
        return np.concatenate([T, V], axis=1).astype(np.float32)
    elif model_name == "M4":
        prod = T * V
        diff = np.abs(T - V)
        cos = np.sum(prod, axis=1, keepdims=True)
        return np.concatenate([T, V, prod, diff, cos], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

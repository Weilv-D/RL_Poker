"""History buffer scaffolding for memory-based policies.

This is a lightweight placeholder to define interfaces for future
sequence-aware models (GRU/Transformer). Not used by default.
"""

from dataclasses import dataclass

import torch


@dataclass
class HistoryConfig:
    """Configuration for history encoding."""

    enabled: bool = False
    window: int = 16
    feature_dim: int = 0


class HistoryBuffer:
    """Fixed-length history buffer for per-env sequences.

    Stores the last K feature vectors per environment. When disabled,
    it becomes a no-op and returns None.
    """

    def __init__(self, num_envs: int, config: HistoryConfig, device: torch.device):
        self.config: HistoryConfig = config
        self.device: torch.device = device
        self.num_envs: int = num_envs
        if not config.enabled:
            self.buffer: torch.Tensor | None = None
            return
        if config.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0 when history is enabled")
        self.buffer = torch.zeros(
            num_envs, config.window, config.feature_dim, device=device, dtype=torch.float32
        )
        self._pos: torch.Tensor = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if self.buffer is None:
            return
        self.buffer[env_ids] = 0.0
        self._pos[env_ids] = 0

    def push(self, features: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Append features for given envs.

        Args:
            features: [B, feature_dim] features to append
            env_ids: Optional indices into buffer. If None, assume full batch order.
        """
        if self.buffer is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pos = self._pos[env_ids]
        self.buffer[env_ids, pos] = features
        self._pos[env_ids] = (pos + 1) % self.config.window

    def get_sequence(self, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Return history sequence for envs in chronological order: [B, window, feature_dim]."""
        if self.buffer is None:
            return None
        if env_ids is None:
            buf = self.buffer
            pos = self._pos
        else:
            buf = self.buffer[env_ids]
            pos = self._pos[env_ids]

        window = self.config.window
        feat_dim = self.config.feature_dim
        idx = (torch.arange(window, device=self.device).unsqueeze(0) + pos.unsqueeze(1)) % window
        idx = idx.unsqueeze(2).expand(-1, -1, feat_dim)
        return buf.gather(1, idx)

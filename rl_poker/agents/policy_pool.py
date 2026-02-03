"""Policy pool for model checkpoint management.

This module provides:
- PolicyPool: A registry for storing and loading model checkpoints
- Support for multiple checkpoint formats (PyTorch, etc.)
- Interface for evaluation against past checkpoints

The policy pool is designed to:
1. Store model checkpoints during training
2. Load checkpoints for evaluation
3. Sample opponents from the pool for self-play
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Protocol, Union
import json
import numpy as np


class PolicyProtocol(Protocol):
    """Protocol that policy models should follow."""

    def act(
        self,
        observation: dict,
        action_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select an action given observation and mask."""
        ...

    def reset(self) -> None:
        """Reset policy state."""
        ...


@dataclass
class CheckpointInfo:
    """Metadata about a stored checkpoint.

    Attributes:
        name: Unique name/identifier for the checkpoint
        path: File path to the checkpoint
        step: Training step when checkpoint was saved
        metadata: Additional metadata (performance stats, etc.)
        created_at: Timestamp when checkpoint was created
    """

    name: str
    path: Path
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "step": self.step,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointInfo":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            path=Path(d["path"]),
            step=d.get("step", 0),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at"),
        )


class PolicyPool:
    """Pool of policy checkpoints for evaluation and self-play.

    The policy pool manages multiple policy checkpoints and provides
    utilities for:
    - Saving/loading checkpoints
    - Sampling opponents for evaluation
    - Tracking checkpoint performance

    Attributes:
        pool_dir: Directory where checkpoints are stored
        checkpoints: Registry of checkpoint info by name
        loader: Function to load a policy from checkpoint path
    """

    def __init__(
        self,
        pool_dir: Union[str, Path],
        loader: Optional[Callable[[Path], PolicyProtocol]] = None,
    ):
        """Initialize the policy pool.

        Args:
            pool_dir: Directory to store checkpoints
            loader: Function to load a policy from a checkpoint path.
                    If None, uses default PyTorch loader.
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, CheckpointInfo] = {}
        self.loader = loader or self._default_loader
        self._rng = np.random.default_rng()

        # Load existing checkpoints from registry
        self._load_registry()

    def _registry_path(self) -> Path:
        """Path to the checkpoint registry file."""
        return self.pool_dir / "registry.json"

    def _load_registry(self) -> None:
        """Load checkpoint registry from disk."""
        registry_path = self._registry_path()
        if registry_path.exists():
            with open(registry_path, "r") as f:
                data = json.load(f)
                for entry in data.get("checkpoints", []):
                    info = CheckpointInfo.from_dict(entry)
                    self.checkpoints[info.name] = info

    def _save_registry(self) -> None:
        """Save checkpoint registry to disk."""
        registry_path = self._registry_path()
        data = {"checkpoints": [info.to_dict() for info in self.checkpoints.values()]}
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _default_loader(self, path: Path) -> PolicyProtocol:
        """Default checkpoint loader using PyTorch.

        This expects checkpoints saved via torch.save() containing
        a state_dict.
        """
        try:
            import torch

            # Load checkpoint
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

            # The checkpoint should contain model info
            # This is a simple wrapper - actual implementation depends on model class
            raise NotImplementedError(
                "Default loader requires model class. "
                "Provide a custom loader function that knows how to instantiate your model."
            )
        except ImportError:
            raise ImportError("PyTorch required for default checkpoint loading")

    def add_checkpoint(
        self,
        name: str,
        policy: Any,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        save_fn: Optional[Callable[[Any, Path], None]] = None,
    ) -> CheckpointInfo:
        """Add a new checkpoint to the pool.

        Args:
            name: Unique name for this checkpoint
            policy: The policy object to save
            step: Training step number
            metadata: Additional metadata to store
            save_fn: Custom save function (uses torch.save by default)

        Returns:
            CheckpointInfo for the saved checkpoint
        """
        from datetime import datetime

        # Create checkpoint path
        checkpoint_path = self.pool_dir / f"{name}.pt"

        # Save checkpoint
        if save_fn:
            save_fn(policy, checkpoint_path)
        else:
            try:
                import torch

                if hasattr(policy, "state_dict"):
                    torch.save(
                        {
                            "state_dict": policy.state_dict(),
                            "step": step,
                            "metadata": metadata or {},
                        },
                        checkpoint_path,
                    )
                else:
                    torch.save(policy, checkpoint_path)
            except ImportError:
                raise ImportError("PyTorch required for default checkpoint saving")

        # Create checkpoint info
        info = CheckpointInfo(
            name=name,
            path=checkpoint_path,
            step=step,
            metadata=metadata or {},
            created_at=datetime.now().isoformat(),
        )

        # Register checkpoint
        self.checkpoints[name] = info
        self._save_registry()

        return info

    def load_checkpoint(self, name: str) -> PolicyProtocol:
        """Load a policy from a checkpoint.

        Args:
            name: Name of the checkpoint to load

        Returns:
            Loaded policy object

        Raises:
            KeyError: If checkpoint name not found
            FileNotFoundError: If checkpoint file missing
        """
        if name not in self.checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found in pool")

        info = self.checkpoints[name]
        if not info.path.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {info.path}")

        return self.loader(info.path)

    def load_by_path(self, path: Union[str, Path]) -> PolicyProtocol:
        """Load a policy directly from a file path.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded policy object
        """
        return self.loader(Path(path))

    def remove_checkpoint(self, name: str, delete_file: bool = True) -> None:
        """Remove a checkpoint from the pool.

        Args:
            name: Name of checkpoint to remove
            delete_file: Whether to delete the checkpoint file
        """
        if name not in self.checkpoints:
            return

        info = self.checkpoints[name]

        if delete_file and info.path.exists():
            info.path.unlink()

        del self.checkpoints[name]
        self._save_registry()

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints in the pool.

        Returns:
            List of CheckpointInfo objects sorted by step
        """
        return sorted(self.checkpoints.values(), key=lambda x: x.step)

    def get_checkpoint_info(self, name: str) -> Optional[CheckpointInfo]:
        """Get info about a specific checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            CheckpointInfo or None if not found
        """
        return self.checkpoints.get(name)

    def sample_checkpoints(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[CheckpointInfo]:
        """Sample random checkpoints from the pool.

        Args:
            n: Number of checkpoints to sample
            rng: Random generator for sampling
            exclude: List of checkpoint names to exclude

        Returns:
            List of sampled CheckpointInfo objects
        """
        if rng is None:
            rng = self._rng

        available = [
            info
            for name, info in self.checkpoints.items()
            if exclude is None or name not in exclude
        ]

        if not available:
            return []

        n = min(n, len(available))
        indices = rng.choice(len(available), size=n, replace=False)
        return [available[i] for i in indices]

    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint by step number.

        Returns:
            Latest CheckpointInfo or None if pool is empty
        """
        if not self.checkpoints:
            return None
        return max(self.checkpoints.values(), key=lambda x: x.step)

    def set_seed(self, seed: int) -> None:
        """Set random seed for sampling.

        Args:
            seed: Random seed
        """
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of checkpoints in the pool."""
        return len(self.checkpoints)

    def __contains__(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        return name in self.checkpoints

    def __repr__(self) -> str:
        return f"PolicyPool(pool_dir={self.pool_dir!r}, n_checkpoints={len(self)})"


class PooledPolicyAgent:
    """Agent wrapper that loads policy from pool.

    This provides a consistent interface for using pooled checkpoints
    as opponents in evaluation.
    """

    def __init__(
        self,
        pool: PolicyPool,
        checkpoint_name: str,
        name: Optional[str] = None,
    ):
        """Initialize pooled policy agent.

        Args:
            pool: PolicyPool to load from
            checkpoint_name: Name of checkpoint to use
            name: Optional display name
        """
        self.pool = pool
        self.checkpoint_name = checkpoint_name
        self.name = name or f"Pool[{checkpoint_name}]"
        self._policy: Optional[PolicyProtocol] = None

    def _ensure_loaded(self) -> PolicyProtocol:
        """Ensure policy is loaded."""
        if self._policy is None:
            self._policy = self.pool.load_checkpoint(self.checkpoint_name)
        return self._policy

    def act(
        self,
        observation: dict,
        action_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select an action using the pooled policy."""
        policy = self._ensure_loaded()
        return policy.act(observation, action_mask, rng)

    def reset(self) -> None:
        """Reset policy state."""
        if self._policy is not None:
            self._policy.reset()

    def __repr__(self) -> str:
        return f"PooledPolicyAgent(checkpoint={self.checkpoint_name!r})"


def create_policy_pool(
    pool_dir: Union[str, Path],
    loader: Optional[Callable[[Path], PolicyProtocol]] = None,
) -> PolicyPool:
    """Factory function to create a policy pool.

    Args:
        pool_dir: Directory for checkpoint storage
        loader: Custom checkpoint loader function

    Returns:
        Configured PolicyPool instance
    """
    return PolicyPool(pool_dir=pool_dir, loader=loader)

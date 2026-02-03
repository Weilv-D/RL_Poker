"""Random legal action agent.

This module provides a simple baseline agent that randomly selects
from available legal moves. Useful for testing and as a baseline
opponent for evaluation.
"""

from typing import Optional, Protocol
import numpy as np


class BaseAgent(Protocol):
    """Protocol defining the agent interface.

    All agents should implement this interface for compatibility
    with the evaluation framework.
    """

    def act(
        self,
        observation: dict,
        action_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select an action given the current observation.

        Args:
            observation: Dictionary containing game state observation
            action_mask: Boolean array indicating legal actions (True = legal)
            rng: Optional random generator for reproducibility

        Returns:
            Action index to take
        """
        ...

    def reset(self) -> None:
        """Reset agent state (if any) at the start of a new game."""
        ...


class RandomAgent:
    """Agent that selects uniformly at random from legal actions.

    This is the simplest baseline agent - it completely ignores
    game state and simply picks a random legal move.

    Attributes:
        name: Agent name for identification
        rng: Random number generator
    """

    def __init__(self, seed: Optional[int] = None, name: str = "RandomAgent"):
        """Initialize the random agent.

        Args:
            seed: Random seed for reproducibility
            name: Agent name for identification
        """
        self.name = name
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        observation: dict,
        action_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Select a random legal action.

        Args:
            observation: Dictionary containing game state (ignored by random agent)
            action_mask: Boolean array indicating legal actions
            rng: Optional random generator (uses internal rng if not provided)

        Returns:
            Randomly selected legal action index

        Raises:
            ValueError: If no legal actions available
        """
        if rng is None:
            rng = self.rng

        # Find indices of legal actions
        legal_indices = np.where(action_mask)[0]

        if len(legal_indices) == 0:
            raise ValueError("No legal actions available")

        return int(rng.choice(legal_indices))

    def reset(self) -> None:
        """Reset agent state (no-op for random agent)."""
        # Random agent has no state to reset
        pass

    def set_seed(self, seed: int) -> None:
        """Set a new random seed.

        Args:
            seed: New random seed
        """
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"RandomAgent(seed={self._seed}, name={self.name!r})"


def create_random_agent(seed: Optional[int] = None) -> RandomAgent:
    """Factory function to create a random agent.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured RandomAgent instance
    """
    return RandomAgent(seed=seed)

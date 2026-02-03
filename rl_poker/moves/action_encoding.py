"""Action encoding for RL environment.

This module provides:
- Fixed action space with MAX_ACTIONS constant
- Move-to-action encoding (move → action index)
- Action-to-move decoding (action index → move)
- Action mask generation for illegal move masking

The action space is a padded list of legal moves with a fixed maximum size.
Invalid actions are masked out.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from .legal_moves import (
    Move,
    MoveType,
    MoveContext,
    PASS_MOVE,
    get_legal_moves,
)
from rl_poker.rules import Card


# Maximum number of actions in the action space
# Empirically chosen based on worst-case analysis:
# - Singles: 13 cards max
# - Pairs: up to 6 pairs from 13 cards
# - Straights: many combinations of 5-13 cards
# - Consecutive pairs: many combinations
# - 3+2: C(4,3) * C(10,2) combinations per quad
# - 4+3: C(4,4) * C(9,3) per quad
# With exemptions, this can grow.
# Testing shows hands with multiple quads can generate 700+ moves.
# Using 1000 as a safe upper bound with margin for tail-hand exemptions.
MAX_ACTIONS = 1000


@dataclass
class ActionSpace:
    """Action space representation for the RL environment.

    Attributes:
        legal_moves: List of legal Move objects
        action_mask: Boolean array of length MAX_ACTIONS (True = valid)
        move_to_action: Dict mapping Move to action index
        action_to_move: Dict mapping action index to Move
    """

    legal_moves: List[Move]
    action_mask: np.ndarray  # Shape: (MAX_ACTIONS,), dtype: bool
    move_to_action: Dict[Move, int]
    action_to_move: Dict[int, Move]


class ActionEncodingError(Exception):
    """Raised when action encoding fails."""

    pass


def encode_action_space(
    cards: List[Card],
    context: Optional[MoveContext] = None,
    max_actions: int = MAX_ACTIONS,
) -> ActionSpace:
    """Encode the legal moves into a fixed-size action space.

    Args:
        cards: List of cards in the player's hand
        context: Move context (previous move, is_tail_hand, etc.)
        max_actions: Maximum number of actions (default: MAX_ACTIONS)

    Returns:
        ActionSpace object with legal moves, mask, and mappings

    Raises:
        ActionEncodingError: If legal moves exceed max_actions
    """
    legal_moves = get_legal_moves(cards, context)

    # Overflow guard: assert legal moves don't exceed max_actions
    if len(legal_moves) > max_actions:
        raise ActionEncodingError(
            f"Legal moves ({len(legal_moves)}) exceed MAX_ACTIONS ({max_actions}). "
            f"Cannot encode action space without truncation."
        )

    # Create action mask (True = valid action)
    action_mask = np.zeros(max_actions, dtype=bool)
    action_mask[: len(legal_moves)] = True

    # Create move-to-action and action-to-move mappings
    move_to_action = {move: i for i, move in enumerate(legal_moves)}
    action_to_move = {i: move for i, move in enumerate(legal_moves)}

    return ActionSpace(
        legal_moves=legal_moves,
        action_mask=action_mask,
        move_to_action=move_to_action,
        action_to_move=action_to_move,
    )


def get_action_mask(
    cards: List[Card],
    context: Optional[MoveContext] = None,
    max_actions: int = MAX_ACTIONS,
) -> np.ndarray:
    """Get the action mask for the current state.

    Args:
        cards: List of cards in the player's hand
        context: Move context
        max_actions: Maximum number of actions

    Returns:
        Boolean numpy array of shape (max_actions,)
    """
    action_space = encode_action_space(cards, context, max_actions)
    return action_space.action_mask


def decode_action(
    action_idx: int,
    action_space: ActionSpace,
) -> Move:
    """Decode an action index to a Move.

    Args:
        action_idx: The action index to decode
        action_space: The ActionSpace object

    Returns:
        The corresponding Move object

    Raises:
        ActionEncodingError: If action index is invalid
    """
    if action_idx < 0 or action_idx >= MAX_ACTIONS:
        raise ActionEncodingError(f"Action index {action_idx} out of range [0, {MAX_ACTIONS})")

    if not action_space.action_mask[action_idx]:
        raise ActionEncodingError(f"Action index {action_idx} is masked (illegal)")

    if action_idx not in action_space.action_to_move:
        raise ActionEncodingError(f"Action index {action_idx} has no corresponding move")

    return action_space.action_to_move[action_idx]


def encode_move(
    move: Move,
    action_space: ActionSpace,
) -> int:
    """Encode a Move to an action index.

    Args:
        move: The Move object to encode
        action_space: The ActionSpace object

    Returns:
        The corresponding action index

    Raises:
        ActionEncodingError: If move is not in the action space
    """
    if move not in action_space.move_to_action:
        raise ActionEncodingError(f"Move {move} not found in action space")

    return action_space.move_to_action[move]


def get_pass_action(action_space: ActionSpace) -> int:
    """Get the action index for PASS move.

    Args:
        action_space: The ActionSpace object

    Returns:
        The action index for PASS

    Raises:
        ActionEncodingError: If PASS is not in the action space
    """
    return encode_move(PASS_MOVE, action_space)


def is_valid_action(action_idx: int, action_space: ActionSpace) -> bool:
    """Check if an action index is valid (not masked).

    Args:
        action_idx: The action index to check
        action_space: The ActionSpace object

    Returns:
        True if the action is valid (unmasked)
    """
    if action_idx < 0 or action_idx >= MAX_ACTIONS:
        return False
    return bool(action_space.action_mask[action_idx])


def sample_random_action(
    action_space: ActionSpace, rng: Optional[np.random.Generator] = None
) -> int:
    """Sample a random valid action.

    Args:
        action_space: The ActionSpace object
        rng: Optional numpy random generator

    Returns:
        A randomly sampled valid action index
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_indices = np.where(action_space.action_mask)[0]
    return int(rng.choice(valid_indices))


def count_legal_actions(action_space: ActionSpace) -> int:
    """Count the number of legal (unmasked) actions.

    Args:
        action_space: The ActionSpace object

    Returns:
        Number of legal actions
    """
    return int(np.sum(action_space.action_mask))


def encode_move_as_vector(move: Move, max_cards: int = 13) -> np.ndarray:
    """Encode a Move as a fixed-size vector for neural network input.

    This is a simple encoding that represents the move as:
    - [0]: Move type (0=PASS, 1=PLAY)
    - [1]: Is exemption (0/1)
    - [2]: Standard type for exemption (HandType value or 0)
    - [3]: Number of cards
    - [4:4+max_cards]: Card presence indicators

    Args:
        move: The Move to encode
        max_cards: Maximum cards in hand (default 13)

    Returns:
        Numpy array of shape (4 + max_cards,)
    """
    vec = np.zeros(4 + max_cards, dtype=np.float32)

    # Move type
    vec[0] = 1.0 if move.move_type == MoveType.PLAY else 0.0

    # Is exemption
    vec[1] = 1.0 if move.is_exemption else 0.0

    # Standard type (for exemption moves)
    vec[2] = float(move.standard_type.value) if move.standard_type is not None else 0.0

    # Number of cards
    vec[3] = float(len(move.cards))

    # We don't encode card details here - that's for observation encoding
    # This is just move metadata

    return vec


# Utility for tracking max observed legal moves (for tuning MAX_ACTIONS)
class LegalMoveStats:
    """Statistics tracker for legal move counts.

    Use this during testing to verify MAX_ACTIONS is sufficient.
    """

    def __init__(self):
        self.max_observed = 0
        self.total_samples = 0
        self.sum_moves = 0
        self.observations = []

    def record(self, num_moves: int) -> None:
        """Record a legal move count observation."""
        self.max_observed = max(self.max_observed, num_moves)
        self.total_samples += 1
        self.sum_moves += num_moves
        self.observations.append(num_moves)

    @property
    def average(self) -> float:
        """Average number of legal moves."""
        if self.total_samples == 0:
            return 0.0
        return self.sum_moves / self.total_samples

    @property
    def within_limit(self) -> bool:
        """Check if max observed is within MAX_ACTIONS."""
        return self.max_observed <= MAX_ACTIONS

    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"LegalMoveStats: max={self.max_observed}, avg={self.average:.1f}, "
            f"samples={self.total_samples}, within_limit={self.within_limit}"
        )


# Global stats tracker for tests
_global_stats = LegalMoveStats()


def record_legal_move_count(num_moves: int) -> None:
    """Record a legal move count to global stats."""
    _global_stats.record(num_moves)


def get_global_stats() -> LegalMoveStats:
    """Get the global legal move stats tracker."""
    return _global_stats


def reset_global_stats() -> None:
    """Reset the global stats tracker."""
    global _global_stats
    _global_stats = LegalMoveStats()

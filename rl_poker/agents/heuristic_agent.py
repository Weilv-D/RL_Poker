"""Heuristic-based policy agent.

This module provides a rule-based agent that uses domain knowledge
to make better decisions than random play. The heuristics are:

1. When leading (no current move to beat):
   - Play smallest single or smallest move to minimize hand size
   - Avoid breaking pairs if possible (preserve pairs for later)

2. When following (must beat current move):
   - Find smallest legal move that beats current
   - Prefer shorter moves (fewer cards) to conserve resources
   - Pass if no winning move is available

3. General preference order:
   - Prefer shedding (playing cards when safe)
   - Avoid playing high-value cards when not necessary
"""

from typing import List, Optional, Tuple
import numpy as np

from rl_poker.rules import (
    Card,
    Rank,
    HandType,
    get_rank_counts,
)
from rl_poker.moves.legal_moves import (
    Move,
    MoveType,
    MoveContext,
    get_legal_moves,
    PASS_MOVE,
)


class HeuristicAgent:
    """Agent that uses heuristic rules for decision-making.

    This agent applies domain-specific knowledge to make better
    decisions than random play.

    Attributes:
        name: Agent name for identification
        rng: Random number generator for tie-breaking
    """

    def __init__(self, seed: Optional[int] = None, name: str = "HeuristicAgent"):
        """Initialize the heuristic agent.

        Args:
            seed: Random seed for reproducibility (used for tie-breaking)
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
        """Select an action using heuristic rules.

        Args:
            observation: Dictionary containing game state
                Expected keys:
                - 'legal_moves': List[Move] - available moves
                - 'hand': List[Card] - current hand
                - 'current_move': Optional[Move] - move to beat (None if leading)
                Or action_space-based observation with:
                - 'action_space': ActionSpace object

            action_mask: Boolean array indicating legal actions
            rng: Optional random generator for tie-breaking

        Returns:
            Selected action index
        """
        if rng is None:
            rng = self.rng

        # Get legal moves from observation
        legal_moves = observation.get("legal_moves", [])
        if not legal_moves:
            # Fall back to action mask indices
            legal_indices = np.where(action_mask)[0]
            if len(legal_indices) == 0:
                raise ValueError("No legal actions available")
            return int(rng.choice(legal_indices))

        # Get current move to beat (None if leading)
        current_move = observation.get("current_move", None)

        # Get current hand
        hand = observation.get("hand", [])

        # Select best move using heuristics
        if current_move is None or current_move.move_type == MoveType.PASS:
            # Leading - choose how to play
            best_move = self._select_lead_move(legal_moves, hand, rng)
        else:
            # Following - must beat or pass
            best_move = self._select_follow_move(legal_moves, current_move, hand, rng)

        # Convert move to action index
        for i, move in enumerate(legal_moves):
            if self._moves_equal(move, best_move):
                return i

        # Fallback: return first legal action
        legal_indices = np.where(action_mask)[0]
        return int(legal_indices[0]) if len(legal_indices) > 0 else 0

    def _moves_equal(self, m1: Move, m2: Move) -> bool:
        """Check if two moves are equal."""
        return (
            m1.cards == m2.cards
            and m1.move_type == m2.move_type
            and m1.is_exemption == m2.is_exemption
        )

    def _select_lead_move(
        self, legal_moves: List[Move], hand: List[Card], rng: np.random.Generator
    ) -> Move:
        """Select a move when leading (no move to beat).

        Strategy:
        1. Play single cards first (lowest rank)
        2. Avoid breaking valuable combinations (pairs, triples)
        3. Play smallest move to shed cards
        """
        # Filter out PASS (we want to play when leading)
        play_moves = [m for m in legal_moves if m.move_type == MoveType.PLAY]

        if not play_moves:
            # No play moves available, must pass (shouldn't happen when leading)
            return PASS_MOVE

        # Categorize moves by type
        singles = [m for m in play_moves if m.hand and m.hand.hand_type == HandType.SINGLE]
        pairs = [m for m in play_moves if m.hand and m.hand.hand_type == HandType.PAIR]
        others = [m for m in play_moves if m not in singles and m not in pairs]

        # Analyze hand for pairs we want to preserve
        rank_counts = get_rank_counts(hand)
        paired_ranks = {r for r, c in rank_counts.items() if c >= 2}

        # Strategy: Play lowest single that doesn't break a pair
        safe_singles = [m for m in singles if m.hand and m.hand.main_rank not in paired_ranks]

        if safe_singles:
            # Play lowest safe single
            return min(safe_singles, key=lambda m: m.hand.main_rank if m.hand else Rank.TWO)

        if singles:
            # If no safe singles, play lowest single
            return min(singles, key=lambda m: m.hand.main_rank if m.hand else Rank.TWO)

        # No singles available - prefer smallest move type
        # Pairs are next best
        if pairs:
            return min(pairs, key=lambda m: m.hand.main_rank if m.hand else Rank.TWO)

        # Play smallest "other" move
        if others:
            # Prefer smaller card count, then lower rank
            return min(
                others, key=lambda m: (len(m.cards), m.hand.main_rank if m.hand else Rank.TWO)
            )

        # Fallback: random play move
        return play_moves[int(rng.integers(len(play_moves)))]

    def _select_follow_move(
        self,
        legal_moves: List[Move],
        current_move: Move,
        hand: List[Card],
        rng: np.random.Generator,
    ) -> Move:
        """Select a move when following (must beat current move or pass).

        Strategy:
        1. Find all moves that beat the current move
        2. Select the smallest winning move (minimize waste)
        3. If no winning moves, pass
        """
        # Filter to play moves that beat current
        beating_moves = [
            m
            for m in legal_moves
            if m.move_type == MoveType.PLAY and self._beats_current(m, current_move)
        ]

        if not beating_moves:
            # No winning moves - must pass
            return PASS_MOVE

        # Select smallest winning move
        # Prefer: fewer cards, then lower main rank
        best_move = min(
            beating_moves,
            key=lambda m: (
                len(m.cards),
                m.hand.main_rank if m.hand else self._get_exemption_main_rank(m),
            ),
        )

        return best_move

    def _beats_current(self, move: Move, current_move: Move) -> bool:
        """Check if move beats current_move.

        Since we're selecting from legal moves, this should always be true
        for non-PASS moves in the legal set. But we verify as safety.
        """
        if move.move_type == MoveType.PASS:
            return False

        if current_move is None or current_move.move_type == MoveType.PASS:
            return True  # Can play anything when leading

        # Current move is an exemption (no hand parsed)
        if current_move.is_exemption:
            current_rank = self._get_exemption_main_rank(current_move)
            if current_rank is None:
                return False
            if move.is_exemption:
                move_rank = self._get_exemption_main_rank(move)
                if move_rank is None:
                    return False
                return move.standard_type == current_move.standard_type and move_rank > current_rank
            if move.hand is None:
                return False
            return (
                move.hand.hand_type == current_move.standard_type
                and move.hand.main_rank > current_rank
            )

        # For exemption moves against standard current move
        if move.is_exemption:
            if current_move.hand is None:
                return False
            move_rank = self._get_exemption_main_rank(move)
            if move_rank is None:
                return False
            return (
                move.standard_type == current_move.hand.hand_type
                and move_rank > current_move.hand.main_rank
            )

        # For standard moves
        if move.hand is None or current_move.hand is None:
            return False

        if move.hand.hand_type != current_move.hand.hand_type:
            return False

        if len(move.cards) != len(current_move.cards):
            return False

        return move.hand.main_rank > current_move.hand.main_rank

    def _get_exemption_main_rank(self, move: Move) -> Optional[Rank]:
        """Get the main rank from an exemption move."""
        if not move.is_exemption:
            return None

        rank_counts = get_rank_counts(list(move.cards))

        if move.standard_type == HandType.THREE_PLUS_TWO:
            for rank, count in rank_counts.items():
                if count == 3:
                    return rank
        elif move.standard_type == HandType.FOUR_PLUS_THREE:
            for rank, count in rank_counts.items():
                if count == 4:
                    return rank

        return None

    def reset(self) -> None:
        """Reset agent state (no-op for heuristic agent)."""
        pass

    def set_seed(self, seed: int) -> None:
        """Set a new random seed.

        Args:
            seed: New random seed
        """
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"HeuristicAgent(seed={self._seed}, name={self.name!r})"


def create_heuristic_agent(seed: Optional[int] = None) -> HeuristicAgent:
    """Factory function to create a heuristic agent.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured HeuristicAgent instance
    """
    return HeuristicAgent(seed=seed)

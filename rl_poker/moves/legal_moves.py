"""Legal move enumeration with tail-hand exemptions.

This module provides:
- Enumeration of all legal moves from a hand
- Tail-hand exemption rules for 3+2 and 4+3 hands
- Strict follow-up matching when previous player used exemption

Tail-hand exemption rules (when player has last move/tail hand):
- 3+2: Can play 3+1 (4 cards) or 3+0 (3 cards) as shorthand
- 4+3: Can play 4+2 (6 cards), 4+1 (5 cards), or 4+0 (4 cards) as shorthand

Strict follow-up matching:
- If previous player used exemption (e.g., played 3+1 instead of 3+2),
  next player MUST play full standard size (5 cards for 3+2, 7 cards for 4+3)
- If next player cannot match full size, they MUST pass
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import combinations

from typing import List, Optional, Set, FrozenSet, Tuple

from rl_poker.rules import (
    Card,
    Rank,
    Hand,
    HandType,
    parse_hand,
    can_beat,
    get_rank_counts,
    is_valid_sequence_rank,
    are_consecutive,
    sort_cards,
)


class MoveType(IntEnum):
    """Types of moves that can be made."""

    PASS = auto()  # Pass turn
    PLAY = auto()  # Play cards


@dataclass(frozen=True)
class Move:
    """A move that can be made in the game.

    Attributes:
        move_type: PASS or PLAY
        cards: Frozenset of cards being played (empty for PASS)
        hand: The parsed Hand object (None for PASS)
        is_exemption: True if this move uses tail-hand exemption
        standard_type: For exemption moves, the standard hand type this represents
    """

    move_type: MoveType
    cards: FrozenSet[Card]
    hand: Optional[Hand] = None
    is_exemption: bool = False
    standard_type: Optional[HandType] = None

    def __hash__(self):
        return hash((self.move_type, self.cards, self.is_exemption, self.standard_type))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (
            self.move_type == other.move_type
            and self.cards == other.cards
            and self.is_exemption == other.is_exemption
            and self.standard_type == other.standard_type
        )

    def __str__(self) -> str:
        if self.move_type == MoveType.PASS:
            return "PASS"
        sorted_cards = sort_cards(list(self.cards))
        cards_str = " ".join(str(c) for c in sorted_cards)
        if self.is_exemption and self.standard_type is not None:
            return f"{self.standard_type.name}[exemption]({cards_str})"
        return f"{self.hand.hand_type.name if self.hand else 'UNKNOWN'}({cards_str})"

    @property
    def size(self) -> int:
        """Number of cards in the move."""
        return len(self.cards)


# Pass move singleton
PASS_MOVE = Move(move_type=MoveType.PASS, cards=frozenset())


@dataclass
class MoveContext:
    """Context for determining legal moves.

    Attributes:
        previous_move: The move to beat (None if leading)
        is_tail_hand: True if this is the player's last move (can use exemptions)
        previous_used_exemption: True if previous player used exemption
    """

    previous_move: Optional[Move] = None
    is_tail_hand: bool = False
    previous_used_exemption: bool = False


def enumerate_all_hands(cards: List[Card]) -> List[Move]:
    """Enumerate all valid hand combinations from the given cards.

    This generates all possible valid hands from the cards, including:
    - Singles
    - Pairs
    - Straights (5+ cards)
    - Consecutive pairs (3+ pairs)
    - Three plus two (3+2)
    - Four plus three (4+3)

    OPTIMIZED: Uses smart enumeration instead of brute-force combinations.

    Args:
        cards: List of cards in the player's hand

    Returns:
        List of Move objects representing all valid plays
    """
    moves = []
    n = len(cards)
    if n == 0:
        return moves

    # Group cards by rank for efficient enumeration
    rank_counts = get_rank_counts(cards)
    cards_by_rank = {}
    for card in cards:
        if card.rank not in cards_by_rank:
            cards_by_rank[card.rank] = []
        cards_by_rank[card.rank].append(card)

    # === Singles ===
    for card in cards:
        hand = Hand(hand_type=HandType.SINGLE, cards=frozenset([card]), main_rank=card.rank)
        moves.append(Move(move_type=MoveType.PLAY, cards=frozenset([card]), hand=hand))

    # === Pairs ===
    for rank, rank_cards in cards_by_rank.items():
        if len(rank_cards) >= 2:
            for combo in combinations(rank_cards, 2):
                hand = Hand(hand_type=HandType.PAIR, cards=frozenset(combo), main_rank=rank)
                moves.append(Move(move_type=MoveType.PLAY, cards=frozenset(combo), hand=hand))

    # === Straights (5+ consecutive cards, 3-K only) ===
    # Only enumerate valid sequence ranks
    sequence_ranks = [r for r in rank_counts.keys() if is_valid_sequence_rank(r)]
    if len(sequence_ranks) >= 5:
        sequence_ranks_sorted = sorted(sequence_ranks, key=lambda r: r.value)
        # Find all consecutive runs
        for start_idx in range(len(sequence_ranks_sorted)):
            for end_idx in range(start_idx + 4, len(sequence_ranks_sorted)):
                run_ranks = sequence_ranks_sorted[start_idx : end_idx + 1]
                # Check if consecutive
                if all(
                    run_ranks[i].value + 1 == run_ranks[i + 1].value
                    for i in range(len(run_ranks) - 1)
                ):
                    # Generate all combinations: pick 1 card from each rank
                    _add_straight_combos(moves, cards_by_rank, run_ranks)

    # === Consecutive Pairs (3+ pairs, 3-K only) ===
    pair_ranks = [
        r for r in rank_counts.keys() if rank_counts[r] >= 2 and is_valid_sequence_rank(r)
    ]
    if len(pair_ranks) >= 3:
        pair_ranks_sorted = sorted(pair_ranks, key=lambda r: r.value)
        for start_idx in range(len(pair_ranks_sorted)):
            for end_idx in range(start_idx + 2, len(pair_ranks_sorted)):
                run_ranks = pair_ranks_sorted[start_idx : end_idx + 1]
                if all(
                    run_ranks[i].value + 1 == run_ranks[i + 1].value
                    for i in range(len(run_ranks) - 1)
                ):
                    _add_consecutive_pair_combos(moves, cards_by_rank, run_ranks)

    # === Three Plus Two (3+2) ===
    triple_ranks = [r for r in rank_counts.keys() if rank_counts[r] >= 3]
    for main_rank in triple_ranks:
        main_cards = cards_by_rank[main_rank]
        other_cards = [c for c in cards if c.rank != main_rank]
        if len(other_cards) >= 2:
            for triple in combinations(main_cards, 3):
                for kicker in combinations(other_cards, 2):
                    all_cards = frozenset(triple) | frozenset(kicker)
                    kicker_ranks = [c.rank for c in kicker]
                    secondary = max(kicker_ranks, key=lambda r: r.value)
                    hand = Hand(
                        hand_type=HandType.THREE_PLUS_TWO,
                        cards=all_cards,
                        main_rank=main_rank,
                        secondary_rank=secondary,
                    )
                    moves.append(Move(move_type=MoveType.PLAY, cards=all_cards, hand=hand))

    # === Four Plus Three (4+3) ===
    quad_ranks = [r for r in rank_counts.keys() if rank_counts[r] >= 4]
    for main_rank in quad_ranks:
        main_cards = cards_by_rank[main_rank]
        other_cards = [c for c in cards if c.rank != main_rank]
        if len(other_cards) >= 3:
            for quad in combinations(main_cards, 4):
                for kicker in combinations(other_cards, 3):
                    all_cards = frozenset(quad) | frozenset(kicker)
                    kicker_ranks = [c.rank for c in kicker]
                    secondary = max(kicker_ranks, key=lambda r: r.value)
                    hand = Hand(
                        hand_type=HandType.FOUR_PLUS_THREE,
                        cards=all_cards,
                        main_rank=main_rank,
                        secondary_rank=secondary,
                    )
                    moves.append(Move(move_type=MoveType.PLAY, cards=all_cards, hand=hand))

    return moves


def _add_straight_combos(moves: List[Move], cards_by_rank: dict, run_ranks: List[Rank]):
    """Helper to add all straight combinations for a run of consecutive ranks."""
    from itertools import product

    card_options = [cards_by_rank[r] for r in run_ranks]
    for combo in product(*card_options):
        all_cards = frozenset(combo)
        highest_rank = run_ranks[-1]
        hand = Hand(hand_type=HandType.STRAIGHT, cards=all_cards, main_rank=highest_rank)
        moves.append(Move(move_type=MoveType.PLAY, cards=all_cards, hand=hand))


def _add_consecutive_pair_combos(moves: List[Move], cards_by_rank: dict, run_ranks: List[Rank]):
    """Helper to add all consecutive pair combinations for a run of consecutive ranks."""
    from itertools import product

    # For each rank, need to pick 2 cards
    pair_options = []
    for r in run_ranks:
        rank_cards = cards_by_rank[r]
        pair_options.append(list(combinations(rank_cards, 2)))

    for combo in product(*pair_options):
        all_cards = frozenset(c for pair in combo for c in pair)
        highest_rank = run_ranks[-1]
        hand = Hand(hand_type=HandType.CONSECUTIVE_PAIRS, cards=all_cards, main_rank=highest_rank)
        moves.append(Move(move_type=MoveType.PLAY, cards=all_cards, hand=hand))


def enumerate_exemption_hands(cards: List[Card]) -> List[Move]:
    """Enumerate exemption hands (3+1, 3+0, 4+2, 4+1, 4+0).

    These are only valid when is_tail_hand is True.

    Args:
        cards: List of cards in the player's hand

    Returns:
        List of Move objects representing exemption plays
    """
    moves = []
    rank_counts = get_rank_counts(cards)
    cards_by_rank = {}
    for card in cards:
        if card.rank not in cards_by_rank:
            cards_by_rank[card.rank] = []
        cards_by_rank[card.rank].append(card)

    # Find ranks with 3+ cards (for 3+N exemptions)
    for rank, count in rank_counts.items():
        if count >= 3:
            rank_cards = cards_by_rank[rank]

            # 3+0 (just the triple)
            for triple in combinations(rank_cards, 3):
                triple_set = frozenset(triple)
                moves.append(
                    Move(
                        move_type=MoveType.PLAY,
                        cards=triple_set,
                        hand=None,
                        is_exemption=True,
                        standard_type=HandType.THREE_PLUS_TWO,
                    )
                )

            # 3+1 (triple + any 1 card)
            other_cards = [c for c in cards if c.rank != rank]
            for triple in combinations(rank_cards, 3):
                for kicker in other_cards:
                    combo_set = frozenset(list(triple) + [kicker])
                    moves.append(
                        Move(
                            move_type=MoveType.PLAY,
                            cards=combo_set,
                            hand=None,
                            is_exemption=True,
                            standard_type=HandType.THREE_PLUS_TWO,
                        )
                    )

    # Find ranks with 4 cards (for 4+N exemptions)
    for rank, count in rank_counts.items():
        if count == 4:
            quad_cards = cards_by_rank[rank]
            other_cards = [c for c in cards if c.rank != rank]

            # 4+0 (just the quad)
            quad_set = frozenset(quad_cards)
            moves.append(
                Move(
                    move_type=MoveType.PLAY,
                    cards=quad_set,
                    hand=None,
                    is_exemption=True,
                    standard_type=HandType.FOUR_PLUS_THREE,
                )
            )

            # 4+1 (quad + any 1 card)
            for kicker in other_cards:
                combo_set = frozenset(quad_cards + [kicker])
                moves.append(
                    Move(
                        move_type=MoveType.PLAY,
                        cards=combo_set,
                        hand=None,
                        is_exemption=True,
                        standard_type=HandType.FOUR_PLUS_THREE,
                    )
                )

            # 4+2 (quad + any 2 cards)
            if len(other_cards) >= 2:
                for kickers in combinations(other_cards, 2):
                    combo_set = frozenset(quad_cards + list(kickers))
                    moves.append(
                        Move(
                            move_type=MoveType.PLAY,
                            cards=combo_set,
                            hand=None,
                            is_exemption=True,
                            standard_type=HandType.FOUR_PLUS_THREE,
                        )
                    )

    return moves


def _get_main_rank_from_exemption(move: Move) -> Optional[Rank]:
    """Extract the main rank from an exemption move.

    For 3+N exemptions, this is the rank of the triple.
    For 4+N exemptions, this is the rank of the quad.
    """
    if not move.is_exemption:
        return None

    rank_counts = get_rank_counts(list(move.cards))

    if move.standard_type == HandType.THREE_PLUS_TWO:
        # Find the rank with 3 cards
        for rank, count in rank_counts.items():
            if count == 3:
                return rank
    elif move.standard_type == HandType.FOUR_PLUS_THREE:
        # Find the rank with 4 cards
        for rank, count in rank_counts.items():
            if count == 4:
                return rank

    return None


def get_legal_moves(
    cards: List[Card],
    context: Optional[MoveContext] = None,
) -> List[Move]:
    """Get all legal moves given a hand and context.

    Args:
        cards: List of cards in the player's hand
        context: Move context (previous move, is_tail_hand, etc.)

    Returns:
        List of legal Move objects (always includes PASS if not leading)
    """
    if not cards:
        return []

    if context is None:
        context = MoveContext()

    # Case 1: Leading (no previous move to beat)
    if context.previous_move is None or context.previous_move.move_type == MoveType.PASS:
        moves = enumerate_all_hands(cards)
        if context.is_tail_hand:
            moves.extend(enumerate_exemption_hands(cards))
        # Remove duplicates (frozenset equality handles this)
        seen = set()
        unique_moves = []
        for move in moves:
            key = (move.cards, move.is_exemption, move.standard_type)
            if key not in seen:
                seen.add(key)
                unique_moves.append(move)
        return unique_moves

    # Case 2: Previous player used exemption - must match FULL standard size
    if context.previous_used_exemption:
        return _get_legal_moves_after_exemption(cards, context)

    # Case 3: Normal follow - must beat previous move with same type/size
    return _get_legal_moves_normal_follow(cards, context)


def _get_legal_moves_after_exemption(
    cards: List[Card],
    context: MoveContext,
) -> List[Move]:
    """Get legal moves when previous player used exemption.

    Next player MUST play full standard size or PASS.
    """
    prev_move = context.previous_move
    if prev_move is None:
        return [PASS_MOVE]

    prev_main_rank = _get_main_rank_from_exemption(prev_move)
    if prev_main_rank is None:
        return [PASS_MOVE]

    # Generate all standard hands of the required type
    all_moves = enumerate_all_hands(cards)

    # Filter to matching type with higher main rank
    valid_moves = []
    for move in all_moves:
        if move.hand is None:
            continue

        # Must be same standard type
        if prev_move.standard_type == HandType.THREE_PLUS_TWO:
            if move.hand.hand_type != HandType.THREE_PLUS_TWO:
                continue
            # Standard 3+2 is 5 cards
            if move.size != 5:
                continue
            if move.hand.main_rank > prev_main_rank:
                valid_moves.append(move)

        elif prev_move.standard_type == HandType.FOUR_PLUS_THREE:
            if move.hand.hand_type != HandType.FOUR_PLUS_THREE:
                continue
            # Standard 4+3 is 7 cards
            if move.size != 7:
                continue
            if move.hand.main_rank > prev_main_rank:
                valid_moves.append(move)

    # Always include PASS
    valid_moves.append(PASS_MOVE)

    return valid_moves


def _get_legal_moves_normal_follow(
    cards: List[Card],
    context: MoveContext,
) -> List[Move]:
    """Get legal moves for normal follow (no exemption in previous).

    Must beat previous move with same type and size.
    """
    prev_move = context.previous_move
    if prev_move is None or prev_move.hand is None:
        return [PASS_MOVE]

    prev_hand = prev_move.hand
    prev_type = prev_hand.hand_type
    prev_size = prev_hand.size

    # Generate all hands
    all_moves = enumerate_all_hands(cards)

    # Also allow exemptions if tail hand
    if context.is_tail_hand:
        all_moves.extend(enumerate_exemption_hands(cards))

    # Filter to matching type/size that beats previous
    valid_moves = []
    for move in all_moves:
        if move.is_exemption:
            # Exemption moves: check if standard type matches and main rank beats
            if move.standard_type != prev_type:
                continue
            move_main_rank = _get_main_rank_from_exemption(move)
            if move_main_rank is None:
                continue
            if move_main_rank > prev_hand.main_rank:
                valid_moves.append(move)
        elif move.hand is not None:
            # Standard moves: must match type and size
            if move.hand.hand_type != prev_type:
                continue
            if move.size != prev_size:
                continue
            if can_beat(move.hand, prev_hand):
                valid_moves.append(move)

    # Always include PASS
    valid_moves.append(PASS_MOVE)

    # Remove duplicates
    seen = set()
    unique_moves = []
    for move in valid_moves:
        key = (move.cards, move.is_exemption, move.standard_type)
        if key not in seen:
            seen.add(key)
            unique_moves.append(move)

    return unique_moves


def can_play_move(
    cards: List[Card],
    move: Move,
    context: Optional[MoveContext] = None,
) -> bool:
    """Check if a specific move can be legally played.

    Args:
        cards: List of cards in the player's hand
        move: The move to check
        context: Move context

    Returns:
        True if the move is legal
    """
    # Check cards are in hand
    if not move.cards.issubset(set(cards)):
        return False

    legal_moves = get_legal_moves(cards, context)
    return any(m.cards == move.cards and m.is_exemption == move.is_exemption for m in legal_moves)


def get_move_count(
    cards: List[Card],
    context: Optional[MoveContext] = None,
) -> int:
    """Get the count of legal moves.

    Args:
        cards: List of cards in the player's hand
        context: Move context

    Returns:
        Number of legal moves
    """
    return len(get_legal_moves(cards, context))


def must_pass(
    cards: List[Card],
    context: Optional[MoveContext] = None,
) -> bool:
    """Check if the player must pass (no valid plays).

    Args:
        cards: List of cards in the player's hand
        context: Move context

    Returns:
        True if the only legal move is PASS
    """
    legal_moves = get_legal_moves(cards, context)
    # If no moves or only PASS, must pass
    return len(legal_moves) == 0 or (
        len(legal_moves) == 1 and legal_moves[0].move_type == MoveType.PASS
    )

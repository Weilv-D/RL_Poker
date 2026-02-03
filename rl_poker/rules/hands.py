"""Hand type detection, parsing, and comparison.

Hand types supported:
- Single: any single card
- Pair: two cards of the same rank
- Straight: 5+ consecutive cards (3-K only, no A/2)
- Consecutive pairs: 3+ consecutive pairs (3-K only, no A/2)
- Three plus two (3+2): three cards of same rank + any 2 cards
- Four plus three (4+3): four cards of same rank + any 3 cards

Comparison rules:
- Same hand type only (except tail-hand exemptions, handled in moves)
- 3+2 and 4+3: compare by main rank (the 3 or 4 cards)
- Sequences (straight, consecutive pairs): compare by highest rank
- Single and pair: compare by rank
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional, Tuple, FrozenSet

from .ranks import (
    Card,
    Rank,
    Suit,
    get_rank_counts,
    is_valid_sequence_rank,
    are_consecutive,
    SEQUENCE_RANKS,
    MIN_SEQUENCE_RANK,
    MAX_SEQUENCE_RANK,
    sort_cards,
    RANK_SYMBOLS,
)


class HandType(IntEnum):
    """Types of hands that can be played."""

    SINGLE = auto()
    PAIR = auto()
    STRAIGHT = auto()  # 5+ consecutive cards
    CONSECUTIVE_PAIRS = auto()  # 3+ consecutive pairs
    THREE_PLUS_TWO = auto()  # 3 of same rank + any 2
    FOUR_PLUS_THREE = auto()  # 4 of same rank + any 3


@dataclass(frozen=True)
class Hand:
    """A classified poker hand.

    Attributes:
        hand_type: The type of hand
        cards: Frozenset of cards in the hand
        main_rank: The primary rank for comparison (main cards in 3+2, 4+3; highest in sequences)
        secondary_rank: Secondary rank (for kickers in 3+2, 4+3), or None
    """

    hand_type: HandType
    cards: FrozenSet[Card]
    main_rank: Rank
    secondary_rank: Optional[Rank] = None

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        sorted_cards = sort_cards(list(self.cards))
        cards_str = " ".join(str(c) for c in sorted_cards)
        return f"{self.hand_type.name}({cards_str})"

    @property
    def size(self) -> int:
        """Number of cards in the hand."""
        return len(self.cards)


class HandParseError(Exception):
    """Raised when cards cannot form a valid hand."""

    pass


def parse_hand(cards: List[Card]) -> Optional[Hand]:
    """Attempt to parse a list of cards into a valid hand.

    Args:
        cards: List of Card objects

    Returns:
        Hand object if valid, None if not a valid hand type

    Note:
        This function returns None for invalid hands rather than raising,
        to make it easy to check if a hand is valid.
    """
    if not cards:
        return None

    n = len(cards)
    rank_counts = get_rank_counts(cards)

    # Single card
    if n == 1:
        return Hand(hand_type=HandType.SINGLE, cards=frozenset(cards), main_rank=cards[0].rank)

    # Pair - exactly 2 cards of same rank
    if n == 2:
        if len(rank_counts) == 1:
            rank = list(rank_counts.keys())[0]
            return Hand(hand_type=HandType.PAIR, cards=frozenset(cards), main_rank=rank)
        return None

    # Check for 3+2 (5 cards: 3 of one rank + 2 of any)
    if n == 5:
        hand = _try_parse_three_plus_two(cards, rank_counts)
        if hand:
            return hand

    # Check for 4+3 (7 cards: 4 of one rank + 3 of any)
    if n == 7:
        hand = _try_parse_four_plus_three(cards, rank_counts)
        if hand:
            return hand

    # Check for straight (5+ consecutive, 3-K only)
    if n >= 5:
        hand = _try_parse_straight(cards, rank_counts)
        if hand:
            return hand

    # Check for consecutive pairs (6+ cards = 3+ pairs consecutive, 3-K only)
    if n >= 6 and n % 2 == 0:
        hand = _try_parse_consecutive_pairs(cards, rank_counts)
        if hand:
            return hand

    return None


def _try_parse_three_plus_two(cards: List[Card], rank_counts: dict) -> Optional[Hand]:
    """Try to parse cards as 3+2 (three of a kind + any 2 cards).

    The "kicker" 2 cards can be any ranks - they don't need to be a pair.
    """
    if len(cards) != 5:
        return None

    # Find the rank with 3 cards
    main_rank = None
    for rank, count in rank_counts.items():
        if count == 3:
            main_rank = rank
            break

    if main_rank is None:
        return None

    # Verify remaining 2 cards exist (any combination is fine)
    remaining_count = sum(c for r, c in rank_counts.items() if r != main_rank)
    if remaining_count != 2:
        return None

    # Get secondary rank (highest of the kickers)
    kicker_ranks = [r for r in rank_counts.keys() if r != main_rank]
    secondary_rank = max(kicker_ranks) if kicker_ranks else None

    return Hand(
        hand_type=HandType.THREE_PLUS_TWO,
        cards=frozenset(cards),
        main_rank=main_rank,
        secondary_rank=secondary_rank,
    )


def _try_parse_four_plus_three(cards: List[Card], rank_counts: dict) -> Optional[Hand]:
    """Try to parse cards as 4+3 (four of a kind + any 3 cards).

    The "kicker" 3 cards can be any ranks - they don't need to match.
    """
    if len(cards) != 7:
        return None

    # Find the rank with 4 cards
    main_rank = None
    for rank, count in rank_counts.items():
        if count == 4:
            main_rank = rank
            break

    if main_rank is None:
        return None

    # Verify remaining 3 cards exist (any combination is fine)
    remaining_count = sum(c for r, c in rank_counts.items() if r != main_rank)
    if remaining_count != 3:
        return None

    # Get secondary rank (highest of the kickers)
    kicker_ranks = [r for r in rank_counts.keys() if r != main_rank]
    secondary_rank = max(kicker_ranks) if kicker_ranks else None

    return Hand(
        hand_type=HandType.FOUR_PLUS_THREE,
        cards=frozenset(cards),
        main_rank=main_rank,
        secondary_rank=secondary_rank,
    )


def _try_parse_straight(cards: List[Card], rank_counts: dict) -> Optional[Hand]:
    """Try to parse cards as a straight (5+ consecutive, 3-K only).

    Rules:
    - Must have exactly 1 card of each rank
    - All ranks must be in SEQUENCE_RANKS (3-K)
    - Ranks must be consecutive
    """
    if len(cards) < 5:
        return None

    # Each rank must appear exactly once
    if any(count != 1 for count in rank_counts.values()):
        return None

    ranks = sorted(rank_counts.keys())

    # All ranks must be valid for sequences (3-K only, no A/2)
    if not all(is_valid_sequence_rank(r) for r in ranks):
        return None

    # Check if consecutive
    if not are_consecutive(ranks):
        return None

    # Main rank is the highest rank in the straight
    main_rank = max(ranks)

    return Hand(hand_type=HandType.STRAIGHT, cards=frozenset(cards), main_rank=main_rank)


def _try_parse_consecutive_pairs(cards: List[Card], rank_counts: dict) -> Optional[Hand]:
    """Try to parse cards as consecutive pairs (3+ consecutive pairs, 3-K only).

    Rules:
    - Must have 6+ cards (3+ pairs)
    - Each rank must appear exactly twice
    - All ranks must be in SEQUENCE_RANKS (3-K)
    - Ranks must be consecutive
    """
    if len(cards) < 6 or len(cards) % 2 != 0:
        return None

    # Each rank must appear exactly twice
    if any(count != 2 for count in rank_counts.values()):
        return None

    # Need at least 3 pairs
    if len(rank_counts) < 3:
        return None

    ranks = sorted(rank_counts.keys())

    # All ranks must be valid for sequences (3-K only, no A/2)
    if not all(is_valid_sequence_rank(r) for r in ranks):
        return None

    # Check if consecutive
    if not are_consecutive(ranks):
        return None

    # Main rank is the highest rank in the consecutive pairs
    main_rank = max(ranks)

    return Hand(hand_type=HandType.CONSECUTIVE_PAIRS, cards=frozenset(cards), main_rank=main_rank)


def is_valid_hand(cards: List[Card]) -> bool:
    """Check if a list of cards forms a valid hand.

    Args:
        cards: List of Card objects

    Returns:
        True if cards form a valid hand type
    """
    return parse_hand(cards) is not None


def can_beat(hand1: Hand, hand2: Hand) -> bool:
    """Check if hand1 can beat hand2.

    Args:
        hand1: The hand attempting to beat
        hand2: The hand to beat

    Returns:
        True if hand1 beats hand2

    Note:
        Hands must be of the same type and size to be compared.
        This function does NOT handle tail-hand exemptions (those are move-level rules).
    """
    # Must be same hand type
    if hand1.hand_type != hand2.hand_type:
        return False

    # Must be same size (important for straights and consecutive pairs)
    if hand1.size != hand2.size:
        return False

    # Compare by main rank
    # For 3+2 and 4+3: main_rank is the 3 or 4 of a kind
    # For sequences: main_rank is the highest card
    # For single/pair: main_rank is the card rank
    return hand1.main_rank > hand2.main_rank


def compare_hands(hand1: Hand, hand2: Hand) -> int:
    """Compare two hands.

    Args:
        hand1: First hand
        hand2: Second hand

    Returns:
        Positive if hand1 > hand2
        Negative if hand1 < hand2
        Zero if equal or incomparable (different types/sizes)

    Note:
        Returns 0 for hands that cannot be compared (different types or sizes).
    """
    # Must be same hand type
    if hand1.hand_type != hand2.hand_type:
        return 0

    # Must be same size
    if hand1.size != hand2.size:
        return 0

    return int(hand1.main_rank) - int(hand2.main_rank)


def get_valid_hand_types() -> List[HandType]:
    """Get all valid hand types."""
    return list(HandType)


def describe_hand_requirements() -> dict:
    """Get a description of requirements for each hand type.

    Returns:
        Dict mapping HandType to description string
    """
    return {
        HandType.SINGLE: "Any single card",
        HandType.PAIR: "Two cards of the same rank",
        HandType.STRAIGHT: "5+ consecutive cards (3-K only, no A/2)",
        HandType.CONSECUTIVE_PAIRS: "3+ consecutive pairs (3-K only, no A/2)",
        HandType.THREE_PLUS_TWO: "Three cards of same rank + any 2 cards",
        HandType.FOUR_PLUS_THREE: "Four cards of same rank + any 3 cards",
    }


# Helper functions for creating hands for testing


def make_cards_from_ranks(ranks: List[Rank], suits: Optional[List[Suit]] = None) -> List[Card]:
    """Create cards from a list of ranks and optional suits.

    If suits not provided, cycles through suits for variety.

    Args:
        ranks: List of Rank values
        suits: Optional list of Suit values (must match length of ranks if provided)

    Returns:
        List of Card objects
    """
    if suits is None:
        suits = [Suit(i % 4) for i in range(len(ranks))]

    if len(ranks) != len(suits):
        raise ValueError("ranks and suits must have same length")

    return [Card(rank=r, suit=s) for r, s in zip(ranks, suits)]


def make_cards_from_string(s: str) -> List[Card]:
    """Parse cards from a string like "3H 4D 5C 6S 7H".

    Args:
        s: Space-separated card strings

    Returns:
        List of Card objects
    """
    return [Card.from_string(cs) for cs in s.split()]

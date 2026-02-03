"""Card rank definitions and utilities.

Rank order (high to low): 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3

This module provides:
- Rank constants and ordering
- Card representation
- Suit definitions
- Comparison utilities
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple


class Rank(IntEnum):
    """Card ranks ordered by strength (higher value = stronger rank).

    Order: 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
    """

    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12  # Highest rank


class Suit(IntEnum):
    """Card suits. Order matters for identifying the lead player (heart 3)."""

    HEART = 0
    DIAMOND = 1
    CLUB = 2
    SPADE = 3


# Rank symbols for display
RANK_SYMBOLS = {
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.TWO: "2",
}

# Suit symbols for display
SUIT_SYMBOLS = {
    Suit.HEART: "♥",
    Suit.DIAMOND: "♦",
    Suit.CLUB: "♣",
    Suit.SPADE: "♠",
}

# Symbol to rank mapping (for parsing)
SYMBOL_TO_RANK = {v: k for k, v in RANK_SYMBOLS.items()}

# Ranks allowed in sequences (3-K only, no A or 2)
SEQUENCE_RANKS = frozenset(
    [
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
    ]
)

# Minimum and maximum sequence ranks
MIN_SEQUENCE_RANK = Rank.THREE
MAX_SEQUENCE_RANK = Rank.KING


@dataclass(frozen=True, order=True)
class Card:
    """A playing card with rank and suit.

    Cards are ordered by rank first (for sorting hands), then by suit.
    Immutable and hashable for use in sets.
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __repr__(self) -> str:
        return f"Card({RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]})"

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse a card from string like '3♥' or '10♠'.

        Args:
            s: Card string in format "RANK+SUIT"

        Returns:
            Card object

        Raises:
            ValueError: If string cannot be parsed
        """
        # Handle suits
        suit_char = s[-1]
        rank_str = s[:-1]

        suit_map = {v: k for k, v in SUIT_SYMBOLS.items()}
        suit_map.update({"H": Suit.HEART, "D": Suit.DIAMOND, "C": Suit.CLUB, "S": Suit.SPADE})
        suit_map.update({"h": Suit.HEART, "d": Suit.DIAMOND, "c": Suit.CLUB, "s": Suit.SPADE})

        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit character: {suit_char}")
        suit = suit_map[suit_char]

        if rank_str not in SYMBOL_TO_RANK:
            raise ValueError(f"Invalid rank: {rank_str}")
        rank = SYMBOL_TO_RANK[rank_str]

        return cls(rank=rank, suit=suit)


def is_valid_sequence_rank(rank: Rank) -> bool:
    """Check if a rank can be part of a sequence (straight or consecutive pairs).

    Only 3-K can be in sequences. A and 2 are strictly prohibited.
    """
    return rank in SEQUENCE_RANKS


def are_consecutive(ranks: List[Rank]) -> bool:
    """Check if a sorted list of unique ranks are consecutive.

    Args:
        ranks: List of ranks (should be sorted and unique)

    Returns:
        True if all ranks are consecutive
    """
    if len(ranks) < 2:
        return True

    for i in range(1, len(ranks)):
        if int(ranks[i]) - int(ranks[i - 1]) != 1:
            return False
    return True


def get_rank_counts(cards: List[Card]) -> dict:
    """Count occurrences of each rank in a list of cards.

    Args:
        cards: List of Card objects

    Returns:
        Dict mapping Rank to count
    """
    counts = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def create_standard_deck() -> List[Card]:
    """Create a standard 52-card deck.

    Returns:
        List of 52 Card objects (13 ranks × 4 suits)
    """
    deck = []
    for rank in Rank:
        for suit in Suit:
            deck.append(Card(rank=rank, suit=suit))
    return deck


def get_heart_three() -> Card:
    """Get the Heart 3 card (used to determine starting player)."""
    return Card(rank=Rank.THREE, suit=Suit.HEART)


def sort_cards(cards: List[Card]) -> List[Card]:
    """Sort cards by rank (ascending), then by suit.

    Args:
        cards: List of Card objects

    Returns:
        New sorted list of cards
    """
    return sorted(cards)


def compare_ranks(rank1: Rank, rank2: Rank) -> int:
    """Compare two ranks.

    Args:
        rank1: First rank
        rank2: Second rank

    Returns:
        Positive if rank1 > rank2, negative if rank1 < rank2, zero if equal
    """
    return int(rank1) - int(rank2)

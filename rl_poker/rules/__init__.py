"""Poker rules implementations.

This module provides:
- Card and rank definitions (ranks.py)
- Hand type detection and comparison (hands.py)
"""

from .ranks import (
    Rank,
    Suit,
    Card,
    RANK_SYMBOLS,
    SUIT_SYMBOLS,
    SEQUENCE_RANKS,
    MIN_SEQUENCE_RANK,
    MAX_SEQUENCE_RANK,
    is_valid_sequence_rank,
    are_consecutive,
    get_rank_counts,
    create_standard_deck,
    get_heart_three,
    sort_cards,
    compare_ranks,
)

from .hands import (
    HandType,
    Hand,
    HandParseError,
    parse_hand,
    is_valid_hand,
    can_beat,
    compare_hands,
    get_valid_hand_types,
    describe_hand_requirements,
    make_cards_from_ranks,
    make_cards_from_string,
)

__all__ = [
    # Ranks
    "Rank",
    "Suit",
    "Card",
    "RANK_SYMBOLS",
    "SUIT_SYMBOLS",
    "SEQUENCE_RANKS",
    "MIN_SEQUENCE_RANK",
    "MAX_SEQUENCE_RANK",
    "is_valid_sequence_rank",
    "are_consecutive",
    "get_rank_counts",
    "create_standard_deck",
    "get_heart_three",
    "sort_cards",
    "compare_ranks",
    # Hands
    "HandType",
    "Hand",
    "HandParseError",
    "parse_hand",
    "is_valid_hand",
    "can_beat",
    "compare_hands",
    "get_valid_hand_types",
    "describe_hand_requirements",
    "make_cards_from_ranks",
    "make_cards_from_string",
]

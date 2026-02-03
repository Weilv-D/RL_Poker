"""Tests for the rules engine (ranks and hands).

Test coverage:
- Rank ordering: 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
- Hand type detection: single, pair, straight, consecutive pairs, 3+2, 4+3
- Sequence constraints: 3-K only, no A/2 in straights or consecutive pairs
- Comparison logic: main rank for 3+2/4+3, highest rank for sequences
"""

import pytest
from rl_poker.rules import (
    Rank,
    Suit,
    Card,
    SEQUENCE_RANKS,
    is_valid_sequence_rank,
    are_consecutive,
    get_rank_counts,
    create_standard_deck,
    get_heart_three,
    sort_cards,
    compare_ranks,
    HandType,
    Hand,
    parse_hand,
    is_valid_hand,
    can_beat,
    compare_hands,
    make_cards_from_ranks,
    make_cards_from_string,
)


class TestRankOrdering:
    """Test that rank ordering is correct: 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3"""

    def test_rank_order_two_is_highest(self):
        assert Rank.TWO > Rank.ACE
        assert Rank.TWO > Rank.KING
        assert Rank.TWO > Rank.THREE

    def test_rank_order_ace_second_highest(self):
        assert Rank.ACE > Rank.KING
        assert Rank.ACE > Rank.QUEEN
        assert Rank.ACE < Rank.TWO

    def test_rank_order_three_is_lowest(self):
        assert Rank.THREE < Rank.FOUR
        assert Rank.THREE < Rank.ACE
        assert Rank.THREE < Rank.TWO

    def test_complete_rank_ordering(self):
        expected_order = [
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
            Rank.ACE,
            Rank.TWO,
        ]
        for i in range(len(expected_order) - 1):
            assert expected_order[i] < expected_order[i + 1]

    def test_compare_ranks_function(self):
        assert compare_ranks(Rank.TWO, Rank.ACE) > 0
        assert compare_ranks(Rank.THREE, Rank.FOUR) < 0
        assert compare_ranks(Rank.KING, Rank.KING) == 0


class TestSequenceConstraints:
    """Test that sequences only allow 3-K, strictly prohibiting A and 2."""

    def test_sequence_ranks_excludes_ace(self):
        assert Rank.ACE not in SEQUENCE_RANKS
        assert not is_valid_sequence_rank(Rank.ACE)

    def test_sequence_ranks_excludes_two(self):
        assert Rank.TWO not in SEQUENCE_RANKS
        assert not is_valid_sequence_rank(Rank.TWO)

    def test_sequence_ranks_includes_three_to_king(self):
        valid_ranks = [
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
        for rank in valid_ranks:
            assert rank in SEQUENCE_RANKS
            assert is_valid_sequence_rank(rank)

    def test_are_consecutive(self):
        assert are_consecutive([Rank.THREE, Rank.FOUR, Rank.FIVE])
        assert are_consecutive([Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING])
        assert not are_consecutive([Rank.THREE, Rank.FIVE])
        assert not are_consecutive([Rank.THREE, Rank.FOUR, Rank.SIX])


class TestCardBasics:
    """Test Card creation and utilities."""

    def test_card_creation(self):
        card = Card(rank=Rank.THREE, suit=Suit.HEART)
        assert card.rank == Rank.THREE
        assert card.suit == Suit.HEART

    def test_card_from_string(self):
        card = Card.from_string("3♥")
        assert card.rank == Rank.THREE
        assert card.suit == Suit.HEART

        card = Card.from_string("10S")
        assert card.rank == Rank.TEN
        assert card.suit == Suit.SPADE

        card = Card.from_string("AH")
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEART

    def test_card_equality_and_hashing(self):
        c1 = Card(rank=Rank.THREE, suit=Suit.HEART)
        c2 = Card(rank=Rank.THREE, suit=Suit.HEART)
        c3 = Card(rank=Rank.THREE, suit=Suit.SPADE)

        assert c1 == c2
        assert c1 != c3
        assert hash(c1) == hash(c2)

        card_set = {c1, c2, c3}
        assert len(card_set) == 2

    def test_heart_three(self):
        heart_three = get_heart_three()
        assert heart_three.rank == Rank.THREE
        assert heart_three.suit == Suit.HEART

    def test_standard_deck(self):
        deck = create_standard_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52

        rank_counts = get_rank_counts(deck)
        assert all(count == 4 for count in rank_counts.values())


class TestHandTypeSingle:
    """Test single card hands."""

    def test_single_card_valid(self):
        cards = [Card(rank=Rank.THREE, suit=Suit.HEART)]
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.SINGLE
        assert hand.main_rank == Rank.THREE
        assert len(hand) == 1

    def test_single_comparison(self):
        low_single = parse_hand([Card(rank=Rank.THREE, suit=Suit.HEART)])
        high_single = parse_hand([Card(rank=Rank.TWO, suit=Suit.SPADE)])

        assert can_beat(high_single, low_single)
        assert not can_beat(low_single, high_single)


class TestHandTypePair:
    """Test pair hands."""

    def test_pair_valid(self):
        cards = [Card(rank=Rank.THREE, suit=Suit.HEART), Card(rank=Rank.THREE, suit=Suit.SPADE)]
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.PAIR
        assert hand.main_rank == Rank.THREE
        assert len(hand) == 2

    def test_pair_invalid_different_ranks(self):
        cards = [Card(rank=Rank.THREE, suit=Suit.HEART), Card(rank=Rank.FOUR, suit=Suit.SPADE)]
        hand = parse_hand(cards)
        assert hand is None

    def test_pair_comparison(self):
        low_pair = parse_hand(make_cards_from_ranks([Rank.THREE, Rank.THREE]))
        high_pair = parse_hand(make_cards_from_ranks([Rank.TWO, Rank.TWO]))

        assert can_beat(high_pair, low_pair)
        assert not can_beat(low_pair, high_pair)


class TestHandTypeStraight:
    """Test straight hands - 5+ consecutive cards, 3-K only, no A/2."""

    def test_straight_5_cards_valid(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.STRAIGHT
        assert hand.main_rank == Rank.SEVEN
        assert len(hand) == 5

    def test_straight_max_valid_3_to_k(self):
        cards = make_cards_from_ranks(
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
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.STRAIGHT
        assert hand.main_rank == Rank.KING

    def test_straight_9_to_k_valid(self):
        cards = make_cards_from_ranks([Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING])
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.STRAIGHT
        assert hand.main_rank == Rank.KING

    def test_straight_with_ace_invalid(self):
        cards = make_cards_from_ranks([Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE])
        hand = parse_hand(cards)
        assert hand is None

    def test_straight_with_two_invalid(self):
        cards = make_cards_from_ranks([Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TWO])
        hand = parse_hand(cards)
        assert hand is None

    def test_straight_ace_two_three_invalid(self):
        cards = make_cards_from_ranks([Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE])
        hand = parse_hand(cards)
        assert hand is None

    def test_straight_4_cards_invalid(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX])
        hand = parse_hand(cards)
        assert hand is None

    def test_straight_not_consecutive_invalid(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.EIGHT])
        hand = parse_hand(cards)
        assert hand is None

    def test_straight_comparison_by_highest_rank(self):
        low_straight = parse_hand(
            make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        )
        high_straight = parse_hand(
            make_cards_from_ranks([Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT])
        )

        assert can_beat(high_straight, low_straight)
        assert not can_beat(low_straight, high_straight)

    def test_straight_different_lengths_incompatible(self):
        straight_5 = parse_hand(
            make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        )
        straight_6 = parse_hand(
            make_cards_from_ranks(
                [Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT]
            )
        )

        assert not can_beat(straight_6, straight_5)
        assert not can_beat(straight_5, straight_6)


class TestHandTypeConsecutivePairs:
    """Test consecutive pairs - 3+ consecutive pairs, 3-K only, no A/2."""

    def test_consecutive_pairs_3_pairs_valid(self):
        cards = make_cards_from_ranks(
            [Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FOUR, Rank.FIVE, Rank.FIVE]
        )
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.CONSECUTIVE_PAIRS
        assert hand.main_rank == Rank.FIVE
        assert len(hand) == 6

    def test_consecutive_pairs_jqk_valid(self):
        cards = make_cards_from_ranks(
            [Rank.JACK, Rank.JACK, Rank.QUEEN, Rank.QUEEN, Rank.KING, Rank.KING]
        )
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.CONSECUTIVE_PAIRS
        assert hand.main_rank == Rank.KING

    def test_consecutive_pairs_with_ace_invalid(self):
        cards = make_cards_from_ranks(
            [Rank.QUEEN, Rank.QUEEN, Rank.KING, Rank.KING, Rank.ACE, Rank.ACE]
        )
        hand = parse_hand(cards)
        assert hand is None

    def test_consecutive_pairs_with_two_invalid(self):
        cards = make_cards_from_ranks(
            [Rank.KING, Rank.KING, Rank.ACE, Rank.ACE, Rank.TWO, Rank.TWO]
        )
        hand = parse_hand(cards)
        assert hand is None

    def test_consecutive_pairs_2_pairs_invalid(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FOUR])
        hand = parse_hand(cards)
        assert hand is None

    def test_consecutive_pairs_not_consecutive_invalid(self):
        cards = make_cards_from_ranks(
            [Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FOUR, Rank.SIX, Rank.SIX]
        )
        hand = parse_hand(cards)
        assert hand is None

    def test_consecutive_pairs_not_all_pairs_invalid(self):
        cards = make_cards_from_ranks(
            [Rank.THREE, Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.FIVE]
        )
        hand = parse_hand(cards)
        assert hand is None

    def test_consecutive_pairs_comparison_by_highest_rank(self):
        low_pairs = parse_hand(
            make_cards_from_ranks(
                [Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FOUR, Rank.FIVE, Rank.FIVE]
            )
        )
        high_pairs = parse_hand(
            make_cards_from_ranks([Rank.FOUR, Rank.FOUR, Rank.FIVE, Rank.FIVE, Rank.SIX, Rank.SIX])
        )

        assert can_beat(high_pairs, low_pairs)
        assert not can_beat(low_pairs, high_pairs)


class TestHandTypeThreePlusTwo:
    """Test 3+2 hands - three of same rank + any 2 cards."""

    def test_three_plus_two_valid(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FIVE])
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.THREE_PLUS_TWO
        assert hand.main_rank == Rank.THREE
        assert len(hand) == 5

    def test_three_plus_two_with_pair_kicker_valid(self):
        cards = make_cards_from_ranks([Rank.KING, Rank.KING, Rank.KING, Rank.FIVE, Rank.FIVE])
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.THREE_PLUS_TWO
        assert hand.main_rank == Rank.KING

    def test_three_plus_two_comparison_by_main_rank_only(self):
        low_hand = parse_hand(
            make_cards_from_ranks([Rank.THREE, Rank.THREE, Rank.THREE, Rank.ACE, Rank.TWO])
        )
        high_hand = parse_hand(
            make_cards_from_ranks([Rank.FOUR, Rank.FOUR, Rank.FOUR, Rank.THREE, Rank.THREE])
        )

        assert can_beat(high_hand, low_hand)
        assert not can_beat(low_hand, high_hand)

    def test_three_plus_two_kicker_does_not_affect_comparison(self):
        hand_low_kicker = parse_hand(
            make_cards_from_ranks([Rank.TEN, Rank.TEN, Rank.TEN, Rank.THREE, Rank.FOUR])
        )
        hand_high_kicker = parse_hand(
            make_cards_from_ranks([Rank.TEN, Rank.TEN, Rank.TEN, Rank.ACE, Rank.TWO])
        )

        assert not can_beat(hand_high_kicker, hand_low_kicker)
        assert not can_beat(hand_low_kicker, hand_high_kicker)
        assert compare_hands(hand_high_kicker, hand_low_kicker) == 0


class TestHandTypeFourPlusThree:
    """Test 4+3 hands - four of same rank + any 3 cards."""

    def test_four_plus_three_valid(self):
        cards = make_cards_from_ranks(
            [Rank.THREE, Rank.THREE, Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]
        )
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.FOUR_PLUS_THREE
        assert hand.main_rank == Rank.THREE
        assert len(hand) == 7

    def test_four_plus_three_with_triple_kicker_valid(self):
        cards = make_cards_from_ranks(
            [Rank.KING, Rank.KING, Rank.KING, Rank.KING, Rank.FIVE, Rank.FIVE, Rank.FIVE]
        )
        hand = parse_hand(cards)

        assert hand is not None
        assert hand.hand_type == HandType.FOUR_PLUS_THREE
        assert hand.main_rank == Rank.KING

    def test_four_plus_three_comparison_by_main_rank_only(self):
        low_hand = parse_hand(
            make_cards_from_ranks(
                [Rank.THREE, Rank.THREE, Rank.THREE, Rank.THREE, Rank.ACE, Rank.TWO, Rank.TWO]
            )
        )
        high_hand = parse_hand(
            make_cards_from_ranks(
                [Rank.FOUR, Rank.FOUR, Rank.FOUR, Rank.FOUR, Rank.THREE, Rank.THREE, Rank.THREE]
            )
        )

        assert can_beat(high_hand, low_hand)
        assert not can_beat(low_hand, high_hand)

    def test_four_plus_three_kicker_does_not_affect_comparison(self):
        hand_low_kicker = parse_hand(
            make_cards_from_ranks(
                [Rank.TEN, Rank.TEN, Rank.TEN, Rank.TEN, Rank.THREE, Rank.FOUR, Rank.FIVE]
            )
        )
        hand_high_kicker = parse_hand(
            make_cards_from_ranks(
                [Rank.TEN, Rank.TEN, Rank.TEN, Rank.TEN, Rank.ACE, Rank.TWO, Rank.TWO]
            )
        )

        assert not can_beat(hand_high_kicker, hand_low_kicker)
        assert not can_beat(hand_low_kicker, hand_high_kicker)
        assert compare_hands(hand_high_kicker, hand_low_kicker) == 0

    def test_four_of_a_kind_not_a_bomb(self):
        four_of_a_kind = make_cards_from_ranks(
            [Rank.THREE, Rank.THREE, Rank.THREE, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]
        )
        single_two = [Card(rank=Rank.TWO, suit=Suit.HEART)]

        hand_4plus3 = parse_hand(four_of_a_kind)
        hand_single = parse_hand(single_two)

        assert not can_beat(hand_4plus3, hand_single)


class TestHandTypeIncompatibility:
    """Test that different hand types cannot beat each other."""

    def test_single_cannot_beat_pair(self):
        single = parse_hand([Card(rank=Rank.TWO, suit=Suit.HEART)])
        pair = parse_hand(make_cards_from_ranks([Rank.THREE, Rank.THREE]))

        assert not can_beat(single, pair)
        assert not can_beat(pair, single)

    def test_pair_cannot_beat_straight(self):
        pair = parse_hand(make_cards_from_ranks([Rank.TWO, Rank.TWO]))
        straight = parse_hand(
            make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        )

        assert not can_beat(pair, straight)
        assert not can_beat(straight, pair)

    def test_three_plus_two_cannot_beat_straight(self):
        three_plus_two = parse_hand(
            make_cards_from_ranks([Rank.TWO, Rank.TWO, Rank.TWO, Rank.ACE, Rank.ACE])
        )
        straight = parse_hand(
            make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        )

        assert not can_beat(three_plus_two, straight)
        assert not can_beat(straight, three_plus_two)


class TestIsValidHand:
    """Test the is_valid_hand convenience function."""

    def test_valid_hands(self):
        assert is_valid_hand([Card(rank=Rank.THREE, suit=Suit.HEART)])
        assert is_valid_hand(make_cards_from_ranks([Rank.THREE, Rank.THREE]))
        assert is_valid_hand(
            make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN])
        )

    def test_invalid_hands(self):
        assert not is_valid_hand([])
        assert not is_valid_hand(make_cards_from_ranks([Rank.THREE, Rank.FOUR]))
        assert not is_valid_hand(make_cards_from_ranks([Rank.THREE, Rank.THREE, Rank.THREE]))


class TestMakeCardsHelpers:
    """Test helper functions for creating test cards."""

    def test_make_cards_from_ranks(self):
        cards = make_cards_from_ranks([Rank.THREE, Rank.FOUR, Rank.FIVE])
        assert len(cards) == 3
        assert cards[0].rank == Rank.THREE
        assert cards[1].rank == Rank.FOUR
        assert cards[2].rank == Rank.FIVE

    def test_make_cards_from_string(self):
        cards = make_cards_from_string("3H 4D 5C")
        assert len(cards) == 3
        assert cards[0] == Card(rank=Rank.THREE, suit=Suit.HEART)
        assert cards[1] == Card(rank=Rank.FOUR, suit=Suit.DIAMOND)
        assert cards[2] == Card(rank=Rank.FIVE, suit=Suit.CLUB)

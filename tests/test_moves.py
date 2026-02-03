"""Tests for legal moves and action encoding.

Tests cover:
- Legal move enumeration (singles, pairs, straights, etc.)
- Tail-hand exemption rules (3+1, 3+0, 4+2, 4+1, 4+0)
- Strict follow-up matching after exemption
- Forced PASS when unable to play
- Action encoding and mask validity
- MAX_ACTIONS overflow guard
"""

import pytest
import numpy as np

from rl_poker.rules import (
    Card,
    Rank,
    Suit,
    HandType,
    parse_hand,
    make_cards_from_ranks,
    make_cards_from_string,
)
from rl_poker.moves.legal_moves import (
    Move,
    MoveType,
    MoveContext,
    PASS_MOVE,
    get_legal_moves,
    enumerate_all_hands,
    enumerate_exemption_hands,
    can_play_move,
    must_pass,
)
from rl_poker.moves.action_encoding import (
    MAX_ACTIONS,
    ActionSpace,
    ActionEncodingError,
    encode_action_space,
    get_action_mask,
    decode_action,
    encode_move,
    is_valid_action,
    sample_random_action,
    count_legal_actions,
    record_legal_move_count,
    get_global_stats,
    reset_global_stats,
)


class TestBasicMoveEnumeration:
    """Test basic legal move enumeration."""

    def test_single_card_moves(self):
        """Test that singles are enumerated correctly."""
        cards = make_cards_from_string("3H 5D 7C")
        moves = enumerate_all_hands(cards)

        # Should have at least 3 singles
        singles = [m for m in moves if m.hand and m.hand.hand_type == HandType.SINGLE]
        assert len(singles) == 3

    def test_pair_moves(self):
        """Test that pairs are enumerated correctly."""
        cards = make_cards_from_string("3H 3D 5C 5S")
        moves = enumerate_all_hands(cards)

        pairs = [m for m in moves if m.hand and m.hand.hand_type == HandType.PAIR]
        assert len(pairs) == 2  # One pair of 3s, one pair of 5s

    def test_straight_moves(self):
        """Test that straights are enumerated correctly."""
        cards = make_cards_from_string("3H 4D 5C 6S 7H")
        moves = enumerate_all_hands(cards)

        straights = [m for m in moves if m.hand and m.hand.hand_type == HandType.STRAIGHT]
        assert len(straights) >= 1
        # The straight 3-4-5-6-7 should exist
        straight_5 = [s for s in straights if len(s.cards) == 5]
        assert len(straight_5) == 1

    def test_consecutive_pairs_moves(self):
        """Test that consecutive pairs are enumerated correctly."""
        cards = make_cards_from_string("3H 3D 4C 4S 5H 5D")
        moves = enumerate_all_hands(cards)

        cons_pairs = [m for m in moves if m.hand and m.hand.hand_type == HandType.CONSECUTIVE_PAIRS]
        assert len(cons_pairs) >= 1

    def test_three_plus_two_moves(self):
        """Test that 3+2 hands are enumerated correctly."""
        cards = make_cards_from_string("3H 3D 3C 5S 7H")
        moves = enumerate_all_hands(cards)

        three_plus_two = [
            m for m in moves if m.hand and m.hand.hand_type == HandType.THREE_PLUS_TWO
        ]
        assert len(three_plus_two) >= 1

    def test_four_plus_three_moves(self):
        """Test that 4+3 hands are enumerated correctly."""
        cards = make_cards_from_string("3H 3D 3C 3S 5H 7D 9C")
        moves = enumerate_all_hands(cards)

        four_plus_three = [
            m for m in moves if m.hand and m.hand.hand_type == HandType.FOUR_PLUS_THREE
        ]
        assert len(four_plus_three) >= 1


class TestLeadingMoves:
    """Test leading (no previous move) scenarios."""

    def test_leading_includes_all_valid_hands(self):
        """When leading, all valid hands should be available."""
        cards = make_cards_from_string("3H 3D 5C 5S 7H")
        moves = get_legal_moves(cards)

        # Should have singles, pairs, and possibly a 3+2 if applicable
        move_types = {m.hand.hand_type for m in moves if m.hand}
        assert HandType.SINGLE in move_types
        assert HandType.PAIR in move_types

    def test_leading_no_pass(self):
        """When leading, PASS should NOT be included."""
        cards = make_cards_from_string("3H 5D 7C")
        moves = get_legal_moves(cards)

        pass_moves = [m for m in moves if m.move_type == MoveType.PASS]
        assert len(pass_moves) == 0


class TestTailHandExemption:
    """Test tail-hand exemption rules."""

    def test_exemption_three_plus_one(self):
        """Test 3+1 exemption (4 cards for 3+2)."""
        cards = make_cards_from_string("3H 3D 3C 5S")  # Triple 3s + one kicker
        context = MoveContext(is_tail_hand=True)

        moves = get_legal_moves(cards, context)

        # Should have exemption moves
        exemption_moves = [m for m in moves if m.is_exemption]
        assert len(exemption_moves) >= 1

        # Should have 3+1 (4 cards)
        three_plus_one = [m for m in exemption_moves if len(m.cards) == 4]
        assert len(three_plus_one) >= 1

    def test_exemption_three_plus_zero(self):
        """Test 3+0 exemption (3 cards for 3+2)."""
        cards = make_cards_from_string("3H 3D 3C 5S")  # Triple 3s + one kicker
        context = MoveContext(is_tail_hand=True)

        moves = get_legal_moves(cards, context)

        # Should have 3+0 (just the triple)
        exemption_moves = [m for m in moves if m.is_exemption]
        three_plus_zero = [m for m in exemption_moves if len(m.cards) == 3]
        assert len(three_plus_zero) >= 1

    def test_exemption_four_plus_two(self):
        """Test 4+2 exemption (6 cards for 4+3)."""
        cards = make_cards_from_string("3H 3D 3C 3S 5H 7D")  # Quad 3s + two kickers
        context = MoveContext(is_tail_hand=True)

        moves = get_legal_moves(cards, context)

        exemption_moves = [
            m for m in moves if m.is_exemption and m.standard_type == HandType.FOUR_PLUS_THREE
        ]
        four_plus_two = [m for m in exemption_moves if len(m.cards) == 6]
        assert len(four_plus_two) >= 1

    def test_exemption_four_plus_one(self):
        """Test 4+1 exemption (5 cards for 4+3)."""
        cards = make_cards_from_string("3H 3D 3C 3S 5H")  # Quad 3s + one kicker
        context = MoveContext(is_tail_hand=True)

        moves = get_legal_moves(cards, context)

        exemption_moves = [
            m for m in moves if m.is_exemption and m.standard_type == HandType.FOUR_PLUS_THREE
        ]
        four_plus_one = [m for m in exemption_moves if len(m.cards) == 5]
        assert len(four_plus_one) >= 1

    def test_exemption_four_plus_zero(self):
        """Test 4+0 exemption (4 cards for 4+3)."""
        cards = make_cards_from_string("3H 3D 3C 3S")  # Just quad 3s
        context = MoveContext(is_tail_hand=True)

        moves = get_legal_moves(cards, context)

        exemption_moves = [
            m for m in moves if m.is_exemption and m.standard_type == HandType.FOUR_PLUS_THREE
        ]
        four_plus_zero = [m for m in exemption_moves if len(m.cards) == 4]
        assert len(four_plus_zero) >= 1

    def test_no_exemption_when_not_tail_hand(self):
        """Exemptions should NOT be available when not tail hand."""
        cards = make_cards_from_string("3H 3D 3C 5S")
        context = MoveContext(is_tail_hand=False)

        moves = get_legal_moves(cards, context)

        exemption_moves = [m for m in moves if m.is_exemption]
        assert len(exemption_moves) == 0


class TestStrictFollowUpMatching:
    """Test strict follow-up matching after exemption.

    Key rule: If previous player used exemption (e.g., 333+4 as 4 cards),
    next player MUST play full standard size (5 cards for 3+2) or PASS.
    """

    def test_must_match_full_size_after_exemption(self):
        """After exemption, must play full standard size."""
        # Previous player played 333+4 (exemption: 3+1, 4 cards)
        prev_cards = make_cards_from_string("3H 3D 3C 4S")
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=None,
            is_exemption=True,
            standard_type=HandType.THREE_PLUS_TWO,
        )

        # Current player has 555+67 (standard 3+2, 5 cards)
        cards = make_cards_from_string("5H 5D 5C 6S 7H")
        context = MoveContext(
            previous_move=prev_move,
            previous_used_exemption=True,
        )

        moves = get_legal_moves(cards, context)

        # Should have the 5-card 3+2
        play_moves = [m for m in moves if m.move_type == MoveType.PLAY]
        three_plus_two = [
            m for m in play_moves if m.hand and m.hand.hand_type == HandType.THREE_PLUS_TWO
        ]
        assert len(three_plus_two) >= 1

        # All play moves should be 5 cards (standard 3+2 size)
        for pm in play_moves:
            assert len(pm.cards) == 5, "Must play full 5-card 3+2 after exemption"

    def test_forced_pass_when_cannot_match_full_size(self):
        """Must PASS if cannot play full standard size after exemption."""
        # Previous player played 333+4 (exemption: 3+1, 4 cards)
        prev_cards = make_cards_from_string("3H 3D 3C 4S")
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=None,
            is_exemption=True,
            standard_type=HandType.THREE_PLUS_TWO,
        )

        # Current player CANNOT make a 3+2 that beats triple 3s
        # They have lower cards
        cards = make_cards_from_string("2H 2D 4C 5S 6H")  # No triple, can't form 3+2
        context = MoveContext(
            previous_move=prev_move,
            previous_used_exemption=True,
        )

        moves = get_legal_moves(cards, context)

        # Only PASS should be available
        assert len(moves) == 1
        assert moves[0].move_type == MoveType.PASS

    def test_must_beat_main_rank_after_exemption(self):
        """After exemption, must beat the main rank with full standard size."""
        # Previous: 333+4 (triple 3s)
        prev_cards = make_cards_from_string("3H 3D 3C 4S")
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=None,
            is_exemption=True,
            standard_type=HandType.THREE_PLUS_TWO,
        )

        # Current has 222+AB (triple 2s) - this should beat triple 3s
        cards = make_cards_from_string("2H 2D 2C AH KS")
        context = MoveContext(
            previous_move=prev_move,
            previous_used_exemption=True,
        )

        moves = get_legal_moves(cards, context)
        play_moves = [m for m in moves if m.move_type == MoveType.PLAY]

        # Triple 2s beats triple 3s
        assert len(play_moves) >= 1


class TestNormalFollow:
    """Test normal follow scenarios (no exemption in previous)."""

    def test_must_match_type_and_size(self):
        """Must beat with same type and size."""
        # Previous: pair of 3s
        prev_cards = make_cards_from_string("3H 3D")
        prev_hand = parse_hand(list(prev_cards))
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=prev_hand,
        )

        # Current has pair of 5s and other cards
        cards = make_cards_from_string("5H 5D 7C 9S")
        context = MoveContext(previous_move=prev_move)

        moves = get_legal_moves(cards, context)

        # Should have pair of 5s and PASS
        play_moves = [m for m in moves if m.move_type == MoveType.PLAY]
        assert len(play_moves) >= 1

        for pm in play_moves:
            assert pm.hand.hand_type == HandType.PAIR
            assert len(pm.cards) == 2

    def test_pass_always_available_when_following(self):
        """PASS is always available when following."""
        prev_cards = make_cards_from_string("3H 3D")
        prev_hand = parse_hand(list(prev_cards))
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=prev_hand,
        )

        cards = make_cards_from_string("5H 5D 7C")
        context = MoveContext(previous_move=prev_move)

        moves = get_legal_moves(cards, context)
        pass_moves = [m for m in moves if m.move_type == MoveType.PASS]
        assert len(pass_moves) == 1

    def test_exemption_allowed_when_following_tail_hand(self):
        """Exemption allowed when following if is_tail_hand."""
        # Previous: standard 3+2 (555+67)
        prev_cards = make_cards_from_string("5H 5D 5C 6S 7H")
        prev_hand = parse_hand(list(prev_cards))
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=prev_hand,
        )

        # Current: has 666 and one kicker (tail hand)
        cards = make_cards_from_string("6H 6D 6C 8S")
        context = MoveContext(
            previous_move=prev_move,
            is_tail_hand=True,
        )

        moves = get_legal_moves(cards, context)

        # Should have exemption moves (3+1 with 666+8)
        exemption_moves = [m for m in moves if m.is_exemption]
        assert len(exemption_moves) >= 1


class TestMustPass:
    """Test must_pass helper function."""

    def test_must_pass_when_no_valid_plays(self):
        """must_pass returns True when only PASS is valid."""
        # Previous: pair of Ks
        prev_cards = make_cards_from_string("KH KD")
        prev_hand = parse_hand(list(prev_cards))
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=prev_hand,
        )

        # Current: only has pair of 3s (can't beat Ks)
        cards = make_cards_from_string("3H 3D 5C")
        context = MoveContext(previous_move=prev_move)

        assert must_pass(cards, context)

    def test_not_must_pass_when_can_play(self):
        """must_pass returns False when valid plays exist."""
        prev_cards = make_cards_from_string("3H 3D")
        prev_hand = parse_hand(list(prev_cards))
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(prev_cards),
            hand=prev_hand,
        )

        cards = make_cards_from_string("5H 5D")  # Can beat pair of 3s
        context = MoveContext(previous_move=prev_move)

        assert not must_pass(cards, context)


class TestActionEncoding:
    """Test action encoding and mask."""

    def test_action_mask_length(self):
        """Action mask length equals MAX_ACTIONS."""
        cards = make_cards_from_string("3H 5D 7C")
        action_space = encode_action_space(cards)

        assert len(action_space.action_mask) == MAX_ACTIONS

    def test_invalid_actions_masked_out(self):
        """Invalid actions should be masked (False)."""
        cards = make_cards_from_string("3H 5D 7C")
        action_space = encode_action_space(cards)

        num_legal = len(action_space.legal_moves)
        # First num_legal actions should be True
        assert all(action_space.action_mask[:num_legal])
        # Rest should be False
        assert not any(action_space.action_mask[num_legal:])

    def test_decode_encode_roundtrip(self):
        """Encoding and decoding should be consistent."""
        cards = make_cards_from_string("3H 5D 7C")
        action_space = encode_action_space(cards)

        for i, move in enumerate(action_space.legal_moves):
            # Encode move to action
            action_idx = encode_move(move, action_space)
            assert action_idx == i

            # Decode action to move
            decoded_move = decode_action(action_idx, action_space)
            assert decoded_move == move

    def test_decode_invalid_action_raises(self):
        """Decoding masked action should raise error."""
        cards = make_cards_from_string("3H 5D")
        action_space = encode_action_space(cards)

        # Last action should be masked
        with pytest.raises(ActionEncodingError):
            decode_action(MAX_ACTIONS - 1, action_space)

    def test_is_valid_action(self):
        """Test is_valid_action helper."""
        cards = make_cards_from_string("3H 5D")
        action_space = encode_action_space(cards)

        assert is_valid_action(0, action_space)  # First action is valid
        assert not is_valid_action(MAX_ACTIONS - 1, action_space)  # Last is invalid
        assert not is_valid_action(-1, action_space)  # Negative is invalid
        assert not is_valid_action(MAX_ACTIONS + 1, action_space)  # Out of range

    def test_sample_random_action(self):
        """Test random action sampling."""
        cards = make_cards_from_string("3H 5D 7C")
        action_space = encode_action_space(cards)

        rng = np.random.default_rng(42)
        for _ in range(10):
            action = sample_random_action(action_space, rng)
            assert is_valid_action(action, action_space)

    def test_count_legal_actions(self):
        """Test counting legal actions."""
        cards = make_cards_from_string("3H 5D 7C")
        action_space = encode_action_space(cards)

        assert count_legal_actions(action_space) == len(action_space.legal_moves)


class TestOverflowGuard:
    """Test MAX_ACTIONS overflow protection."""

    def test_overflow_raises_error(self):
        """Should raise error if legal moves exceed MAX_ACTIONS."""
        # This test verifies the guard exists - actual overflow is rare
        cards = make_cards_from_string("3H 5D 7C")

        # With artificially low max_actions, should raise
        with pytest.raises(ActionEncodingError):
            encode_action_space(cards, max_actions=1)

    def test_record_max_observed(self):
        """Test recording max observed legal moves."""
        reset_global_stats()

        # Record some observations
        record_legal_move_count(10)
        record_legal_move_count(50)
        record_legal_move_count(30)

        stats = get_global_stats()
        assert stats.max_observed == 50
        assert stats.total_samples == 3
        assert stats.within_limit

    def test_max_actions_sufficient_for_typical_hands(self):
        """MAX_ACTIONS should be sufficient for typical hands."""
        reset_global_stats()

        # Test with various hand configurations
        test_hands = [
            "3H 4D 5C 6S 7H 8D 9C 10S JH QD KC",  # 11 cards
            "3H 3D 3C 5S 5H 5D 7C 7S 9H 9D AD AH",  # 12 cards with pairs/triples
            "2H 2D 2C 2S AH AD AC AS KH KD KC KS QH",  # 13 cards with quads
        ]

        for hand_str in test_hands:
            cards = make_cards_from_string(hand_str)
            action_space = encode_action_space(cards)
            record_legal_move_count(len(action_space.legal_moves))

        stats = get_global_stats()
        assert stats.within_limit, (
            f"Max observed {stats.max_observed} exceeds MAX_ACTIONS {MAX_ACTIONS}"
        )


class TestSpecificExemptionScenario:
    """Test the specific exemption scenario from the plan: 333+4 → next must play 5-card 3+2."""

    def test_333_plus_4_exemption_scenario(self):
        """Specific test case: 333+4 → next player must play 5-card 3+2 (e.g., 555+67)."""
        # Player A plays 333+4 (exemption: 3+1)
        player_a_cards = make_cards_from_string("3H 3D 3C 4S")
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(player_a_cards),
            hand=None,
            is_exemption=True,
            standard_type=HandType.THREE_PLUS_TWO,
        )

        # Player B has cards that CAN form a 3+2 beating triple 3s
        player_b_cards = make_cards_from_string("5H 5D 5C 6S 7H 8D")
        context = MoveContext(
            previous_move=prev_move,
            previous_used_exemption=True,
        )

        moves = get_legal_moves(player_b_cards, context)

        # Should have 3+2 plays with 5 cards
        play_moves = [m for m in moves if m.move_type == MoveType.PLAY]
        assert len(play_moves) >= 1

        for pm in play_moves:
            # Must be full 5-card 3+2
            assert len(pm.cards) == 5
            assert pm.hand is not None
            assert pm.hand.hand_type == HandType.THREE_PLUS_TWO
            # Main rank must beat 3
            assert pm.hand.main_rank > Rank.THREE

        # PASS should also be available
        pass_moves = [m for m in moves if m.move_type == MoveType.PASS]
        assert len(pass_moves) == 1

    def test_cannot_play_exemption_to_counter_exemption(self):
        """After exemption, cannot counter with another exemption - must use full size."""
        # Player A plays 333+4 (exemption)
        player_a_cards = make_cards_from_string("3H 3D 3C 4S")
        prev_move = Move(
            move_type=MoveType.PLAY,
            cards=frozenset(player_a_cards),
            hand=None,
            is_exemption=True,
            standard_type=HandType.THREE_PLUS_TWO,
        )

        # Player B has 555 but only one kicker - cannot form full 3+2
        player_b_cards = make_cards_from_string("5H 5D 5C 6S")  # Only 4 cards, can't form 3+2
        context = MoveContext(
            previous_move=prev_move,
            previous_used_exemption=True,
            is_tail_hand=True,  # Even with tail hand, can't use exemption here
        )

        moves = get_legal_moves(player_b_cards, context)

        # Should only have PASS (can't form full 5-card 3+2)
        play_moves = [m for m in moves if m.move_type == MoveType.PLAY]
        assert len(play_moves) == 0, "Cannot counter exemption without full standard hand"


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_hand(self):
        """Empty hand should return no moves."""
        cards = []
        moves = get_legal_moves(cards)
        assert len(moves) == 0

    def test_single_card_hand(self):
        """Single card hand can only play that single."""
        cards = make_cards_from_string("3H")
        moves = get_legal_moves(cards)

        assert len(moves) == 1
        assert moves[0].hand.hand_type == HandType.SINGLE

    def test_pass_move_singleton(self):
        """PASS move should be consistent."""
        assert PASS_MOVE.move_type == MoveType.PASS
        assert len(PASS_MOVE.cards) == 0

    def test_move_equality(self):
        """Test Move equality."""
        cards1 = frozenset(make_cards_from_string("3H 3D"))
        cards2 = frozenset(make_cards_from_string("3H 3D"))
        hand = parse_hand(list(cards1))

        move1 = Move(move_type=MoveType.PLAY, cards=cards1, hand=hand)
        move2 = Move(move_type=MoveType.PLAY, cards=cards2, hand=hand)

        assert move1 == move2

    def test_move_str_representation(self):
        """Test Move string representation."""
        assert str(PASS_MOVE) == "PASS"

        cards = make_cards_from_string("3H 3D")
        hand = parse_hand(list(cards))
        move = Move(move_type=MoveType.PLAY, cards=frozenset(cards), hand=hand)
        assert "PAIR" in str(move)


class TestActionMaskIntegration:
    """Integration tests for action mask with legal moves."""

    def test_action_mask_matches_legal_moves(self):
        """Action mask should match legal move availability."""
        test_scenarios = [
            # Leading
            (make_cards_from_string("3H 5D 7C 9S"), None),
            # Following
            (
                make_cards_from_string("5H 5D 7C"),
                MoveContext(
                    previous_move=Move(
                        move_type=MoveType.PLAY,
                        cards=frozenset(make_cards_from_string("3H 3D")),
                        hand=parse_hand(make_cards_from_string("3H 3D")),
                    )
                ),
            ),
        ]

        for cards, context in test_scenarios:
            legal_moves = get_legal_moves(cards, context)
            action_space = encode_action_space(cards, context)

            # Number of True values in mask should equal legal moves
            assert np.sum(action_space.action_mask) == len(legal_moves)

            # Each legal move should be in the action space
            for move in legal_moves:
                assert move in action_space.move_to_action


# Run stats summary at end of test session
def test_final_stats_summary():
    """Print final stats summary (run last)."""
    stats = get_global_stats()
    print(f"\n{stats.summary()}")
    # Assertion to ensure we stay within limits
    if stats.total_samples > 0:
        assert stats.within_limit, (
            f"CRITICAL: Max observed ({stats.max_observed}) exceeds MAX_ACTIONS ({MAX_ACTIONS})"
        )

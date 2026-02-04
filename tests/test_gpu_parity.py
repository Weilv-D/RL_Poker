"""Sanity checks for GPU action masks against CPU legality.

These tests focus on soundness: GPU-legal actions should be legal in the CPU engine.
"""

from typing import List
import random

import numpy as np
import torch

from rl_poker.moves.gpu_action_mask import (
    GPUActionMaskComputer,
    ACTION_TYPE_PASS,
    ACTION_TYPE_SINGLE,
    ACTION_TYPE_PAIR,
    ACTION_TYPE_STRAIGHT,
    ACTION_TYPE_CONSEC_PAIRS,
    ACTION_TYPE_THREE_TWO,
    ACTION_TYPE_FOUR_THREE,
)
from rl_poker.rules import Card, Rank, Suit, create_standard_deck
from rl_poker.rules.hands import HandType
from rl_poker.moves.legal_moves import Move, MoveType, MoveContext, can_play_move


def _select_cards_for_action(hand: List[Card], required_counts: np.ndarray) -> List[Card] | None:
    cards_by_rank = {r: [] for r in Rank}
    for card in hand:
        cards_by_rank[card.rank].append(card)
    for cards in cards_by_rank.values():
        cards.sort(key=lambda c: c.suit.value)

    selected = []
    for rank in Rank:
        need = int(required_counts[rank.value])
        if need == 0:
            continue
        if len(cards_by_rank[rank]) < need:
            return None
        selected.extend(cards_by_rank[rank][:need])
    return selected


def _gpu_action_to_move(
    action_idx: int, hand: List[Card], mask_comp: GPUActionMaskComputer
) -> Move | None:
    if action_idx == 0:
        return Move(move_type=MoveType.PASS, cards=frozenset())

    required = mask_comp.action_required_counts[action_idx].cpu().numpy()
    cards = _select_cards_for_action(hand, required)
    if cards is None:
        return None

    is_exempt = bool(mask_comp.action_is_exemption[action_idx].item())
    std_type = int(mask_comp.action_standard_types[action_idx].item())

    if is_exempt:
        if std_type == ACTION_TYPE_THREE_TWO:
            standard = HandType.THREE_PLUS_TWO
        elif std_type == ACTION_TYPE_FOUR_THREE:
            standard = HandType.FOUR_PLUS_THREE
        else:
            standard = None
        return Move(
            move_type=MoveType.PLAY,
            cards=frozenset(cards),
            hand=None,
            is_exemption=True,
            standard_type=standard,
        )

    hand_obj = None
    if cards:
        from rl_poker.rules.hands import parse_hand

        hand_obj = parse_hand(cards)
    return Move(move_type=MoveType.PLAY, cards=frozenset(cards), hand=hand_obj)


def _make_random_hand(rng: random.Random, size: int = 13) -> List[Card]:
    deck = create_standard_deck()
    rng.shuffle(deck)
    return deck[:size]


def test_gpu_mask_leading_moves_are_cpu_legal():
    device = torch.device("cpu")
    mask_comp = GPUActionMaskComputer(device)
    rng = random.Random(123)

    for _ in range(10):
        hand = _make_random_hand(rng)
        hand_cards = torch.zeros(52, dtype=torch.bool, device=device)
        rank_counts = torch.zeros(13, dtype=torch.int32, device=device)
        for card in hand:
            idx = card.suit.value * 13 + card.rank.value
            hand_cards[idx] = True
            rank_counts[card.rank.value] += 1

        can_pass = torch.tensor([False], device=device)
        cards_remaining = torch.tensor([len(hand)], device=device)
        first_move = torch.tensor([False], device=device)
        has_h3 = torch.tensor(
            [any(c.rank == Rank.THREE and c.suit == Suit.HEART for c in hand)], device=device
        )

        mask = mask_comp.compute_mask_batched(
            hand_cards.unsqueeze(0),
            rank_counts.unsqueeze(0),
            can_pass,
            cards_remaining,
            first_move,
            has_h3,
        )[0]

        legal_idxs = torch.where(mask)[0].tolist()
        sample = legal_idxs[:30]
        for idx in sample:
            move = _gpu_action_to_move(idx, hand, mask_comp)
            assert move is not None
            context = MoveContext(previous_move=None, is_tail_hand=(len(hand) == len(move.cards)))
            assert can_play_move(hand, move, context)


def test_gpu_mask_following_single_is_cpu_legal():
    device = torch.device("cpu")
    mask_comp = GPUActionMaskComputer(device)
    rng = random.Random(456)

    prev_card = Card(rank=Rank.FIVE, suit=Suit.HEART)
    from rl_poker.rules.hands import parse_hand

    prev_hand = parse_hand([prev_card])
    prev_move = Move(move_type=MoveType.PLAY, cards=frozenset([prev_card]), hand=prev_hand)

    for _ in range(10):
        hand = _make_random_hand(rng)
        hand_cards = torch.zeros(52, dtype=torch.bool, device=device)
        rank_counts = torch.zeros(13, dtype=torch.int32, device=device)
        for card in hand:
            idx = card.suit.value * 13 + card.rank.value
            hand_cards[idx] = True
            rank_counts[card.rank.value] += 1

        can_pass = torch.tensor([True], device=device)
        cards_remaining = torch.tensor([len(hand)], device=device)
        first_move = torch.tensor([False], device=device)
        has_h3 = torch.tensor(
            [any(c.rank == Rank.THREE and c.suit == Suit.HEART for c in hand)], device=device
        )

        mask = mask_comp.compute_mask_batched(
            hand_cards.unsqueeze(0),
            rank_counts.unsqueeze(0),
            can_pass,
            cards_remaining,
            first_move,
            has_h3,
        )
        mask = mask_comp.apply_following_constraint(
            mask,
            prev_action_type=torch.tensor([ACTION_TYPE_SINGLE], device=device),
            prev_action_rank=torch.tensor([Rank.FIVE.value], device=device),
            prev_action_length=torch.tensor([1], device=device),
            prev_action_is_exemption=torch.tensor([False], device=device),
            prev_action_standard_type=torch.tensor([ACTION_TYPE_SINGLE], device=device),
            is_leading=torch.tensor([False], device=device),
        )[0]

        legal_idxs = torch.where(mask)[0].tolist()[:20]
        context = MoveContext(
            previous_move=prev_move, is_tail_hand=False, previous_used_exemption=False
        )
        for idx in legal_idxs:
            move = _gpu_action_to_move(idx, hand, mask_comp)
            assert move is not None
            assert can_play_move(hand, move, context)

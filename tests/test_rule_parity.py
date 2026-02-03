"""Deep parity test between CPU engine and GPU environment.

Compares turn flow, pass handling, ranks/finish order, and rank counts after
each action for randomly played games. Suit-level differences are ignored
for composite actions where GPU actions are rank-only.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch

from rl_poker.engine.game_state import GameState as CPUGameState
from rl_poker.moves.gpu_action_mask import (
    ACTION_TABLE,
    ACTION_TYPE_CONSEC_PAIRS,
    ACTION_TYPE_FOUR_THREE,
    ACTION_TYPE_FOUR_THREE_EXEMPT,
    ACTION_TYPE_PAIR,
    ACTION_TYPE_PASS,
    ACTION_TYPE_SINGLE,
    ACTION_TYPE_STRAIGHT,
    ACTION_TYPE_THREE_TWO,
    ACTION_TYPE_THREE_TWO_EXEMPT,
)
from rl_poker.moves.legal_moves import Move, MoveType
from rl_poker.rl.gpu_env import GPUPokerEnv
from rl_poker.rules.hands import HandType


def _rank_counts(cards) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for card in cards:
        counts[card.rank.value] = counts.get(card.rank.value, 0) + 1
    return counts


def _action_key_from_index(action_idx: int) -> Tuple[str, object]:
    atype, data = ACTION_TABLE[action_idx]
    if atype == "PASS":
        return ("PASS", None)
    if atype in ("SINGLE", "PAIR"):
        card_idx = next(iter(data))
        rank = card_idx % 13
        return (atype, rank)
    return (atype, data)


def _action_key_from_move(move: Move) -> Tuple[str, object]:
    if move.move_type == MoveType.PASS:
        return ("PASS", None)

    cards = list(move.cards)
    counts = _rank_counts(cards)

    if move.is_exemption:
        if move.standard_type == HandType.THREE_PLUS_TWO:
            main_rank = next(r for r, c in counts.items() if c == 3)
            kickers = []
            for r, c in counts.items():
                if r == main_rank:
                    continue
                kickers.extend([r] * c)
            return ("THREE_TWO_EXEMPT", (main_rank, tuple(sorted(kickers))))
        if move.standard_type == HandType.FOUR_PLUS_THREE:
            main_rank = next(r for r, c in counts.items() if c == 4)
            kickers = []
            for r, c in counts.items():
                if r == main_rank:
                    continue
                kickers.extend([r] * c)
            return ("FOUR_THREE_EXEMPT", (main_rank, tuple(sorted(kickers))))

    hand = move.hand
    if hand is None:
        raise AssertionError("Expected hand for non-exemption move")

    if hand.hand_type == HandType.SINGLE:
        return ("SINGLE", hand.main_rank.value)
    if hand.hand_type == HandType.PAIR:
        return ("PAIR", hand.main_rank.value)
    if hand.hand_type == HandType.STRAIGHT:
        ranks = sorted({c.rank.value for c in cards})
        return ("STRAIGHT", tuple(ranks))
    if hand.hand_type == HandType.CONSECUTIVE_PAIRS:
        ranks = sorted({c.rank.value for c in cards})
        return ("CONSEC_PAIRS", tuple(ranks))
    if hand.hand_type == HandType.THREE_PLUS_TWO:
        main_rank = hand.main_rank.value
        kickers = []
        for r, c in counts.items():
            if r == main_rank:
                continue
            kickers.extend([r] * c)
        return ("THREE_PLUS_TWO", (main_rank, tuple(sorted(kickers))))
    if hand.hand_type == HandType.FOUR_PLUS_THREE:
        main_rank = hand.main_rank.value
        kickers = []
        for r, c in counts.items():
            if r == main_rank:
                continue
            kickers.extend([r] * c)
        return ("FOUR_PLUS_THREE", (main_rank, tuple(sorted(kickers))))

    raise AssertionError(f"Unhandled hand type: {hand.hand_type}")


def _move_action_info(move: Move) -> Tuple[int, int, int, bool, int]:
    if move.move_type == MoveType.PASS:
        return (ACTION_TYPE_PASS, -1, 0, False, ACTION_TYPE_PASS)

    cards = list(move.cards)
    counts = _rank_counts(cards)
    length = len(cards)

    if move.is_exemption:
        if move.standard_type == HandType.THREE_PLUS_TWO:
            main_rank = next(r for r, c in counts.items() if c == 3)
            return (
                ACTION_TYPE_THREE_TWO_EXEMPT,
                main_rank,
                length,
                True,
                ACTION_TYPE_THREE_TWO,
            )
        if move.standard_type == HandType.FOUR_PLUS_THREE:
            main_rank = next(r for r, c in counts.items() if c == 4)
            return (
                ACTION_TYPE_FOUR_THREE_EXEMPT,
                main_rank,
                length,
                True,
                ACTION_TYPE_FOUR_THREE,
            )

    hand = move.hand
    if hand is None:
        raise AssertionError("Expected hand for non-exemption move")

    if hand.hand_type == HandType.SINGLE:
        return (ACTION_TYPE_SINGLE, hand.main_rank.value, length, False, ACTION_TYPE_SINGLE)
    if hand.hand_type == HandType.PAIR:
        return (ACTION_TYPE_PAIR, hand.main_rank.value, length, False, ACTION_TYPE_PAIR)
    if hand.hand_type == HandType.STRAIGHT:
        ranks = sorted({c.rank.value for c in cards})
        return (ACTION_TYPE_STRAIGHT, max(ranks), length, False, ACTION_TYPE_STRAIGHT)
    if hand.hand_type == HandType.CONSECUTIVE_PAIRS:
        ranks = sorted({c.rank.value for c in cards})
        return (ACTION_TYPE_CONSEC_PAIRS, max(ranks), length, False, ACTION_TYPE_CONSEC_PAIRS)
    if hand.hand_type == HandType.THREE_PLUS_TWO:
        return (ACTION_TYPE_THREE_TWO, hand.main_rank.value, length, False, ACTION_TYPE_THREE_TWO)
    if hand.hand_type == HandType.FOUR_PLUS_THREE:
        return (ACTION_TYPE_FOUR_THREE, hand.main_rank.value, length, False, ACTION_TYPE_FOUR_THREE)

    raise AssertionError(f"Unhandled hand type: {hand.hand_type}")


def _assert_state_alignment(cpu_state: CPUGameState, gpu_state) -> None:
    # Current player and lead player
    assert int(gpu_state.current_player.item()) == cpu_state.current_player
    assert int(gpu_state.lead_player.item()) == cpu_state.lead_player

    # Passes and first move
    assert int(gpu_state.consecutive_passes.item()) == cpu_state.pass_count
    assert bool(gpu_state.first_move.item()) == cpu_state.first_move

    # Player status and rank counts
    cpu_rank_counts = torch.zeros((4, 13), dtype=torch.int32)
    cpu_cards_remaining = torch.zeros((4,), dtype=torch.int32)
    for p in range(4):
        counts = _rank_counts(cpu_state.players[p].hand)
        for r, c in counts.items():
            cpu_rank_counts[p, r] = c
        cpu_cards_remaining[p] = len(cpu_state.players[p].hand)

        assert bool(gpu_state.has_passed[0, p].item()) == cpu_state.players[p].has_passed
        assert bool(gpu_state.has_finished[0, p].item()) == cpu_state.players[p].has_finished
        gpu_rank = int(gpu_state.finish_rank[0, p].item())
        cpu_rank = cpu_state.players[p].finish_rank or 0
        assert gpu_rank == cpu_rank

    gpu_rank_counts = gpu_state.rank_counts[0].cpu()
    assert torch.equal(cpu_rank_counts, gpu_rank_counts)
    assert torch.equal(cpu_cards_remaining, gpu_state.cards_remaining[0].cpu())

    # Previous action info
    if cpu_state.current_move is None:
        assert int(gpu_state.prev_action_rank.item()) == -1
        assert int(gpu_state.prev_action_type.item()) == ACTION_TYPE_PASS
        assert bool(gpu_state.prev_action_is_exemption.item()) is False
    else:
        atype, rank, length, is_exempt, std_type = _move_action_info(cpu_state.current_move)
        assert int(gpu_state.prev_action_type.item()) == atype
        assert int(gpu_state.prev_action_rank.item()) == rank
        assert int(gpu_state.prev_action_length.item()) == length
        assert bool(gpu_state.prev_action_is_exemption.item()) == is_exempt
        assert int(gpu_state.prev_action_standard_type.item()) == std_type
        assert int(gpu_state.prev_player.item()) == cpu_state.lead_player

    # Done flag
    assert bool(gpu_state.done.item()) == cpu_state.is_game_over()


def test_gpu_cpu_game_flow_parity():
    device = torch.device("cpu")
    env = GPUPokerEnv(num_envs=1, device=device)
    rng = random.Random(2024)

    for game_idx in range(5):
        seed = rng.randint(0, 10_000_000)
        cpu_state = CPUGameState.new_game(seed=seed)
        hands = [p.hand for p in cpu_state.players]
        gpu_state = env.state_from_hands(hands)

        steps = 0
        while not cpu_state.is_game_over():
            obs, mask = env.get_obs_and_mask(gpu_state)
            mask = mask[0]

            cpu_moves = cpu_state.get_legal_moves()
            cpu_key_to_move = {_action_key_from_move(m): m for m in cpu_moves}
            cpu_keys = set(cpu_key_to_move.keys())

            gpu_action_idxs = torch.where(mask)[0].tolist()
            gpu_keys = {_action_key_from_index(i) for i in gpu_action_idxs}

            assert cpu_keys == gpu_keys

            action_idx = rng.choice(gpu_action_idxs)
            action_key = _action_key_from_index(action_idx)
            move = cpu_key_to_move[action_key]
            cpu_state.apply_move(move)

            actions = torch.tensor([action_idx], device=device, dtype=torch.long)
            gpu_state, _, _ = env.step(gpu_state, actions)

            _assert_state_alignment(cpu_state, gpu_state)

            steps += 1
            assert steps < 400, "Game exceeded expected max steps"

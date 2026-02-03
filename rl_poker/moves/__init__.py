"""Poker move definitions and action spaces.

This module provides:
- Legal move enumeration with tail-hand exemptions (legal_moves.py)
- Action encoding for RL environment (action_encoding.py)
"""

from .legal_moves import (
    Move,
    MoveType,
    MoveContext,
    PASS_MOVE,
    get_legal_moves,
    enumerate_all_hands,
    enumerate_exemption_hands,
    can_play_move,
    get_move_count,
    must_pass,
)

from .action_encoding import (
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
    encode_move_as_vector,
    LegalMoveStats,
    record_legal_move_count,
    get_global_stats,
    reset_global_stats,
)

from .gpu_action_mask import (
    GPUActionMaskComputer,
    NUM_ACTIONS,
    ACTION_TYPE_PASS,
    ACTION_TYPE_SINGLE,
    ACTION_TYPE_PAIR,
    ACTION_TYPE_STRAIGHT,
    ACTION_TYPE_CONSEC_PAIRS,
    ACTION_TYPE_THREE_TWO,
    ACTION_TYPE_FOUR_THREE,
    card_to_idx,
    idx_to_card,
    cards_to_tensor,
)

__all__ = [
    # Legal moves
    "Move",
    "MoveType",
    "MoveContext",
    "PASS_MOVE",
    "get_legal_moves",
    "enumerate_all_hands",
    "enumerate_exemption_hands",
    "can_play_move",
    "get_move_count",
    "must_pass",
    # Action encoding (CPU)
    "MAX_ACTIONS",
    "ActionSpace",
    "ActionEncodingError",
    "encode_action_space",
    "get_action_mask",
    "decode_action",
    "encode_move",
    "is_valid_action",
    "sample_random_action",
    "count_legal_actions",
    "encode_move_as_vector",
    "LegalMoveStats",
    "record_legal_move_count",
    "get_global_stats",
    "reset_global_stats",
    # GPU action mask
    "GPUActionMaskComputer",
    "NUM_ACTIONS",
    "ACTION_TYPE_PASS",
    "ACTION_TYPE_SINGLE",
    "ACTION_TYPE_PAIR",
    "ACTION_TYPE_STRAIGHT",
    "ACTION_TYPE_CONSEC_PAIRS",
    "ACTION_TYPE_THREE_TWO",
    "ACTION_TYPE_FOUR_THREE",
    "card_to_idx",
    "idx_to_card",
    "cards_to_tensor",
]

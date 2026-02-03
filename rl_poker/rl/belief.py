"""Belief utilities for behavior-based posterior updates."""

from __future__ import annotations

import torch

from rl_poker.moves.gpu_action_mask import (
    ACTION_TYPE_CONSEC_PAIRS,
    ACTION_TYPE_STRAIGHT,
    GPUActionMaskComputer,
)


def build_response_rank_weights(
    mask_computer: GPUActionMaskComputer, block_size: int = 256
) -> torch.Tensor:
    """Build approximate rank-level response weights for PASS updates.

    For each previous action index, this computes how strongly each rank is
    implicated by *any* legal response action (ignoring private card availability).

    Returns:
        Tensor [num_actions, 13] with values in [0, 1].
    """
    device = mask_computer.device
    num_actions = mask_computer.num_actions

    action_types = mask_computer.action_types.to(device)
    action_ranks = mask_computer.action_ranks.to(device)
    action_lengths = mask_computer.action_lengths.to(device)
    action_is_exempt = mask_computer.action_is_exemption.to(device)
    action_std = mask_computer.action_standard_types.to(device)
    required_counts = mask_computer.action_required_counts.to(device).float()

    act_type = action_types.view(1, -1)
    act_rank = action_ranks.view(1, -1)
    act_len = action_lengths.view(1, -1)
    act_is_exempt = action_is_exempt.view(1, -1)
    act_std = action_std.view(1, -1)

    weights = torch.zeros((num_actions, 13), device=device, dtype=torch.float32)

    for start in range(0, num_actions, block_size):
        end = min(start + block_size, num_actions)
        prev_type = action_types[start:end].view(-1, 1)
        prev_rank = action_ranks[start:end].view(-1, 1)
        prev_len = action_lengths[start:end].view(-1, 1)
        prev_is_exempt = action_is_exempt[start:end].view(-1, 1)
        prev_std = action_std[start:end].view(-1, 1)

        higher_rank = act_rank > prev_rank
        same_length = act_len == prev_len
        needs_length_match = (prev_type == ACTION_TYPE_STRAIGHT) | (
            prev_type == ACTION_TYPE_CONSEC_PAIRS
        )

        valid_after_exempt = (~act_is_exempt) & (act_std == prev_std) & higher_rank
        valid_standard = (
            (~act_is_exempt)
            & (act_type == prev_type)
            & higher_rank
            & (same_length | ~needs_length_match)
        )
        valid_exempt = act_is_exempt & (act_std == prev_type) & higher_rank

        valid_follow = torch.where(prev_is_exempt, valid_after_exempt, valid_standard | valid_exempt)

        weights[start:end] = valid_follow.float() @ required_counts

    # No penalty when there is no previous action (PASS/leading)
    if num_actions > 0:
        weights[0] = 0.0

    max_w = weights.max(dim=1, keepdim=True).values
    weights = torch.where(max_w > 0, weights / max_w, weights)

    return weights

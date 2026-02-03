"""Utility functions for PPO training."""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: [T, B] rewards
        values: [T+1, B] value estimates (includes bootstrap)
        dones: [T, B] done flags
        gamma, gae_lambda: GAE parameters

    Returns:
        advantages: [T, B]
        returns: [T, B]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])

    for t in reversed(range(T)):
        next_values = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * not_done * lastgaelam

    returns = advantages + values[:-1]
    return advantages, returns

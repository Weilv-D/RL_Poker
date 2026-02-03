"""Policy network definitions for PPO training."""

from typing import override

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Actor-critic network for PPO."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 256):
        super().__init__()

        self.shared: nn.Sequential = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head: nn.Linear = nn.Linear(hidden_size, num_actions)
        self.value_head: nn.Linear = nn.Linear(hidden_size, 1)

    @override
    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: [batch, obs_dim]
            action_mask: [batch, num_actions] bool

        Returns:
            logits: [batch, num_actions] (masked with -inf for invalid)
            values: [batch]
        """
        features: torch.Tensor = self.shared(obs)
        logits: torch.Tensor = self.policy_head(features)

        # Mask invalid actions
        logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=obs.device))

        values: torch.Tensor = self.value_head(features).squeeze(-1)

        return logits, values

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from policy.

        Returns:
            actions: [batch] sampled action indices
            log_probs: [batch] log probabilities
            values: [batch] value estimates
        """
        logits, values = self.forward(obs, action_mask)

        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions: torch.Tensor = dist.sample()
        log_probs: torch.Tensor = dist.log_prob(actions)

        return actions, log_probs, values

    def evaluate_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: [batch]
            values: [batch]
            entropy: scalar
        """
        logits, values = self.forward(obs, action_mask)

        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs: torch.Tensor = dist.log_prob(actions)
        entropy: torch.Tensor = dist.entropy().mean()

        return log_probs, values, entropy

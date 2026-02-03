"""Recurrent policy scaffolding (not enabled by default).

Defines a GRU-based actor-critic that can consume history sequences.
This is a placeholder for Phase 3 and is not wired into training yet.
"""

from typing import override

import torch
import torch.nn as nn


class RecurrentPolicyNetwork(nn.Module):
    """GRU-based actor-critic network.

    Inputs can be:
        - obs: [B, obs_dim]
        - seq: [B, T, feat_dim] optional history features

    If seq is provided, the GRU processes it and the final hidden state
    is concatenated with obs before the heads.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        history_feat_dim: int,
        hidden_size: int = 256,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.gru: nn.GRU = nn.GRU(history_feat_dim, gru_hidden, batch_first=True)
        self.shared: nn.Sequential = nn.Sequential(
            nn.Linear(obs_dim + gru_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head: nn.Linear = nn.Linear(hidden_size, num_actions)
        self.value_head: nn.Linear = nn.Linear(hidden_size, 1)

    @override
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq is None:
            raise ValueError("Sequence input required for recurrent policy")
        _, h = self.gru(seq)
        h = h.squeeze(0)
        features: torch.Tensor = torch.cat([obs, h], dim=-1)
        features = self.shared(features)
        logits: torch.Tensor = self.policy_head(features)
        logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=obs.device))
        values: torch.Tensor = self.value_head(features).squeeze(-1)
        return logits, values

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor, seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, action_mask, seq)
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions: torch.Tensor = dist.sample()
        log_probs: torch.Tensor = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, action_mask, seq)
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs: torch.Tensor = dist.log_prob(actions)
        entropy: torch.Tensor = dist.entropy().mean()
        return log_probs, values, entropy

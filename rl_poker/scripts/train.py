#!/usr/bin/env python3
"""GPU-native PPO training for RL Poker.

This module provides a high-performance training loop that runs entirely on GPU:
- Vectorized environments simulated in parallel
- Batched action mask computation
- PPO algorithm with self-play

Key features:
- ~10,000+ SPS (steps per second) on modern GPUs
- TensorBoard logging
- Checkpoint saving/loading
- Configurable hyperparameters

Usage:
    python -m rl_poker.scripts.train --help
    python -m rl_poker.scripts.train --num-envs 128 --total-timesteps 2000000
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from rl_poker.moves.gpu_action_mask import GPUActionMaskComputer, NUM_ACTIONS


# ============================================================================
# GPU Vectorized Environment
# ============================================================================


@dataclass
class GameState:
    """Batched game state on GPU.

    All tensors have shape [batch, ...] for N parallel games.

    Card encoding: card_idx = suit * 13 + rank
    - suit: 0=Heart, 1=Diamond, 2=Club, 3=Spade
    - rank: 0=3, 1=4, ..., 10=K, 11=A, 12=2
    """

    # Player hands: [batch, 4, 52] bool
    hands: torch.Tensor

    # Cards per rank per player: [batch, 4, 13] int
    rank_counts: torch.Tensor

    # Current player index: [batch] int in [0,3]
    current_player: torch.Tensor

    # Previous play tracking
    prev_action: torch.Tensor  # [batch] action index that was played
    prev_action_type: torch.Tensor  # [batch] type of the action (SINGLE, PAIR, etc.)
    prev_action_rank: torch.Tensor  # [batch] main rank of that action
    prev_action_length: torch.Tensor  # [batch] length of the action (for straights)
    prev_action_is_exemption: torch.Tensor  # [batch] bool
    prev_action_standard_type: torch.Tensor  # [batch] standard type for exemptions
    prev_player: torch.Tensor  # [batch] who played it
    lead_player: torch.Tensor  # [batch] current lead player
    consecutive_passes: torch.Tensor  # [batch] passes since last play
    has_passed: torch.Tensor  # [batch, 4] bool
    has_finished: torch.Tensor  # [batch, 4] bool
    finish_rank: torch.Tensor  # [batch, 4] int (0 if not finished)
    next_rank: torch.Tensor  # [batch] next rank to assign (1-4)
    first_move: torch.Tensor  # [batch] bool

    # Game state
    cards_remaining: torch.Tensor  # [batch, 4] cards left per player
    done: torch.Tensor  # [batch] bool
    winner: torch.Tensor  # [batch] int, -1 if not done


class GPUPokerEnv:
    """Fully vectorized poker environment on GPU.

    Rules:
    - 4 players, each gets 13 cards from standard 52-card deck
    - Player with Heart 3 leads first
    - Players take turns clockwise
    - Must play a hand that beats the previous, or pass
    - After 3 consecutive passes, lead player can play anything
    - First player to empty their hand wins
    """

    NUM_PLAYERS = 4
    CARDS_PER_PLAYER = 13
    HEART_THREE_IDX = 0  # card_idx for Heart 3 = suit(0)*13 + rank(0)

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device

        # Action mask computer
        self.mask_computer = GPUActionMaskComputer(device)
        self.num_actions = self.mask_computer.num_actions

        # Observation dimension
        # [hand(52) + rank_counts(13) + other_rank_counts(3*13) + context(5)]
        self.obs_dim = 52 + 13 + 39 + 5

        # Pre-compute helper tensors
        self._init_tensors()

    def _init_tensors(self):
        """Pre-compute static tensors for efficiency."""
        # Rank lookup for cards: card_idx % 13 = rank
        self.card_to_rank = torch.arange(52, device=self.device) % 13

        # Suit lookup for cards: card_idx // 13 = suit
        self.card_to_suit = torch.arange(52, device=self.device) // 13

        # Suit combination table for pairs
        self.suit_combo_table = torch.tensor(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], device=self.device, dtype=torch.long
        )

    def reset(self) -> GameState:
        """Reset all environments with new random deals.

        FULLY VECTORIZED using argsort for batched shuffling.
        """
        B = self.num_envs

        # Vectorized shuffle: argsort of random values
        rand = torch.rand(B, 52, device=self.device)
        perms = rand.argsort(dim=1)  # [B, 52] random permutations

        # Create hands tensor
        hands = torch.zeros((B, 4, 52), dtype=torch.bool, device=self.device)

        # Deal cards to each player (vectorized)
        for p in range(4):
            # Cards for player p are at indices [p*13 : (p+1)*13]
            player_cards = perms[:, p * 13 : (p + 1) * 13]  # [B, 13]
            # Create index tensors for scatter
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, 13)
            hands[batch_idx, p, player_cards] = True

        # Compute rank counts efficiently
        # Reshape hands to group cards by rank: [B, 4, 13, 4] where last dim is suits
        hands_by_rank = hands.view(B, 4, 4, 13).permute(0, 1, 3, 2)  # [B, 4, 13, 4]
        rank_counts = hands_by_rank.sum(dim=3).to(torch.int32)  # [B, 4, 13]

        # Find starting player (who has Heart 3)
        has_heart_three = hands[:, :, self.HEART_THREE_IDX]  # [B, 4]
        starting_player = has_heart_three.int().argmax(dim=1)  # [B]

        return GameState(
            hands=hands,
            rank_counts=rank_counts,
            current_player=starting_player,
            prev_action=torch.zeros(B, dtype=torch.long, device=self.device),
            prev_action_type=torch.zeros(B, dtype=torch.long, device=self.device),
            prev_action_rank=torch.full((B,), -1, dtype=torch.long, device=self.device),
            prev_action_length=torch.zeros(B, dtype=torch.long, device=self.device),
            prev_action_is_exemption=torch.zeros(B, dtype=torch.bool, device=self.device),
            prev_action_standard_type=torch.zeros(B, dtype=torch.long, device=self.device),
            prev_player=torch.full((B,), -1, dtype=torch.long, device=self.device),
            lead_player=starting_player.clone(),
            consecutive_passes=torch.zeros(B, dtype=torch.long, device=self.device),
            has_passed=torch.zeros((B, 4), dtype=torch.bool, device=self.device),
            has_finished=torch.zeros((B, 4), dtype=torch.bool, device=self.device),
            finish_rank=torch.zeros((B, 4), dtype=torch.long, device=self.device),
            next_rank=torch.ones(B, dtype=torch.long, device=self.device),
            first_move=torch.ones(B, dtype=torch.bool, device=self.device),
            cards_remaining=torch.full((B, 4), 13, dtype=torch.long, device=self.device),
            done=torch.zeros(B, dtype=torch.bool, device=self.device),
            winner=torch.full((B,), -1, dtype=torch.long, device=self.device),
        )

    def get_obs_and_mask(self, state: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observations and action masks for current players.

        Returns:
            obs: [batch, obs_dim] float
            mask: [batch, num_actions] bool
        """
        B = self.num_envs

        # Gather current player's hand and rank counts
        p_idx = state.current_player.view(B, 1, 1)
        p_idx_2d = state.current_player.view(B, 1)

        my_hand = state.hands.gather(1, p_idx.expand(-1, 1, 52)).squeeze(1)  # [B, 52]
        my_rank_counts = state.rank_counts.gather(1, p_idx.expand(-1, 1, 13)).squeeze(1)  # [B, 13]
        my_cards_remaining = state.cards_remaining.gather(1, p_idx_2d).squeeze(1)  # [B]

        # Other players' rank counts (visible information)
        # Rotate so index 0 is current player
        other_rank_counts = []
        for offset in [1, 2, 3]:
            other_p = (state.current_player + offset) % 4
            other_rc = state.rank_counts.gather(1, other_p.view(B, 1, 1).expand(-1, 1, 13)).squeeze(
                1
            )
            other_rank_counts.append(other_rc)
        other_rank_counts = torch.cat(other_rank_counts, dim=1)  # [B, 39]

        # Context features
        # is_leading: True when starting new round (no previous play to beat)
        # This happens when prev_action_rank == -1 (set on reset or after 3 passes)
        is_leading = state.prev_action_rank < 0
        context = torch.stack(
            [
                state.prev_action.float() / self.num_actions,  # Normalized prev action
                (state.prev_action_rank.float() + 1) / 14,  # Normalized rank
                state.consecutive_passes.float() / 4,  # Normalized passes
                is_leading.float(),  # Is leading
                state.cards_remaining.sum(dim=1).float() / 52,  # Total cards remaining
            ],
            dim=1,
        )  # [B, 5]

        # Combine observation
        obs = torch.cat(
            [
                my_hand.float(),
                my_rank_counts.float() / 4,  # Normalize to [0, 1]
                other_rank_counts.float() / 4,
                context,
            ],
            dim=1,
        )

        # Compute action mask
        can_pass = ~is_leading
        has_heart_three = my_hand[:, self.HEART_THREE_IDX]
        mask = self.mask_computer.compute_mask_batched(
            my_hand,
            my_rank_counts,
            can_pass,
            my_cards_remaining,
            state.first_move,
            has_heart_three,
        )

        # Apply following constraint: must beat previous play
        mask = self.mask_computer.apply_following_constraint(
            mask,
            prev_action_type=state.prev_action_type,
            prev_action_rank=state.prev_action_rank,
            prev_action_length=state.prev_action_length,
            prev_action_is_exemption=state.prev_action_is_exemption,
            prev_action_standard_type=state.prev_action_standard_type,
            is_leading=is_leading,
        )

        return obs, mask

    def step(
        self, state: GameState, actions: torch.Tensor
    ) -> Tuple[GameState, torch.Tensor, torch.Tensor]:
        """Execute actions and return new state.

        Args:
            state: Current game state
            actions: [batch] action indices

        Returns:
            new_state: Updated state
            rewards: [batch, 4] rewards per player
            dones: [batch] episode done flags
        """
        B = self.num_envs
        batch_idx = torch.arange(B, device=self.device)
        current_player = state.current_player
        not_done = ~state.done

        # Clone mutable state
        new_hands = state.hands.clone()
        new_rank_counts = state.rank_counts.clone()
        new_cards_remaining = state.cards_remaining.clone()
        new_prev_action = state.prev_action.clone()
        new_prev_type = state.prev_action_type.clone()
        new_prev_rank = state.prev_action_rank.clone()
        new_prev_length = state.prev_action_length.clone()
        new_prev_is_exemption = state.prev_action_is_exemption.clone()
        new_prev_standard_type = state.prev_action_standard_type.clone()
        new_prev_player = state.prev_player.clone()
        new_lead_player = state.lead_player.clone()
        new_passes = state.consecutive_passes.clone()
        new_has_passed = state.has_passed.clone()
        new_has_finished = state.has_finished.clone()
        new_finish_rank = state.finish_rank.clone()
        new_next_rank = state.next_rank.clone()
        new_first_move = state.first_move.clone()
        new_done = state.done.clone()
        new_winner = state.winner.clone()

        # Get action info
        action_types, action_ranks, action_lengths = self.mask_computer.get_action_info(actions)
        action_is_exemption = self.mask_computer.action_is_exemption[actions]
        action_standard_types = self.mask_computer.action_standard_types[actions]

        played = (actions != 0) & not_done
        is_pass = (actions == 0) & not_done

        # Update prev action tracking for played actions
        new_prev_action = torch.where(played, actions, new_prev_action)
        new_prev_type = torch.where(played, action_types, new_prev_type)
        new_prev_rank = torch.where(played, action_ranks, new_prev_rank)
        new_prev_length = torch.where(played, action_lengths, new_prev_length)
        new_prev_is_exemption = torch.where(played, action_is_exemption, new_prev_is_exemption)
        new_prev_standard_type = torch.where(played, action_standard_types, new_prev_standard_type)
        new_prev_player = torch.where(played, current_player, new_prev_player)
        new_lead_player = torch.where(played, current_player, new_lead_player)

        # First move flag clears after any play
        new_first_move = torch.where(played, torch.zeros_like(new_first_move), new_first_move)

        # Handle PASS
        pass_envs = batch_idx[is_pass]
        pass_players = current_player[is_pass]
        new_has_passed[pass_envs, pass_players] = True
        new_passes = torch.where(is_pass, new_passes + 1, new_passes)

        # Handle PLAY: reset pass flags for that env
        played_envs = batch_idx[played]
        new_has_passed[played_envs] = False
        new_passes = torch.where(played, torch.zeros_like(new_passes), new_passes)

        # Remove required cards for played actions
        req_counts = self.mask_computer.action_required_counts[actions].clone()
        req_counts = req_counts * played.unsqueeze(1).to(req_counts.dtype)

        # Ensure Heart 3 is used on first move when required
        needs_h3 = played & state.first_move & (req_counts[:, 0] > 0)
        envs = batch_idx[needs_h3]
        players = current_player[needs_h3]
        h_idx = self.HEART_THREE_IDX
        has_h3 = new_hands[envs, players, h_idx]
        envs_use = envs[has_h3]
        players_use = players[has_h3]
        new_hands[envs_use, players_use, h_idx] = False
        new_rank_counts[envs_use, players_use, 0] -= 1
        new_cards_remaining[envs_use, players_use] -= 1
        req_counts[envs_use, 0] -= 1

        # Vectorized removal by rank/suit order
        current_hand = new_hands[batch_idx, current_player]  # [B, 52]
        hand_by_rank = current_hand.view(B, 4, 13).permute(0, 2, 1)  # [B, 13, 4]
        hand_int = hand_by_rank.to(torch.int32)
        take_mask = hand_by_rank & (hand_int.cumsum(dim=2) <= req_counts.unsqueeze(2))
        removed_counts = take_mask.sum(dim=2)  # [B, 13]
        new_hand_by_rank = hand_by_rank & ~take_mask
        new_hand_flat = new_hand_by_rank.permute(0, 2, 1).contiguous().view(B, 52)
        new_hands[batch_idx, current_player] = new_hand_flat
        new_rank_counts[batch_idx, current_player] -= removed_counts
        new_cards_remaining[batch_idx, current_player] -= removed_counts.sum(dim=1)

        # Mark finished players and assign ranks
        finished_now = played & (new_cards_remaining[batch_idx, current_player] == 0)
        finished_now = finished_now & (~new_has_finished[batch_idx, current_player])
        envs = batch_idx[finished_now]
        players = current_player[finished_now]
        new_has_finished[envs, players] = True
        new_finish_rank[envs, players] = new_next_rank[envs]
        # Update winner for first finisher (rank 1)
        is_first = new_finish_rank[envs, players] == 1
        new_winner[envs[is_first]] = players[is_first]
        new_next_rank[envs] += 1

        # Check for game over (third player finished)
        finished_count = new_has_finished.sum(dim=1)
        game_over = (finished_count >= 3) & ~new_done
        # For game-over environments, assign rank 4 to remaining (unfinished) players
        game_over_mask = game_over.unsqueeze(1)  # [B, 1]
        remaining_mask = ~new_has_finished & game_over_mask  # [B, 4]
        new_finish_rank = torch.where(
            remaining_mask, torch.full_like(new_finish_rank, 4), new_finish_rank
        )
        new_has_finished = torch.where(game_over_mask, torch.ones_like(new_has_finished), new_has_finished)
        new_done = torch.where(game_over, torch.ones_like(new_done), new_done)

        # New lead round when all other active players passed
        active = ~new_has_finished
        lead_mask = torch.zeros_like(active)
        lead_mask[batch_idx, new_lead_player] = True
        all_others_passed = ((new_has_passed | ~active) | lead_mask).all(dim=1) & ~new_done
        envs = batch_idx[all_others_passed]
        new_has_passed[envs] = False
        new_passes[envs] = 0
        new_prev_action[envs] = 0
        new_prev_type[envs] = 0
        new_prev_rank[envs] = -1
        new_prev_length[envs] = 0
        new_prev_is_exemption[envs] = False
        new_prev_standard_type[envs] = 0

        # Advance to next active player who has not passed
        new_current = (current_player + 1) % 4
        for _ in range(4):
            active_next = ~new_has_finished.gather(1, new_current.view(B, 1)).squeeze(1)
            not_passed_next = ~new_has_passed.gather(1, new_current.view(B, 1)).squeeze(1)
            ok = active_next & not_passed_next
            new_current = torch.where(ok, new_current, (new_current + 1) % 4)

        lead_active = ~new_has_finished[batch_idx, new_lead_player]
        first_active = active.float().argmax(dim=1)
        new_current = torch.where(all_others_passed & lead_active, new_lead_player, new_current)
        new_current = torch.where(all_others_passed & ~lead_active, first_active, new_current)
        new_lead_player = torch.where(all_others_passed & ~lead_active, first_active, new_lead_player)

        # Compute rewards
        rewards = torch.zeros((B, 4), device=self.device)
        score_map = torch.tensor([0, 2, 1, -1, -2], device=self.device, dtype=torch.float)
        rewards = torch.where(game_over.unsqueeze(1), score_map[new_finish_rank], rewards)

        new_state = GameState(
            hands=new_hands,
            rank_counts=new_rank_counts,
            current_player=new_current,
            prev_action=new_prev_action,
            prev_action_type=new_prev_type,
            prev_action_rank=new_prev_rank,
            prev_action_length=new_prev_length,
            prev_action_is_exemption=new_prev_is_exemption,
            prev_action_standard_type=new_prev_standard_type,
            prev_player=new_prev_player,
            lead_player=new_lead_player,
            consecutive_passes=new_passes,
            has_passed=new_has_passed,
            has_finished=new_has_finished,
            finish_rank=new_finish_rank,
            next_rank=new_next_rank,
            first_move=new_first_move,
            cards_remaining=new_cards_remaining,
            done=new_done,
            winner=new_winner,
        )

        return new_state, rewards, new_done

    def reset_done_envs(self, state: GameState, dones: torch.Tensor) -> GameState:
        """Reset environments that are done."""
        if not dones.any():
            return state

        fresh_state = self.reset()

        # Merge: use fresh state where done, keep old where not done
        def merge(old, new, dones):
            expand_shape = [-1] + [1] * (old.dim() - 1)
            mask = dones.view(*expand_shape).expand_as(old)
            return torch.where(mask, new, old)

        return GameState(
            hands=merge(state.hands, fresh_state.hands, dones),
            rank_counts=merge(state.rank_counts, fresh_state.rank_counts, dones),
            current_player=merge(state.current_player, fresh_state.current_player, dones),
            prev_action=merge(state.prev_action, fresh_state.prev_action, dones),
            prev_action_type=merge(state.prev_action_type, fresh_state.prev_action_type, dones),
            prev_action_rank=merge(state.prev_action_rank, fresh_state.prev_action_rank, dones),
            prev_action_length=merge(
                state.prev_action_length, fresh_state.prev_action_length, dones
            ),
            prev_action_is_exemption=merge(
                state.prev_action_is_exemption, fresh_state.prev_action_is_exemption, dones
            ),
            prev_action_standard_type=merge(
                state.prev_action_standard_type, fresh_state.prev_action_standard_type, dones
            ),
            prev_player=merge(state.prev_player, fresh_state.prev_player, dones),
            lead_player=merge(state.lead_player, fresh_state.lead_player, dones),
            consecutive_passes=merge(
                state.consecutive_passes, fresh_state.consecutive_passes, dones
            ),
            has_passed=merge(state.has_passed, fresh_state.has_passed, dones),
            has_finished=merge(state.has_finished, fresh_state.has_finished, dones),
            finish_rank=merge(state.finish_rank, fresh_state.finish_rank, dones),
            next_rank=merge(state.next_rank, fresh_state.next_rank, dones),
            first_move=merge(state.first_move, fresh_state.first_move, dones),
            cards_remaining=merge(state.cards_remaining, fresh_state.cards_remaining, dones),
            done=merge(state.done, fresh_state.done, dones),
            winner=merge(state.winner, fresh_state.winner, dones),
        )


# ============================================================================
# Policy Network
# ============================================================================


class PolicyNetwork(nn.Module):
    """Actor-critic network for PPO."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: [batch, obs_dim]
            action_mask: [batch, num_actions] bool

        Returns:
            logits: [batch, num_actions] (masked with -inf for invalid)
            values: [batch]
        """
        features = self.shared(obs)
        logits = self.policy_head(features)

        # Mask invalid actions
        logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=obs.device))

        values = self.value_head(features).squeeze(-1)

        return logits, values

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from policy.

        Returns:
            actions: [batch] sampled action indices
            log_probs: [batch] log probabilities
            values: [batch] value estimates
        """
        logits, values = self.forward(obs, action_mask)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values

    def evaluate_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: [batch]
            values: [batch]
            entropy: scalar
        """
        logits, values = self.forward(obs, action_mask)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy


# ============================================================================
# PPO Training
# ============================================================================


@dataclass
class TrainConfig:
    """Training configuration."""

    # Environment
    num_envs: int = 128

    # Training
    total_timesteps: int = 2_000_000
    rollout_steps: int = 128

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    num_minibatches: int = 8
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    hidden_size: int = 256

    # Logging
    log_interval: int = 5
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # Misc
    seed: int = 42
    cuda: bool = True


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    lastgaelam = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = values[t + 1]
        else:
            next_values = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * not_done * lastgaelam

    returns = advantages + values[:-1]
    return advantages, returns


def train(config: TrainConfig):
    """Main training loop."""

    # Setup device
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    # Create environment
    env = GPUPokerEnv(config.num_envs, device)
    print(f"Environments: {config.num_envs}")
    print(f"Observation dim: {env.obs_dim}")
    print(f"Action space: {env.num_actions}")

    # Create network
    network = PolicyNetwork(env.obs_dim, env.num_actions, config.hidden_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)

    # Logging
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    run_name = f"rl_poker_{config.seed}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(config.log_dir, run_name))

    # Training state
    state = env.reset()
    total_steps = 0
    total_games = 0
    total_wins = {i: 0 for i in range(4)}
    start_time = time.time()
    sps = 0.0

    # Calculate update counts
    batch_size = config.num_envs * config.rollout_steps
    num_updates = config.total_timesteps // batch_size
    minibatch_size = batch_size // config.num_minibatches

    print(f"\nTraining for {num_updates} updates...")
    print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")

    for update in range(1, num_updates + 1):
        # Storage for rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        mask_buf = []

        # Collect rollout
        for step in range(config.rollout_steps):
            obs, action_mask = env.get_obs_and_mask(state)

            with torch.no_grad():
                actions, log_probs, values = network.get_action(obs, action_mask)

            new_state, rewards, dones = env.step(state, actions)

            # Get current player's reward
            player_rewards = rewards.gather(1, state.current_player.unsqueeze(1)).squeeze(1)

            # Store
            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(log_probs)
            rew_buf.append(player_rewards)
            done_buf.append(dones)
            val_buf.append(values)
            mask_buf.append(action_mask)

            # Track wins
            if dones.any():
                done_idx = dones.nonzero().squeeze(-1)
                for i in done_idx:
                    winner = new_state.winner[i].item()
                    if winner >= 0:
                        total_wins[winner] += 1
                total_games += dones.sum().item()

            # Reset done envs
            state = env.reset_done_envs(new_state, dones)
            total_steps += config.num_envs

        # Get bootstrap value
        with torch.no_grad():
            last_obs, last_mask = env.get_obs_and_mask(state)
            _, _, last_value = network.get_action(last_obs, last_mask)

        # Stack buffers
        obs_buf = torch.stack(obs_buf)  # [T, B, obs_dim]
        act_buf = torch.stack(act_buf)  # [T, B]
        logp_buf = torch.stack(logp_buf)  # [T, B]
        rew_buf = torch.stack(rew_buf)  # [T, B]
        done_buf = torch.stack(done_buf)  # [T, B]
        val_buf = torch.stack(val_buf)  # [T, B]
        mask_buf = torch.stack(mask_buf)  # [T, B, A]

        # Add bootstrap value
        val_with_bootstrap = torch.cat([val_buf, last_value.unsqueeze(0)], dim=0)

        # Compute GAE
        advantages, returns = compute_gae(
            rew_buf, val_with_bootstrap, done_buf, config.gamma, config.gae_lambda
        )

        # Flatten for minibatch updates
        T, B = obs_buf.shape[:2]
        obs_flat = obs_buf.view(T * B, -1)
        act_flat = act_buf.view(T * B)
        logp_flat = logp_buf.view(T * B)
        ret_flat = returns.view(T * B)
        adv_flat = advantages.view(T * B)
        mask_flat = mask_buf.view(T * B, -1)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # PPO update
        indices = torch.randperm(T * B, device=device)
        loss = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)

        for epoch in range(config.ppo_epochs):
            for start in range(0, T * B, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_flat[mb_idx]
                mb_act = act_flat[mb_idx]
                mb_logp_old = logp_flat[mb_idx]
                mb_ret = ret_flat[mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_mask = mask_flat[mb_idx]

                # Get current policy outputs
                new_logp, new_val, entropy = network.evaluate_actions(mb_obs, mb_mask, mb_act)

                # Policy loss
                ratio = torch.exp(new_logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                # Total loss
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
                optimizer.step()

        # Logging
        elapsed = time.time() - start_time
        sps = total_steps / elapsed

        if update % config.log_interval == 0:
            win_rate = total_wins[0] / max(1, total_games) * 100

            print(
                f"Update {update}/{num_updates} | "
                f"Steps: {total_steps:,} | "
                f"Games: {total_games} | "
                f"SPS: {sps:.0f} | "
                f"Win%: {win_rate:.1f}% | "
                f"Loss: {loss.item():.4f}"
            )

            writer.add_scalar("charts/SPS", sps, total_steps)
            writer.add_scalar("charts/games", total_games, total_steps)
            writer.add_scalar("charts/win_rate_p0", win_rate, total_steps)
            writer.add_scalar("losses/total", loss.item(), total_steps)
            writer.add_scalar("losses/policy", policy_loss.item(), total_steps)
            writer.add_scalar("losses/value", value_loss.item(), total_steps)
            writer.add_scalar("losses/entropy", entropy.item(), total_steps)

        # Save checkpoint
        if update % config.save_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"{run_name}_step_{total_steps}.pt")
            torch.save(
                {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": total_steps,
                    "games": total_games,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(config.checkpoint_dir, f"{run_name}_final.pt")
    torch.save(
        {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": total_steps,
            "games": total_games,
        },
        final_path,
    )

    print(f"\nTraining complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Total games: {total_games}")
    print(f"Final SPS: {sps:.0f}")
    print(f"Saved: {final_path}")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL Poker agent with PPO")

    # Environment
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")

    # Training
    parser.add_argument(
        "--total-timesteps", type=int, default=2_000_000, help="Total training timesteps"
    )
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")

    # PPO
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Network
    parser.add_argument("--hidden-size", type=int, default=256)

    # Logging
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()

    config = TrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        cuda=not args.no_cuda,
    )

    train(config)


if __name__ == "__main__":
    main()

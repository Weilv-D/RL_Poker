#!/usr/bin/env python3
"""GPU-native PPO training for RL Poker with opponent pool.

This module provides a high-performance training loop that runs entirely on GPU:
- Vectorized environments simulated in parallel
- Batched action mask computation
- PPO algorithm with fixed learner seat and sampled opponents (PSRO-lite)

Key features:
- ~10,000+ SPS (steps per second) on modern GPUs
- TensorBoard logging
- Checkpoint saving/loading
- Configurable hyperparameters
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torch.nn.functional as F

from rl_poker.rl import (
    GameState,
    GPUPokerEnv,
    PolicyNetwork,
    compute_gae,
    OpponentPool,
    GPURandomPolicy,
    GPUHeuristicPolicy,
    HistoryConfig,
    HistoryBuffer,
    RecurrentPolicyNetwork,
    build_response_rank_weights,
)


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

    # Opponent pool
    pool_max_size: int = 16
    pool_ema_beta: float = 0.05
    pool_psro_beta: float = 3.0
    pool_min_prob: float = 0.05
    pool_add_interval: int = 10
    pool_seed: int = 42
    pool_use_random: bool = True
    pool_use_heuristic: bool = True
    pool_heuristic_styles: str = "conservative,aggressive,rush,counter,variance"
    # Shaping reward
    shaping_alpha: float = 0.1
    shaping_anneal_updates: int = 200
    # Memory/belief
    use_recurrent: bool = True
    history_window: int = 16
    reveal_opponent_ranks: bool = False
    gru_hidden: int = 128
    belief_use_behavior: bool = True
    belief_decay: float = 0.98
    belief_play_bonus: float = 0.5
    belief_pass_penalty: float = 0.3
    belief_temp: float = 2.0

    # Logging
    log_interval: int = 5
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # Misc
    seed: int = 42
    cuda: bool = True


def _parse_heuristic_styles(styles: str) -> list[str]:
    if not styles:
        return []
    return [s.strip() for s in styles.split(",") if s.strip()]


def train(config: TrainConfig):
    """Main training loop."""

    # Setup device
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed
    _ = torch.manual_seed(config.seed)
    if device.type == "cuda":
        _ = torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create environment
    env = GPUPokerEnv(
        config.num_envs, device, reveal_opponent_ranks=config.reveal_opponent_ranks
    )
    # History feature setup
    num_action_types = int(env.mask_computer.action_types.max().item()) + 1
    max_action_length = int(env.mask_computer.action_lengths.max().item())
    history_feature_dim = 4 + num_action_types + 4  # player_onehot + type_onehot + 4 scalars
    history_config = HistoryConfig(
        enabled=config.use_recurrent,
        window=config.history_window,
        feature_dim=history_feature_dim,
    )
    history_buffer = HistoryBuffer(config.num_envs, history_config, device)

    augmented_obs_dim = env.obs_dim + 39 + 3 + 13

    print(f"Environments: {config.num_envs}")
    print(f"Observation dim (base): {env.obs_dim}")
    print(f"Observation dim (augmented): {augmented_obs_dim}")
    print(f"Action space: {env.num_actions}")

    # Public played counts for belief features
    public_played_counts = torch.zeros(config.num_envs, 13, device=device, dtype=torch.float32)
    opp_rank_logits = torch.zeros(config.num_envs, 4, 13, device=device, dtype=torch.float32)
    rank_positions = torch.arange(13, device=device)
    rank_dist = torch.abs(rank_positions.unsqueeze(0) - rank_positions.unsqueeze(1)).float()
    rank_affinity = torch.exp(-rank_dist / max(config.belief_temp, 1e-6))
    rank_affinity = rank_affinity / rank_affinity.sum(dim=1, keepdim=True).clamp(min=1e-6)
    response_rank_weights = build_response_rank_weights(env.mask_computer)

    def build_history_features(actions: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
        action_types = env.mask_computer.action_types[actions]
        action_ranks = env.mask_computer.action_ranks[actions].clamp(min=0)
        action_lengths = env.mask_computer.action_lengths[actions]
        action_is_exempt = env.mask_computer.action_is_exemption[actions].float()
        is_pass = (actions == 0).float()

        player_onehot = F.one_hot(players, num_classes=4).float()
        type_onehot = F.one_hot(action_types, num_classes=num_action_types).float()
        action_rank_norm = (action_ranks.float() / 12.0).unsqueeze(1)
        action_len_norm = (action_lengths.float() / max_action_length).unsqueeze(1)
        scalars = torch.cat(
            [action_rank_norm, action_len_norm, action_is_exempt.unsqueeze(1), is_pass.unsqueeze(1)],
            dim=1,
        )
        return torch.cat([player_onehot, type_onehot, scalars], dim=1)

    def augment_obs(obs: torch.Tensor, state: GameState) -> torch.Tensor:
        B = obs.shape[0]
        p_idx = state.current_player
        my_rank_counts = state.rank_counts.gather(
            1, p_idx.view(B, 1, 1).expand(-1, 1, 13)
        ).squeeze(1)
        remaining_rank_counts = (4.0 - my_rank_counts.float() - public_played_counts).clamp(min=0)

        belief_list: list[torch.Tensor] = []
        other_remaining_list: list[torch.Tensor] = []
        for offset in [1, 2, 3]:
            other_p = (p_idx + offset) % 4
            other_cards = state.cards_remaining.gather(1, other_p.view(B, 1)).squeeze(1)
            other_remaining_list.append(other_cards.float())
            logits = opp_rank_logits[torch.arange(B, device=device), other_p]
            weights = torch.softmax(logits, dim=1)
            weighted = remaining_rank_counts * weights
            norm = weighted.sum(dim=1).clamp(min=1.0)
            belief = weighted * (other_cards.float() / norm).unsqueeze(1)
            belief_list.append(belief)

        belief = torch.cat(belief_list, dim=1) / 4.0
        other_remaining = torch.stack(other_remaining_list, dim=1) / 13.0
        played_norm = public_played_counts / 4.0

        return torch.cat([obs, belief, other_remaining, played_norm], dim=1)

    augmented_obs_dim = env.obs_dim + 39 + 3 + 13

    # Create network
    if config.use_recurrent:
        network = RecurrentPolicyNetwork(
            augmented_obs_dim,
            env.num_actions,
            history_feature_dim,
            hidden_size=config.hidden_size,
            gru_hidden=config.gru_hidden,
        ).to(device)
        rec_net = cast(RecurrentPolicyNetwork, network)
        policy_net: PolicyNetwork | None = None
    else:
        network = PolicyNetwork(augmented_obs_dim, env.num_actions, config.hidden_size).to(device)
        policy_net = cast(PolicyNetwork, network)
        rec_net = None
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)

    # Opponent pool
    pool = OpponentPool(
        obs_dim=augmented_obs_dim,
        num_actions=env.num_actions,
        hidden_size=config.hidden_size,
        device=device,
        max_size=config.pool_max_size,
        ema_beta=config.pool_ema_beta,
        psro_beta=config.pool_psro_beta,
        min_prob=config.pool_min_prob,
        seed=config.pool_seed,
        recurrent=config.use_recurrent,
        history_feature_dim=history_feature_dim,
        gru_hidden=config.gru_hidden,
    )

    if config.pool_use_random:
        pool.add_fixed("random", GPURandomPolicy(seed=config.pool_seed), protected=True)

    if config.pool_use_heuristic:
        for style in _parse_heuristic_styles(config.pool_heuristic_styles):
            noise = 0.3 if style == "variance" else 0.0
            pool.add_fixed(
                f"heuristic_{style}",
                GPUHeuristicPolicy(env.mask_computer, style=style, noise_scale=noise),
                protected=True,
            )

    if not pool.entries:
        raise ValueError("Opponent pool is empty. Enable random or heuristic opponents.")

    # Initial opponent assignments: [num_envs, 3] for players 1,2,3
    opponent_ids = pool.sample_opponents(config.num_envs, seats=3)

    # Logging
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    run_name = f"rl_poker_{config.seed}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(config.log_dir, run_name))

    # Training state
    state = env.reset()
    if history_config.enabled:
        history_buffer.reset_envs(torch.arange(config.num_envs, device=device))
    public_played_counts.zero_()
    total_steps = 0
    total_games = 0
    total_wins = {i: 0 for i in range(4)}
    start_time = time.time()
    sps = 0.0

    # Calculate update counts
    batch_size = config.num_envs * config.rollout_steps
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.total_timesteps < batch_size:
        print(
            f"Warning: total_timesteps ({config.total_timesteps}) < batch_size ({batch_size}); running a single update."
        )
        num_updates = 1
    else:
        num_updates = max(1, config.total_timesteps // batch_size)
        if config.total_timesteps % batch_size != 0:
            print(
                f"Warning: total_timesteps ({config.total_timesteps}) not divisible by batch_size ({batch_size}); effective timesteps = {num_updates * batch_size}."
            )

    if config.num_minibatches <= 0:
        raise ValueError("num_minibatches must be >= 1")
    if config.num_minibatches > batch_size:
        raise ValueError(
            f"num_minibatches ({config.num_minibatches}) > batch_size ({batch_size}); reduce num_minibatches or increase batch size."
        )
    if batch_size % config.num_minibatches != 0:
        print(
            f"Warning: batch_size ({batch_size}) not divisible by num_minibatches ({config.num_minibatches}); last minibatch will be smaller."
        )
    minibatch_size = max(1, batch_size // config.num_minibatches)

    print(f"\nTraining for {num_updates} updates...")
    print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")

    def shaping_alpha(update_idx: int) -> float:
        if config.shaping_alpha <= 0 or config.shaping_anneal_updates <= 0:
            return 0.0
        progress = min(update_idx / config.shaping_anneal_updates, 1.0)
        return config.shaping_alpha * (1.0 - progress)

    def compute_shaping_bonus(
        opp_ids: np.ndarray, learner_scores: np.ndarray, opponent_scores: np.ndarray, alpha: float
    ) -> np.ndarray:
        if alpha <= 0:
            return np.zeros_like(learner_scores)

        bonuses = np.zeros_like(learner_scores, dtype=np.float32)
        for env_idx in range(opp_ids.shape[0]):
            opp_ev = []
            for seat_idx in range(opp_ids.shape[1]):
                entry = pool.entries[int(opp_ids[env_idx, seat_idx])]
                opp_ev.append(entry.stats.ev_ema)
            baseline = float(np.mean(opp_ev)) if opp_ev else 0.0
            episode_adv = float(learner_scores[env_idx] - np.mean(opponent_scores[env_idx]))
            bonuses[env_idx] = alpha * (episode_adv - baseline)
        return bonuses

    def update_pool_and_assignments(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        new_state: GameState,
        last_step_idx_buf: np.ndarray,
        rew_buf_ref: list[torch.Tensor],
        current_step: int,
        step_rew_ref: torch.Tensor | None,
    ) -> None:
        nonlocal total_games, total_wins, opponent_ids

        if not dones.any():
            return

        done_idx = dones.nonzero().squeeze(-1)
        done_idx_cpu = done_idx.cpu().numpy()

        # Update win counts
        for i in done_idx:
            winner = new_state.winner[i].item()
            if winner >= 0:
                total_wins[winner] += 1
        total_games += dones.sum().item()

        # Update opponent pool stats
        opp_ids_done = opponent_ids[done_idx_cpu]
        learner_scores = rewards[done_idx, 0].detach().cpu().numpy()
        opp_scores = rewards[done_idx, 1:4].detach().cpu().numpy()
        pool.update_stats(opp_ids_done, learner_scores, opp_scores)

        # Apply shaping reward to last learner step (before resampling opponents)
        alpha = shaping_alpha(update)
        if alpha > 0:
            bonuses = compute_shaping_bonus(opp_ids_done, learner_scores, opp_scores, alpha)
            for env_idx, bonus in zip(done_idx_cpu, bonuses):
                env_i = int(env_idx)
                last_idx = int(last_step_idx_buf[env_i])
                if last_idx == current_step and step_rew_ref is not None:
                    step_rew_ref[env_i] += float(bonus)
                elif 0 <= last_idx < len(rew_buf_ref):
                    rew_buf_ref[last_idx][env_i] += float(bonus)

        # Clear last step indices for finished envs
        last_step_idx_buf[done_idx_cpu] = -1

        # Resample opponents for finished envs
        opponent_ids[done_idx_cpu] = pool.sample_opponents(len(done_idx_cpu), seats=3)

    def select_opponent_actions(
        actions: torch.Tensor,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        current_player: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        current_cpu = current_player.cpu().numpy()
        active_cpu = active_mask.cpu().numpy()

        for player_id in (1, 2, 3):
            envs = np.where(active_cpu & (current_cpu == player_id))[0]
            if envs.size == 0:
                continue

            opp_ids = opponent_ids[envs, player_id - 1]
            for opp_id in np.unique(opp_ids):
                subset = envs[opp_ids == opp_id]
                if subset.size == 0:
                    continue
                idx = torch.tensor(subset, device=device, dtype=torch.long)
                policy = pool.entries[int(opp_id)].policy
                if config.use_recurrent:
                    seq = history_buffer.get_sequence(idx)
                    assert seq is not None
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx], seq=seq)
                else:
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx])

        return actions

    for update in range(1, num_updates + 1):
        # Storage for rollout
        obs_buf_list: list[torch.Tensor] = []
        act_buf_list: list[torch.Tensor] = []
        logp_buf_list: list[torch.Tensor] = []
        rew_buf_list: list[torch.Tensor] = []
        done_buf_list: list[torch.Tensor] = []
        val_buf_list: list[torch.Tensor] = []
        mask_buf_list: list[torch.Tensor] = []
        seq_buf_list: list[torch.Tensor] | None = [] if config.use_recurrent else None
        last_step_idx = -np.ones(config.num_envs, dtype=np.int32)

        # Collect rollout (one learner action per env per step)
        for step in range(config.rollout_steps):
            pending = torch.ones(config.num_envs, dtype=torch.bool, device=device)
            step_obs = torch.zeros(config.num_envs, augmented_obs_dim, device=device)
            step_act = torch.zeros(config.num_envs, dtype=torch.long, device=device)
            step_logp = torch.zeros(config.num_envs, device=device)
            step_val = torch.zeros(config.num_envs, device=device)
            step_rew = torch.zeros(config.num_envs, device=device)
            step_done = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
            step_mask = torch.zeros(config.num_envs, env.num_actions, dtype=torch.bool, device=device)
            step_seq: torch.Tensor | None = None
            if config.use_recurrent:
                step_seq = torch.zeros(
                    config.num_envs,
                    history_config.window,
                    history_feature_dim,
                    device=device,
                )

            while pending.any():
                obs, action_mask = env.get_obs_and_mask(state)
                obs = augment_obs(obs, state)
                actions = torch.zeros(config.num_envs, dtype=torch.long, device=device)

                learner_envs = pending & (state.current_player == 0)
                has_learner = bool(learner_envs.any().item())
                if has_learner:
                    idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
                    with torch.no_grad():
                        if config.use_recurrent:
                            assert rec_net is not None
                            seq_l = history_buffer.get_sequence(idx)
                            assert seq_l is not None
                            actions_l, logp_l, val_l = rec_net.get_action(
                                obs[idx], action_mask[idx], seq_l
                            )
                        else:
                            assert policy_net is not None
                            actions_l, logp_l, val_l = policy_net.get_action(
                                obs[idx], action_mask[idx]
                            )
                            seq_l = None
                    actions[idx] = actions_l

                opponent_active = pending & (state.current_player != 0)
                if opponent_active.any():
                    actions = select_opponent_actions(
                        actions, obs, action_mask, state.current_player, opponent_active
                    )

                new_state, rewards, dones = env.step(state, actions, active_mask=pending)
                total_steps += int(pending.sum().item())

                # Update public played counts, belief logits, and history
                played = (actions != 0) & pending
                if played.any():
                    action_counts = env.mask_computer.action_required_counts[actions].float()
                    public_played_counts[played] += action_counts[played]
                    if config.belief_use_behavior:
                        players = state.current_player
                        opp_rank_logits[played, players[played]] *= config.belief_decay
                        evidence = action_counts[played] @ rank_affinity
                        opp_rank_logits[played, players[played]] += (
                            config.belief_play_bonus * evidence
                        )

                if history_config.enabled:
                    env_ids = pending.nonzero(as_tuple=False).squeeze(-1)
                    if env_ids.numel() > 0:
                        feats = build_history_features(actions[env_ids], state.current_player[env_ids])
                        history_buffer.push(feats, env_ids=env_ids)

                passed = (actions == 0) & pending
                if passed.any() and config.belief_use_behavior:
                    players = state.current_player
                    prev_rank = state.prev_action_rank
                    valid = prev_rank >= 0
                    if valid.any():
                        envs = passed & valid
                        if envs.any():
                            prev_actions = state.prev_action[envs]
                            penalty = response_rank_weights[prev_actions] @ rank_affinity
                            opp_rank_logits[envs, players[envs]] -= (
                                config.belief_pass_penalty * penalty
                            )

                if has_learner:
                    step_obs[idx] = obs[idx]
                    step_act[idx] = actions[idx]
                    step_logp[idx] = logp_l
                    step_val[idx] = val_l
                    step_rew[idx] = rewards[idx, 0]
                    step_done[idx] = dones[idx]
                    step_mask[idx] = action_mask[idx]
                    if config.use_recurrent:
                        assert step_seq is not None
                        assert seq_l is not None
                        step_seq[idx] = seq_l
                    pending[idx] = False
                    idx_cpu = idx.detach().cpu().numpy()
                    last_step_idx[idx_cpu] = step

                if dones.any():
                    done_idx = dones.nonzero().squeeze(-1)
                    done_idx_cpu = done_idx.detach().cpu().numpy()
                    learner_envs_cpu = learner_envs.detach().cpu().numpy()
                    for env_id in done_idx_cpu:
                        if not learner_envs_cpu[env_id]:
                            last_idx = int(last_step_idx[env_id])
                            if 0 <= last_idx < len(rew_buf_list):
                                rew_buf_list[last_idx][env_id] = rewards[env_id, 0]
                                done_buf_list[last_idx][env_id] = True

                update_pool_and_assignments(
                    rewards,
                    dones,
                    new_state,
                    last_step_idx,
                    rew_buf_list,
                    step,
                    step_rew,
                )

                if dones.any():
                    done_idx = dones.nonzero().squeeze(-1)
                    public_played_counts[done_idx] = 0.0
                    opp_rank_logits[done_idx] = 0.0
                    if history_config.enabled:
                        history_buffer.reset_envs(done_idx)
                    state = env.reset_done_envs(new_state, dones)
                else:
                    state = new_state

            obs_buf_list.append(step_obs)
            act_buf_list.append(step_act)
            logp_buf_list.append(step_logp)
            rew_buf_list.append(step_rew)
            done_buf_list.append(step_done)
            val_buf_list.append(step_val)
            mask_buf_list.append(step_mask)
            if config.use_recurrent:
                assert seq_buf_list is not None
                assert step_seq is not None
                seq_buf_list.append(step_seq)

        # Bootstrap value: advance opponents until learner's turn, then evaluate value
        last_value = torch.zeros(config.num_envs, device=device)
        pending = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        while pending.any():
            obs, action_mask = env.get_obs_and_mask(state)
            obs = augment_obs(obs, state)

            learner_envs = pending & (state.current_player == 0)
            has_learner = bool(learner_envs.any().item())
            if has_learner:
                idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
                with torch.no_grad():
                    if config.use_recurrent:
                        assert rec_net is not None
                        seq = history_buffer.get_sequence(idx)
                        assert seq is not None
                        _, _, values = rec_net.get_action(obs[idx], action_mask[idx], seq)
                    else:
                        assert policy_net is not None
                        _, _, values = policy_net.get_action(obs[idx], action_mask[idx])
                last_value[idx] = values
                pending[idx] = False

            opponent_active = pending & (state.current_player != 0)
            if opponent_active.any():
                actions = torch.zeros(config.num_envs, dtype=torch.long, device=device)
                actions = select_opponent_actions(
                    actions, obs, action_mask, state.current_player, opponent_active
                )
                new_state, rewards, dones = env.step(state, actions, active_mask=opponent_active)
                total_steps += int(opponent_active.sum().item())

                played = (actions != 0) & opponent_active
                if played.any():
                    action_counts = env.mask_computer.action_required_counts[actions].float()
                    public_played_counts[played] += action_counts[played]
                    if config.belief_use_behavior:
                        players = state.current_player
                        opp_rank_logits[played, players[played]] *= config.belief_decay
                        evidence = action_counts[played] @ rank_affinity
                        opp_rank_logits[played, players[played]] += (
                            config.belief_play_bonus * evidence
                        )

                passed = (actions == 0) & opponent_active
                if passed.any() and config.belief_use_behavior:
                    players = state.current_player
                    prev_rank = state.prev_action_rank
                    valid = prev_rank >= 0
                    if valid.any():
                        envs = passed & valid
                        if envs.any():
                            prev_actions = state.prev_action[envs]
                            penalty = response_rank_weights[prev_actions] @ rank_affinity
                            opp_rank_logits[envs, players[envs]] -= (
                                config.belief_pass_penalty * penalty
                            )

                if history_config.enabled:
                    env_ids = opponent_active.nonzero(as_tuple=False).squeeze(-1)
                    if env_ids.numel() > 0:
                        feats = build_history_features(actions[env_ids], state.current_player[env_ids])
                        history_buffer.push(feats, env_ids=env_ids)

                if dones.any():
                    done_idx = dones.nonzero().squeeze(-1)
                    done_idx_cpu = done_idx.detach().cpu().numpy()
                    for env_id in done_idx_cpu:
                        last_idx = int(last_step_idx[env_id])
                        if 0 <= last_idx < len(rew_buf_list):
                            rew_buf_list[last_idx][env_id] = rewards[env_id, 0]
                            done_buf_list[last_idx][env_id] = True

                update_pool_and_assignments(
                    rewards,
                    dones,
                    new_state,
                    last_step_idx,
                    rew_buf_list,
                    -1,
                    None,
                )

                if dones.any():
                    done_idx = dones.nonzero().squeeze(-1)
                    public_played_counts[done_idx] = 0.0
                    opp_rank_logits[done_idx] = 0.0
                    if history_config.enabled:
                        history_buffer.reset_envs(done_idx)
                    state = env.reset_done_envs(new_state, dones)
                else:
                    state = new_state

        # Stack buffers
        obs_buf_tensor = torch.stack(obs_buf_list)  # [T, B, obs_dim]
        act_buf_tensor = torch.stack(act_buf_list)  # [T, B]
        logp_buf_tensor = torch.stack(logp_buf_list)  # [T, B]
        rew_buf_tensor = torch.stack(rew_buf_list)  # [T, B]
        done_buf_tensor = torch.stack(done_buf_list)  # [T, B]
        val_buf_tensor = torch.stack(val_buf_list)  # [T, B]
        mask_buf_tensor = torch.stack(mask_buf_list)  # [T, B, A]
        seq_buf_tensor: torch.Tensor | None = None
        if config.use_recurrent:
            assert seq_buf_list is not None
            seq_buf_tensor = torch.stack(seq_buf_list)  # [T, B, W, F]

        # Add bootstrap value
        val_with_bootstrap = torch.cat([val_buf_tensor, last_value.unsqueeze(0)], dim=0)

        # Compute GAE
        advantages, returns = compute_gae(
            rew_buf_tensor, val_with_bootstrap, done_buf_tensor, config.gamma, config.gae_lambda
        )

        # Flatten for minibatch updates
        T, B = obs_buf_tensor.shape[:2]
        obs_flat = obs_buf_tensor.view(T * B, -1)
        act_flat = act_buf_tensor.view(T * B)
        logp_flat = logp_buf_tensor.view(T * B)
        ret_flat = returns.view(T * B)
        adv_flat = advantages.view(T * B)
        mask_flat = mask_buf_tensor.view(T * B, -1)
        seq_flat: torch.Tensor | None = None
        if config.use_recurrent:
            assert seq_buf_tensor is not None
            seq_flat = seq_buf_tensor.view(T * B, history_config.window, history_feature_dim)

        # Normalize advantages (skip if variance is too small)
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std(unbiased=False)
        adv_norm_skipped = False
        if adv_std > 1e-6:
            adv_flat = (adv_flat - adv_mean) / (adv_std + 1e-8)
        else:
            adv_norm_skipped = True

        # PPO update
        indices = torch.randperm(T * B, device=device)
        loss = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
        mb_count = 0
        sum_loss = 0.0
        sum_policy = 0.0
        sum_value = 0.0
        sum_entropy = 0.0

        for epoch in range(config.ppo_epochs):
            for start in range(0, T * B, minibatch_size):
                end = min(start + minibatch_size, T * B)
                mb_idx = indices[start:end]

                mb_obs = obs_flat[mb_idx]
                mb_act = act_flat[mb_idx]
                mb_logp_old = logp_flat[mb_idx]
                mb_ret = ret_flat[mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_mask = mask_flat[mb_idx]

                # Get current policy outputs
                if config.use_recurrent:
                    assert rec_net is not None
                    assert seq_flat is not None
                    mb_seq = seq_flat[mb_idx]
                    new_logp, new_val, entropy = rec_net.evaluate_actions(
                        mb_obs, mb_mask, mb_act, mb_seq
                    )
                else:
                    assert policy_net is not None
                    new_logp, new_val, entropy = policy_net.evaluate_actions(
                        mb_obs, mb_mask, mb_act
                    )

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

                mb_count += 1
                sum_loss += float(loss.item())
                sum_policy += float(policy_loss.item())
                sum_value += float(value_loss.item())
                sum_entropy += float(entropy.item())

        # Logging
        elapsed = time.time() - start_time
        sps = total_steps / elapsed if elapsed > 0 else 0.0

        if update % config.log_interval == 0:
            win_rate = total_wins[0] / max(1, total_games) * 100
            pool_evs = [entry.stats.ev_ema for entry in pool.entries]
            pool_mean_ev = float(np.mean(pool_evs)) if pool_evs else 0.0
            pool_min_ev = float(np.min(pool_evs)) if pool_evs else 0.0
            denom = max(1, mb_count)
            mean_loss = sum_loss / denom
            mean_policy_loss = sum_policy / denom
            mean_value_loss = sum_value / denom
            mean_entropy = sum_entropy / denom

            print(
                f"Update {update}/{num_updates} | Steps: {total_steps:,} | Games: {total_games} | SPS: {sps:.0f} | Win%: {win_rate:.1f}% | Loss: {mean_loss:.4f} | Pool: {len(pool.entries)}"
            )
            if adv_norm_skipped:
                print("Warning: advantage std too small; skipped normalization for this update.")

            writer.add_scalar("charts/SPS", sps, total_steps)
            writer.add_scalar("charts/games", total_games, total_steps)
            writer.add_scalar("charts/win_rate_p0", win_rate, total_steps)
            writer.add_scalar("pool/size", len(pool.entries), total_steps)
            writer.add_scalar("pool/ev_mean", pool_mean_ev, total_steps)
            writer.add_scalar("pool/ev_min", pool_min_ev, total_steps)
            writer.add_scalar("losses/total", mean_loss, total_steps)
            writer.add_scalar("losses/policy", mean_policy_loss, total_steps)
            writer.add_scalar("losses/value", mean_value_loss, total_steps)
            writer.add_scalar("losses/entropy", mean_entropy, total_steps)

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

        # Add snapshot to pool
        if config.pool_add_interval > 0 and update % config.pool_add_interval == 0:
            pool.add_snapshot(f"snapshot_{total_steps}", network.state_dict(), added_step=total_steps)

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

    print("\nTraining complete!")
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

    # Opponent pool
    parser.add_argument("--pool-max-size", type=int, default=16)
    parser.add_argument("--pool-ema-beta", type=float, default=0.05)
    parser.add_argument("--pool-psro-beta", type=float, default=3.0)
    parser.add_argument("--pool-min-prob", type=float, default=0.05)
    parser.add_argument("--pool-add-interval", type=int, default=10)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--pool-no-random", action="store_true")
    parser.add_argument("--pool-no-heuristic", action="store_true")
    parser.add_argument("--pool-heuristic-styles", type=str, default="conservative,aggressive")
    # Shaping reward
    parser.add_argument("--shaping-alpha", type=float, default=0.1)
    parser.add_argument("--shaping-anneal-updates", type=int, default=200)
    # Memory/belief
    parser.add_argument("--no-recurrent", action="store_true")
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--reveal-opponent-ranks", action="store_true")
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--belief-no-behavior", action="store_true")
    parser.add_argument("--belief-decay", type=float, default=0.98)
    parser.add_argument("--belief-play-bonus", type=float, default=0.5)
    parser.add_argument("--belief-pass-penalty", type=float, default=0.3)
    parser.add_argument("--belief-temp", type=float, default=2.0)

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
        pool_max_size=args.pool_max_size,
        pool_ema_beta=args.pool_ema_beta,
        pool_psro_beta=args.pool_psro_beta,
        pool_min_prob=args.pool_min_prob,
        pool_add_interval=args.pool_add_interval,
        pool_seed=args.pool_seed,
        pool_use_random=not args.pool_no_random,
        pool_use_heuristic=not args.pool_no_heuristic,
        pool_heuristic_styles=args.pool_heuristic_styles,
        shaping_alpha=args.shaping_alpha,
        shaping_anneal_updates=args.shaping_anneal_updates,
        use_recurrent=not args.no_recurrent,
        history_window=args.history_window,
        reveal_opponent_ranks=args.reveal_opponent_ranks,
        gru_hidden=args.gru_hidden,
        belief_use_behavior=not args.belief_no_behavior,
        belief_decay=args.belief_decay,
        belief_play_bonus=args.belief_play_bonus,
        belief_pass_penalty=args.belief_pass_penalty,
        belief_temp=args.belief_temp,
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

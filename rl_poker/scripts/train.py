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
import re
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None
    log_dir: str = "runs"

    # Misc
    seed: int = 42
    cuda: bool = True
    resume_path: str | None = None
    resume_update: int | None = None


def _parse_heuristic_styles(styles: str) -> list[str]:
    if not styles:
        return []
    return [s.strip() for s in styles.split(",") if s.strip()]


def train(config: TrainConfig):
    """Main training loop."""

    # Setup device
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    other_offsets = torch.tensor([1, 2, 3], device=device)
    rank_positions = torch.arange(13, device=device)
    rank_dist = torch.abs(rank_positions.unsqueeze(0) - rank_positions.unsqueeze(1)).float()
    rank_affinity = torch.exp(-rank_dist / max(config.belief_temp, 1e-6))
    rank_affinity = rank_affinity / rank_affinity.sum(dim=1, keepdim=True).clamp(min=1e-6)
    response_rank_weights = build_response_rank_weights(env.mask_computer)

    player_onehot_table = torch.eye(4, device=device, dtype=torch.float32)
    action_types_table = env.mask_computer.action_types
    action_ranks_table = env.mask_computer.action_ranks.clamp(min=0)
    action_lengths_table = env.mask_computer.action_lengths
    action_is_exempt_table = env.mask_computer.action_is_exemption.float().unsqueeze(1)
    action_type_onehot_table = F.one_hot(action_types_table, num_classes=num_action_types).float()
    action_rank_norm_table = (action_ranks_table.float() / 12.0).unsqueeze(1)
    action_len_norm_table = (action_lengths_table.float() / max_action_length).unsqueeze(1)
    is_pass_table = (torch.arange(env.num_actions, device=device) == 0).float().unsqueeze(1)

    def build_history_features(actions: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
        player_onehot = player_onehot_table[players]
        type_onehot = action_type_onehot_table[actions]
        scalars = torch.cat(
            [
                action_rank_norm_table[actions],
                action_len_norm_table[actions],
                action_is_exempt_table[actions],
                is_pass_table[actions],
            ],
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

        other_p = (p_idx.view(B, 1) + other_offsets.view(1, 3)) % 4  # [B, 3]
        other_cards = state.cards_remaining.gather(1, other_p)  # [B, 3]
        logits = opp_rank_logits.gather(1, other_p.unsqueeze(-1).expand(-1, -1, 13))  # [B, 3, 13]
        weights = torch.softmax(logits, dim=2)
        weighted = remaining_rank_counts.unsqueeze(1) * weights  # [B, 3, 13]
        norm = weighted.sum(dim=2).clamp(min=1.0)
        belief = weighted * (other_cards.float() / norm).unsqueeze(-1)  # [B, 3, 13]

        belief = belief.reshape(B, -1) / 4.0
        other_remaining = other_cards.float() / 13.0
        played_norm = public_played_counts / 4.0

        return torch.cat([obs, belief, other_remaining, played_norm], dim=1)

    # Create network
    rec_net: RecurrentPolicyNetwork | None = None
    policy_net: PolicyNetwork | None = None
    if config.use_recurrent:
        network = RecurrentPolicyNetwork(
            augmented_obs_dim,
            env.num_actions,
            history_feature_dim,
            hidden_size=config.hidden_size,
            gru_hidden=config.gru_hidden,
        ).to(device)
        rec_net = network
    else:
        network = PolicyNetwork(augmented_obs_dim, env.num_actions, config.hidden_size).to(device)
        policy_net = network
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
    opponent_ids = torch.as_tensor(
        pool.sample_opponents(config.num_envs, seats=3), device=device, dtype=torch.long
    )

    # Logging
    checkpoint_root = os.path.abspath(config.checkpoint_dir)
    os.makedirs(checkpoint_root, exist_ok=True)
    def _sanitize_prefix(prefix: str) -> str:
        prefix = prefix.strip().replace(" ", "-")
        prefix = re.sub(r"[^A-Za-z0-9._-]", "_", prefix)
        return prefix or "run"

    def _next_checkpoint_id(prefix: str, root: str) -> int:
        prefix = _sanitize_prefix(prefix)
        max_id = 0
        for fname in os.listdir(root):
            stem, _ = os.path.splitext(fname)
            if not stem.startswith(prefix + "_"):
                continue
            tail = stem[len(prefix) + 1 :]
            # Expect tail like: 001_step_123 or 001_final
            if "_step_" in tail:
                suffix = tail.split("_step_")[0]
            elif tail.endswith("_final"):
                suffix = tail[: -len("_final")]
            else:
                suffix = tail
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
        return max_id + 1

    run_prefix = config.run_name or None
    if run_prefix:
        run_prefix = _sanitize_prefix(run_prefix)
        # Avoid nested <checkpoint>/<run>/<run> when checkpoint_dir already points to run folder
        base_name = os.path.basename(checkpoint_root.rstrip(os.sep))
        if base_name == run_prefix:
            run_dir = checkpoint_root
        else:
            run_dir = os.path.join(checkpoint_root, run_prefix)
        os.makedirs(run_dir, exist_ok=True)
        next_ckpt_id = _next_checkpoint_id(run_prefix, run_dir)
    else:
        base_name = os.path.basename(checkpoint_root.rstrip(os.sep))
        if base_name and base_name not in {"checkpoints", "checkpoint", "ckpt"}:
            run_prefix = _sanitize_prefix(base_name)
            next_ckpt_id = _next_checkpoint_id(run_prefix, checkpoint_root)
        else:
            run_prefix = "rl_poker"
            next_ckpt_id = _next_checkpoint_id(run_prefix, checkpoint_root)
        run_dir = checkpoint_root

    def _derive_prefix(path: str) -> str:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        m = re.match(r"^(.*)_\\d+_(step_|final)", stem)
        if m:
            return m.group(1)
        if "_step_" in stem:
            return stem.split("_step_")[0]
        if stem.endswith("_final"):
            return stem[: -len("_final")]
        return stem

    # Training state
    state = env.reset()
    if history_config.enabled:
        history_buffer.reset_envs(torch.arange(config.num_envs, device=device))
    public_played_counts.zero_()
    total_steps = 0
    total_games = 0
    total_wins = {i: 0 for i in range(4)}
    start_update = 1

    if config.resume_path:
        ckpt = torch.load(config.resume_path, map_location=device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Checkpoint {config.resume_path} is not a dict")
        net_state = ckpt.get("network")
        opt_state = ckpt.get("optimizer")
        if isinstance(net_state, dict):
            network.load_state_dict(net_state)
        else:
            raise ValueError("Checkpoint missing network state_dict")
        if isinstance(opt_state, dict):
            optimizer.load_state_dict(opt_state)
        total_steps = int(ckpt.get("step", 0))
        total_games = int(ckpt.get("games", 0))
        ckpt_wins = ckpt.get("wins")
        if isinstance(ckpt_wins, dict) and len(ckpt_wins) == 4:
            total_wins = {int(k): int(v) for k, v in ckpt_wins.items()}
        elif isinstance(ckpt_wins, list) and len(ckpt_wins) == 4:
            total_wins = {i: int(ckpt_wins[i]) for i in range(4)}
        else:
            total_games = 0
            total_wins = {i: 0 for i in range(4)}
            print("Warning: checkpoint missing win counts; Win% will be tracked from resume.")

        if config.resume_update is not None:
            start_update = int(config.resume_update) + 1
        else:
            ckpt_update = ckpt.get("update")
            if isinstance(ckpt_update, int):
                start_update = ckpt_update + 1
            else:
                print(
                    "Warning: checkpoint missing update index. "
                    "Pass --resume-update to continue without redoing updates."
                )
                start_update = 1

        run_prefix = _derive_prefix(config.resume_path)
        run_dir = os.path.dirname(os.path.abspath(config.resume_path))
        next_ckpt_id = _next_checkpoint_id(run_prefix, run_dir)
        if config.run_name:
            print("Note: --run-name ignored when resuming from checkpoint.")
        print(f"Resuming from {config.resume_path}")
        print(f"Start update: {start_update}")
        print(f"Total steps loaded: {total_steps:,}")

    # Setup logging file by run-name directory
    log_root = os.path.abspath(config.log_dir)
    os.makedirs(log_root, exist_ok=True)
    log_run_dir = os.path.join(log_root, run_prefix)
    os.makedirs(log_run_dir, exist_ok=True)
    log_path = os.path.join(log_run_dir, f"{run_prefix}_{next_ckpt_id:03d}.log")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    try:
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"Logging to {log_path}")
    except Exception as exc:
        print(f"Warning: could not open log file at {log_path}: {exc}")
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
            return np.zeros_like(learner_scores, dtype=np.float32)

        ev_ema = np.array([entry.stats.ev_ema for entry in pool.entries], dtype=np.float32)
        baseline = ev_ema[opp_ids].mean(axis=1) if opp_ids.size > 0 else 0.0
        episode_adv = learner_scores - opponent_scores.mean(axis=1)
        bonuses = alpha * (episode_adv - baseline)
        return bonuses.astype(np.float32)

    def update_pool_and_assignments(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        new_state: GameState,
        last_step_idx_buf: torch.Tensor,
        rew_buf_ref: torch.Tensor,
        current_step: int,
        step_rew_ref: torch.Tensor | None,
    ) -> None:
        nonlocal total_games, total_wins, opponent_ids

        if not dones.any():
            return

        done_idx = dones.nonzero().squeeze(-1)

        # Update win counts
        winners = new_state.winner[done_idx].detach().cpu().numpy()
        if winners.size > 0:
            valid_w = winners[winners >= 0]
            if valid_w.size > 0:
                counts = np.bincount(valid_w, minlength=4)
                for i in range(4):
                    total_wins[i] += int(counts[i])
        total_games += dones.sum().item()

        # Update opponent pool stats
        opp_ids_done = opponent_ids[done_idx].detach().cpu().numpy()
        learner_scores = rewards[done_idx, 0].detach().cpu().numpy()
        opp_scores = rewards[done_idx, 1:4].detach().cpu().numpy()
        pool.update_stats(opp_ids_done, learner_scores, opp_scores)

        # Apply shaping reward to last learner step (before resampling opponents)
        alpha = shaping_alpha(update)
        if alpha > 0:
            bonuses = compute_shaping_bonus(opp_ids_done, learner_scores, opp_scores, alpha)
            bonuses_t = torch.as_tensor(bonuses, device=device)
            last_idx = last_step_idx_buf[done_idx]
            if step_rew_ref is not None and current_step >= 0:
                mask_step = last_idx == current_step
                if mask_step.any():
                    env_ids = done_idx[mask_step]
                    step_rew_ref[env_ids] += bonuses_t[mask_step]
                mask_buf = (last_idx >= 0) & ~mask_step
            else:
                mask_buf = last_idx >= 0
            if mask_buf.any():
                env_ids = done_idx[mask_buf]
                step_ids = last_idx[mask_buf]
                rew_buf_ref[step_ids, env_ids] += bonuses_t[mask_buf]

        # Clear last step indices for finished envs
        last_step_idx_buf[done_idx] = -1

        # Resample opponents for finished envs
        opponent_ids[done_idx] = torch.as_tensor(
            pool.sample_opponents(done_idx.numel(), seats=3), device=device, dtype=torch.long
        )

    def select_opponent_actions(
        actions: torch.Tensor,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        current_player: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        for player_id in (1, 2, 3):
            env_mask = active_mask & (current_player == player_id)
            if not env_mask.any():
                continue
            envs = env_mask.nonzero(as_tuple=False).squeeze(-1)
            opp_ids = opponent_ids[envs, player_id - 1]
            unique_ids = torch.unique(opp_ids)
            for opp_id in unique_ids:
                subset_mask = opp_ids == opp_id
                if not subset_mask.any():
                    continue
                idx = envs[subset_mask]
                policy = pool.entries[int(opp_id.item())].policy
                if config.use_recurrent:
                    seq = history_buffer.get_sequence(idx)
                    assert seq is not None
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx], seq=seq)
                else:
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx])

        return actions

    # Storage for rollout (reused each update)
    obs_buf = torch.zeros(
        config.rollout_steps, config.num_envs, augmented_obs_dim, device=device
    )
    act_buf = torch.zeros(
        config.rollout_steps, config.num_envs, dtype=torch.long, device=device
    )
    logp_buf = torch.zeros(config.rollout_steps, config.num_envs, device=device)
    rew_buf = torch.zeros(config.rollout_steps, config.num_envs, device=device)
    done_buf = torch.zeros(
        config.rollout_steps, config.num_envs, dtype=torch.bool, device=device
    )
    val_buf = torch.zeros(config.rollout_steps, config.num_envs, device=device)
    mask_buf = torch.zeros(
        config.rollout_steps, config.num_envs, env.num_actions, dtype=torch.bool, device=device
    )
    seq_buf = None
    if config.use_recurrent:
        seq_buf = torch.zeros(
            config.rollout_steps,
            config.num_envs,
            history_config.window,
            history_feature_dim,
            device=device,
        )

    for update in range(start_update, num_updates + 1):
        # Reset rollout buffers
        obs_buf.zero_()
        act_buf.zero_()
        logp_buf.zero_()
        rew_buf.zero_()
        done_buf.zero_()
        val_buf.zero_()
        mask_buf.zero_()
        if config.use_recurrent:
            assert seq_buf is not None
            seq_buf.zero_()
        last_step_idx = torch.full(
            (config.num_envs,), -1, device=device, dtype=torch.long
        )

        with torch.inference_mode():
            # Collect rollout (one learner action per env per step)
            for step in range(config.rollout_steps):
                pending = torch.ones(config.num_envs, dtype=torch.bool, device=device)
                step_obs = obs_buf[step]
                step_act = act_buf[step]
                step_logp = logp_buf[step]
                step_val = val_buf[step]
                step_rew = rew_buf[step]
                step_done = done_buf[step]
                step_mask = mask_buf[step]
                step_obs.zero_()
                step_act.zero_()
                step_logp.zero_()
                step_val.zero_()
                step_rew.zero_()
                step_done.zero_()
                step_mask.zero_()
                step_seq: torch.Tensor | None = None
                if config.use_recurrent:
                    assert seq_buf is not None
                    step_seq = seq_buf[step]
                    step_seq.zero_()

                actions = torch.zeros(config.num_envs, dtype=torch.long, device=device)
                while pending.any():
                    actions.zero_()
                    obs, action_mask = env.get_obs_and_mask(state)
                    obs = augment_obs(obs, state)
                    idx: torch.Tensor | None = None
                    logp_l: torch.Tensor | None = None
                    val_l: torch.Tensor | None = None
                    seq_l: torch.Tensor | None = None

                    learner_envs = pending & (state.current_player == 0)
                    has_learner = bool(learner_envs.any().item())
                    if has_learner:
                        idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
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
                        assert idx is not None
                        assert logp_l is not None
                        assert val_l is not None
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
                        last_step_idx[idx] = step

                    if dones.any():
                        done_idx = dones.nonzero().squeeze(-1)
                        non_learner = done_idx[~learner_envs[done_idx]]
                        if non_learner.numel() > 0:
                            last_idx = last_step_idx[non_learner]
                            valid = last_idx >= 0
                            if valid.any():
                                env_ids = non_learner[valid]
                                step_ids = last_idx[valid]
                                rew_buf[step_ids, env_ids] = rewards[env_ids, 0]
                                done_buf[step_ids, env_ids] = True

                    update_pool_and_assignments(
                        rewards,
                        dones,
                        new_state,
                        last_step_idx,
                        rew_buf,
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

            # Bootstrap value: advance opponents until learner's turn, then evaluate value
            last_value = torch.zeros(config.num_envs, device=device)
            pending = torch.ones(config.num_envs, dtype=torch.bool, device=device)
            actions = torch.zeros(config.num_envs, dtype=torch.long, device=device)
            while pending.any():
                actions.zero_()
                obs, action_mask = env.get_obs_and_mask(state)
                obs = augment_obs(obs, state)

                learner_envs = pending & (state.current_player == 0)
                has_learner = bool(learner_envs.any().item())
                if has_learner:
                    idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
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
                        last_idx = last_step_idx[done_idx]
                        valid = last_idx >= 0
                        if valid.any():
                            env_ids = done_idx[valid]
                            step_ids = last_idx[valid]
                            rew_buf[step_ids, env_ids] = rewards[env_ids, 0]
                            done_buf[step_ids, env_ids] = True

                    update_pool_and_assignments(
                        rewards,
                        dones,
                        new_state,
                        last_step_idx,
                        rew_buf,
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

        # Rollout buffers (preallocated)
        obs_buf_tensor = obs_buf
        act_buf_tensor = act_buf
        logp_buf_tensor = logp_buf
        rew_buf_tensor = rew_buf
        done_buf_tensor = done_buf
        val_buf_tensor = val_buf
        mask_buf_tensor = mask_buf
        seq_buf_tensor: torch.Tensor | None = None
        if config.use_recurrent:
            assert seq_buf is not None
            seq_buf_tensor = seq_buf  # [T, B, W, F]

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
        for _epoch in range(config.ppo_epochs):
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
                _ = nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
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
            denom = max(1, mb_count)
            mean_loss = sum_loss / denom
            print(
                f"Update {update}/{num_updates} | Steps: {total_steps:,} | Games: {total_games} | SPS: {sps:.0f} | Win%: {win_rate:.1f}% | Loss: {mean_loss:.4f} | Pool: {len(pool.entries)}"
            )
            if adv_norm_skipped:
                print("Warning: advantage std too small; skipped normalization for this update.")

        # Save checkpoint
        if update % config.save_interval == 0:
            ckpt_path = os.path.join(run_dir, f"{run_prefix}_{next_ckpt_id:03d}_step_{total_steps}.pt")
            torch.save(
                {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": total_steps,
                    "games": total_games,
                    "wins": total_wins,
                    "update": update,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            next_ckpt_id += 1

        # Add snapshot to pool
        if config.pool_add_interval > 0 and update % config.pool_add_interval == 0:
            pool.add_snapshot_if_stronger(
                f"snapshot_{total_steps}", network.state_dict(), added_step=total_steps, candidate_ev=0.0
            )

    # Final save
    final_path = os.path.join(run_dir, f"{run_prefix}_{next_ckpt_id:03d}_final.pt")
    torch.save(
        {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": total_steps,
            "games": total_games,
            "wins": total_wins,
            "update": num_updates,
        },
        final_path,
    )

    print("\nTraining complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Total games: {total_games}")
    print(f"Final SPS: {sps:.0f}")
    print(f"Saved: {final_path}")

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
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name prefix (e.g., garlic -> garlic_001, garlic_002 ...)",
    )
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for training logs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--resume-update",
        type=int,
        default=None,
        help="Update index to resume from (if checkpoint lacks update)",
    )

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
        run_name=args.run_name,
        log_dir=args.log_dir,
        seed=args.seed,
        cuda=not args.no_cuda,
        resume_path=args.resume,
        resume_update=args.resume_update,
    )

    train(config)


if __name__ == "__main__":
    main()

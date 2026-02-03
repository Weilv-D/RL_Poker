#!/usr/bin/env python3
"""GPU evaluation for RL Poker policies.

Runs vectorized games on GPU against a configurable opponent pool.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from rl_poker.rl import (
    GPUPokerEnv,
    PolicyNetwork,
    RecurrentPolicyNetwork,
    OpponentPool,
    GPURandomPolicy,
    GPUHeuristicPolicy,
    HistoryConfig,
    HistoryBuffer,
)


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_ratings(ratings: List[float], ranks: List[int], k_factor: float = 32) -> List[float]:
    n_players = len(ratings)
    new_ratings = list(ratings)
    rank_map = {i: ranks[i] for i in range(n_players)}

    for i in range(n_players):
        total_expected = 0.0
        total_actual = 0.0
        for j in range(n_players):
            if i == j:
                continue
            expected = calculate_expected_score(ratings[i], ratings[j])
            total_expected += expected
            if rank_map[i] < rank_map[j]:
                actual = 1.0
            elif rank_map[i] > rank_map[j]:
                actual = 0.0
            else:
                actual = 0.5
            total_actual += actual
        rating_change = k_factor * (total_actual - total_expected)
        new_ratings[i] = ratings[i] + rating_change
    return new_ratings


@dataclass
class EvalConfig:
    episodes: int = 100
    num_envs: int = 128
    use_recurrent: bool = True
    history_window: int = 16
    reveal_opponent_ranks: bool = False
    hidden_size: int = 256
    gru_hidden: int = 128
    pool_max_size: int = 16
    pool_seed: int = 42
    pool_use_random: bool = True
    pool_use_heuristic: bool = True
    pool_heuristic_styles: str = "conservative,aggressive"
    snapshot_dir: str | None = None
    snapshot_glob: str = "*_step_*.pt"
    snapshot_max: int = 8
    elo_k: float = 32.0


def _parse_heuristic_styles(styles: str) -> List[str]:
    if not styles:
        return []
    return [s.strip() for s in styles.split(",") if s.strip()]


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", None)
    state_dict = ckpt["network"]
    return config, state_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Poker policy on GPU")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--no-recurrent", action="store_true")
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--reveal-opponent-ranks", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--pool-max-size", type=int, default=16)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--pool-no-random", action="store_true")
    parser.add_argument("--pool-no-heuristic", action="store_true")
    parser.add_argument("--pool-heuristic-styles", type=str, default="conservative,aggressive")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--snapshot-glob", type=str, default="*_step_*.pt")
    parser.add_argument("--snapshot-max", type=int, default=8)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    cfg = EvalConfig(
        episodes=args.episodes,
        num_envs=args.num_envs,
        use_recurrent=not args.no_recurrent,
        history_window=args.history_window,
        reveal_opponent_ranks=args.reveal_opponent_ranks,
        hidden_size=args.hidden_size,
        gru_hidden=args.gru_hidden,
        pool_max_size=args.pool_max_size,
        pool_seed=args.pool_seed,
        pool_use_random=not args.pool_no_random,
        pool_use_heuristic=not args.pool_no_heuristic,
        pool_heuristic_styles=args.pool_heuristic_styles,
        snapshot_dir=args.snapshot_dir,
        snapshot_glob=args.snapshot_glob,
        snapshot_max=args.snapshot_max,
        elo_k=args.elo_k,
    )

    ckpt_config, state_dict = load_checkpoint(args.checkpoint, device)
    if ckpt_config is not None:
        cfg.use_recurrent = getattr(ckpt_config, "use_recurrent", cfg.use_recurrent)
        cfg.history_window = getattr(ckpt_config, "history_window", cfg.history_window)
        cfg.reveal_opponent_ranks = getattr(
            ckpt_config, "reveal_opponent_ranks", cfg.reveal_opponent_ranks
        )
        cfg.hidden_size = getattr(ckpt_config, "hidden_size", cfg.hidden_size)
        cfg.gru_hidden = getattr(ckpt_config, "gru_hidden", cfg.gru_hidden)

    env = GPUPokerEnv(cfg.num_envs, device, reveal_opponent_ranks=cfg.reveal_opponent_ranks)

    num_action_types = int(env.mask_computer.action_types.max().item()) + 1
    max_action_length = int(env.mask_computer.action_lengths.max().item())
    history_feature_dim = 4 + num_action_types + 4
    history_cfg = HistoryConfig(
        enabled=cfg.use_recurrent, window=cfg.history_window, feature_dim=history_feature_dim
    )
    history_buffer = HistoryBuffer(cfg.num_envs, history_cfg, device)

    public_played_counts = torch.zeros(cfg.num_envs, 13, device=device, dtype=torch.float32)

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

    def augment_obs(obs: torch.Tensor, state) -> torch.Tensor:
        B = obs.shape[0]
        p_idx = state.current_player
        my_rank_counts = state.rank_counts.gather(
            1, p_idx.view(B, 1, 1).expand(-1, 1, 13)
        ).squeeze(1)
        remaining_rank_counts = (4.0 - my_rank_counts.float() - public_played_counts).clamp(min=0)
        total_unknown = remaining_rank_counts.sum(dim=1).clamp(min=1.0)

        beliefs = []
        other_remaining = []
        for offset in [1, 2, 3]:
            other_p = (p_idx + offset) % 4
            other_cards = state.cards_remaining.gather(1, other_p.view(B, 1)).squeeze(1)
            other_remaining.append(other_cards.float())
            frac = (other_cards.float() / total_unknown).unsqueeze(1)
            beliefs.append(remaining_rank_counts * frac)

        belief = torch.cat(beliefs, dim=1) / 4.0
        other_remaining = torch.stack(other_remaining, dim=1) / 13.0
        played_norm = public_played_counts / 4.0

        return torch.cat([obs, belief, other_remaining, played_norm], dim=1)

    augmented_obs_dim = env.obs_dim + 39 + 3 + 13

    if cfg.use_recurrent:
        network = RecurrentPolicyNetwork(
            augmented_obs_dim,
            env.num_actions,
            history_feature_dim,
            hidden_size=cfg.hidden_size,
            gru_hidden=cfg.gru_hidden,
        ).to(device)
    else:
        network = PolicyNetwork(augmented_obs_dim, env.num_actions, cfg.hidden_size).to(device)

    network.load_state_dict(state_dict)
    network.eval()

    pool = OpponentPool(
        obs_dim=augmented_obs_dim,
        num_actions=env.num_actions,
        hidden_size=cfg.hidden_size,
        device=device,
        max_size=cfg.pool_max_size,
        seed=cfg.pool_seed,
        recurrent=cfg.use_recurrent,
        history_feature_dim=history_feature_dim,
        gru_hidden=cfg.gru_hidden,
    )

    if cfg.pool_use_random:
        pool.add_fixed("random", GPURandomPolicy(seed=cfg.pool_seed), protected=True)
    if cfg.pool_use_heuristic:
        for style in _parse_heuristic_styles(cfg.pool_heuristic_styles):
            pool.add_fixed(
                f"heuristic_{style}", GPUHeuristicPolicy(env.mask_computer, style=style), protected=True
            )

    if cfg.snapshot_dir:
        snapshot_paths = sorted(Path(cfg.snapshot_dir).glob(cfg.snapshot_glob))
        if cfg.snapshot_max > 0:
            snapshot_paths = snapshot_paths[-cfg.snapshot_max :]
        for path in snapshot_paths:
            ckpt = torch.load(path, map_location=device)
            pool.add_snapshot(path.stem, ckpt["network"], added_step=ckpt.get("step", 0))

    if not pool.entries:
        raise ValueError("Opponent pool is empty. Enable random/heuristic or snapshot dir.")

    opponent_ids = pool.sample_opponents(cfg.num_envs, seats=3)

    state = env.reset()
    public_played_counts.zero_()
    if history_cfg.enabled:
        history_buffer.reset_envs(torch.arange(cfg.num_envs, device=device))

    scores = []
    ranks = []
    elo_ratings = [1000.0] * 4
    episodes_done = 0

    with torch.no_grad():
        while episodes_done < cfg.episodes:
            obs, action_mask = env.get_obs_and_mask(state)
            obs = augment_obs(obs, state)
            actions = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

            learner_envs = state.current_player == 0
            if learner_envs.any():
                idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
                if cfg.use_recurrent:
                    seq = history_buffer.get_sequence(idx)
                    actions_l, _, _ = network.get_action(obs[idx], action_mask[idx], seq)
                else:
                    actions_l, _, _ = network.get_action(obs[idx], action_mask[idx])
                actions[idx] = actions_l

            opponent_envs = state.current_player != 0
            if opponent_envs.any():
                current_cpu = state.current_player.cpu().numpy()
                active_cpu = opponent_envs.cpu().numpy()
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
                        if cfg.use_recurrent:
                            seq = history_buffer.get_sequence(idx)
                            actions[idx] = policy.select_actions(obs[idx], action_mask[idx], seq=seq)
                        else:
                            actions[idx] = policy.select_actions(obs[idx], action_mask[idx])

            new_state, rewards, dones = env.step(state, actions)

            played = actions != 0
            if played.any():
                action_counts = env.mask_computer.action_required_counts[actions]
                public_played_counts[played] += action_counts[played].float()

            if history_cfg.enabled:
                env_ids = torch.arange(cfg.num_envs, device=device)
                feats = build_history_features(actions, state.current_player)
                history_buffer.push(feats, env_ids=env_ids)

            if dones.any():
                done_idx = dones.nonzero().squeeze(-1)
                scores.extend(rewards[done_idx, 0].detach().cpu().numpy().tolist())
                batch_ranks = new_state.finish_rank[done_idx].detach().cpu().numpy().tolist()
                ranks.extend([r[0] for r in batch_ranks])
                for r in batch_ranks:
                    elo_ratings = update_elo_ratings(elo_ratings, r, k_factor=cfg.elo_k)
                episodes_done += len(done_idx)
                public_played_counts[done_idx] = 0.0
                if history_cfg.enabled:
                    history_buffer.reset_envs(done_idx)
                state = env.reset_done_envs(new_state, dones)
                opponent_ids[done_idx.cpu().numpy()] = pool.sample_opponents(len(done_idx), seats=3)
            else:
                state = new_state

    scores = np.array(scores[: cfg.episodes], dtype=np.float32)
    ranks = np.array(ranks[: cfg.episodes], dtype=np.int32)
    win_rate = np.mean(ranks <= 2) * 100 if ranks.size > 0 else 0.0
    avg_rank = np.mean(ranks) if ranks.size > 0 else 0.0

    print("\nEvaluation results")
    print(f"Episodes: {cfg.episodes}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Win rate (top2): {win_rate:.1f}%")
    print(f"Average rank: {avg_rank:.2f}")
    print(f"Elo (seat0/1/2/3): {[round(r, 1) for r in elo_ratings]}")


if __name__ == "__main__":
    main()

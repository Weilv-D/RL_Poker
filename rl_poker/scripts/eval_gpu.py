#!/usr/bin/env python3
"""GPU evaluation for RL Poker policies.

Runs vectorized games on GPU against a configurable opponent pool.
Supports evaluating a single checkpoint or a directory of checkpoints.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast
import json

import numpy as np
import torch
import torch.nn.functional as F

from rl_poker.rl import (
    GameState,
    GPUPokerEnv,
    PolicyNetwork,
    RecurrentPolicyNetwork,
    OpponentPool,
    GPURandomPolicy,
    GPUHeuristicPolicy,
    HistoryConfig,
    HistoryBuffer,
    build_response_rank_weights,
)
from rl_poker.scripts.train import TrainConfig as TrainConfig


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_ratings(ratings: list[float], ranks: list[int], k_factor: float = 32) -> list[float]:
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
    pool_heuristic_styles: str = "conservative,aggressive,rush,counter,variance"
    snapshot_dir: str | None = None
    snapshot_glob: str = "*_step_*.pt"
    snapshot_max: int = 8
    elo_k: float = 32.0
    belief_use_behavior: bool = True
    belief_decay: float = 0.98
    belief_play_bonus: float = 0.5
    belief_pass_penalty: float = 0.3
    belief_temp: float = 2.0
    progress_interval: int = 50


def _parse_heuristic_styles(styles: str) -> list[str]:
    if not styles:
        return []
    return [s.strip() for s in styles.split(",") if s.strip()]


def load_checkpoint(path: str, device: torch.device) -> tuple[object | None, dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict")
    config = ckpt.get("config", None)
    state_dict = ckpt.get("network")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing 'network' state_dict")
    return config, cast(dict[str, torch.Tensor], state_dict)


def evaluate_checkpoint(path: str, cfg: EvalConfig, device: torch.device) -> dict[str, object]:
    ckpt_config, state_dict = load_checkpoint(path, device)
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
    opp_rank_logits = torch.zeros(cfg.num_envs, 4, 13, device=device, dtype=torch.float32)
    rank_positions = torch.arange(13, device=device)
    rank_dist = torch.abs(rank_positions.unsqueeze(0) - rank_positions.unsqueeze(1)).float()
    rank_affinity = torch.exp(-rank_dist / max(cfg.belief_temp, 1e-6))
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

    rec_net: RecurrentPolicyNetwork | None = None
    policy_net: PolicyNetwork | None = None
    if cfg.use_recurrent:
        network = RecurrentPolicyNetwork(
            augmented_obs_dim,
            env.num_actions,
            history_feature_dim,
            hidden_size=cfg.hidden_size,
            gru_hidden=cfg.gru_hidden,
        ).to(device)
        rec_net = network
    else:
        network = PolicyNetwork(augmented_obs_dim, env.num_actions, cfg.hidden_size).to(device)
        policy_net = network

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
            noise = 0.3 if style == "variance" else 0.0
            pool.add_fixed(
                f"heuristic_{style}",
                GPUHeuristicPolicy(env.mask_computer, style=style, noise_scale=noise),
                protected=True,
            )

    if cfg.snapshot_dir:
        snapshot_paths = sorted(Path(cfg.snapshot_dir).glob(cfg.snapshot_glob))
        if cfg.snapshot_max > 0:
            snapshot_paths = snapshot_paths[-cfg.snapshot_max :]
        for snap_path in snapshot_paths:
            ckpt = torch.load(snap_path, map_location=device)
            pool.add_snapshot(snap_path.stem, ckpt["network"], added_step=ckpt.get("step", 0))

    if not pool.entries:
        raise ValueError("Opponent pool is empty. Enable random/heuristic or snapshot dir.")

    opponent_ids = torch.as_tensor(
        pool.sample_opponents(cfg.num_envs, seats=3), device=device, dtype=torch.long
    )

    state = env.reset()
    public_played_counts.zero_()
    if history_cfg.enabled:
        history_buffer.reset_envs(torch.arange(cfg.num_envs, device=device))

    scores: list[float] = []
    ranks: list[int] = []
    elo_ratings = [1000.0] * 4
    episodes_done = 0
    progress_steps: list[int] = []
    progress_mean_score: list[float] = []
    progress_win_rate: list[float] = []
    progress_avg_rank: list[float] = []
    progress_elo: list[list[float]] = []

    opp_games = np.zeros(len(pool.entries), dtype=np.int64)
    opp_score_sum = np.zeros(len(pool.entries), dtype=np.float64)
    opp_rank_sum = np.zeros(len(pool.entries), dtype=np.float64)

    def maybe_record_progress() -> None:
        if cfg.episodes <= 0:
            return
        if episodes_done == 0:
            return
        if episodes_done % cfg.progress_interval != 0 and episodes_done != cfg.episodes:
            return
        recent_scores = scores[-cfg.progress_interval :] if scores else []
        recent_ranks = ranks[-cfg.progress_interval :] if ranks else []
        if recent_scores:
            mean_score = float(np.mean(recent_scores))
        else:
            mean_score = 0.0
        if recent_ranks:
            win_rate = float(np.mean(np.array(recent_ranks) <= 2) * 100.0)
            avg_rank = float(np.mean(recent_ranks))
        else:
            win_rate = 0.0
            avg_rank = 0.0
        progress_steps.append(episodes_done)
        progress_mean_score.append(mean_score)
        progress_win_rate.append(win_rate)
        progress_avg_rank.append(avg_rank)
        progress_elo.append([round(r, 2) for r in elo_ratings])

    def select_opponent_actions(
        actions: torch.Tensor,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        current_player: torch.Tensor,
    ) -> torch.Tensor:
        for player_id in (1, 2, 3):
            env_mask = current_player == player_id
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
                if cfg.use_recurrent:
                    seq = history_buffer.get_sequence(idx)
                    assert seq is not None
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx], seq=seq)
                else:
                    actions[idx] = policy.select_actions(obs[idx], action_mask[idx])
        return actions

    with torch.inference_mode():
        while episodes_done < cfg.episodes:
            obs, action_mask = env.get_obs_and_mask(state)
            obs = augment_obs(obs, state)
            actions = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

            learner_envs = state.current_player == 0
            if learner_envs.any():
                idx = learner_envs.nonzero(as_tuple=False).squeeze(-1)
                if cfg.use_recurrent:
                    assert rec_net is not None
                    seq = history_buffer.get_sequence(idx)
                    assert seq is not None
                    actions_l, _, _ = rec_net.get_action(obs[idx], action_mask[idx], seq)
                else:
                    assert policy_net is not None
                    actions_l, _, _ = policy_net.get_action(obs[idx], action_mask[idx])
                actions[idx] = actions_l

            opponent_envs = state.current_player != 0
            if opponent_envs.any():
                actions = select_opponent_actions(actions, obs, action_mask, state.current_player)

            new_state, rewards, dones = env.step(state, actions)

            played = actions != 0
            if played.any():
                action_counts = env.mask_computer.action_required_counts[actions].float()
                public_played_counts[played] += action_counts[played]
                if cfg.belief_use_behavior:
                    players = state.current_player
                    opp_rank_logits[played, players[played]] *= cfg.belief_decay
                    evidence = action_counts[played] @ rank_affinity
                    opp_rank_logits[played, players[played]] += (
                        cfg.belief_play_bonus * evidence
                    )

            passed = actions == 0
            if passed.any() and cfg.belief_use_behavior:
                players = state.current_player
                prev_rank = state.prev_action_rank
                valid = prev_rank >= 0
                if valid.any():
                    envs = passed & valid
                    if envs.any():
                        prev_actions = state.prev_action[envs]
                        penalty = response_rank_weights[prev_actions] @ rank_affinity
                        opp_rank_logits[envs, players[envs]] -= (
                            cfg.belief_pass_penalty * penalty
                        )

            if history_cfg.enabled:
                env_ids = torch.arange(cfg.num_envs, device=device)
                feats = build_history_features(actions, state.current_player)
                history_buffer.push(feats, env_ids=env_ids)

            if dones.any():
                done_idx = dones.nonzero().squeeze(-1)
                batch_scores = rewards[done_idx, 0].detach().cpu().numpy()
                scores.extend(batch_scores.tolist())
                batch_ranks_full = new_state.finish_rank[done_idx].detach().cpu().numpy()
                batch_ranks = [r[0] for r in batch_ranks_full]
                ranks.extend(batch_ranks)
                for r in batch_ranks_full:
                    elo_ratings = update_elo_ratings(elo_ratings, r, k_factor=cfg.elo_k)
                episodes_done += len(done_idx)

                opp_ids_done = opponent_ids[done_idx].detach().cpu().numpy()
                for i in range(opp_ids_done.shape[0]):
                    for seat in range(opp_ids_done.shape[1]):
                        oid = int(opp_ids_done[i, seat])
                        opp_games[oid] += 1
                        opp_score_sum[oid] += float(batch_scores[i])
                        opp_rank_sum[oid] += float(batch_ranks[i])

                maybe_record_progress()
                public_played_counts[done_idx] = 0.0
                opp_rank_logits[done_idx] = 0.0
                if history_cfg.enabled:
                    history_buffer.reset_envs(done_idx)
                state = env.reset_done_envs(new_state, dones)
                opponent_ids[done_idx] = torch.as_tensor(
                    pool.sample_opponents(len(done_idx), seats=3),
                    device=device,
                    dtype=torch.long,
                )
            else:
                state = new_state

    scores = np.array(scores[: cfg.episodes], dtype=np.float32)
    ranks = np.array(ranks[: cfg.episodes], dtype=np.int32)
    win_rate = np.mean(ranks <= 2) * 100 if ranks.size > 0 else 0.0
    avg_rank = np.mean(ranks) if ranks.size > 0 else 0.0
    win_rate_top1 = np.mean(ranks == 1) * 100 if ranks.size > 0 else 0.0
    rank_hist = [int((ranks == i).sum()) for i in range(1, 5)] if ranks.size > 0 else [0, 0, 0, 0]

    opp_breakdown = []
    for idx, entry in enumerate(pool.entries):
        games = int(opp_games[idx])
        opp_breakdown.append(
            {
                "name": entry.name,
                "games": games,
                "avg_score": float(opp_score_sum[idx] / games) if games > 0 else 0.0,
                "avg_rank": float(opp_rank_sum[idx] / games) if games > 0 else 0.0,
                "protected": entry.protected,
            }
        )

    return {
        "checkpoint": str(path),
        "episodes": cfg.episodes,
        "mean_score": float(scores.mean()) if scores.size > 0 else 0.0,
        "win_rate": float(win_rate),
        "win_rate_top1": float(win_rate_top1),
        "avg_rank": float(avg_rank),
        "score_std": float(scores.std()) if scores.size > 0 else 0.0,
        "score_min": float(scores.min()) if scores.size > 0 else 0.0,
        "score_max": float(scores.max()) if scores.size > 0 else 0.0,
        "rank_hist": rank_hist,
        "elo": [round(r, 1) for r in elo_ratings],
        "progress": {
            "episodes": progress_steps,
            "mean_score": progress_mean_score,
            "win_rate": progress_win_rate,
            "avg_rank": progress_avg_rank,
            "elo": progress_elo,
        },
        "opponent_breakdown": opp_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Poker policy on GPU")
    default_root = Path(__file__).resolve().parents[1] / "checkpoints"
    default_ckpt_dir = str(default_root)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--checkpoint-glob", type=str, default="*_step_*.pt")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for log directory")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for evaluation logs")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--plot", type=str, default=None, help="Save plot image (png/svg)")
    parser.add_argument("--no-recurrent", action="store_true")
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--reveal-opponent-ranks", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--pool-max-size", type=int, default=16)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--pool-no-random", action="store_true")
    parser.add_argument("--pool-no-heuristic", action="store_true")
    parser.add_argument("--pool-heuristic-styles", type=str, default="conservative,aggressive,rush,counter,variance")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--snapshot-glob", type=str, default="*_step_*.pt")
    parser.add_argument("--snapshot-max", type=int, default=8)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--belief-no-behavior", action="store_true")
    parser.add_argument("--belief-decay", type=float, default=0.98)
    parser.add_argument("--belief-play-bonus", type=float, default=0.5)
    parser.add_argument("--belief-pass-penalty", type=float, default=0.3)
    parser.add_argument("--belief-temp", type=float, default=2.0)
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_dir is None:
        raise SystemExit("Provide --checkpoint or --checkpoint-dir")

    def _sanitize_prefix(prefix: str) -> str:
        prefix = prefix.strip().replace(" ", "-")
        return re.sub(r"[^A-Za-z0-9._-]", "_", prefix) or "eval"

    def _derive_prefix_from_checkpoint(path: str) -> str:
        stem = Path(path).stem
        m = re.match(r"^(.*)_\\d+_(step_|final)", stem)
        if m:
            return m.group(1)
        return stem

    def _allocate_eval_log(prefix: str, root: str) -> str:
        os.makedirs(root, exist_ok=True)
        prefix = _sanitize_prefix(prefix)
        max_id = 0
        for fname in os.listdir(root):
            stem, _ = os.path.splitext(fname)
            if not stem.startswith(prefix + "_eval_"):
                continue
            tail = stem[len(prefix) + len("_eval_") :]
            if tail.isdigit():
                max_id = max(max_id, int(tail))
        return os.path.join(root, f"{prefix}_eval_{max_id + 1:03d}.log")

    run_prefix = args.run_name
    if not run_prefix and args.checkpoint:
        run_prefix = _derive_prefix_from_checkpoint(args.checkpoint)
    if not run_prefix and args.checkpoint_dir:
        base = Path(args.checkpoint_dir).resolve().name
        if base not in {"checkpoints", "checkpoint", "ckpt"}:
            run_prefix = base
    if not run_prefix:
        run_prefix = "eval"

    log_run_dir = os.path.join(os.path.abspath(args.log_dir), _sanitize_prefix(run_prefix))
    log_path = _allocate_eval_log(run_prefix, log_run_dir)

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
        belief_use_behavior=not args.belief_no_behavior,
        belief_decay=args.belief_decay,
        belief_play_bonus=args.belief_play_bonus,
        belief_pass_penalty=args.belief_pass_penalty,
        belief_temp=args.belief_temp,
        progress_interval=args.progress_interval,
    )

    results = []
    if args.checkpoint:
        results.append(evaluate_checkpoint(args.checkpoint, cfg, device))
    else:
        ckpts = sorted(Path(args.checkpoint_dir).rglob(args.checkpoint_glob))
        for path in ckpts:
            results.append(evaluate_checkpoint(str(path), cfg, device))

    if args.output_json:
        output_path = Path(args.output_json)
    elif args.checkpoint_dir:
        output_path = Path(args.checkpoint_dir) / "eval_results.json"
    else:
        output_path = None

    if output_path:
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Saved eval results to {output_path}")

    for res in results:
        print("\nEvaluation results")
        print(f"Checkpoint: {res['checkpoint']}")
        print(f"Episodes: {res['episodes']}")
        print(f"Mean score: {res['mean_score']:.3f}")
        print(f"Win rate (top1): {res.get('win_rate_top1', 0.0):.1f}%")
        print(f"Win rate (top2): {res['win_rate']:.1f}%")
        print(f"Average rank: {res['avg_rank']:.2f}")
        print(f"Score std: {res.get('score_std', 0.0):.3f}")
        print(f"Elo (seat0/1/2/3): {res['elo']}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plot skipped (matplotlib not available): {exc}")
            return

        for res in results:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(Path(res["checkpoint"]).name)

            # Rank histogram
            rank_hist = res.get("rank_hist", [0, 0, 0, 0])
            axes[0, 0].bar([1, 2, 3, 4], rank_hist, color="#4C78A8")
            axes[0, 0].set_title("Rank Distribution")
            axes[0, 0].set_xlabel("Rank")
            axes[0, 0].set_ylabel("Count")

            # Score summary
            mean_score = res.get("mean_score", 0.0)
            std_score = res.get("score_std", 0.0)
            axes[0, 1].bar([0], [mean_score], color="#72B7B2", width=0.6)
            axes[0, 1].errorbar([0], [mean_score], yerr=[std_score], color="black", capsize=6)
            axes[0, 1].set_title("Score Mean ± Std")
            axes[0, 1].set_xticks([0])
            axes[0, 1].set_xticklabels(["score"])

            # Progress curves
            progress = res.get("progress", {})
            ep = progress.get("episodes", [])
            mean_score = progress.get("mean_score", [])
            win_rate = progress.get("win_rate", [])
            axes[1, 0].plot(ep, mean_score, label="Mean Score", color="#54A24B")
            axes[1, 0].set_title("Mean Score (Progress)")
            axes[1, 0].set_xlabel("Episodes")
            axes[1, 0].set_ylabel("Mean Score")

            axes[1, 1].plot(ep, win_rate, label="Win% (Top2)", color="#E45756")
            axes[1, 1].set_title("Win Rate (Progress)")
            axes[1, 1].set_xlabel("Episodes")
            axes[1, 1].set_ylabel("Win%")

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = Path(args.plot)
            if len(results) > 1:
                plot_path = plot_path.with_name(
                    plot_path.stem + "_" + Path(res["checkpoint"]).stem + plot_path.suffix
                )
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from rl_poker.rl import GPUPokerEnv, PolicyNetwork, compute_gae


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

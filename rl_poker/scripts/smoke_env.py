#!/usr/bin/env python3
"""Smoke test for the RL Poker AEC environment.

This script runs N episodes of the environment with random actions
to verify basic functionality:
- Environment reset works
- Agent iteration works
- Action masking is correct
- Episodes complete without errors
- Observations have correct shapes

Usage:
    python -m rl_poker.scripts.smoke_env --episodes 5
    python -m rl_poker.scripts.smoke_env --episodes 10 --seed 42 --verbose
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np

from rl_poker.envs import RLPokerAECEnv, NUM_CARDS, MOVE_VECTOR_SIZE
from rl_poker.moves.action_encoding import MAX_ACTIONS
from rl_poker.engine.game_state import NUM_PLAYERS


def run_episode(
    env: RLPokerAECEnv,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run a single episode with random actions.

    Args:
        env: The environment instance
        seed: Random seed for the episode
        verbose: If True, print detailed output

    Returns:
        Dict with episode statistics
    """
    stats = {
        "total_steps": 0,
        "rewards": {},
        "action_mask_errors": 0,
        "observation_errors": 0,
    }

    # Reset environment
    obs, info = env.reset(seed=seed)

    # Create RNG for action sampling
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n=== Episode Start (seed={seed}) ===")
        env.render()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        stats["total_steps"] += 1

        # Validate observation shape
        if observation is not None:
            try:
                assert "hand" in observation, "Missing 'hand' in observation"
                assert "last_move" in observation, "Missing 'last_move' in observation"
                assert "remaining_cards" in observation, "Missing 'remaining_cards' in observation"
                assert "action_mask" in observation, "Missing 'action_mask' in observation"

                assert observation["hand"].shape == (NUM_CARDS,), (
                    f"Hand shape mismatch: {observation['hand'].shape} != ({NUM_CARDS},)"
                )
                assert observation["last_move"].shape == (MOVE_VECTOR_SIZE,), (
                    f"Last move shape mismatch: {observation['last_move'].shape} != ({MOVE_VECTOR_SIZE},)"
                )
                assert observation["remaining_cards"].shape == (NUM_PLAYERS,), (
                    f"Remaining cards shape mismatch: {observation['remaining_cards'].shape} != ({NUM_PLAYERS},)"
                )
                assert observation["action_mask"].shape == (MAX_ACTIONS,), (
                    f"Action mask shape mismatch: {observation['action_mask'].shape} != ({MAX_ACTIONS},)"
                )
            except AssertionError as e:
                stats["observation_errors"] += 1
                if verbose:
                    print(f"  Observation error: {e}")

        if termination or truncation:
            action = None
            stats["rewards"][agent] = reward
            if verbose:
                print(f"  {agent}: DONE (reward={reward})")
        else:
            # Get action mask from observation or info
            if observation is not None and "action_mask" in observation:
                mask = observation["action_mask"]
            elif "action_mask" in info:
                mask = info["action_mask"]
            else:
                mask = None
                stats["action_mask_errors"] += 1
                if verbose:
                    print(f"  Warning: No action mask found for {agent}")

            # Sample valid action
            if mask is not None:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    # Fallback - should not happen
                    action = 0
                    stats["action_mask_errors"] += 1
                    if verbose:
                        print(f"  Warning: No valid actions for {agent}")
                else:
                    action = int(rng.choice(valid_actions))
            else:
                # No mask - just use action 0 (usually PASS)
                action = 0

            if verbose:
                print(f"  {agent}: action={action}")

        env.step(action)

    if verbose:
        print(f"=== Episode End: {stats['total_steps']} steps ===")
        print(f"  Rewards: {stats['rewards']}")

    return stats


def validate_observation_space(env: RLPokerAECEnv) -> bool:
    """Validate that observation space is correctly defined.

    Args:
        env: The environment instance

    Returns:
        True if valid, False otherwise
    """
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)

        # Check that action_mask is aligned with MAX_ACTIONS
        if "action_mask" not in obs_space.spaces:
            print(f"Error: 'action_mask' not in observation space for {agent}")
            return False

        action_mask_space = obs_space.spaces["action_mask"]
        if action_mask_space.shape != (MAX_ACTIONS,):
            print(f"Error: action_mask shape {action_mask_space.shape} != ({MAX_ACTIONS},)")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test for RL Poker AEC environment")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (default: None for random)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    args = parser.parse_args()

    print(f"Running {args.episodes} episode(s)...")
    print(f"  MAX_ACTIONS: {MAX_ACTIONS}")
    print(f"  NUM_CARDS: {NUM_CARDS}")
    print(f"  MOVE_VECTOR_SIZE: {MOVE_VECTOR_SIZE}")

    # Create environment
    env = RLPokerAECEnv(render_mode="ansi" if args.verbose else None)

    # Validate observation space
    print("\nValidating observation space...")
    if not validate_observation_space(env):
        print("FAILED: Observation space validation")
        sys.exit(1)
    print("  OK: Observation space valid")

    # Run episodes
    total_stats = {
        "episodes": 0,
        "total_steps": 0,
        "total_action_mask_errors": 0,
        "total_observation_errors": 0,
        "all_rewards": [],
    }

    start_time = time.time()

    for episode in range(args.episodes):
        episode_seed = None if args.seed is None else args.seed + episode

        if args.verbose:
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

        stats = run_episode(env, seed=episode_seed, verbose=args.verbose)

        total_stats["episodes"] += 1
        total_stats["total_steps"] += stats["total_steps"]
        total_stats["total_action_mask_errors"] += stats["action_mask_errors"]
        total_stats["total_observation_errors"] += stats["observation_errors"]
        total_stats["all_rewards"].append(stats["rewards"])

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n=== Summary ===")
    print(f"  Episodes: {total_stats['episodes']}")
    print(f"  Total steps: {total_stats['total_steps']}")
    print(f"  Time: {elapsed:.2f}s ({elapsed / total_stats['episodes']:.3f}s/episode)")
    print(f"  Action mask errors: {total_stats['total_action_mask_errors']}")
    print(f"  Observation errors: {total_stats['total_observation_errors']}")

    # Check for errors
    errors = total_stats["total_action_mask_errors"] + total_stats["total_observation_errors"]

    if errors > 0:
        print(f"\nFAILED: {errors} error(s) detected")
        sys.exit(1)

    print("\nPASSED: All episodes completed successfully")
    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

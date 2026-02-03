#!/usr/bin/env python
"""Evaluation script for testing agents against baselines.

This script runs evaluation episodes between agents and collects
performance statistics including:
- Average score per player position
- Win rate (finishing 1st/2nd vs 3rd/4th)
- Rank distribution
- Elo rating tracking across runs

Usage:
    python -m rl_poker.scripts.evaluate --episodes 10 --opponents random,heuristic
    python -m rl_poker.scripts.evaluate --episodes 100 --opponents heuristic,heuristic,heuristic
    python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic,policy_pool
    python -m rl_poker.scripts.evaluate --help
"""

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import sys
import numpy as np

from rl_poker.engine.game_state import GameState, NUM_PLAYERS
from rl_poker.agents.random_agent import create_random_agent
from rl_poker.agents.heuristic_agent import create_heuristic_agent
from rl_poker.agents.policy_pool import PolicyPool, PooledPolicyAgent, create_policy_pool, PolicyProtocol


# Default K-factor for Elo updates
DEFAULT_ELO_K = 32
# Default starting Elo
DEFAULT_STARTING_ELO = 1000.0


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A vs player B using Elo formula.

    Args:
        rating_a: Elo rating of player A
        rating_b: Elo rating of player B

    Returns:
        Expected score (probability of winning) for player A
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_ratings(
    ratings: list[float],
    ranks: dict[int, int],
    k_factor: float = DEFAULT_ELO_K,
) -> list[float]:
    """Update Elo ratings after a 4-player game.

    Uses a pairwise comparison approach: each player is compared against
    all other players, with 1 point for winning (better rank), 0.5 for tie,
    0 for losing.

    Args:
        ratings: Current Elo ratings for each player (list of 4 floats)
        ranks: Dict mapping player_id to rank (1-4, where 1 is best)
        k_factor: K-factor for Elo update (higher = more volatile)

    Returns:
        New Elo ratings after the game
    """
    n_players = len(ratings)
    new_ratings = list(ratings)

    for i in range(n_players):
        total_expected = 0.0
        total_actual = 0.0

        for j in range(n_players):
            if i == j:
                continue

            # Expected score against player j
            expected = calculate_expected_score(ratings[i], ratings[j])
            total_expected += expected

            # Actual score: 1 if better rank, 0.5 if tie, 0 if worse
            rank_i = ranks.get(i, 4)
            rank_j = ranks.get(j, 4)
            if rank_i < rank_j:
                actual = 1.0  # Won (lower rank is better)
            elif rank_i > rank_j:
                actual = 0.0  # Lost
            else:
                actual = 0.5  # Tie
            total_actual += actual

        # Update rating based on total expected vs actual across all opponents
        rating_change = k_factor * (total_actual - total_expected)
        new_ratings[i] = ratings[i] + rating_change

    return new_ratings


@dataclass
class EloTracker:
    """Track Elo ratings across evaluation runs.

    Attributes:
        ratings: Current Elo ratings by position
        rating_history: History of ratings after each episode
        k_factor: K-factor for Elo updates
    """

    ratings: list[float] = field(default_factory=lambda: [DEFAULT_STARTING_ELO] * NUM_PLAYERS)
    rating_history: list[list[float]] = field(default_factory=list)
    k_factor: float = DEFAULT_ELO_K

    def update(self, ranks: dict[int, int]) -> list[float]:
        """Update ratings after an episode.

        Args:
            ranks: Dict mapping player_id to rank (1-4)

        Returns:
            New ratings
        """
        self.ratings = update_elo_ratings(self.ratings, ranks, self.k_factor)
        self.rating_history.append(list(self.ratings))
        return self.ratings

    def get_rating(self, position: int) -> float:
        """Get current rating for a position."""
        return self.ratings[position]

    def get_rating_change(self, position: int) -> float:
        """Get total rating change for a position."""
        if not self.rating_history:
            return 0.0
        return self.ratings[position] - DEFAULT_STARTING_ELO

    def summary(self, agent_names: list[str]) -> str:
        """Generate Elo summary string."""
        lines = ["\nElo Ratings:"]
        for pos in range(NUM_PLAYERS):
            name = agent_names[pos] if pos < len(agent_names) else f"Player {pos}"
            rating = self.ratings[pos]
            change = self.get_rating_change(pos)
            sign = "+" if change >= 0 else ""
            lines.append(f"  {name}: {rating:.0f} ({sign}{change:.0f})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "ratings": self.ratings,
            "rating_history": self.rating_history,
            "k_factor": self.k_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EloTracker":
        """Create from dictionary."""
        tracker = cls()
        ratings = data.get("ratings")
        if isinstance(ratings, list):
            tracker.ratings = [float(r) for r in ratings]
        else:
            tracker.ratings = [DEFAULT_STARTING_ELO] * NUM_PLAYERS

        history = data.get("rating_history")
        if isinstance(history, list):
            tracker.rating_history = [list(map(float, row)) for row in history if isinstance(row, list)]
        else:
            tracker.rating_history = []

        k_factor = data.get("k_factor")
        if isinstance(k_factor, (int, float)):
            tracker.k_factor = float(k_factor)
        else:
            tracker.k_factor = DEFAULT_ELO_K
        return tracker


@dataclass
class EvaluationStats:
    """Statistics from evaluation episodes.

    Attributes:
        total_episodes: Number of episodes run
        scores_by_position: Dict mapping position (0-3) to list of scores
        ranks_by_position: Dict mapping position (0-3) to list of ranks
        wins_by_position: Dict mapping position (0-3) to win count (1st or 2nd place)
        elo_tracker: Optional Elo rating tracker
    """

    total_episodes: int = 0
    scores_by_position: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    ranks_by_position: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    wins_by_position: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    elo_tracker: EloTracker | None = None

    def __post_init__(self):
        if self.elo_tracker is None:
            self.elo_tracker = EloTracker()

    def record_episode(self, scores: dict[int, int], ranks: dict[int, int]) -> None:
        """Record results from a single episode.

        Args:
            scores: Dict mapping player_id to score
            ranks: Dict mapping player_id to rank (1-4)
        """
        self.total_episodes += 1
        for player_id in range(NUM_PLAYERS):
            score = scores.get(player_id, 0)
            rank = ranks.get(player_id, 4)
            self.scores_by_position[player_id].append(score)
            self.ranks_by_position[player_id].append(rank)
            if rank <= 2:  # 1st or 2nd place
                self.wins_by_position[player_id] += 1

        # Update Elo ratings
        if self.elo_tracker is not None:
            self.elo_tracker.update(ranks)

    def get_average_score(self, position: int) -> float:
        """Get average score for a position."""
        scores = self.scores_by_position[position]
        return sum(scores) / len(scores) if scores else 0.0

    def get_win_rate(self, position: int) -> float:
        """Get win rate (top 2 finish) for a position."""
        return (
            self.wins_by_position[position] / self.total_episodes
            if self.total_episodes > 0
            else 0.0
        )

    def get_rank_distribution(self, position: int) -> dict[int, float]:
        """Get rank distribution for a position."""
        ranks = self.ranks_by_position[position]
        if not ranks:
            return {1: 0, 2: 0, 3: 0, 4: 0}
        total = len(ranks)
        return {r: ranks.count(r) / total for r in [1, 2, 3, 4]}

    def get_elo_rating(self, position: int) -> float:
        """Get current Elo rating for a position."""
        if self.elo_tracker is None:
            return DEFAULT_STARTING_ELO
        return self.elo_tracker.get_rating(position)

    def summary(self, agent_names: list[str]) -> str:
        """Generate summary string.

        Args:
            agent_names: List of agent names by position

        Returns:
            Formatted summary string
        """
        lines = [
            f"\n{'=' * 60}",
            f"Evaluation Summary ({self.total_episodes} episodes)",
            f"{'=' * 60}",
        ]

        for pos in range(NUM_PLAYERS):
            name = agent_names[pos] if pos < len(agent_names) else f"Player {pos}"
            avg_score = self.get_average_score(pos)
            win_rate = self.get_win_rate(pos)
            rank_dist = self.get_rank_distribution(pos)
            elo = self.get_elo_rating(pos)
            elo_change = self.elo_tracker.get_rating_change(pos) if self.elo_tracker else 0.0
            elo_sign = "+" if elo_change >= 0 else ""

            lines.append(f"\n{name} (Position {pos}):")
            lines.append(f"  Average Score: {avg_score:+.2f}")
            lines.append(f"  Win Rate (Top 2): {win_rate:.1%}")
            lines.append(
                f"  Rank Distribution: 1st={rank_dist[1]:.1%}, 2nd={rank_dist[2]:.1%}, 3rd={rank_dist[3]:.1%}, 4th={rank_dist[4]:.1%}"
            )
            lines.append(f"  Elo Rating: {elo:.0f} ({elo_sign}{elo_change:.0f})")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)


def create_agent(
    agent_type: str,
    seed: int | None = None,
    position: int = 0,
    policy_pool: PolicyPool | None = None,
    checkpoint_name: str | None = None,
) -> PolicyProtocol:
    """Create an agent of the specified type.

    Args:
        agent_type: Type of agent ('random', 'heuristic', 'policy_pool')
        seed: Random seed for the agent
        position: Player position (0-3) for naming
        policy_pool: PolicyPool instance for policy_pool agent type
        checkpoint_name: Specific checkpoint to load from pool (if None, uses latest)

    Returns:
        Agent instance

    Raises:
        ValueError: If agent type is unknown or policy_pool missing for 'policy_pool' type
    """
    _ = position
    agent_type = agent_type.lower().strip()

    if agent_type == "random":
        return create_random_agent(seed=seed)
    elif agent_type == "heuristic":
        return create_heuristic_agent(seed=seed)
    elif agent_type == "policy_pool":
        if policy_pool is None:
            raise ValueError("policy_pool argument required for 'policy_pool' agent type")
        if len(policy_pool) == 0:
            raise ValueError("PolicyPool is empty - no checkpoints available")
        # Use specified checkpoint or get latest
        if checkpoint_name is not None:
            if checkpoint_name not in policy_pool:
                raise ValueError(f"Checkpoint '{checkpoint_name}' not found in pool")
            name = checkpoint_name
        else:
            latest = policy_pool.get_latest_checkpoint()
            if latest is None:
                raise ValueError("No checkpoints in policy pool")
            name = latest.name
        return PooledPolicyAgent(pool=policy_pool, checkpoint_name=name)
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: random, heuristic, policy_pool"
        )


def run_episode(
    agents: list[PolicyProtocol],
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[dict[int, int], dict[int, int], int]:
    """Run a single evaluation episode.

    Args:
        agents: List of 4 agents (one per position)
        seed: Random seed for game state
        verbose: Whether to print move-by-move details

    Returns:
        Tuple of (scores_dict, ranks_dict, total_moves)
    """
    state = GameState.new_game(seed=seed)
    move_count = 0

    if verbose:
        print(f"\n--- Starting Episode (seed={seed}) ---")
        print(state)

    while not state.is_game_over():
        current_player = state.current_player
        agent = agents[current_player]

        # Get legal moves
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            raise RuntimeError(f"No legal moves for player {current_player}")

        # Create observation for agent
        observation = {
            "legal_moves": legal_moves,
            "hand": state.get_player_hand(current_player),
            "current_move": state.current_move,
            "current_player": current_player,
            "player_card_counts": [p.card_count for p in state.players],
        }

        # Create action mask
        action_mask = np.zeros(len(legal_moves), dtype=bool)
        action_mask[:] = True  # All moves in legal_moves are valid

        # Get action from agent
        action_idx = agent.act(observation, action_mask)

        # Ensure action is in range
        if action_idx < 0 or action_idx >= len(legal_moves):
            action_idx = 0

        move = legal_moves[action_idx]

        if verbose:
            print(f"Player {current_player}: {move}")

        # Apply move
        state.apply_move(move)
        move_count += 1

    if verbose:
        print(f"\n--- Episode Complete ({move_count} moves) ---")
        print(f"Scores: {state.get_scores()}")
        print(f"Rankings: {state.get_rankings()}")

    return state.get_scores(), state.get_rankings(), move_count


def evaluate(
    opponents: list[str],
    episodes: int = 10,
    seed: int | None = None,
    verbose: bool = False,
    policy_pool_dir: str | Path | None = None,
    policy_loader: Callable[[Path], PolicyProtocol] | None = None,
) -> EvaluationStats:
    """Run evaluation episodes.

    Args:
        opponents: List of opponent types (up to 4). If less than 4,
                  fills remaining with the last specified type.
                  Supported types: random, heuristic, policy_pool
        episodes: Number of episodes to run
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
        policy_pool_dir: Directory containing policy pool checkpoints
                        (required if using 'policy_pool' opponent type)
        policy_loader: Custom loader function for policy pool checkpoints

    Returns:
        EvaluationStats with results
    """
    # Ensure we have exactly 4 agent types
    while len(opponents) < NUM_PLAYERS:
        opponents.append(opponents[-1] if opponents else "random")
    opponents = opponents[:NUM_PLAYERS]

    # Check if policy_pool is needed
    needs_pool = any(o.lower().strip() == "policy_pool" for o in opponents)
    policy_pool = None
    if needs_pool:
        if policy_pool_dir is None:
            raise ValueError(
                "policy_pool_dir required when using 'policy_pool' opponent type. Use --pool-dir to specify the checkpoint directory."
            )
        policy_pool = create_policy_pool(pool_dir=policy_pool_dir, loader=policy_loader)
        if len(policy_pool) == 0:
            print(
                f"Warning: PolicyPool at '{policy_pool_dir}' is empty. Falling back to random agent for policy_pool positions."
            )

    # Create agents
    rng = np.random.default_rng(seed)
    agents = []
    for i, agent_type in enumerate(opponents):
        agent_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        try:
            agents.append(
                create_agent(
                    agent_type,
                    seed=agent_seed,
                    position=i,
                    policy_pool=policy_pool,
                )
            )
        except ValueError as e:
            if "PolicyPool is empty" in str(e) or "No checkpoints" in str(e):
                # Fallback to random if pool is empty
                print(f"  Position {i}: Falling back to random agent ({e})")
                agents.append(create_random_agent(seed=agent_seed))
                opponents[i] = "random (fallback)"
            else:
                raise

    agent_names = [f"{opponents[i].capitalize()} ({i})" for i in range(NUM_PLAYERS)]

    stats = EvaluationStats()

    print(f"Running {episodes} episodes with agents: {', '.join(opponents)}")

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 2**31)) if seed is not None else None

        # Reset agents
        for agent in agents:
            if hasattr(agent, "reset"):
                agent.reset()

        try:
            scores, ranks, _ = run_episode(agents, seed=ep_seed, verbose=verbose)
            stats.record_episode(scores, ranks)

            if (ep + 1) % max(1, episodes // 10) == 0:
                print(f"  Completed {ep + 1}/{episodes} episodes...")

        except Exception as e:
            print(f"  Episode {ep + 1} failed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    print(stats.summary(agent_names))
    return stats


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate agents in RL Poker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rl_poker.scripts.evaluate --episodes 10 --opponents random,heuristic
  python -m rl_poker.scripts.evaluate --episodes 100 --opponents heuristic,heuristic,heuristic,heuristic
  python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,random,heuristic,heuristic --seed 42
  python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic,policy_pool --pool-dir ./checkpoints
        """,
    )

    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes to run (default: 10)",
    )

    parser.add_argument(
        "--opponents",
        "-o",
        type=str,
        default="random,random,random,random",
        help="Comma-separated list of opponent types: random, heuristic, policy_pool (default: random,random,random,random)",
    )

    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed move-by-move output"
    )

    parser.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Directory containing policy pool checkpoints (required for policy_pool opponent type)",
    )

    args = parser.parse_args()

    # Parse opponents
    opponents = [o.strip() for o in args.opponents.split(",")]

    # Validate opponent types
    valid_types = {"random", "heuristic", "policy_pool"}
    for opp in opponents:
        if opp.lower() not in valid_types:
            print(f"Error: Unknown opponent type '{opp}'. Valid types: {', '.join(valid_types)}")
            sys.exit(1)

    # Check if policy_pool requires pool-dir
    needs_pool = any(o.lower() == "policy_pool" for o in opponents)
    if needs_pool and args.pool_dir is None:
        print("Error: --pool-dir is required when using 'policy_pool' opponent type")
        sys.exit(1)

    # Run evaluation
    try:
        evaluate(
            opponents=opponents,
            episodes=args.episodes,
            seed=args.seed,
            verbose=args.verbose,
            policy_pool_dir=args.pool_dir,
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

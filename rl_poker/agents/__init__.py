"""RL agents for poker.

This module provides:
- RandomAgent: Baseline agent that plays random legal moves
- HeuristicAgent: Rule-based agent with domain knowledge
- PolicyPool: Checkpoint management for model evaluation
- PooledPolicyAgent: Agent wrapper for pooled checkpoints
"""

from .random_agent import (
    BaseAgent,
    RandomAgent,
    create_random_agent,
)

from .heuristic_agent import (
    HeuristicAgent,
    create_heuristic_agent,
)

from .policy_pool import (
    PolicyProtocol,
    PolicyPool,
    PooledPolicyAgent,
    CheckpointInfo,
    create_policy_pool,
)

__all__ = [
    # Protocols
    "BaseAgent",
    "PolicyProtocol",
    # Random agent
    "RandomAgent",
    "create_random_agent",
    # Heuristic agent
    "HeuristicAgent",
    "create_heuristic_agent",
    # Policy pool
    "PolicyPool",
    "PooledPolicyAgent",
    "CheckpointInfo",
    "create_policy_pool",
]

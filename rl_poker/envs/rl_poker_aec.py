"""PettingZoo AEC environment for 4-player poker.

This module provides a PettingZoo AEC (Agent Environment Cycle) environment
wrapper over the game engine, supporting:
- reset(), step(), observe(), agent_iter() interface
- Observation space with hand encoding, last move, remaining cards, action mask
- Sparse reward only at game end: +2 (1st), +1 (2nd), -1 (3rd), -2 (4th)
- Seeding support for deterministic reset (reproducibility)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from rl_poker.engine.game_state import (
    GameState,
    GamePhase,
    NUM_PLAYERS,
    CARDS_PER_PLAYER,
    RANK_SCORES,
)
from rl_poker.moves.action_encoding import (
    MAX_ACTIONS,
    encode_action_space,
    decode_action,
    ActionSpace,
)
from rl_poker.moves.legal_moves import Move, MoveType, PASS_MOVE
from rl_poker.rules import Rank, Suit, Card


# Constants for encoding
NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52  # 13 ranks * 4 suits

# Move encoding dimensions
MOVE_VECTOR_SIZE = 17  # 4 metadata + 13 card presence indicators


def card_to_index(card: Card) -> int:
    """Convert a card to a unique index (0-51)."""
    return int(card.rank) * NUM_SUITS + int(card.suit)


def encode_hand(cards: List[Card]) -> np.ndarray:
    """Encode a hand as a one-hot vector of 52 bits.

    Args:
        cards: List of cards in hand

    Returns:
        np.ndarray of shape (52,) with 1s for cards present
    """
    encoding = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in cards:
        encoding[card_to_index(card)] = 1.0
    return encoding


def encode_move(move: Optional[Move]) -> np.ndarray:
    """Encode a move as a fixed-size vector.

    Encoding format:
    - [0]: Is valid move (0 if None/PASS, 1 if actual play)
    - [1]: Move type (0=PASS/None, 1=PLAY)
    - [2]: Is exemption (0/1)
    - [3]: Number of cards (0-7)
    - [4:17]: Card presence indicators (13 ranks, any suit = 1)

    Args:
        move: The move to encode (None for no move)

    Returns:
        np.ndarray of shape (MOVE_VECTOR_SIZE,)
    """
    encoding = np.zeros(MOVE_VECTOR_SIZE, dtype=np.float32)

    if move is None or move.move_type == MoveType.PASS:
        return encoding

    encoding[0] = 1.0  # Valid move
    encoding[1] = 1.0  # PLAY type
    encoding[2] = 1.0 if move.is_exemption else 0.0
    encoding[3] = float(len(move.cards)) / 7.0  # Normalized by max cards

    # Encode card ranks (any suit counts)
    for card in move.cards:
        encoding[4 + int(card.rank)] = 1.0

    return encoding


class RLPokerAECEnv(AECEnv):
    """PettingZoo AEC environment for 4-player poker.

    This environment wraps the game engine and provides:
    - Standard AEC API: reset(), step(), observe(), agent_iter()
    - Dict observation space with action mask
    - Sparse rewards at game end only
    - Seeding for deterministic reset

    Observation space (Dict):
        - "hand": one-hot encoding of player's cards (52,)
        - "last_move": encoding of the current move to beat (MOVE_VECTOR_SIZE,)
        - "remaining_cards": cards remaining per player (4,)
        - "action_mask": boolean mask for legal actions (MAX_ACTIONS,)

    Action space: Discrete(MAX_ACTIONS)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "rl_poker_v1",
        "is_parallelizable": False,
    }

    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the environment.

        Args:
            render_mode: Render mode ("human", "ansi", or None)
        """
        super().__init__()
        self.render_mode = render_mode

        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(NUM_PLAYERS)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # Define spaces
        self._action_spaces = {
            agent: spaces.Discrete(MAX_ACTIONS) for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "hand": spaces.Box(low=0, high=1, shape=(NUM_CARDS,), dtype=np.float32),
                    "last_move": spaces.Box(
                        low=0, high=1, shape=(MOVE_VECTOR_SIZE,), dtype=np.float32
                    ),
                    "remaining_cards": spaces.Box(
                        low=0, high=CARDS_PER_PLAYER, shape=(NUM_PLAYERS,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8),
                }
            )
            for agent in self.possible_agents
        }

        # Internal state
        self._game_state: Optional[GameState] = None
        self._action_spaces_cache: Dict[str, ActionSpace] = {}
        self._seed: Optional[int] = None
        self._cumulative_rewards: Dict[str, float] = {}

        # Agent tracking (not using agent_selector for more direct control)
        self._current_agent: str = self.possible_agents[0]

        # Standard PettingZoo attributes
        self.agents: List[str] = []
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict[str, Any]] = {}

    @property
    def agent_selection(self) -> str:
        """Get the currently selected agent."""
        return self._current_agent

    def observation_space(self, agent: str) -> spaces.Dict:
        """Get observation space for an agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        """Get action space for an agent."""
        return self._action_spaces[agent]

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed for the environment.

        Args:
            seed: Random seed (None for random)
        """
        self._seed = seed

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed (overrides seed() if provided)
            options: Optional reset options (unused)

        Returns:
            Tuple of (observation, info) for the first agent
        """
        # Use provided seed or fallback to stored seed
        if seed is not None:
            self._seed = seed

        # Create new game
        self._game_state = GameState.new_game(self._seed)

        # Reset agents
        self.agents = list(self.possible_agents)

        # Clear action space cache
        self._action_spaces_cache.clear()

        # Initialize rewards and terminations
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Set up starting player from game state
        current_player_idx = self._game_state.current_player
        starting_agent = self.possible_agents[current_player_idx]
        self._current_agent = starting_agent

        # Build action space for starting agent
        self._update_action_space(starting_agent)

        # Get initial observation
        observation = self.observe(starting_agent)
        info = self._get_info(starting_agent)

        return observation, info

    def step(self, action: Optional[int]) -> None:
        """Execute an action for the current agent.

        Args:
            action: Action index (0 to MAX_ACTIONS-1), or None if agent is done
        """
        if self._game_state is None:
            raise RuntimeError("Must call reset() before step()")

        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]

        # Handle terminated/truncated agents
        if self.terminations[agent] or self.truncations[agent]:
            # Use _was_dead_step pattern
            self._was_dead_step(action)
            return

        # Validate action
        if action is None:
            raise ValueError(f"Action cannot be None for active agent {agent}")

        # Get action space and decode action
        action_space = self._get_action_space(agent)

        if not action_space.action_mask[action]:
            raise ValueError(f"Action {action} is masked (illegal) for {agent}")

        move = decode_action(action, action_space)

        # Apply move to game state
        game_over = self._game_state.apply_move(move, player_idx)

        # Clear rewards (rewards are sparse, only at game end)
        for a in self.agents:
            self.rewards[a] = 0.0

        # Handle game over
        if game_over:
            self._handle_game_over()
        else:
            # Update to next player
            self._advance_to_next_player()

    def _was_dead_step(self, action: Optional[int]) -> None:
        """Handle step for a dead (terminated/truncated) agent.

        This is called when an agent that has already terminated takes a step
        (typically with action=None to acknowledge termination).
        """
        agent = self.agent_selection

        # Remove this agent from the active agents list
        if agent in self.agents:
            self.agents.remove(agent)

        # Clear rewards for this agent after they've been received
        self.rewards[agent] = 0.0

        # Advance to next agent if any remain
        if self.agents:
            self._advance_to_next_agent()

    def _handle_game_over(self) -> None:
        """Handle game over - assign final rewards.

        Note: We do NOT clear self.agents here. Instead, we let all agents
        iterate one more time to receive their final rewards via last().
        The agents list is cleared when all agents have acknowledged their
        termination via _was_dead_step.
        """
        if self._game_state is None:
            return

        scores = self._game_state.get_scores()

        # Assign final rewards and mark all agents as terminated
        for agent in self.possible_agents:
            player_idx = self.agent_name_mapping[agent]
            self.rewards[agent] = float(scores.get(player_idx, 0))
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.terminations[agent] = True

        # Keep agents list intact so they can all receive their final rewards
        # via agent_iter -> last() -> done=True -> step(None)

    def _advance_to_next_player(self) -> None:
        """Advance to the next active player."""
        if self._game_state is None:
            return

        # Get current player from game state
        current_player_idx = self._game_state.current_player
        next_agent = self.possible_agents[current_player_idx]

        # Update current agent
        self._current_agent = next_agent

        # Update action space for next agent
        self._update_action_space(next_agent)

    def _advance_to_next_agent(self) -> None:
        """Advance to the next agent in order (used for dead step)."""
        current_idx = self.agent_name_mapping.get(self._current_agent, 0)
        next_idx = (current_idx + 1) % NUM_PLAYERS
        self._current_agent = self.possible_agents[next_idx]

    def _update_action_space(self, agent: str) -> None:
        """Update the cached action space for an agent."""
        if self._game_state is None:
            return

        player_idx = self.agent_name_mapping[agent]
        player = self._game_state.players[player_idx]

        if player.has_finished:
            # Player is done - empty action space
            self._action_spaces_cache[agent] = ActionSpace(
                legal_moves=[],
                action_mask=np.zeros(MAX_ACTIONS, dtype=bool),
                move_to_action={},
                action_to_move={},
            )
            return

        # Get legal moves and encode action space
        context = self._game_state.get_move_context(player_idx)
        legal_moves = self._game_state.get_legal_moves(player_idx)

        # Create action space
        action_mask = np.zeros(MAX_ACTIONS, dtype=bool)
        action_mask[: len(legal_moves)] = True

        move_to_action = {move: i for i, move in enumerate(legal_moves)}
        action_to_move = {i: move for i, move in enumerate(legal_moves)}

        self._action_spaces_cache[agent] = ActionSpace(
            legal_moves=legal_moves,
            action_mask=action_mask,
            move_to_action=move_to_action,
            action_to_move=action_to_move,
        )

    def _get_action_space(self, agent: str) -> ActionSpace:
        """Get the cached action space for an agent."""
        if agent not in self._action_spaces_cache:
            self._update_action_space(agent)
        return self._action_spaces_cache[agent]

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Get observation for an agent.

        Args:
            agent: Agent name

        Returns:
            Observation dict with hand, last_move, remaining_cards, action_mask
        """
        if self._game_state is None:
            # Return empty observation
            return {
                "hand": np.zeros(NUM_CARDS, dtype=np.float32),
                "last_move": np.zeros(MOVE_VECTOR_SIZE, dtype=np.float32),
                "remaining_cards": np.zeros(NUM_PLAYERS, dtype=np.float32),
                "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
            }

        player_idx = self.agent_name_mapping[agent]
        player = self._game_state.players[player_idx]

        # Encode hand
        hand_encoding = encode_hand(player.hand)

        # Encode last move (current move to beat)
        last_move_encoding = encode_move(self._game_state.current_move)

        # Remaining cards per player
        remaining_cards = np.array(
            [p.card_count for p in self._game_state.players],
            dtype=np.float32,
        )

        # Action mask
        action_space = self._get_action_space(agent)
        action_mask = action_space.action_mask.astype(np.int8)

        return {
            "hand": hand_encoding,
            "last_move": last_move_encoding,
            "remaining_cards": remaining_cards,
            "action_mask": action_mask,
        }

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get info dict for an agent."""
        if self._game_state is None:
            return {}

        player_idx = self.agent_name_mapping[agent]
        action_space = self._get_action_space(agent)

        return {
            "action_mask": action_space.action_mask,
            "legal_moves": action_space.legal_moves,
            "num_legal_moves": len(action_space.legal_moves),
        }

    def last(
        self,
        observe: bool = True,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], float, bool, bool, Dict[str, Any]]:
        """Get the last observation, reward, termination, truncation, info.

        Args:
            observe: If True, return observation; otherwise return None

        Returns:
            Tuple of (observation, reward, termination, truncation, info)
        """
        agent = self.agent_selection

        observation = self.observe(agent) if observe else None
        reward = self.rewards.get(agent, 0.0)
        termination = self.terminations.get(agent, False)
        truncation = self.truncations.get(agent, False)
        info = self._get_info(agent)

        return observation, reward, termination, truncation, info

    def render(self) -> Optional[str]:
        """Render the environment.

        Returns:
            String representation if render_mode is "ansi", None otherwise
        """
        if self._game_state is None:
            return None

        if self.render_mode == "human" or self.render_mode == "ansi":
            output = str(self._game_state)
            if self.render_mode == "human":
                print(output)
            return output

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        self._game_state = None
        self._action_spaces_cache.clear()


# Factory function for creating environment
def env(**kwargs) -> RLPokerAECEnv:
    """Create an RLPokerAECEnv environment.

    Args:
        **kwargs: Keyword arguments passed to RLPokerAECEnv.__init__

    Returns:
        RLPokerAECEnv instance
    """
    return RLPokerAECEnv(**kwargs)

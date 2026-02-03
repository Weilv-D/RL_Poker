"""Game state management, turn flow, and scoring.

This module provides:
- GameState: Main game state tracking class
- Dealing cards to 4 players
- Lead player identification (player with Heart 3)
- Turn order management
- PASS logic and "new lead" when all others pass
- Player exit detection and ranking
- Stop condition: game ends when third player finishes
- Scoring: +2 (1st), +1 (2nd), -1 (3rd), -2 (4th), zero-sum

Game flow:
1. Deal 13 cards to each of 4 players
2. Find player with Heart 3 - they lead first
3. First move MUST contain Heart 3
4. Play proceeds clockwise (player 0 -> 1 -> 2 -> 3 -> 0)
5. Players can PASS or play a valid move that beats the current lead
6. When all other active players pass, last player to play gets "new lead"
7. When a player's hand is empty, they exit and their rank is locked
8. Game ends when 3 players finish (4th is automatically last)
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Optional, Dict, Tuple, Set
import random
import copy

from rl_poker.rules import (
    Card,
    Rank,
    Suit,
    create_standard_deck,
    get_heart_three,
    sort_cards,
)
from rl_poker.rules.ranks import get_rank_counts
from rl_poker.moves.legal_moves import (
    Move,
    MoveType,
    MoveContext,
    PASS_MOVE,
    get_legal_moves,
)


class GamePhase(IntEnum):
    """Phases of the game."""

    DEALING = auto()  # Cards being dealt
    PLAYING = auto()  # Game in progress
    FINISHED = auto()  # Game over


# Scores for each rank position
RANK_SCORES = {
    1: 2,  # 1st place: +2
    2: 1,  # 2nd place: +1
    3: -1,  # 3rd place: -1
    4: -2,  # 4th place: -2
}

# Number of players
NUM_PLAYERS = 4

# Cards per player
CARDS_PER_PLAYER = 13


@dataclass
class PlayerState:
    """State of a single player.

    Attributes:
        player_id: Player identifier (0-3)
        hand: List of cards in hand
        has_passed: Whether player passed in current round
        has_finished: Whether player has emptied their hand
        finish_rank: Rank when finished (1-4), None if not finished
    """

    player_id: int
    hand: List[Card] = field(default_factory=list)
    has_passed: bool = False
    has_finished: bool = False
    finish_rank: Optional[int] = None

    @property
    def card_count(self) -> int:
        """Number of cards remaining in hand."""
        return len(self.hand)

    @property
    def is_active(self) -> bool:
        """Whether player is still active in the game."""
        return not self.has_finished


@dataclass
class GameState:
    """Complete game state.

    Attributes:
        players: List of PlayerState objects (one per player)
        phase: Current game phase
        current_player: Index of current player (0-3)
        lead_player: Index of player who played the current lead move
        current_move: The move currently on the table to beat (None = new lead)
        previous_used_exemption: Whether the current move used tail-hand exemption
        pass_count: Number of consecutive passes
        next_rank: Next rank to assign when a player finishes (1-4)
        first_move: Whether this is the very first move of the game
        rng: Random number generator for shuffling
    """

    players: List[PlayerState] = field(default_factory=list)
    phase: GamePhase = GamePhase.DEALING
    current_player: int = 0
    lead_player: int = 0
    current_move: Optional[Move] = None
    previous_used_exemption: bool = False
    pass_count: int = 0
    next_rank: int = 1
    first_move: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        """Initialize players if not already set."""
        if not self.players:
            self.players = [PlayerState(player_id=i) for i in range(NUM_PLAYERS)]

    @classmethod
    def new_game(cls, seed: Optional[int] = None) -> "GameState":
        """Create a new game with shuffled and dealt cards.

        Args:
            seed: Random seed for reproducibility

        Returns:
            New GameState ready to play
        """
        state = cls()
        state.rng = random.Random(seed)
        state.deal()
        return state

    @classmethod
    def from_hands(cls, hands: List[List[Card]]) -> "GameState":
        """Create a game state from pre-defined hands (for testing).

        Args:
            hands: List of 4 hands (each a list of cards)

        Returns:
            New GameState with the given hands
        """
        if len(hands) != NUM_PLAYERS:
            raise ValueError(f"Expected {NUM_PLAYERS} hands, got {len(hands)}")

        state = cls()
        for i, hand in enumerate(hands):
            state.players[i].hand = list(hand)

        state._setup_after_deal()
        return state

    def deal(self) -> None:
        """Shuffle deck and deal cards to all players."""
        # Create and shuffle deck
        deck = create_standard_deck()
        self.rng.shuffle(deck)

        # Deal 13 cards to each player
        for i in range(NUM_PLAYERS):
            start = i * CARDS_PER_PLAYER
            end = start + CARDS_PER_PLAYER
            self.players[i].hand = deck[start:end]

        self._setup_after_deal()

    def _setup_after_deal(self) -> None:
        """Set up game state after dealing."""
        # Find lead player (who has Heart 3)
        heart_three = get_heart_three()
        for i, player in enumerate(self.players):
            if heart_three in player.hand:
                self.current_player = i
                self.lead_player = i
                break

        self.phase = GamePhase.PLAYING
        self.first_move = True
        self.current_move = None
        self.previous_used_exemption = False
        self.pass_count = 0
        self.next_rank = 1

    def get_current_player(self) -> PlayerState:
        """Get the current player's state."""
        return self.players[self.current_player]

    def get_player_hand(self, player_id: int) -> List[Card]:
        """Get a player's current hand."""
        return self.players[player_id].hand

    def is_game_over(self) -> bool:
        """Check if game is over (3 players have finished)."""
        return self.phase == GamePhase.FINISHED

    def get_active_player_count(self) -> int:
        """Count players still active in the game."""
        return sum(1 for p in self.players if p.is_active)

    def _is_tail_hand(self, player_id: int) -> bool:
        """Check if this would be the player's last move (tail hand)."""
        # Tail-hand exemption applies ONLY when the player's remaining cards
        # are fewer than the standard hand size for 3+2 (5) or 4+3 (7),
        # and the player can use an exemption to play ALL remaining cards.
        cards = self.players[player_id].hand
        count = len(cards)

        if count < 3:
            return False

        rank_counts = get_rank_counts(cards)
        has_triple = any(c >= 3 for c in rank_counts.values())
        has_quad = any(c == 4 for c in rank_counts.values())

        # 3+2 exemptions: only when 3 or 4 cards remain
        if count in (3, 4) and has_triple:
            return True

        # 4+3 exemptions: only when 4, 5, or 6 cards remain
        if count in (4, 5, 6) and has_quad:
            return True

        return False

    def get_move_context(self, player_id: Optional[int] = None) -> MoveContext:
        """Get the move context for the current player.

        Args:
            player_id: Player ID (default: current player)

        Returns:
            MoveContext for legal move enumeration
        """
        if player_id is None:
            player_id = self.current_player

        player = self.players[player_id]

        return MoveContext(
            previous_move=self.current_move,
            is_tail_hand=self._is_tail_hand(player_id),
            previous_used_exemption=self.previous_used_exemption,
        )

    def get_legal_moves(self, player_id: Optional[int] = None) -> List[Move]:
        """Get all legal moves for a player.

        Args:
            player_id: Player ID (default: current player)

        Returns:
            List of legal moves
        """
        if player_id is None:
            player_id = self.current_player

        player = self.players[player_id]

        if player.has_finished:
            return []

        context = self.get_move_context(player_id)
        moves = get_legal_moves(player.hand, context)

        # First move must contain Heart 3
        if self.first_move and player_id == self.current_player:
            heart_three = get_heart_three()
            moves = [m for m in moves if heart_three in m.cards]

        return moves

    def is_legal_move(self, move: Move, player_id: Optional[int] = None) -> bool:
        """Check if a move is legal for a player.

        Args:
            move: The move to check
            player_id: Player ID (default: current player)

        Returns:
            True if the move is legal
        """
        if player_id is None:
            player_id = self.current_player

        legal_moves = self.get_legal_moves(player_id)

        # Check by cards and exemption status
        for legal_move in legal_moves:
            if legal_move.cards == move.cards and legal_move.is_exemption == move.is_exemption:
                return True
        return False

    def apply_move(self, move: Move, player_id: Optional[int] = None) -> bool:
        """Apply a move to the game state.

        Args:
            move: The move to apply
            player_id: Player ID (default: current player)

        Returns:
            True if game is over after this move
        """
        if player_id is None:
            player_id = self.current_player

        if self.phase != GamePhase.PLAYING:
            raise ValueError(f"Cannot apply move in phase {self.phase}")

        if player_id != self.current_player:
            raise ValueError(f"Not player {player_id}'s turn (current: {self.current_player})")

        if not self.is_legal_move(move, player_id):
            raise ValueError(f"Illegal move: {move}")

        player = self.players[player_id]

        if move.move_type == MoveType.PASS:
            self._handle_pass(player_id)
        else:
            self._handle_play(player_id, move)

        # Check for game over
        if self._check_game_over():
            self.phase = GamePhase.FINISHED
            return True

        # Advance to next player
        self._advance_turn()

        return False

    def _handle_pass(self, player_id: int) -> None:
        """Handle a PASS move."""
        player = self.players[player_id]
        player.has_passed = True
        self.pass_count += 1

    def _handle_play(self, player_id: int, move: Move) -> None:
        """Handle a PLAY move."""
        player = self.players[player_id]

        # Remove cards from hand
        cards_to_remove = set(move.cards)
        player.hand = [c for c in player.hand if c not in cards_to_remove]

        # Update game state
        self.current_move = move
        self.lead_player = player_id
        self.previous_used_exemption = move.is_exemption
        self.first_move = False

        # Reset pass tracking (new lead established)
        self.pass_count = 0
        for p in self.players:
            p.has_passed = False

        # Check if player finished
        if player.card_count == 0:
            player.has_finished = True
            player.finish_rank = self.next_rank
            self.next_rank += 1

    def _check_game_over(self) -> bool:
        """Check if game is over (3 players finished)."""
        finished_count = sum(1 for p in self.players if p.has_finished)

        if finished_count >= 3:
            # Assign 4th place to remaining player
            for player in self.players:
                if not player.has_finished:
                    player.has_finished = True
                    player.finish_rank = 4
            return True

        return False

    def _advance_turn(self) -> None:
        """Advance to the next player's turn."""
        lead_still_active = self.players[self.lead_player].is_active

        all_others_passed = all(
            p.has_passed or not p.is_active or i == self.lead_player
            for i, p in enumerate(self.players)
        )

        if all_others_passed:
            self._start_new_lead_round(lead_still_active)
            return

        self._advance_to_next_active_player()

    def _start_new_lead_round(self, lead_still_active: bool) -> None:
        """Start a new lead round when all others have passed."""
        self.current_move = None
        self.previous_used_exemption = False
        self.pass_count = 0
        for p in self.players:
            p.has_passed = False

        if lead_still_active:
            self.current_player = self.lead_player
        else:
            for i in range(NUM_PLAYERS):
                if self.players[i].is_active:
                    self.current_player = i
                    self.lead_player = i
                    return

    def _advance_to_next_active_player(self) -> None:
        """Move turn to the next active player who hasn't passed."""
        next_player = (self.current_player + 1) % NUM_PLAYERS
        for _ in range(NUM_PLAYERS):
            player = self.players[next_player]
            if player.is_active and not player.has_passed:
                self.current_player = next_player
                return
            next_player = (next_player + 1) % NUM_PLAYERS

        raise RuntimeError("Could not find next player")

    def get_scores(self) -> Dict[int, int]:
        """Get the scores for all players.

        Returns:
            Dict mapping player_id to score
        """
        scores = {}
        for player in self.players:
            if player.finish_rank is not None:
                scores[player.player_id] = RANK_SCORES[player.finish_rank]
            else:
                scores[player.player_id] = 0
        return scores

    def get_rankings(self) -> Dict[int, int]:
        """Get the rankings for all players.

        Returns:
            Dict mapping player_id to rank (1-4)
        """
        rankings = {}
        for player in self.players:
            if player.finish_rank is not None:
                rankings[player.player_id] = player.finish_rank
        return rankings

    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        new_state = GameState()
        new_state.players = [
            PlayerState(
                player_id=p.player_id,
                hand=list(p.hand),
                has_passed=p.has_passed,
                has_finished=p.has_finished,
                finish_rank=p.finish_rank,
            )
            for p in self.players
        ]
        new_state.phase = self.phase
        new_state.current_player = self.current_player
        new_state.lead_player = self.lead_player
        new_state.current_move = self.current_move
        new_state.previous_used_exemption = self.previous_used_exemption
        new_state.pass_count = self.pass_count
        new_state.next_rank = self.next_rank
        new_state.first_move = self.first_move
        # Note: RNG state is not copied - use seed for reproducibility
        return new_state

    def __str__(self) -> str:
        """String representation of game state."""
        lines = [f"GameState (phase={self.phase.name})"]
        lines.append(f"  Current player: {self.current_player}")
        lines.append(f"  Lead player: {self.lead_player}")
        lines.append(f"  Current move: {self.current_move}")
        lines.append(f"  First move: {self.first_move}")
        for player in self.players:
            status = "FINISHED" if player.has_finished else "active"
            if player.has_passed:
                status += " (passed)"
            cards = ", ".join(str(c) for c in sort_cards(player.hand))
            rank_info = f" [Rank {player.finish_rank}]" if player.finish_rank else ""
            lines.append(f"  Player {player.player_id} ({status}{rank_info}): {cards}")
        return "\n".join(lines)


def play_random_game(seed: Optional[int] = None) -> Tuple[GameState, List[Tuple[int, Move]]]:
    """Play a complete game with random moves.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (final_state, move_history)
    """
    rng = random.Random(seed)
    state = GameState.new_game(seed)
    history = []

    while not state.is_game_over():
        player_id = state.current_player
        legal_moves = state.get_legal_moves()

        if not legal_moves:
            raise RuntimeError(f"No legal moves for player {player_id}")

        # Pick random move
        move = rng.choice(legal_moves)
        history.append((player_id, move))
        state.apply_move(move)

    return state, history

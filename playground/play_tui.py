#!/usr/bin/env python3
"""
Play a human vs AI game using a trained checkpoint.
Optimized for performance, interaction, appearance, and logic.

Local Usage Guide (not for GitHub):
1) Quick launch (TUI wizard):
   .venv/bin/python scripts/play_human_vs_ai.py --tui

2) Pick a specific run folder:
   .venv/bin/python scripts/play_human_vs_ai.py --tui --ckpt-dir checkpoints/star

3) Direct CLI (no wizard):
   .venv/bin/python scripts/play_human_vs_ai.py --mode human1_ai3 --checkpoint checkpoints/star/star_008_step_110879895.pt

4) Watch AI vs AI:
   .venv/bin/python scripts/play_human_vs_ai.py --mode ai4 --checkpoint checkpoints/star/star_008_step_110879895.pt

Input tips:
- Enter / p / pass: pass
- "0 1 2": play by hand indices (shown on UI)
- "3H 4D": play by card names
- "list": show legal action IDs
- "quit": exit
"""

from __future__ import annotations

import argparse
import os
import time
import signal
import random
import logging
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Set, Union

import torch
import torch.nn.functional as F
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.style import Style
from rich import box
from rich.prompt import Prompt

from rl_poker.rl import (
    GPUPokerEnv,
    PolicyNetwork,
    RecurrentPolicyNetwork,
    HistoryConfig,
    HistoryBuffer,
    GPURandomPolicy,
    GPUHeuristicPolicy,
    build_response_rank_weights,
)
from rl_poker.moves.gpu_action_mask import ACTION_TABLE, GPUActionMaskComputer
from rl_poker.rules.ranks import Card, Rank, Suit
from rl_poker.scripts.train import TrainConfig


# ==============================================================================
# Constants & Config
# ==============================================================================

RANK_STRS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUIT_STRS = ["H", "D", "C", "S"]
SUIT_SYMBOLS = ["\u2665", "\u2666", "\u2663", "\u2660"]

# Colors for TUI - High Contrast / Visibility
COLOR_HEART = "red1"      # Bright red
COLOR_DIAMOND = "red1"    # Bright red
COLOR_CLUB = "green1"     # Bright green (distinct from black/white)
COLOR_SPADE = "cyan1"     # Bright cyan (distinct from white)
COLOR_BG_CARD = "black"  # or "white" if using black text, but terminals usually dark
COLOR_HIGHLIGHT = "yellow"

console = Console()
logger = logging.getLogger(__name__)

# ==============================================================================
# UI Helpers
# ==============================================================================

def get_card_rich_text(idx: int) -> Text:
    """Return a Rich Text object for a card with symbol and color."""
    rank = idx % 13
    suit = idx // 13
    
    rank_char = RANK_STRS[rank]
    symbol = SUIT_SYMBOLS[suit]
    
    if suit == 0:  # Heart
        style = f"bold {COLOR_HEART}"
    elif suit == 1:  # Diamond
        style = f"bold {COLOR_DIAMOND}"
    elif suit == 2:  # Club
        style = f"bold {COLOR_CLUB}"
    else:  # Spade
        style = f"bold {COLOR_SPADE}"
    
    # Big Card Layout (3 lines) for better visibility
    # Width 7 characters (Visual)
    # Line 1: "R     " (2+5)
    # Line 2: "   S   " (3+1+3)
    # Line 3: "     R" (5+2)
    
    # Use Fullwidth characters for single digits/letters to make them look "Bigger"
    # 10 is 2 chars wide, matches fullwidth width roughly
    full_map = {
        "3": "３", "4": "４", "5": "５", "6": "６", "7": "７",
        "8": "８", "9": "９", "J": "Ｊ", "Q": "Ｑ", "K": "Ｋ",
        "A": "Ａ", "2": "２"
    }
    disp_rank = full_map.get(rank_char, rank_char)
    
    # Manual construction for alignment safety
    line1 = f"{disp_rank}     "
    line2 = f"   {symbol}   "
    line3 = f"     {disp_rank}"
    
    return Text(f"{line1}\n{line2}\n{line3}", style=style)


def render_hand_visual(card_idxs: List[int], indices: Optional[List[int]] = None) -> Table:
    """Render a horizontal list of cards."""
    # Use tighter padding in grid, let Panel handle card spacing
    grid = Table.grid(padding=(0, 1))
    
    # If indices provided, add index row
    if indices is not None:
        idx_row = []
        for i in indices:
            # Match the card width (~9 with border) for the index centering
            idx_row.append(Text(f"{i}", justify="center", style="bold white"))
        grid.add_row(*idx_row)
        
    card_row = []
    for idx in card_idxs:
        # Create a mini panel for each card
        t = get_card_rich_text(idx)
        # padding=(0, 1) to conserve vertical space while keeping width
        # Card content 3 lines high + 0 padding = 3 lines.
        # Border adds 2 lines. Total height 5 lines.
        card_row.append(Panel(t, expand=False, padding=(0,1), border_style="white"))
    
    grid.add_row(*card_row)
    return grid


def describe_action(action_idx: int) -> str:
    """Get text description of an action."""
    if action_idx == 0:
        return "PASS"
    
    atype, data = ACTION_TABLE[action_idx]
    
    if atype == "SINGLE":
        card_idx = next(iter(data))
        return f"SINGLE {card_idx_to_str(card_idx)}"
    
    if atype == "PAIR":
        cards = sorted(list(data))
        return f"PAIR {card_idx_to_str(cards[0])} {card_idx_to_str(cards[1])}"
        
    if atype == "STRAIGHT":
        ranks = [RANK_STRS[r] for r in data]
        return f"STRAIGHT {'-'.join(ranks)}"
        
    if "CONSEC_PAIRS" in str(atype):
        ranks = [RANK_STRS[r] for r in data]
        return f"PAIRS {'-'.join(ranks)}"
        
    if "THREE" in str(atype) or "FOUR" in str(atype):
        # Generic handler for complex types
        # 3+2, 4+3, etc.
        # Structure is usually (main_rank, kickers)
        main_rank, kickers = data
        main = RANK_STRS[main_rank]
        if isinstance(kickers, (list, tuple)):
            k_str = " ".join(RANK_STRS[k] for k in kickers) if kickers else ""
            return f"{atype} {main} + {k_str}"
        return f"{atype} {main}"
        
    return str(atype)


def describe_action_visual(action_idx: int) -> Union[Text, Table]:
    """Get a visual representation of the action (cards)."""
    if action_idx == 0:
        return Text("PASS", style="dim italic")
    
    try:
        atype, data = ACTION_TABLE[action_idx]
        cards = []
        
        if atype == "SINGLE":
            return render_hand_visual(list(data))
        if atype == "PAIR":
            return render_hand_visual(sorted(list(data)))
            
        # For others, we just return text because we don't know suits
        return Text(describe_action(action_idx), style="cyan")
    except Exception as exc:
        if os.getenv("RL_POKER_DEBUG") == "1":
            console.print(f"[dim]describe_action_visual failed for {action_idx}: {exc}[/dim]")
        else:
            logger.exception("describe_action_visual failed for action %s", action_idx)
        return Text(str(action_idx))


def card_idx_to_str(idx: int) -> str:
    rank = idx % 13
    suit = idx // 13
    return f"{RANK_STRS[rank]}{SUIT_STRS[suit]}"

# ==============================================================================
# Model & Logic
# ==============================================================================

def build_pair_lookup(mask_computer: GPUActionMaskComputer) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    pair_cards = mask_computer.pair_card_idx
    for i in range(pair_cards.shape[0]):
        card1 = int(pair_cards[i, 0].item())
        card2 = int(pair_cards[i, 1].item())
        key = (min(card1, card2), max(card1, card2))
        lookup[key] = mask_computer.pair_start + i
    return lookup

def parse_cards_or_indices(tokens: Iterable[str], displayed_hand: List[int]) -> List[int]:
    """Parse user input into card indices."""
    selected_indices: List[int] = []
    
    # Case 1: All tokens are integers -> Treat as hand indices
    if all(t.isdigit() for t in tokens):
        for t in tokens:
            idx = int(t)
            if idx < 0 or idx >= len(displayed_hand):
                raise ValueError(f"Index {idx} out of range")
            selected_indices.append(displayed_hand[idx])
        return selected_indices

    # Case 2: Treat as card names (e.g. "3H", "KS")
    for token in tokens:
        token = token.strip().upper()
        # Symbol normalization
        token = token.replace("\u2665", "H").replace("\u2666", "D").replace("\u2663", "C").replace("\u2660", "S")
        try:
            card = Card.from_string(token)
            idx = int(card.suit.value) * 13 + int(card.rank.value)
            selected_indices.append(idx)
        except Exception:
             # Try mixed mode? e.g. "3" -> Index 3?
             # If exact match fails, try parsing as int
             if token.isdigit():
                 idx = int(token)
                 if 0 <= idx < len(displayed_hand):
                     selected_indices.append(displayed_hand[idx])
                     continue
             raise ValueError(f"Unknown card: {token}")
             
    return selected_indices


def parse_ai_hand_spec(spec: str) -> List[int]:
    """Parse a custom AI hand spec into absolute card indices [0-51].

    Accepts:
      - Card strings: "3H 10S AS", unicode suits allowed (♥♦♣♠)
      - Absolute indices: "0 1 2" (0..51)
      - Mixed separators: spaces/commas
    """
    if not spec:
        raise ValueError("Empty AI hand spec.")
    tokens = [t.strip() for t in spec.replace(",", " ").split() if t.strip()]
    if not tokens:
        raise ValueError("Empty AI hand spec.")

    card_idxs: List[int] = []
    for token in tokens:
        token = token.strip()
        # Normalize unicode suits to letters
        token = (
            token.replace("\u2665", "H")
            .replace("\u2666", "D")
            .replace("\u2663", "C")
            .replace("\u2660", "S")
        )
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= 52:
                raise ValueError(f"Card index out of range: {idx}")
            card_idxs.append(idx)
            continue
        # Try parse as card string
        try:
            card = Card.from_string(token.upper())
            idx = int(card.suit.value) * 13 + int(card.rank.value)
            card_idxs.append(idx)
        except Exception as exc:
            raise ValueError(f"Invalid card token: {token}") from exc

    # Validate uniqueness and size
    unique = set(card_idxs)
    if len(unique) != len(card_idxs):
        raise ValueError("Duplicate cards in AI hand spec.")
    if len(card_idxs) != 13:
        raise ValueError(f"AI hand must have exactly 13 cards, got {len(card_idxs)}.")

    return card_idxs


def build_custom_hands(
    ai_seat: int, ai_card_idxs: List[int]
) -> List[List[Card]]:
    """Build full 4-player hands with a fixed AI hand and random human hands."""
    if ai_seat not in (0, 1, 2, 3):
        raise ValueError(f"Invalid AI seat: {ai_seat}")

    # Build full deck indices
    deck = list(range(52))
    ai_set = set(ai_card_idxs)
    if len(ai_set) != 13:
        raise ValueError("AI hand must contain 13 unique cards.")
    if not ai_set.issubset(deck):
        raise ValueError("AI hand contains invalid card indices.")

    # Remove AI cards from deck
    remaining = [i for i in deck if i not in ai_set]
    if len(remaining) != 39:
        raise ValueError("Remaining deck size mismatch after removing AI cards.")

    random.shuffle(remaining)

    def idx_to_card(idx: int) -> Card:
        suit = Suit(idx // 13)
        rank = Rank(idx % 13)
        return Card(rank=rank, suit=suit)

    hands: List[List[Card]] = [[] for _ in range(4)]
    hands[ai_seat] = [idx_to_card(i) for i in ai_card_idxs]

    # Deal remaining cards to human seats in order
    pos = 0
    for seat in range(4):
        if seat == ai_seat:
            continue
        hand_idxs = remaining[pos : pos + 13]
        pos += 13
        hands[seat] = [idx_to_card(i) for i in hand_idxs]

    return hands


def parse_ai_checkpoints(spec: Optional[str]) -> Tuple[List[str], Dict[int, str]]:
    """Parse ai checkpoint spec.

    Formats:
      - "a.pt" (single) 
      - "a.pt,b.pt,c.pt" (list)
      - "0:a.pt,2:b.pt" (seat mapping)
    """
    if not spec:
        return [], {}
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        return [], {}
    if any(":" in t for t in tokens):
        mapping: Dict[int, str] = {}
        for token in tokens:
            if ":" not in token:
                raise ValueError("Seat mapping requires seat:path format for all entries.")
            seat_str, path = token.split(":", 1)
            seat = int(seat_str)
            if seat not in (0, 1, 2, 3):
                raise ValueError(f"Invalid seat index: {seat}")
            if seat in mapping:
                raise ValueError(f"Duplicate seat entry: {seat}")
            mapping[seat] = path
        return [], mapping
    return tokens, {}


def resolve_ai_paths(
    ai_seats: Set[int],
    default_path: Optional[str],
    list_paths: List[str],
    map_paths: Dict[int, str],
) -> Dict[int, str]:
    if map_paths:
        if default_path is None and any(seat not in map_paths for seat in ai_seats):
            raise ValueError("Missing default checkpoint for unmapped AI seats.")
        return {seat: map_paths.get(seat, default_path) for seat in ai_seats}
    if not list_paths:
        if default_path is None:
            raise ValueError("No checkpoint provided.")
        return {seat: default_path for seat in ai_seats}
    if len(list_paths) == 1:
        return {seat: list_paths[0] for seat in ai_seats}
    if len(list_paths) == len(ai_seats):
        return {seat: path for seat, path in zip(sorted(ai_seats), list_paths)}
    if len(list_paths) == 4:
        seat_paths = {i: list_paths[i] for i in range(4)}
        return {seat: seat_paths[seat] for seat in ai_seats}
    raise ValueError("Invalid --ai-checkpoints count. Provide 1, 4, or one per AI seat.")


def action_from_cards(
    card_idxs: List[int],
    mask: torch.Tensor,
    mask_computer: GPUActionMaskComputer,
    pair_lookup: Dict[Tuple[int, int], int],
) -> Optional[int]:
    
    if not card_idxs:
        # User entered 'pass' or empty -> treated elsewhere. 
        # If this function called with empty list, maybe logic error.
        return 0

    card_idxs = sorted(card_idxs)
    num_cards = len(card_idxs)
    
    # 1. Single
    if num_cards == 1:
        action = mask_computer.single_start + card_idxs[0]
        if bool(mask[0, action]):
            return int(action)
        return None

    # 2. Pair
    if num_cards == 2:
        # Check ranks match
        if (card_idxs[0] % 13) == (card_idxs[1] % 13):
            key = (card_idxs[0], card_idxs[1])
            action = pair_lookup.get(key)
            if action is not None and bool(mask[0, action]):
                return int(action)
        return None

    # 3. Complex types
    # Logic:
    # We iterate over all valid actions in the mask.
    # For each valid action, we check if the hand composition matches the action requirements.
    
    # Optimization: Filter by length first
    valid_actions = mask[0].nonzero(as_tuple=False).squeeze(-1)
    
    # Get action requirements
    counts = torch.zeros(13, dtype=torch.long, device=mask.device)
    for idx in card_idxs:
        counts[idx % 13] += 1
    
    candidates = []
    
    for action in valid_actions.tolist():
        # Check Length
        if int(mask_computer.action_card_counts[action].item()) != num_cards:
            continue
            
        # Check Rank Counts
        req_counts = mask_computer.action_required_counts[action]
        if torch.equal(req_counts, counts):
            # Check Special Constraints (Flush/Straight Flush)
            atype, _ = ACTION_TABLE[action]
            # atype_str = str(atype)
            
            # Straight Flush vs Straight handled by game logic, usually higher action ID is better.
            candidates.append(action)

    if not candidates:
        return None

    if len(candidates) == 1:
        return int(candidates[0])

    # If multiple candidates, try to match exact action data by rank signature
    rank_counts = counts.tolist()
    ranks = [r for r, c in enumerate(rank_counts) if c > 0]

    expected: List[Tuple[str, Tuple]] = []

    # Straight (all single ranks, consecutive)
    if all(rank_counts[r] == 1 for r in ranks) and len(ranks) >= 5:
        sorted_ranks = sorted(ranks)
        if all(sorted_ranks[i] + 1 == sorted_ranks[i + 1] for i in range(len(sorted_ranks) - 1)):
            expected.append(("STRAIGHT", tuple(sorted_ranks)))

    # Consecutive pairs (all ranks appear twice, consecutive)
    if all(rank_counts[r] == 2 for r in ranks) and len(ranks) >= 3:
        sorted_ranks = sorted(ranks)
        if all(sorted_ranks[i] + 1 == sorted_ranks[i + 1] for i in range(len(sorted_ranks) - 1)):
            expected.append(("CONSEC_PAIRS", tuple(sorted_ranks)))

    # Three + kickers
    if any(c == 3 for c in rank_counts):
        main = next(r for r, c in enumerate(rank_counts) if c == 3)
        kickers = []
        for r, c in enumerate(rank_counts):
            if r == main:
                continue
            kickers.extend([r] * c)
        kickers = tuple(sorted(kickers))
        if num_cards <= 4:
            expected.append(("THREE_TWO_EXEMPT", (main, kickers)))
        if num_cards == 5:
            expected.append(("THREE_PLUS_TWO", (main, kickers)))

    # Four + kickers
    if any(c == 4 for c in rank_counts):
        main = next(r for r, c in enumerate(rank_counts) if c == 4)
        kickers = []
        for r, c in enumerate(rank_counts):
            if r == main:
                continue
            kickers.extend([r] * c)
        kickers = tuple(sorted(kickers))
        if num_cards <= 6:
            expected.append(("FOUR_THREE_EXEMPT", (main, kickers)))
        if num_cards == 7:
            expected.append(("FOUR_PLUS_THREE", (main, kickers)))

    if expected:
        for action in candidates:
            atype, data = ACTION_TABLE[action]
            for exp_type, exp_data in expected:
                if atype == exp_type and data == exp_data:
                    return int(action)

    # Fallback: choose the lowest legal action for stability
    return int(min(candidates))


# ==============================================================================
# TUI Layout Managers
# ==============================================================================

class GameUI:
    def __init__(self, human_seats: Set[int], ai_seats: Set[int], model_names: Dict[int, str]):
        self.human_seats = human_seats
        self.ai_seats = ai_seats
        self.model_names = model_names
        self.console = Console()

    def create_layout(self, state, prev_desc: str, action_history: deque) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="log_area", size=6),
        )
        
        # Header
        layout["header"].update(self._make_header())
        
        # Main Table
        # Balanced layout with enlarged areas
        # Fixed sizes reduced to Ensure Table has space even on smaller screens
        layout["main"].split_column(
            Layout(name="opponents", size=8),
            Layout(name="table_center", ratio=1),
            Layout(name="player_me", size=16),
        )
        
        # Opponents and Me
        self._update_players(layout, state, prev_desc)
        
        # Table Center (Last Played)
        layout["table_center"].update(self._make_table_center(state, prev_desc))
        
        # Footer / Log
        layout["log_area"].split_row(
            Layout(name="history", ratio=2),
            Layout(name="controls", ratio=1)
        )
        layout["history"].update(self._make_history_panel(action_history))
        layout["controls"].update(self._make_controls_panel())
        
        return layout

    def _make_header(self) -> Panel:
        title = Text("RL Poker: Big Two (Human vs AI)", style="bold white on blue", justify="center")
        return Panel(title, box=box.HEAVY)

    def _update_players(self, layout, state, prev_desc: str):
        # We want to arrange players: Top (Opposite), Left/Right, Bottom (Me).
        # Standard Big2 is 4 players.
        # Fixed mapping: 0,1,2,3.
        # Assuming Human is seat 0 (or defined in human_seats).
        # Let's verify 'state.current_player' to highlight.
        
        cur_p = int(state.current_player.item())
        lead_p = int(state.lead_player.item())
        is_new_lead = (prev_desc == "New lead")
        
        # Find "My" seat (first human seat or 0)
        my_seat = next(iter(self.human_seats)) if self.human_seats else 0
        
        # Relative positions: 
        # Bottom: Me
        # Right: (Me+1)%4
        # Top: (Me+2)%4
        # Left: (Me+3)%4
        
        seat_map = {
            "bottom": my_seat,
            "right": (my_seat + 1) % 4,
            "top": (my_seat + 2) % 4,
            "left": (my_seat + 3) % 4
        }
        
        # Generate panels
        panels = {}
        for pos, seat in seat_map.items():
            name = self.model_names.get(seat, f"Player {seat}")
            if seat in self.human_seats:
                name = f"{name} (Human)"
            
            count = int(state.cards_remaining[0, seat].item())
            is_turn = (seat == cur_p)
            has_lead = (seat == lead_p) and is_new_lead
            
            panels[pos] = self._create_player_panel(seat, name, count, is_turn, has_lead, state)

        # "opponents" layout: Left, Top, Right
        opp_grid = Table.grid(expand=True)
        opp_grid.add_column(justify="center", ratio=1)
        opp_grid.add_column(justify="center", ratio=1)
        opp_grid.add_column(justify="center", ratio=1)
        
        opp_grid.add_row(panels["left"], panels["top"], panels["right"])
        
        layout["opponents"].update(Panel(opp_grid, border_style="dim"))
        layout["player_me"].update(panels["bottom"])

    def _create_player_panel(self, seat: int, name: str, cards_cnt: int, is_turn: bool, is_lead: bool, state) -> Panel:
        # Style
        border = "dim"
        if is_turn:
            border = "bold yellow"
            name = f"▶ {name}"
        if is_lead:
            name += " [★LEAD]"
            
        content = []
        content.append(Text(name, justify="center"))
        
        # Card Visuals
        # For opponents, show card backs? Or just count.
        # For Human, show cards.
        
        if seat in self.human_seats:
            # Show actual cards
            hand = state.hands[0, seat].nonzero(as_tuple=False).squeeze(-1).tolist()
            hand = sorted(hand, key=lambda x: (x % 13, x // 13))
            index_map = {card: i for i, card in enumerate(hand)}
            
            # Cards are wider now (~8-9 chars), so reducing chunk size prevents wrapping
            # Fits ~8 cards per line on standard terminals
            BYTES_PER_LINE = 8
            card_chunks = [hand[i:i+BYTES_PER_LINE] for i in range(0, len(hand), BYTES_PER_LINE)]
            for i, chunk in enumerate(card_chunks):
                # We need indices relative to the sorted hand for selection
                indices = [index_map[c] for c in chunk]
                content.append(render_hand_visual(chunk, indices))
                # Add spacing between rows if not the last row
                if i < len(card_chunks) - 1:
                     content.append(Text(""))
        else:
            # Opponent: Show count
            if cards_cnt == 0:
                txt = Text("WINNER", style="bold gold1")
            else:
                txt = Text(f"🂠  " * cards_cnt, style="red" if cards_cnt < 5 else "white", overflow="fold")
                if cards_cnt > 13: # Fallback for bug
                    txt = Text(f"{cards_cnt} cards")
            content.append(Align.center(txt))
            
        return Panel(Group(*content), border_style=border, title=f"P{seat}")

    def _make_table_center(self, state, prev_desc: str) -> Panel:
        consecutive_passes = int(state.consecutive_passes.item())
        
        grid = Table.grid(expand=True, padding=1)
        grid.add_column(justify="center")
        
        grid.add_row("[underline]Pile[/underline]")
        grid.add_row("")
        
        if prev_desc == "New lead":
            grid.add_row(Text("Waiting for Lead...", style="italic dim"))
        else:
            # Use visual card representation
            action_idx = int(state.prev_action.item())
            # describe_action_visual handles PASS (0) and cards
            vis = describe_action_visual(action_idx)
            # Center the visual
            if isinstance(vis, Table):
                # Tables in a grid cell are tricky to center unless wrapped in Align
                grid.add_row(Align.center(vis))
            else:
                grid.add_row(Align.center(vis))
            
        grid.add_row("")
        if consecutive_passes > 0:
            grid.add_row(f"Consecutive Passes: [bold red]{consecutive_passes}[/bold red]")
            
        return Panel(Align.center(grid), border_style="blue", title="Table")

    def _make_history_panel(self, history: deque) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=4)
        table.add_column()
        
        for seat, action in history:
            table.add_row(f"P{seat}", action)
            
        return Panel(table, title="History", border_style="white")

    def _make_controls_panel(self) -> Panel:
        # Enhanced instructions for clarity
        help_text = """[bold underline]Controls[/bold underline]
[bold cyan]0 1 2 ...[/]  : Select cards by index
[bold cyan]3H 4D ...[/]  : Select cards by name
[bold cyan]Enter[/]      : Pass turn
[bold yellow]list[/]       : Show valid moves
[bold red]quit[/]       : Exit Game
"""
        return Panel(Text.from_markup(help_text), title="Help", border_style="white")


def build_settlement_panel(
    state,
    rewards: torch.Tensor,
    human_seats: Set[int],
    model_names: Dict[int, str],
) -> Panel:
    ranks = state.finish_rank[0].tolist()
    rows = []
    for seat in range(4):
        rank = int(ranks[seat])
        name = model_names.get(seat, "Human" if seat in human_seats else f"P{seat}")
        role = "Human" if seat in human_seats else "AI"
        rows.append((rank, seat, role, name))

    rows.sort(key=lambda r: r[0])

    table = Table(title="Final Results", box=box.SIMPLE, show_header=True)
    table.add_column("Rank", justify="right")
    table.add_column("Seat", justify="right")
    table.add_column("Role")
    table.add_column("Model")

    for rank, seat, role, name in rows:
        table.add_row(str(rank), f"P{seat}", role, name)

    return Panel(table, title="Settlement", border_style="green")

# ==============================================================================
# Checkpoint Loading (Preserved & Optimized)
# ==============================================================================

def load_checkpoint_safe(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        # Fallback for safe globals if needed, though weights_only=False covers most
        from torch.serialization import safe_globals
        with safe_globals([TrainConfig]):
            return torch.load(path, map_location=device, weights_only=False)

def extract_config_from_ckpt(ckpt) -> dict:
    # Default config
    cfg = {
        "use_recurrent": True,
        "history_window": 32,
        "reveal_opponent_ranks": False,
        "hidden_size": 256,
        "gru_hidden": 128,
        "belief_use_behavior": True,
        "belief_decay": 0.98,
        "belief_play_bonus": 0.5,
        "belief_pass_penalty": 0.3,
        "belief_temp": 2.0,
    }
    
    ckpt_cfg = ckpt.get("config", None)
    if ckpt_cfg:
        for k in cfg.keys():
            if hasattr(ckpt_cfg, k):
                val = getattr(ckpt_cfg, k)
                if val is not None:
                    cfg[k] = val
    return cfg

def _list_checkpoints(ckpt_dir: str) -> List[Path]:
    root = Path(ckpt_dir).expanduser()
    if not root.exists():
        return []
    return sorted(root.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def configure_interactive(args):
    """Interactive configuration wizard using Rich."""
    console.print(Panel("[bold cyan]Game Setup[/bold cyan]", box=box.HEAVY))

    # 1. Select Mode
    modes = ["human1_ai3", "human3_ai1", "ai4"]
    mode_display = {
        "human1_ai3": "1 Human vs 3 AI",
        "human3_ai1": "3 Humans vs 1 AI",
        "ai4": "4 AI (Watch Mode)"
    }
    
    console.print("\n[bold]Select Game Mode:[/bold]")
    for i, m in enumerate(modes):
        console.print(f" {i+1}. {mode_display[m]}")
    
    while True:
        choice = Prompt.ask("Choice", default="1")
        if choice.isdigit() and 1 <= int(choice) <= len(modes):
            args.mode = modes[int(choice)-1]
            break
        console.print("[red]Invalid choice[/red]")

    # 2. Seat Selection
    if args.mode == "human1_ai3":
        args.human_seat = int(Prompt.ask("Your Seat (0-3)", default="0", choices=["0","1","2","3"]))
    elif args.mode == "human3_ai1":
        args.ai_seat = int(Prompt.ask("AI Seat (0-3)", default="3", choices=["0","1","2","3"]))

    # 3. Opponent Type
    op_types = ["policy", "heuristic", "random"]
    console.print("\n[bold]Select Opponent Type:[/bold]")
    for i, t in enumerate(op_types):
        console.print(f" {i+1}. {t.capitalize()}")
        
    while True:
        choice = Prompt.ask("Choice", default="1")
        if choice.isdigit() and 1 <= int(choice) <= len(op_types):
            args.opponent = op_types[int(choice)-1]
            break

    # 4. Checkpoint Selection (If Policy)
    ckpts = []
    if args.opponent == "policy":
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.exists():
            console.print(f"[yellow]Checkpoint directory {ckpt_dir} not found.[/yellow]")
        else:
            ckpts = _list_checkpoints(str(ckpt_dir))
            if not ckpts:
                console.print("[red]No checkpoints found![/red]")
            else:
                console.print(f"\n[bold]Select Checkpoint (found {len(ckpts)}):[/bold]")
                # Show top 5 newest
                options = ckpts[:10]
                for i, p in enumerate(options):
                    # Show readable time
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))
                    rel = p.relative_to(ckpt_dir)
                    console.print(f" {i+1}. {rel} [dim]({mtime})[/dim]")
                
                while True:
                    choice = Prompt.ask("Choice", default="1")
                    if choice.isdigit() and 1 <= int(choice) <= len(options):
                        args.checkpoint = str(options[int(choice)-1])
                        break
                    console.print("[red]Invalid choice[/red]")

    # 4b. Optional per-seat model selection
    if args.opponent == "policy" and ckpts:
        if args.mode == "human1_ai3":
            ai_seats = {0, 1, 2, 3} - {args.human_seat}
        elif args.mode == "human3_ai1":
            ai_seats = {args.ai_seat}
        else:
            ai_seats = {0, 1, 2, 3}

        if len(ai_seats) > 1:
            same = Prompt.ask("Use same model for all AI seats? (y/n)", default="y", choices=["y", "n"])
            if same == "n":
                selections = []
                for seat in sorted(ai_seats):
                    console.print(f"\n[bold]Select checkpoint for seat {seat}[/bold]")
                    options = ckpts[:10]
                    for i, p in enumerate(options):
                        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))
                        rel = p.relative_to(ckpt_dir)
                        console.print(f" {i+1}. {rel} [dim]({mtime})[/dim]")
                    while True:
                        choice = Prompt.ask("Choice", default="1")
                        if choice.isdigit() and 1 <= int(choice) <= len(options):
                            selections.append(f"{seat}:{str(options[int(choice)-1])}")
                            break
                        console.print("[red]Invalid choice[/red]")
                args.ai_checkpoints = ",".join(selections)

    # 4b. AI delay
    try:
        args.ai_delay = float(Prompt.ask("AI move delay (seconds)", default=str(args.ai_delay)))
        if args.ai_delay < 0:
            args.ai_delay = 0.0
    except Exception:
        console.print("[yellow]Invalid delay; using default.[/yellow]")

    # 5. Review
    console.print(f"\n[dim]Configured: Mode={args.mode}, Opponent={args.opponent}[/dim]")
    console.print(f"[dim]Checkpoint: {Path(args.checkpoint).name if args.checkpoint else 'None'}[/dim]\n")


# ==============================================================================
# Main Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Poker: Human vs AI (Enhanced)")
    default_ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    
    parser.add_argument("--checkpoint", type=str, help="Path to common checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default=str(default_ckpt_dir))
    
    # Mode Settings
    parser.add_argument("--tui", action="store_true", help="Use interactive setup wizard")
    parser.add_argument("--mode", default="human1_ai3", choices=["human1_ai3", "human3_ai1", "ai4"])
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--ai-seat", type=int, default=3)
    
    # Opponent Settings
    parser.add_argument("--opponent", default="policy", choices=["policy", "heuristic", "random"])
    parser.add_argument("--ai-checkpoints", type=str, help="csv or seat:path for specific models")
    parser.add_argument("--heuristic-style", default="aggressive")
    parser.add_argument(
        "--ai-hand",
        type=str,
        default=None,
        help="Custom AI hand for human3_ai1 mode. Format: '3H 4D 5S ...' or '0 1 2 ...' (13 cards)",
    )
    
    # Tech Specs
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--ai-delay", type=float, default=0.8, help="Thinking time for AI visualization")
    
    args = parser.parse_args()
    
    if args.tui:
        try:
            configure_interactive(args)
        except KeyboardInterrupt:
            console.print("\n[dim]Setup cancelled.[/dim]")
            return

    # Seed early so env.reset and any shuffles are deterministic when requested
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Device Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    console.print(f"[bold green]Running on {device}[/bold green]")
    
    # 1. Resolve Checkpoints
    # (Simplified logic from original script for clarity)
    ai_seats = set()
    human_seats = set()
    
    if args.mode == "human1_ai3":
        human_seats.add(args.human_seat)
        ai_seats = {0, 1, 2, 3} - human_seats
    elif args.mode == "human3_ai1":
        ai_seats.add(args.ai_seat)
        human_seats = {0, 1, 2, 3} - ai_seats
    else: # ai4
        ai_seats = {0, 1, 2, 3}

    # Resolve paths
    final_paths: Dict[int, str] = {}
    ai_list, ai_map = parse_ai_checkpoints(args.ai_checkpoints)
    if args.opponent != "policy" and (ai_list or ai_map):
        console.print("[yellow]Warning: --ai-checkpoints ignored for non-policy opponents.[/yellow]")
        ai_list, ai_map = [], {}

    if args.opponent == "policy":
        base_path = args.checkpoint
        if not base_path and not (ai_list or ai_map):
            ckpts = sorted(_list_checkpoints(args.ckpt_dir), key=lambda p: p.stat().st_mtime)
            if ckpts:
                base_path = str(ckpts[0])
                console.print(f"[yellow]Auto-selected latest checkpoint: {base_path}[/yellow]")
            else:
                console.print("[red]No checkpoint provided or found.[/red]")
                return

        final_paths = resolve_ai_paths(ai_seats, base_path, ai_list, ai_map)

        # Expand and validate
        for seat, path in list(final_paths.items()):
            p = os.path.expanduser(path)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Checkpoint not found for seat {seat}: {p}")
            final_paths[seat] = p

    # 2. Load Models
    policies = {}
    model_names = {}
    
    # Global Config (from first loaded model)
    env_cfg = None

    if args.opponent == "policy":
        with console.status("Loading AI Models..."):
            cache = {}
            for seat, path in final_paths.items():
                if path not in cache:
                    ckpt = load_checkpoint_safe(path, device)
                    cache[path] = ckpt
                
                ckpt = cache[path]
                # Extract Config if first
                cfg = extract_config_from_ckpt(ckpt)
                if env_cfg is None:
                    env_cfg = cfg
                else:
                    # Basic config consistency check across checkpoints
                    for k, v in env_cfg.items():
                        if cfg.get(k) != v:
                            raise ValueError(f"Checkpoint config mismatch for {k}: {v} vs {cfg.get(k)}")
                
                # Load Net
                state_dict = ckpt["network"]
                
                # Env dimensions needed to init net
                # We need to peek at env first or assume standard dimensions
                # Let's create a dummy env to get dimensions
                # Optimized: We do this once outside loop
                pass
                
                # Save for later init
                policies[seat] = ("policy_deferred", state_dict, cfg)
                model_names[seat] = Path(path).stem[:15]
    elif args.opponent == "random":
        env_cfg = extract_config_from_ckpt({}) # Defaults
        policy = GPURandomPolicy(seed=args.seed)
        for s in ai_seats:
            policies[s] = policy
            model_names[s] = "Random"
    else:
        env_cfg = extract_config_from_ckpt({})
        # Policy inited after env
        for s in ai_seats:
            policies[s] = ("heuristic", args.heuristic_style)
            model_names[s] = f"Heur-{args.heuristic_style}"
            
    # 3. Initialize Env
    console.print("Initializing Environment...")
    env = GPUPokerEnv(1, device, reveal_opponent_ranks=bool(env_cfg["reveal_opponent_ranks"]))
    num_players = env.NUM_PLAYERS
    cards_per_player = env.CARDS_PER_PLAYER

    # Shared dimensions
    num_actions = env.num_actions
    max_act_len = int(env.mask_computer.action_lengths.max().item())
    num_act_types = int(env.mask_computer.action_types.max().item()) + 1
    scalar_dim = 4
    hist_dim = num_players + num_act_types + scalar_dim

    # Init Policies that needed Env
    if args.opponent == "heuristic":
        pol = GPUHeuristicPolicy(env.mask_computer, style=args.heuristic_style)
        for s in ai_seats:
            policies[s] = pol

    elif args.opponent == "policy":
        # Init Networks
        use_recurrent = env_cfg["use_recurrent"]
        hidden = env_cfg["hidden_size"]
        gru = env_cfg["gru_hidden"]
        
        # Dimensions
        obs_dim = env.obs_dim
        belief_dim = (num_players - 1) * cards_per_player
        oth_norm_dim = num_players - 1
        played_norm_dim = cards_per_player
        aug_dim = obs_dim + belief_dim + oth_norm_dim + played_norm_dim
        
        for seat, (ptype, state_dict, p_cfg) in list(policies.items()):
            if ptype == "policy_deferred":
                if use_recurrent:
                    net = RecurrentPolicyNetwork(aug_dim, num_actions, hist_dim, hidden_size=hidden, gru_hidden=gru)
                else:
                    net = PolicyNetwork(aug_dim, num_actions, hidden)
                
                net.to(device)
                net.load_state_dict(state_dict)
                net.eval()
                policies[seat] = net

    # 4. Aux Structures (History, Beliefs)
    history_cfg = HistoryConfig(enabled=env_cfg["use_recurrent"], window=env_cfg["history_window"], feature_dim=hist_dim)
    history_buffer = HistoryBuffer(1, history_cfg, device)
    
    # Belief trackers
    public_played_counts = torch.zeros(1, cards_per_player, device=device)
    opp_rank_logits = torch.zeros(1, num_players, cards_per_player, device=device)
    
    # Precompute affinity for belief update
    rank_pos = torch.arange(cards_per_player, device=device).float()
    rank_dist = torch.abs(rank_pos.unsqueeze(0) - rank_pos.unsqueeze(1))
    temp = max(env_cfg["belief_temp"], 1e-6)
    rank_affinity = torch.exp(-rank_dist / temp)
    rank_affinity /= rank_affinity.sum(dim=1, keepdim=True).clamp(min=1e-6)
    
    response_rank_weights = build_response_rank_weights(env.mask_computer)
    pair_lookup = build_pair_lookup(env.mask_computer)
    
    # Precomputer History Feature Tables
    # (Simplified: Copy from original for exact logic)
    act_rank_norm = (env.mask_computer.action_ranks.clamp(min=0).float() / 12.0).unsqueeze(1)
    act_len_norm = (env.mask_computer.action_lengths.float() / max_act_len).unsqueeze(1)
    act_exempt = env.mask_computer.action_is_exemption.float().unsqueeze(1)
    act_is_pass = (torch.arange(num_actions, device=device) == 0).float().unsqueeze(1)
    act_type_oh = F.one_hot(env.mask_computer.action_types, num_classes=num_act_types).float()
    
    def get_history_features(actions, players):
        p_oh = F.one_hot(players, num_classes=num_players).float()
        t_oh = act_type_oh[actions]
        scalars = torch.cat([act_rank_norm[actions], act_len_norm[actions], act_exempt[actions], act_is_pass[actions]], dim=1)
        return torch.cat([p_oh, t_oh, scalars], dim=1)

    other_offsets = torch.arange(1, num_players, device=device)
    
    def get_augmented_obs(obs, state):
        B = obs.shape[0]
        curr = state.current_player
        
        # My rank counts
        my_counts = state.rank_counts.gather(
            1, curr.view(B, 1, 1).expand(-1, 1, cards_per_player)
        ).squeeze(1)
        remaining = (4.0 - my_counts.float() - public_played_counts).clamp(min=0)
        
        # Others
        oth_idx = (curr.view(B, 1) + other_offsets.view(1, -1)) % num_players
        oth_cards = state.cards_remaining.gather(1, oth_idx)
        
        # Beliefs
        logits = opp_rank_logits.gather(1, oth_idx.unsqueeze(-1).expand(-1, -1, cards_per_player))
        weights = torch.softmax(logits, dim=2)
        weighted = remaining.unsqueeze(1) * weights
        norm = weighted.sum(dim=2).clamp(min=1.0)
        belief = weighted * (oth_cards.float() / norm).unsqueeze(-1)
        
        # Flatten and cat
        belief_flat = belief.reshape(B, -1) / 4.0
        oth_norm = oth_cards.float() / float(cards_per_player)
        played_norm = public_played_counts / 4.0
        
        return torch.cat([obs, belief_flat, oth_norm, played_norm], dim=1)

    # 5. Game Loop
    ui = GameUI(human_seats, ai_seats, model_names)
    while True:
        state = env.reset()
        if args.ai_hand:
            if args.mode != "human3_ai1":
                raise ValueError("--ai-hand is only supported in human3_ai1 mode.")
            ai_card_idxs = parse_ai_hand_spec(args.ai_hand)
            hands = build_custom_hands(args.ai_seat, ai_card_idxs)
            state = env.state_from_hands(hands)

        history_buffer.reset_envs(torch.tensor([0], device=device))
        public_played_counts.zero_()
        opp_rank_logits.zero_()
        action_history = deque(maxlen=8)

        game_result = None
        final_layout = None

        with Live(refresh_per_second=4, screen=True, auto_refresh=False, console=console) as live:
            while True:
                # Determine current state description
                prev_act = int(state.prev_action.item())
                if state.prev_action_rank.item() < 0:
                    desc = "New lead"
                else:
                    desc = describe_action(prev_act)

                current_player = int(state.current_player.item())

                # Render via Live
                layout = ui.create_layout(state, desc, action_history)
                live.update(layout)
                live.refresh()

                # === PLAYER TURN ===
                obs, mask = env.get_obs_and_mask(state)
                aug_obs = get_augmented_obs(obs, state)

                action = 0

                if current_player in human_seats:
                    live.stop()
                    console.clear()
                    console.print(layout)
                    console.print(f"[bold yellow]>> Your Turn (P{current_player})[/bold yellow]")

                    while True:
                        # Get Hand
                        hand_mask = state.hands[0, current_player]
                        hand_idxs = hand_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                        sorted_hand = sorted(hand_idxs, key=lambda x: (x % 13, x // 13))

                        raw = console.input("Action (Enter=pass)> ").strip()
                        if not raw:
                            if bool(mask[0, 0]):
                                action = 0
                                break
                            console.print("[red]Cannot pass (New Lead or Must Play)[/red]")
                            continue

                        cmd = raw.lower()

                        if cmd in ("quit", "exit"):
                            return

                        if cmd in ("l", "list"):
                            valid_ids = mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
                            t = Table(title="Legal Actions")
                            t.add_column("ID")
                            t.add_column("Desc")
                            for vid in valid_ids[:20]:
                                t.add_row(str(vid), describe_action(vid))
                            console.print(t)
                            if len(valid_ids) > 20:
                                console.print("...")
                            continue

                        if cmd in ("p", "pass"):
                            if bool(mask[0, 0]):
                                action = 0
                                break
                            console.print("[red]Cannot pass (New Lead or Must Play)[/red]")
                            continue

                        tokens = raw.replace(",", " ").split()
                        try:
                            selected = parse_cards_or_indices(tokens, sorted_hand)
                            if len(set(selected)) != len(selected):
                                console.print("[red]Duplicate cards[/red]")
                                continue

                            # Verify ownership
                            valid_ownership = True
                            for c in selected:
                                if c not in sorted_hand:
                                    console.print(f"[red]You don't hold {card_idx_to_str(c)}[/red]")
                                    valid_ownership = False
                            if not valid_ownership:
                                continue

                            act = action_from_cards(selected, mask, env.mask_computer, pair_lookup)
                            if act is None:
                                console.print("[red]Invalid move or not allowed now[/red]")
                                continue

                            action = act
                            break
                        except ValueError as ve:
                            console.print(f"[red]{ve}[/red]")
                            continue

                    live.start()

                else:
                    # AI Turn
                    if args.ai_delay > 0:
                        time.sleep(args.ai_delay)

                    policy_obj = policies[current_player]

                    with torch.no_grad():
                        if args.opponent == "policy":
                            net = policy_obj
                            if env_cfg["use_recurrent"]:
                                seq = history_buffer.get_sequence(torch.tensor([0], device=device))
                                act_tensor, _, _ = net.get_action(aug_obs, mask, seq)
                            else:
                                act_tensor, _, _ = net.get_action(aug_obs, mask)
                            action = int(act_tensor.item())
                        elif args.opponent == "random":
                            action = int(policy_obj.select_actions(aug_obs, mask).item())
                        else:
                            action = int(policy_obj.select_actions(aug_obs, mask).item())

                # === EXECUTE ACTION ===
                action_history.append((current_player, describe_action(action)))

                act_tensor = torch.tensor([action], device=device, dtype=torch.long)
                new_state, rewards, dones = env.step(state, act_tensor)

                # === UPDATE BELIEFS ===
                if action != 0:
                    req = env.mask_computer.action_required_counts[action].float()
                    public_played_counts[0] += req

                    if env_cfg["belief_use_behavior"]:
                        opp_rank_logits[0, current_player] *= env_cfg["belief_decay"]
                        evidence = req @ rank_affinity
                        opp_rank_logits[0, current_player] += env_cfg["belief_play_bonus"] * evidence
                elif env_cfg["belief_use_behavior"]:
                    prev_r = state.prev_action_rank.item()
                    if prev_r >= 0:
                        prev_a = state.prev_action
                        penalty = response_rank_weights[prev_a] @ rank_affinity
                        opp_rank_logits[0, current_player] -= env_cfg["belief_pass_penalty"] * penalty[0]

                # === UPDATE HISTORY ===
                if env_cfg["use_recurrent"]:
                    player_tensor = torch.tensor([current_player], device=device)
                    feats = get_history_features(act_tensor, player_tensor)
                    history_buffer.push(feats, env_ids=torch.tensor([0], device=device))

                state = new_state

                if bool(dones.item()):
                    final_layout = ui.create_layout(state, describe_action(action), action_history)
                    live.update(final_layout)
                    live.refresh()
                    game_result = (state, rewards, action)
                    break

        if game_result and final_layout is not None:
            console.clear()
            console.print(final_layout)

            settlement = build_settlement_panel(game_result[0], game_result[1], human_seats, model_names)
            console.print(settlement)

        again = Prompt.ask("Play again? (y/n)", default="n", choices=["y", "n"])
        if again != "y":
            break
            
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")

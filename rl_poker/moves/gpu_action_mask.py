"""GPU-accelerated action mask computation.

This module provides:
- Fixed action space encoding (all possible actions pre-enumerated)
- GPU-accelerated action mask computation using PyTorch
- Batched mask computation for multiple environments

Key insight: Instead of dynamically enumerating legal moves (CPU-intensive),
we pre-define ALL possible actions and compute a binary mask on GPU.

Action space structure (fixed indices):
- 0: PASS
- 1-52: Singles (13 ranks × 4 suits)
- 53-130: Pairs (13 ranks × C(4,2)=6 suit combinations)
- 131+: Straights, consecutive pairs, 3+2, 4+3 (pre-enumerated)
"""

import torch
from itertools import combinations_with_replacement
from typing import List, Tuple
from dataclasses import dataclass

from rl_poker.rules.ranks import Rank, Suit, Card


# ============================================================================
# Fixed Action Space Definition
# ============================================================================


# Card encoding: 0-51 for standard deck (4 suits × 13 ranks)
# card_idx = suit * 13 + rank
def card_to_idx(card: Card) -> int:
    """Convert Card to index 0-51."""
    return card.suit.value * 13 + card.rank.value


def idx_to_card(idx: int) -> Card:
    """Convert index 0-51 to Card."""
    suit = Suit(idx // 13)
    rank = Rank(idx % 13)
    return Card(rank=rank, suit=suit)


# All 52 cards as indices
ALL_CARDS = list(range(52))

# Pre-compute all possible actions
# Action 0: PASS
# Actions 1-52: Singles
# Actions 53-130: Pairs (13 ranks × 6 suit combinations)
# Actions 131+: Complex hands

# Pair suit combinations: C(4,2) = 6
PAIR_SUIT_COMBOS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _build_action_tables():
    """Build lookup tables for all possible actions."""
    actions = []  # List of (action_type, data)

    # Action 0: PASS
    actions.append(("PASS", frozenset()))

    # Actions 1-52: Singles
    for card_idx in range(52):
        actions.append(("SINGLE", frozenset([card_idx])))

    # Actions 53-130: Pairs (13 ranks × 6 suit combos = 78)
    for rank in range(13):
        for s1, s2 in PAIR_SUIT_COMBOS:
            card1 = s1 * 13 + rank
            card2 = s2 * 13 + rank
            actions.append(("PAIR", frozenset([card1, card2])))

    # Straights (5+ consecutive ranks, ranks 0-10 are valid sequence ranks)
    # Full lengths: 5 to 11 (3-K)
    for length in range(5, 12):
        for start_rank in range(11 - length + 1):  # Valid starts
            ranks = list(range(start_rank, start_rank + length))
            actions.append(("STRAIGHT", tuple(ranks)))  # Store ranks only

    # Consecutive pairs (3+ pairs, ranks 0-10)
    for num_pairs in range(3, 12):
        for start_rank in range(11 - num_pairs + 1):
            ranks = list(range(start_rank, start_rank + num_pairs))
            actions.append(("CONSEC_PAIRS", tuple(ranks)))

    # 3+2: 3 of main + any 2 cards (kicker ranks may repeat)
    for main_rank in range(13):
        other_ranks = [r for r in range(13) if r != main_rank]
        for k1, k2 in combinations_with_replacement(other_ranks, 2):
            actions.append(("THREE_PLUS_TWO", (main_rank, (k1, k2))))

    # 4+3: 4 of main + any 3 cards (kicker ranks may repeat)
    for main_rank in range(13):
        other_ranks = [r for r in range(13) if r != main_rank]
        for k1, k2, k3 in combinations_with_replacement(other_ranks, 3):
            actions.append(("FOUR_PLUS_THREE", (main_rank, (k1, k2, k3))))

    # Tail-hand exemptions: 3+0, 3+1
    for main_rank in range(13):
        other_ranks = [r for r in range(13) if r != main_rank]
        actions.append(("THREE_TWO_EXEMPT", (main_rank, ())))
        for k1 in other_ranks:
            actions.append(("THREE_TWO_EXEMPT", (main_rank, (k1,))))

    # Tail-hand exemptions: 4+0, 4+1, 4+2
    for main_rank in range(13):
        other_ranks = [r for r in range(13) if r != main_rank]
        actions.append(("FOUR_THREE_EXEMPT", (main_rank, ())))
        for k1 in other_ranks:
            actions.append(("FOUR_THREE_EXEMPT", (main_rank, (k1,))))
        for k1, k2 in combinations_with_replacement(other_ranks, 2):
            actions.append(("FOUR_THREE_EXEMPT", (main_rank, (k1, k2))))

    return actions


# Build action tables at module load
ACTION_TABLE = _build_action_tables()
NUM_ACTIONS = len(ACTION_TABLE)

# Create reverse lookup: action_type -> list of action indices
ACTION_TYPE_INDICES = {}
for idx, (atype, data) in enumerate(ACTION_TABLE):
    if atype not in ACTION_TYPE_INDICES:
        ACTION_TYPE_INDICES[atype] = []
    ACTION_TYPE_INDICES[atype].append(idx)


# ============================================================================
# GPU Action Mask Computation
# ============================================================================

# Action type constants (for following constraint)
ACTION_TYPE_PASS = 0
ACTION_TYPE_SINGLE = 1
ACTION_TYPE_PAIR = 2
ACTION_TYPE_STRAIGHT = 3
ACTION_TYPE_CONSEC_PAIRS = 4
ACTION_TYPE_THREE_TWO = 5
ACTION_TYPE_FOUR_THREE = 6
ACTION_TYPE_THREE_TWO_EXEMPT = 7
ACTION_TYPE_FOUR_THREE_EXEMPT = 8


@dataclass
class GPUActionMaskComputer:
    """GPU-accelerated action mask computation.

    Keeps tensors on GPU to avoid CPU-GPU transfer overhead.
    """

    device: torch.device

    # Pre-computed tensors for fast mask computation
    single_card_idx: torch.Tensor  # [52] - which card each single action needs
    pair_card_idx: torch.Tensor  # [78, 2] - which 2 cards each pair needs
    straight_rank_masks: torch.Tensor  # [num_straights, 13] - which ranks needed
    consec_pair_rank_masks: torch.Tensor  # [num_consec, 13] - which ranks needed (2+ cards)
    three_two_ranks: torch.Tensor  # [num_3+2, 3] - (main_rank, kicker1, kicker2)
    four_three_ranks: torch.Tensor  # [num_4+3, 4] - (main_rank, kicker1, kicker2, kicker3)

    # For following constraint: type and rank of each action
    action_types: torch.Tensor  # [num_actions] - type of each action
    action_ranks: torch.Tensor  # [num_actions] - main rank of each action (-1 for PASS)
    action_lengths: torch.Tensor  # [num_actions] - length for straights/consec_pairs (0 otherwise)

    # Action index ranges
    pass_idx: int = 0
    single_start: int = 1
    single_end: int = 53  # exclusive
    pair_start: int = 53
    pair_end: int = 131  # 53 + 78

    def __init__(self, device: torch.device):
        self.device = device
        self._build_tensors()
        self._build_action_metadata()

    def _build_tensors(self):
        """Build pre-computed tensors for mask computation."""
        # Singles: action i (1-52) needs card i-1
        self.single_card_idx = torch.arange(52, device=self.device)

        # Pairs: build [78, 2] tensor of card indices
        pair_cards = []
        for rank in range(13):
            for s1, s2 in PAIR_SUIT_COMBOS:
                card1 = s1 * 13 + rank
                card2 = s2 * 13 + rank
                pair_cards.append([card1, card2])
        self.pair_card_idx = torch.tensor(pair_cards, device=self.device, dtype=torch.long)
        self.pair_end = self.pair_start + len(pair_cards)

        # Straights: [num_straights, 13] boolean mask of ranks needed
        straight_masks = []
        straight_indices = ACTION_TYPE_INDICES.get("STRAIGHT", [])
        for idx in straight_indices:
            _, ranks = ACTION_TABLE[idx]
            mask = [1 if r in ranks else 0 for r in range(13)]
            straight_masks.append(mask)
        if straight_masks:
            self.straight_rank_masks = torch.tensor(
                straight_masks, device=self.device, dtype=torch.bool
            )
            self.straight_start = straight_indices[0]
            self.straight_end = straight_indices[-1] + 1
        else:
            self.straight_rank_masks = torch.zeros((0, 13), device=self.device, dtype=torch.bool)
            self.straight_start = self.straight_end = self.pair_end

        # Consecutive pairs
        consec_masks = []
        consec_indices = ACTION_TYPE_INDICES.get("CONSEC_PAIRS", [])
        for idx in consec_indices:
            _, ranks = ACTION_TABLE[idx]
            mask = [1 if r in ranks else 0 for r in range(13)]
            consec_masks.append(mask)
        if consec_masks:
            self.consec_pair_rank_masks = torch.tensor(
                consec_masks, device=self.device, dtype=torch.bool
            )
            self.consec_start = consec_indices[0]
            self.consec_end = consec_indices[-1] + 1
        else:
            self.consec_pair_rank_masks = torch.zeros((0, 13), device=self.device, dtype=torch.bool)
            self.consec_start = self.consec_end = self.straight_end

        # 3+2: [num, 3] with (main_rank, kicker1, kicker2)
        three_two = []
        three_two_indices = ACTION_TYPE_INDICES.get("THREE_PLUS_TWO", [])
        for idx in three_two_indices:
            _, (main_rank, kicker_ranks) = ACTION_TABLE[idx]
            three_two.append([main_rank, kicker_ranks[0], kicker_ranks[1]])
        if three_two:
            self.three_two_ranks = torch.tensor(three_two, device=self.device, dtype=torch.long)
            self.three_two_start = three_two_indices[0]
            self.three_two_end = three_two_indices[-1] + 1
        else:
            self.three_two_ranks = torch.zeros((0, 3), device=self.device, dtype=torch.long)
            self.three_two_start = self.three_two_end = self.consec_end

        # 4+3: [num, 4] with (main_rank, kicker1, kicker2, kicker3)
        four_three = []
        four_three_indices = ACTION_TYPE_INDICES.get("FOUR_PLUS_THREE", [])
        for idx in four_three_indices:
            _, (main_rank, kicker_ranks) = ACTION_TABLE[idx]
            four_three.append([main_rank, kicker_ranks[0], kicker_ranks[1], kicker_ranks[2]])
        if four_three:
            self.four_three_ranks = torch.tensor(four_three, device=self.device, dtype=torch.long)
            self.four_three_start = four_three_indices[0]
            self.four_three_end = four_three_indices[-1] + 1
        else:
            self.four_three_ranks = torch.zeros((0, 4), device=self.device, dtype=torch.long)
            self.four_three_start = self.four_three_end = self.three_two_end

        self.num_actions = len(ACTION_TABLE)

        # Build card removal tensors for step() vectorization
        self._build_card_removal_tensors()

    def _build_card_removal_tensors(self):
        """Build tensors that map actions to card removals for fully vectorized step().

        For each action, we pre-compute which cards and how many to remove.
        This enables the step() function to be fully GPU-vectorized.
        """
        # For straights: precompute card indices to remove
        # straight_cards[i] = [13] tensor where value is suit*13+rank if removing, -1 otherwise
        # Actually, we need to know which ranks, and pick one card per rank at runtime
        # But we can precompute the ranks needed for each straight

        # For 3+2 and 4+3, the ranks are already stored in three_two_ranks and four_three_ranks

        # Precompute number of cards to remove for each action type
        self.action_card_counts = torch.zeros(
            self.num_actions, dtype=torch.long, device=self.device
        )
        self.action_card_counts[0] = 0  # PASS
        self.action_card_counts[self.single_start : self.single_end] = 1  # Singles
        self.action_card_counts[self.pair_start : self.pair_end] = 2  # Pairs

        # Straights: length varies
        if self.straight_rank_masks.shape[0] > 0:
            self.action_card_counts[self.straight_start : self.straight_end] = (
                self.straight_rank_masks.sum(dim=1)
            )

        # Consec pairs: 2 * num_pairs
        if self.consec_pair_rank_masks.shape[0] > 0:
            self.action_card_counts[self.consec_start : self.consec_end] = (
                self.consec_pair_rank_masks.sum(dim=1) * 2
            )

        # 3+2: 5 cards
        if self.three_two_ranks.shape[0] > 0:
            self.action_card_counts[self.three_two_start : self.three_two_end] = 5

        # 4+3: 7 cards
        if self.four_three_ranks.shape[0] > 0:
            self.action_card_counts[self.four_three_start : self.four_three_end] = 7

    def _build_action_metadata(self) -> None:
        """Build action type, rank, length, and requirement metadata."""
        num_actions = len(ACTION_TABLE)
        action_types = torch.zeros(num_actions, dtype=torch.long, device=self.device)
        action_ranks = torch.full((num_actions,), -1, dtype=torch.long, device=self.device)
        action_lengths = torch.zeros(num_actions, dtype=torch.long, device=self.device)
        action_is_exemption = torch.zeros(num_actions, dtype=torch.bool, device=self.device)
        action_standard_types = torch.zeros(num_actions, dtype=torch.long, device=self.device)
        required_counts = torch.zeros((num_actions, 13), dtype=torch.long, device=self.device)

        for idx, (atype, data) in enumerate(ACTION_TABLE):
            if atype == "PASS":
                action_types[idx] = ACTION_TYPE_PASS
                action_standard_types[idx] = ACTION_TYPE_PASS
                continue

            if atype == "SINGLE":
                card_idx = next(iter(data))
                rank = card_idx % 13
                action_types[idx] = ACTION_TYPE_SINGLE
                action_ranks[idx] = rank
                action_lengths[idx] = 1
                action_standard_types[idx] = ACTION_TYPE_SINGLE
                required_counts[idx, rank] = 1
                continue

            if atype == "PAIR":
                card_idx = next(iter(data))
                rank = card_idx % 13
                action_types[idx] = ACTION_TYPE_PAIR
                action_ranks[idx] = rank
                action_lengths[idx] = 2
                action_standard_types[idx] = ACTION_TYPE_PAIR
                required_counts[idx, rank] = 2
                continue

            if atype == "STRAIGHT":
                ranks = list(data)
                action_types[idx] = ACTION_TYPE_STRAIGHT
                action_ranks[idx] = max(ranks)
                action_lengths[idx] = len(ranks)
                action_standard_types[idx] = ACTION_TYPE_STRAIGHT
                for r in ranks:
                    required_counts[idx, r] = 1
                continue

            if atype == "CONSEC_PAIRS":
                ranks = list(data)
                action_types[idx] = ACTION_TYPE_CONSEC_PAIRS
                action_ranks[idx] = max(ranks)
                action_lengths[idx] = len(ranks) * 2
                action_standard_types[idx] = ACTION_TYPE_CONSEC_PAIRS
                for r in ranks:
                    required_counts[idx, r] = 2
                continue

            if atype == "THREE_PLUS_TWO":
                main_rank, kicker_ranks = data
                action_types[idx] = ACTION_TYPE_THREE_TWO
                action_ranks[idx] = main_rank
                action_lengths[idx] = 5
                action_standard_types[idx] = ACTION_TYPE_THREE_TWO
                required_counts[idx, main_rank] = 3
                for r in kicker_ranks:
                    required_counts[idx, r] += 1
                continue

            if atype == "FOUR_PLUS_THREE":
                main_rank, kicker_ranks = data
                action_types[idx] = ACTION_TYPE_FOUR_THREE
                action_ranks[idx] = main_rank
                action_lengths[idx] = 7
                action_standard_types[idx] = ACTION_TYPE_FOUR_THREE
                required_counts[idx, main_rank] = 4
                for r in kicker_ranks:
                    required_counts[idx, r] += 1
                continue

            if atype == "THREE_TWO_EXEMPT":
                main_rank, kicker_ranks = data
                action_types[idx] = ACTION_TYPE_THREE_TWO_EXEMPT
                action_ranks[idx] = main_rank
                action_lengths[idx] = 3 + len(kicker_ranks)
                action_is_exemption[idx] = True
                action_standard_types[idx] = ACTION_TYPE_THREE_TWO
                required_counts[idx, main_rank] = 3
                for r in kicker_ranks:
                    required_counts[idx, r] += 1
                continue

            if atype == "FOUR_THREE_EXEMPT":
                main_rank, kicker_ranks = data
                action_types[idx] = ACTION_TYPE_FOUR_THREE_EXEMPT
                action_ranks[idx] = main_rank
                action_lengths[idx] = 4 + len(kicker_ranks)
                action_is_exemption[idx] = True
                action_standard_types[idx] = ACTION_TYPE_FOUR_THREE
                required_counts[idx, main_rank] = 4
                for r in kicker_ranks:
                    required_counts[idx, r] += 1
                continue

            raise ValueError(f"Unknown action type: {atype}")

        self.action_types = action_types
        self.action_ranks = action_ranks
        self.action_lengths = action_lengths
        self.action_is_exemption = action_is_exemption
        self.action_standard_types = action_standard_types
        self.action_required_counts = required_counts
        self.action_requires_rank_three = required_counts[:, 0] > 0

    def _build_following_tensors(self):
        """Build tensors for following constraint (must beat previous play)."""
        num_actions = self.num_actions

        # Initialize arrays
        action_types = torch.zeros(num_actions, dtype=torch.long, device=self.device)
        action_ranks = torch.full((num_actions,), -1, dtype=torch.long, device=self.device)
        action_lengths = torch.zeros(num_actions, dtype=torch.long, device=self.device)

        # PASS (action 0)
        action_types[0] = ACTION_TYPE_PASS

        # Singles (actions 1-52): rank = (action - 1) % 13
        single_actions = torch.arange(self.single_start, self.single_end, device=self.device)
        action_types[self.single_start : self.single_end] = ACTION_TYPE_SINGLE
        action_ranks[self.single_start : self.single_end] = (
            single_actions - self.single_start
        ) % 13
        action_lengths[self.single_start : self.single_end] = 1

        # Pairs (actions 53-130): rank = (action - 53) // 6
        pair_actions = torch.arange(self.pair_start, self.pair_end, device=self.device)
        action_types[self.pair_start : self.pair_end] = ACTION_TYPE_PAIR
        action_ranks[self.pair_start : self.pair_end] = (pair_actions - self.pair_start) // 6
        action_lengths[self.pair_start : self.pair_end] = 2

        # Straights: extract highest rank from each straight
        if self.straight_rank_masks.shape[0] > 0:
            action_types[self.straight_start : self.straight_end] = ACTION_TYPE_STRAIGHT
            # Highest rank in each straight (rightmost True in mask)
            for i, mask in enumerate(self.straight_rank_masks):
                ranks_in_straight = mask.nonzero().squeeze(-1)
                highest_rank = ranks_in_straight.max().item()
                action_ranks[self.straight_start + i] = highest_rank
                action_lengths[self.straight_start + i] = len(ranks_in_straight)

        # Consecutive pairs: extract highest rank
        if self.consec_pair_rank_masks.shape[0] > 0:
            action_types[self.consec_start : self.consec_end] = ACTION_TYPE_CONSEC_PAIRS
            for i, mask in enumerate(self.consec_pair_rank_masks):
                ranks_in_consec = mask.nonzero().squeeze(-1)
                highest_rank = ranks_in_consec.max().item()
                action_ranks[self.consec_start + i] = highest_rank
                action_lengths[self.consec_start + i] = len(ranks_in_consec)

        # 3+2: main rank is what matters
        if self.three_two_ranks.shape[0] > 0:
            action_types[self.three_two_start : self.three_two_end] = ACTION_TYPE_THREE_TWO
            action_ranks[self.three_two_start : self.three_two_end] = self.three_two_ranks[:, 0]
            action_lengths[self.three_two_start : self.three_two_end] = 5

        # 4+3: main rank is what matters
        if self.four_three_ranks.shape[0] > 0:
            action_types[self.four_three_start : self.four_three_end] = ACTION_TYPE_FOUR_THREE
            action_ranks[self.four_three_start : self.four_three_end] = self.four_three_ranks[:, 0]
            action_lengths[self.four_three_start : self.four_three_end] = 7

        self.action_types = action_types
        self.action_ranks = action_ranks
        self.action_lengths = action_lengths

    def apply_following_constraint(
        self,
        mask: torch.Tensor,  # [batch, num_actions] current valid mask
        prev_action_type: torch.Tensor,  # [batch] type of previous action
        prev_action_rank: torch.Tensor,  # [batch] rank of previous action
        prev_action_length: torch.Tensor,  # [batch] length of previous action (for straights)
        prev_action_is_exemption: torch.Tensor,  # [batch] whether previous was exemption
        prev_action_standard_type: torch.Tensor,  # [batch] standard type for previous
        is_leading: torch.Tensor,  # [batch] whether player is leading (no constraint)
    ) -> torch.Tensor:
        """Apply following constraint: must beat previous play.

        Rules:
        - If leading, no constraint
        - If previous was exemption: only standard 3+2 or 4+3 allowed
        - Otherwise: same type + higher rank (length match for straights/consec pairs)
        - Exemption moves can be used to beat standard moves when tail-hand
        """
        filtered = mask.clone()

        following = ~is_leading

        action_types_exp = self.action_types.unsqueeze(0)
        action_ranks_exp = self.action_ranks.unsqueeze(0)
        action_lengths_exp = self.action_lengths.unsqueeze(0)
        action_is_exempt_exp = self.action_is_exemption.unsqueeze(0)
        action_std_exp = self.action_standard_types.unsqueeze(0)

        prev_type_exp = prev_action_type.unsqueeze(1)
        prev_rank_exp = prev_action_rank.unsqueeze(1)
        prev_len_exp = prev_action_length.unsqueeze(1)
        prev_is_exempt_exp = prev_action_is_exemption.unsqueeze(1)
        prev_std_exp = prev_action_standard_type.unsqueeze(1)

        higher_rank = action_ranks_exp > prev_rank_exp

        same_length = action_lengths_exp == prev_len_exp
        is_straight_type = prev_type_exp == ACTION_TYPE_STRAIGHT
        is_consec_type = prev_type_exp == ACTION_TYPE_CONSEC_PAIRS
        needs_length_match = is_straight_type | is_consec_type

        # Case A: previous was exemption -> only standard full size allowed
        valid_after_exempt = (
            (~action_is_exempt_exp) & (action_std_exp == prev_std_exp) & higher_rank
        )

        # Case B: previous was standard
        valid_standard = (
            (~action_is_exempt_exp)
            & (action_types_exp == prev_type_exp)
            & higher_rank
            & (same_length | ~needs_length_match)
        )
        valid_exempt = action_is_exempt_exp & (action_std_exp == prev_type_exp) & higher_rank

        valid_follow = torch.where(
            prev_is_exempt_exp, valid_after_exempt, valid_standard | valid_exempt
        )

        valid_follow[:, 0] = mask[:, 0]
        following_exp = following.unsqueeze(1)
        filtered = torch.where(following_exp, filtered & valid_follow, filtered)

        return filtered

    def get_action_info(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get type, rank, and length for given actions.

        Args:
            action: [batch] action indices

        Returns:
            types: [batch] action types
            ranks: [batch] action ranks
            lengths: [batch] action lengths
        """
        return (
            self.action_types[action],
            self.action_ranks[action],
            self.action_lengths[action],
        )

    def compute_mask_batched(
        self,
        hand_cards: torch.Tensor,  # [batch, 52] bool - which cards player has
        rank_counts: torch.Tensor,  # [batch, 13] int - count per rank
        can_pass: torch.Tensor,  # [batch] bool - whether PASS is allowed
        cards_remaining: torch.Tensor,  # [batch] int - cards remaining for current player
        first_move: torch.Tensor,  # [batch] bool - first move of game
        has_heart_three: torch.Tensor,  # [batch] bool - current player has heart 3
    ) -> torch.Tensor:
        """Compute action masks for a batch of hands.

        Args:
            hand_cards: [batch, 52] boolean tensor of cards in hand
            rank_counts: [batch, 13] count of cards per rank
            can_pass: [batch] whether PASS is allowed
            cards_remaining: [batch] cards remaining for current player
            first_move: [batch] whether this is the first move
            has_heart_three: [batch] whether player holds heart 3

        Returns:
            [batch, num_actions] boolean action mask
        """
        batch_size = rank_counts.shape[0]
        required = self.action_required_counts  # [num_actions, 13]
        mask = (rank_counts.unsqueeze(1) >= required.unsqueeze(0)).all(dim=2)

        # PASS
        mask[:, 0] = can_pass

        # Tail-hand exemptions: only when all remaining cards are used
        length_ok = cards_remaining.unsqueeze(1) == self.action_lengths.unsqueeze(0)
        mask = torch.where(self.action_is_exemption.unsqueeze(0), mask & length_ok, mask)

        # First move must include Heart 3
        first_move_exp = first_move.unsqueeze(1)
        require_three = self.action_requires_rank_three.unsqueeze(0)
        has_h3 = has_heart_three.unsqueeze(1)
        not_pass = self.action_types.unsqueeze(0) != ACTION_TYPE_PASS
        mask = torch.where(first_move_exp, mask & require_three & has_h3 & not_pass, mask)

        return mask


def cards_to_tensor(cards: List[Card], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert list of Cards to GPU tensors.

    Returns:
        hand_cards: [52] boolean tensor
        rank_counts: [13] int tensor
    """
    hand_cards = torch.zeros(52, dtype=torch.bool, device=device)
    rank_counts = torch.zeros(13, dtype=torch.int32, device=device)

    for card in cards:
        idx = card_to_idx(card)
        hand_cards[idx] = True
        rank_counts[card.rank.value] += 1

    return hand_cards, rank_counts


# ============================================================================
# Benchmark
# ============================================================================


def benchmark():
    """Compare CPU vs GPU action mask computation."""
    import time
    from rl_poker.moves.legal_moves import get_legal_moves, MoveContext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total fixed actions: {NUM_ACTIONS}")

    # Create test hands
    test_cards = [
        Card(rank=Rank(r % 13), suit=Suit(r // 13))
        for r in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 13 cards
    ]

    # CPU benchmark
    print("\n--- CPU (get_legal_moves) ---")
    moves = get_legal_moves(test_cards, MoveContext())
    t0 = time.time()
    for _ in range(1000):
        moves = get_legal_moves(test_cards, MoveContext())
    t1 = time.time()
    print(f"1000 calls: {(t1 - t0) * 1000:.1f}ms ({1000 / (t1 - t0):.0f}/sec)")
    print(f"Legal moves found: {len(moves)}")

    # GPU benchmark
    print("\n--- GPU (batched mask computation) ---")
    computer = GPUActionMaskComputer(device)

    # Single hand
    hand_cards, rank_counts = cards_to_tensor(test_cards, device)
    hand_cards = hand_cards.unsqueeze(0)
    rank_counts = rank_counts.unsqueeze(0)
    can_pass = torch.tensor([False], device=device)
    cards_remaining = torch.tensor([len(test_cards)], device=device)
    first_move = torch.tensor([False], device=device)
    has_heart_three = hand_cards[:, 0]

    mask = computer.compute_mask_batched(
        hand_cards,
        rank_counts,
        can_pass,
        cards_remaining,
        first_move,
        has_heart_three,
    )

    # Warmup
    for _ in range(10):
        mask = computer.compute_mask_batched(
            hand_cards,
            rank_counts,
            can_pass,
            cards_remaining,
            first_move,
            has_heart_three,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None

    t0 = time.time()
    for _ in range(1000):
        mask = computer.compute_mask_batched(
            hand_cards,
            rank_counts,
            can_pass,
            cards_remaining,
            first_move,
            has_heart_three,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()
    print(f"1000 calls (batch=1): {(t1 - t0) * 1000:.1f}ms ({1000 / (t1 - t0):.0f}/sec)")
    print(f"Valid actions: {mask.sum().item()}")

    # Batched (simulate 64 environments)
    batch_size = 64
    hand_cards_batch = hand_cards.expand(batch_size, -1).contiguous()
    rank_counts_batch = rank_counts.expand(batch_size, -1).contiguous()
    can_pass_batch = can_pass.expand(batch_size).contiguous()
    cards_remaining_batch = cards_remaining.expand(batch_size).contiguous()
    first_move_batch = first_move.expand(batch_size).contiguous()
    has_heart_three_batch = has_heart_three.expand(batch_size).contiguous()

    # Warmup
    for _ in range(10):
        mask = computer.compute_mask_batched(
            hand_cards_batch,
            rank_counts_batch,
            can_pass_batch,
            cards_remaining_batch,
            first_move_batch,
            has_heart_three_batch,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None

    t0 = time.time()
    for _ in range(1000):
        mask = computer.compute_mask_batched(
            hand_cards_batch,
            rank_counts_batch,
            can_pass_batch,
            cards_remaining_batch,
            first_move_batch,
            has_heart_three_batch,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()
    print(
        f"1000 calls (batch={batch_size}): {(t1 - t0) * 1000:.1f}ms ({batch_size * 1000 / (t1 - t0):.0f} masks/sec)"
    )


if __name__ == "__main__":
    benchmark()

"""Opponent policies and PSRO-lite pool for GPU training."""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np
import torch

from rl_poker.moves.gpu_action_mask import GPUActionMaskComputer
from rl_poker.rl.policy import PolicyNetwork


class OpponentPolicy(Protocol):
    """Protocol for GPU-compatible opponent policies."""

    def select_actions(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select an action for each env in the batch."""
        ...


class PolicyNetworkOpponent:
    """Frozen policy wrapper for a PolicyNetwork snapshot."""

    def __init__(self, network: PolicyNetwork):
        self.network = network
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

    def select_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            actions, _, _ = self.network.get_action(obs, action_mask)
        return actions


class RecurrentPolicyOpponent:
    """Frozen policy wrapper for a RecurrentPolicyNetwork snapshot."""

    def __init__(self, network):
        self.network = network
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False

    def select_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        if seq is None:
            raise ValueError("Sequence input required for recurrent opponent policy")
        with torch.no_grad():
            actions, _, _ = self.network.get_action(obs, action_mask, seq)
        return actions


class GPURandomPolicy:
    """Uniform random policy over legal actions (GPU)."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = action_mask.float()
        # Fallback if mask invalid (should not happen)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        probs = mask / denom
        # Use CPU RNG for reproducibility, then move to device
        probs_cpu = probs.cpu().numpy()
        actions = [self.rng.choice(probs_cpu.shape[1], p=p) for p in probs_cpu]
        return torch.tensor(actions, device=action_mask.device, dtype=torch.long)


class GPUHeuristicPolicy:
    """Deterministic heuristic policy using GPU action metadata.

    Styles:
        - "conservative": prefer shorter, lower-rank plays
        - "aggressive": prefer longer, higher-rank plays
    """

    def __init__(
        self,
        mask_computer: GPUActionMaskComputer,
        style: str = "conservative",
        pass_penalty: float = 1000.0,
        exemption_penalty: float = 1.0,
    ):
        if style not in {"conservative", "aggressive"}:
            raise ValueError(f"Unknown style: {style}")

        self.mask_computer = mask_computer
        self.style = style
        self.pass_penalty = pass_penalty
        self.exemption_penalty = exemption_penalty
        self._base_scores = self._build_base_scores()

    def _build_base_scores(self) -> torch.Tensor:
        lengths = self.mask_computer.action_lengths.float()
        ranks = self.mask_computer.action_ranks.float().clamp(min=0)
        exemptions = self.mask_computer.action_is_exemption.float()

        base = lengths * 10.0 + ranks + exemptions * self.exemption_penalty
        if self.style == "aggressive":
            base = -base

        # PASS score handled per-batch
        base[self.mask_computer.pass_idx] = 0.0
        return base

    def select_actions(
        self, obs: torch.Tensor, action_mask: torch.Tensor, seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Expand base scores to batch
        scores = self._base_scores.unsqueeze(0).expand(action_mask.shape[0], -1).clone()

        # Mask illegal actions
        scores = scores.masked_fill(~action_mask, float("inf"))

        pass_idx = self.mask_computer.pass_idx
        has_play = (action_mask.sum(dim=1) - action_mask[:, pass_idx].long()) > 0
        if has_play.any():
            scores[has_play, pass_idx] = scores[has_play, pass_idx] + self.pass_penalty

        actions = torch.argmin(scores, dim=1)
        return actions


@dataclass
class OpponentStats:
    """EMA stats for PSRO-lite sampling."""

    ev_ema: float = 0.0
    games: int = 0

    def update(self, ev: float, ema_beta: float) -> None:
        if self.games == 0:
            self.ev_ema = ev
        else:
            self.ev_ema = (1.0 - ema_beta) * self.ev_ema + ema_beta * ev
        self.games += 1


@dataclass
class OpponentEntry:
    name: str
    policy: OpponentPolicy
    stats: OpponentStats = field(default_factory=OpponentStats)
    protected: bool = False
    added_step: int = 0


class OpponentPool:
    """Opponent pool with PSRO-lite sampling."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_size: int,
        device: torch.device,
        max_size: int = 16,
        ema_beta: float = 0.05,
        psro_beta: float = 3.0,
        min_prob: float = 0.05,
        seed: Optional[int] = None,
        recurrent: bool = False,
        history_feature_dim: int = 0,
        gru_hidden: int = 128,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.device = device
        self.max_size = max_size
        self.ema_beta = ema_beta
        self.psro_beta = psro_beta
        self.min_prob = min_prob
        self.rng = np.random.default_rng(seed)
        self.entries: List[OpponentEntry] = []
        self.recurrent = recurrent
        self.history_feature_dim = history_feature_dim
        self.gru_hidden = gru_hidden

    def add_fixed(self, name: str, policy: OpponentPolicy, protected: bool = True) -> None:
        self.entries.append(OpponentEntry(name=name, policy=policy, protected=protected))

    def add_snapshot(self, name: str, state_dict: dict, added_step: int = 0) -> None:
        if self.recurrent:
            from rl_poker.rl.recurrent import RecurrentPolicyNetwork

            if self.history_feature_dim <= 0:
                raise ValueError("history_feature_dim must be set for recurrent pool snapshots")
            network = RecurrentPolicyNetwork(
                self.obs_dim,
                self.num_actions,
                self.history_feature_dim,
                hidden_size=self.hidden_size,
                gru_hidden=self.gru_hidden,
            ).to(self.device)
            network.load_state_dict(state_dict)
            policy = RecurrentPolicyOpponent(network)
        else:
            network = PolicyNetwork(self.obs_dim, self.num_actions, self.hidden_size).to(self.device)
            network.load_state_dict(state_dict)
            policy = PolicyNetworkOpponent(network)

        self.entries.append(
            OpponentEntry(
                name=name,
                policy=policy,
                protected=False,
                added_step=added_step,
            )
        )
        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        if len(self.entries) <= self.max_size:
            return

        protected = [e for e in self.entries if e.protected]
        others = [e for e in self.entries if not e.protected]
        # Keep the hardest opponents (lowest ev_ema) when pruning
        others.sort(key=lambda e: e.stats.ev_ema)
        keep = protected + others[: max(0, self.max_size - len(protected))]
        self.entries = keep

    def compute_sampling_weights(self) -> np.ndarray:
        n = len(self.entries)
        if n == 0:
            raise ValueError("OpponentPool is empty")

        ev = np.array([e.stats.ev_ema for e in self.entries], dtype=np.float32)
        pressure = np.clip(-ev, 0.0, None)

        if pressure.sum() <= 0:
            weights = np.ones(n, dtype=np.float32) / n
        else:
            logits = self.psro_beta * pressure
            logits -= logits.max()
            weights = np.exp(logits).astype(np.float32)
            weights /= weights.sum()

        min_prob = min(self.min_prob, 1.0 / n)
        weights = weights * (1.0 - min_prob * n) + min_prob
        weights /= weights.sum()
        return weights

    def sample_indices(self, num_samples: int) -> np.ndarray:
        weights = self.compute_sampling_weights()
        return self.rng.choice(len(self.entries), size=num_samples, p=weights)

    def sample_opponents(self, num_envs: int, seats: int = 3) -> np.ndarray:
        """Sample opponent ids for each env and seat.

        Returns array of shape [num_envs, seats].
        """
        total = num_envs * seats
        ids = self.sample_indices(total)
        return ids.reshape(num_envs, seats)

    def update_stats(
        self,
        opponent_ids: np.ndarray,
        learner_scores: Sequence[float],
        opponent_scores: np.ndarray,
    ) -> None:
        """Update EV EMA for opponents.

        Args:
            opponent_ids: [N, seats] ids for opponents per finished env
            learner_scores: [N] learner final scores
            opponent_scores: [N, seats] opponent final scores
        """
        if opponent_ids.size == 0:
            return

        for env_idx in range(opponent_ids.shape[0]):
            learner_score = float(learner_scores[env_idx])
            for seat_idx in range(opponent_ids.shape[1]):
                opp_id = int(opponent_ids[env_idx, seat_idx])
                opp_score = float(opponent_scores[env_idx, seat_idx])
                ev = learner_score - opp_score
                self.entries[opp_id].stats.update(ev, self.ema_beta)

    def describe(self) -> str:
        lines = ["OpponentPool:"]
        for idx, entry in enumerate(self.entries):
            lines.append(
                f"  [{idx}] {entry.name}: ev_ema={entry.stats.ev_ema:.3f}, "
                f"games={entry.stats.games}, protected={entry.protected}"
            )
        return "\n".join(lines)

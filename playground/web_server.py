#!/usr/bin/env python3
import os
import time
import random
import threading
import uuid
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections import deque
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from rl_poker.rl import (
    GPUPokerEnv,
    RecurrentPolicyNetwork,
    PolicyNetwork,
    HistoryConfig,
    HistoryBuffer,
    GPUHeuristicPolicy,
    GPURandomPolicy,
    build_response_rank_weights
)
from rl_poker.moves.gpu_action_mask import ACTION_TABLE, GPUActionMaskComputer
from rl_poker.rules.ranks import Card, Rank, Suit
from rl_poker.scripts.train import TrainConfig  # Needed for serialization

# ==============================================================================
# Helpers (Copied/Adapted from play_human_vs_ai.py)
# ==============================================================================

def load_checkpoint_safe(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        from torch.serialization import safe_globals
        with safe_globals([TrainConfig]):
            return torch.load(path, map_location=device, weights_only=False)

def build_pair_lookup(mask_computer):
    lookup = {}
    pair_cards = mask_computer.pair_card_idx
    for i in range(pair_cards.shape[0]):
        card1 = int(pair_cards[i, 0].item())
        card2 = int(pair_cards[i, 1].item())
        key = (min(card1, card2), max(card1, card2))
        lookup[key] = mask_computer.pair_start + i
    return lookup

RANK_STRS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUIT_STRS = ["♥", "♦", "♣", "♠"]

def _card_sort_key(idx: int):
    if idx >= 52:
        return (99, idx)
    return (idx % 13, idx // 13)

def sort_card_indices(card_indices: List[int], reverse: bool = False) -> List[int]:
    return sorted(card_indices, key=_card_sort_key, reverse=reverse)

def describe_action_simple(action_idx: int) -> str:
    if action_idx == 0:
        return "不出"
    atype, data = ACTION_TABLE[action_idx]
    if atype == "SINGLE":
        return "单张"
    if atype == "PAIR":
        return "对子"
    if atype == "STRAIGHT":
        return "顺子"
    if atype == "CONSEC_PAIRS":
        return "连对"
    if atype == "THREE_PLUS_TWO":
        return "三带二"
    if atype == "FOUR_PLUS_THREE":
        return "四带三"
    if atype == "THREE_TWO_EXEMPT":
        return "三带二(豁免)"
    if atype == "FOUR_THREE_EXEMPT":
        return "四带三(豁免)"
    return str(atype)


# ==============================================================================
# Game Session Class
# ==============================================================================

class GameSession:
    def __init__(self, checkpoint_path=None, mode="policy"):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Avoid global default tensor type to prevent side effects.
        
        print(f"Initializing GameSession on {self.device}...")
        self.lock = threading.Lock()
        self.last_access = time.time()
        
        # 1. Setup Env/Config
        self.mode = mode
        
        if checkpoint_path:
             self.ckpt_path = Path(checkpoint_path)
        else:
            # Auto-find best
            ckpt_dir = Path("checkpoints/star")
            if not ckpt_dir.exists():
                ckpt_dir = Path("checkpoints")
            ckpts = sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            self.ckpt_path = ckpts[0] if ckpts else None

        if self.ckpt_path and self.mode == "policy":
            print(f"Loading checkpoint: {self.ckpt_path}")
            self.checkpoint = load_checkpoint_safe(self.ckpt_path, self.device)
        else:
            self.mode = "random" # Fallback
            print("Using Random AI (No checkpoint or explicit random mode)")

        # Config extraction
        self.cfg = {
            "use_recurrent": True,
            "hidden_size": 256,
            "gru_hidden": 128,
            "reveal_opponent_ranks": False,
            "history_window": 32,
            "belief_use_behavior": True,
            "belief_decay": 0.98,
            "belief_play_bonus": 0.5,
            "belief_pass_penalty": 0.3,
            "belief_temp": 2.0
        }
        if self.mode == "policy":
            loaded_cfg = self.checkpoint.get("config", None)
            if loaded_cfg:
                for k in self.cfg:
                    if hasattr(loaded_cfg, k):
                        val = getattr(loaded_cfg, k)
                        if val is not None:
                            self.cfg[k] = val

        self.env = GPUPokerEnv(1, self.device, reveal_opponent_ranks=bool(self.cfg["reveal_opponent_ranks"]))
        self.num_players = self.env.NUM_PLAYERS
        self.cards_per_player = self.env.CARDS_PER_PLAYER
        
        # 2. Init Policies
        self.policies = {}
        # Human is 0. AI are 1, 2, 3.
        self.human_seat = 0
        self.ai_seats = [1, 2, 3]

        # Player Names (Zen Style)
        zen_names = [
            "云隐", "墨客", "松涛", "竹韵", "兰心", "梅骨", "菊傲", 
            "清风", "明月", "听雨", "观海", "卧龙", "凤雏", "逍遥", 
            "无为", "素心", "淡然", "归隐", "流云", "止水"
        ]
        self.player_names = random.sample(zen_names, 4)
        # Ensure Human (0) always gets a specific one? Or just random. Random is fine.
        # Maybe give Human a special one? No, random is fair.
        
        # Dimensions
        num_actions = self.env.num_actions
        obs_dim = self.env.obs_dim
        # Aug dim calculation
        belief_dim = (self.num_players - 1) * self.cards_per_player
        oth_norm_dim = self.num_players - 1
        played_norm_dim = self.cards_per_player
        aug_dim = obs_dim + belief_dim + oth_norm_dim + played_norm_dim
        
        num_act_types = int(self.env.mask_computer.action_types.max().item()) + 1
        scalar_dim = 4
        hist_dim = self.num_players + num_act_types + scalar_dim
        
        if self.mode == "policy":
            state_dict = self.checkpoint["network"]
            for s in self.ai_seats:
                if self.cfg["use_recurrent"]:
                    net = RecurrentPolicyNetwork(aug_dim, num_actions, hist_dim, 
                                                 hidden_size=self.cfg["hidden_size"], 
                                                 gru_hidden=self.cfg["gru_hidden"])
                else:
                    net = PolicyNetwork(aug_dim, num_actions, self.cfg["hidden_size"])
                net.to(self.device)
                net.load_state_dict(state_dict)
                net.eval()
                self.policies[s] = net
        else:
            for s in self.ai_seats:
                self.policies[s] = GPURandomPolicy() # Or heuristic

        # 3. Aux Structures
        history_cfg = HistoryConfig(enabled=self.cfg["use_recurrent"], window=self.cfg["history_window"], feature_dim=hist_dim)
        self.history_buffer = HistoryBuffer(1, history_cfg, self.device)
        self.pair_lookup = build_pair_lookup(self.env.mask_computer)
        
        # Belief trackers
        self.public_played_counts = torch.zeros(1, self.cards_per_player, device=self.device)
        self.opp_rank_logits = torch.zeros(1, self.num_players, self.cards_per_player, device=self.device)
        
        # Precompute affinity
        rank_pos = torch.arange(self.cards_per_player, device=self.device).float()
        rank_dist = torch.abs(rank_pos.unsqueeze(0) - rank_pos.unsqueeze(1))
        temp = max(self.cfg["belief_temp"], 1e-6)
        rank_affinity = torch.exp(-rank_dist / temp)
        self.rank_affinity = rank_affinity / rank_affinity.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.response_rank_weights = build_response_rank_weights(self.env.mask_computer)
        
        # Helpers for feature extraction
        self.act_rank_norm = (self.env.mask_computer.action_ranks.clamp(min=0).float() / 12.0).unsqueeze(1)
        max_act_len = int(self.env.mask_computer.action_lengths.max().item())
        self.act_len_norm = (self.env.mask_computer.action_lengths.float() / max_act_len).unsqueeze(1)
        self.act_exempt = self.env.mask_computer.action_is_exemption.float().unsqueeze(1)
        self.act_is_pass = (torch.arange(num_actions, device=self.device) == 0).float().unsqueeze(1)
        num_act_types = int(self.env.mask_computer.action_types.max().item()) + 1
        self.act_type_oh = F.one_hot(self.env.mask_computer.action_types, num_classes=num_act_types).float()
        self.other_offsets = torch.arange(1, self.num_players, device=self.device)

        self.reset_game(started=False)

    def touch(self) -> None:
        self.last_access = time.time()

    def reset_game(self, started: bool = False):
        self.state = self.env.reset()
        self.history_buffer.reset_envs(torch.tensor([0], device=self.device))
        self.public_played_counts.zero_()
        self.opp_rank_logits.zero_()
        self.history_log = deque(maxlen=None) # No limit for full history
        self.history_log.append("对局开始")
        
        # Game State Vars
        self.last_played_indices = []
        self.game_over = False
        self.game_started = bool(started)
        if not self.game_started:
            self.history_log.append("等待开局")

    def start_game(self) -> bool:
        if self.game_over:
            return False
        if self.game_started:
            return False
        self.game_started = True
        self.history_log.append("开局")
        return True
        
    def get_frontend_state(self):
        s = self.state
        curr = int(s.current_player.item())
        
        # Hands
        my_hand_mask = s.hands[0, self.human_seat]
        my_hand_tensor = my_hand_mask.nonzero(as_tuple=False).squeeze(-1)
        # 处理空手牌或单张情况
        if my_hand_tensor.dim() == 0:
            my_hand = [int(my_hand_tensor.item())] if my_hand_tensor.numel() > 0 else []
        else:
            my_hand = my_hand_tensor.tolist()
        
        # Sort cards by engine order: rank ascending (3..2), suit ascending (♥..♠).
        my_hand = sort_card_indices(my_hand)
        
        # Counts
        counts = {}
        for i in range(4):
            counts[i] = int(s.cards_remaining[0, i].item())
            
        # Table
        prev_act = int(s.prev_action.item())
        is_new_lead = (s.prev_action_rank.item() < 0)
        
        table_cards = []
        if not is_new_lead and prev_act != 0:
            # Need to decode action to cards? 
            # The env doesn't store the exact cards played in state, just the action ID.
            # We can reconstruct representative cards from action ID for visualization,
            # OR we can cache what was actually played if we tracked it.
            # ACTION_TABLE[prev_act] gives (type, data).
            # If data is concrete card list (Single, Pair), easy.
            # If data is abstract (Straight rank 5), we don't know suits.
            # HOWEVER, for the frontend, showing representative suits is OK, or we can improve tracking later.
            try:
                atype, data = ACTION_TABLE[prev_act]
                if atype in ("SINGLE", "PAIR"):
                    table_cards = list(data)
                elif atype == "STRAIGHT":
                     # data is tuple of ranks. Pick suits based on... random? or just default to Hearts/Spades?
                     # Let's just pick first available of that rank? No, that's misleading.
                     # For now, let's decode using a heuristic or just show Empty if complex.
                     # To do it right: we need to track `last_played_cards` in our wrapper.
                     pass 
            except:
                pass
        
        # Use our tracked actual cards if available
        if not is_new_lead and prev_act != 0 and self.last_played_indices:
             table_cards = self.last_played_indices
        elif prev_act == 0 or is_new_lead:
             table_cards = []

        return {
            "current_player": curr,
            "game_over": self.game_over,
            "game_started": self.game_started,
            "cards_remaining": counts,
            "player_names": self.player_names,
            "my_hand": my_hand,
            "is_new_lead": is_new_lead,
            "table_cards": table_cards,  # List of card indices [0-51]
            "history": list(self.history_log),
            "consecutive_passes": int(s.consecutive_passes.item())
        }

    # Helper logic mostly from script
    def _action_from_cards(self, card_idxs, mask):
        # ... (Same logic as in script) ...
        # Minimal reimplementation
        if not card_idxs: return 0
        card_idxs = sorted(card_idxs)
        num = len(card_idxs)
        
        # 1. Single
        if num == 1:
            act = self.env.mask_computer.single_start + card_idxs[0]
            if mask[0, act]: return int(act)
        
        # 2. Pair
        if num == 2 and (card_idxs[0]%13 == card_idxs[1]%13):
            key = (card_idxs[0], card_idxs[1])
            if key in self.pair_lookup:
                act = self.pair_lookup[key]
                if mask[0, act]: return int(act)
                
        # 3. Search All
        valid_actions = mask[0].nonzero(as_tuple=False).squeeze(-1)
        counts = torch.zeros(13, dtype=torch.long, device=self.device)
        for c in card_idxs: counts[c%13] += 1
        
        candidates = []
        for act in valid_actions.tolist():
            if int(self.env.mask_computer.action_card_counts[act].item()) != num: continue
            if torch.equal(self.env.mask_computer.action_required_counts[act], counts):
                candidates.append(act)
        
        if candidates: return int(candidates[0]) # Heuristic: First match
        return None

    def step_action(self, action_idx: int, played_repr_cards: List[int] = None):
        action_tensor = torch.tensor([action_idx], device=self.device, dtype=torch.long)
        curr = int(self.state.current_player.item())
        prev_hand = None
        if action_idx != 0:
            prev_hand = self.state.hands[0, curr].clone()
        
        # Aux updates
        if action_idx != 0:
            req = self.env.mask_computer.action_required_counts[action_idx].float()
            self.public_played_counts[0] += req
            # Belief update
            if self.cfg["belief_use_behavior"]:
                self.opp_rank_logits[0, curr] *= self.cfg["belief_decay"]
                evidence = req @ self.rank_affinity
                self.opp_rank_logits[0, curr] += self.cfg["belief_play_bonus"] * evidence
        elif self.cfg["belief_use_behavior"] and not self.state.prev_action_rank.item() < 0:
             # Pass penalty
             prev_a = self.state.prev_action
             penalty = self.response_rank_weights[prev_a] @ self.rank_affinity
             self.opp_rank_logits[0, curr] -= self.cfg["belief_pass_penalty"] * penalty[0]
             
        # History
        if self.cfg["use_recurrent"]:
            # Feature extraction
            curr_tensor = torch.tensor([curr], device=self.device)
            p_oh = F.one_hot(curr_tensor, num_classes=self.num_players).float()
            t_oh = self.act_type_oh[action_tensor]
            scalars = torch.cat([self.act_rank_norm[action_tensor], self.act_len_norm[action_tensor], 
                                 self.act_exempt[action_tensor], self.act_is_pass[action_tensor]], dim=1)
            feats = torch.cat([p_oh, t_oh, scalars], dim=1)
            self.history_buffer.push(feats, torch.tensor([0], device=self.device))
            
        # Step
        self.state, rewards, dones = self.env.step(self.state, action_tensor)

        # Resolve actually removed cards for display (keeps UI consistent with env)
        if action_idx != 0 and prev_hand is not None:
            new_hand = self.state.hands[0, curr]
            removed_mask = prev_hand & ~new_hand
            removed_tensor = removed_mask.nonzero(as_tuple=False).squeeze(-1)
            if removed_tensor.dim() == 0:
                removed_cards = [int(removed_tensor.item())] if removed_tensor.numel() > 0 else []
            else:
                removed_cards = removed_tensor.tolist()
            if removed_cards:
                self.last_played_indices = sort_card_indices(removed_cards)
            elif played_repr_cards:
                self.last_played_indices = sort_card_indices(played_repr_cards)

        # Log
        if self.player_names and len(self.player_names) > curr:
            player_name = self.player_names[curr]
        else:
            player_name = "玩家" if curr == self.human_seat else f"电脑{curr}"
        desc = describe_action_simple(action_idx)
        
        if action_idx != 0:
             # Show detailed cards
             card_strs = []
             for c_idx in self.last_played_indices:
                 r = c_idx % 13
                 s = c_idx // 13
                 s_char = SUIT_STRS[s] if 0 <= s < 4 else "?"
                 r_str = RANK_STRS[r] if 0 <= r < 13 else "?"
                 card_strs.append(f"{s_char}{r_str}")
             
             played_str = " ".join(card_strs)
             self.history_log.append(f"{player_name}: {played_str} ({len(self.last_played_indices)}张)")
        else:
             self.history_log.append(f"{player_name}: {desc}")
        
        if dones.item():
            self.game_over = True
            ranks = self.state.finish_rank[0].tolist()
            winners = sorted(range(4), key=lambda x: ranks[x])
            name_order = []
            for seat in winners:
                if self.player_names and len(self.player_names) > seat:
                    name_order.append(self.player_names[seat])
                else:
                    name_order.append(f"P{seat}")
            self.history_log.append(f"游戏结束! 排名: {', '.join(name_order)}")


    def get_aug_obs(self):
        obs, mask = self.env.get_obs_and_mask(self.state)
        # Augment
        B = 1
        curr = self.state.current_player
        my_counts = self.state.rank_counts.gather(
            1, curr.view(B, 1, 1).expand(-1, 1, self.cards_per_player)
        ).squeeze(1)
        remaining = (4.0 - my_counts.float() - self.public_played_counts).clamp(min=0)
        oth_idx = (curr.view(B, 1) + self.other_offsets.view(1, -1)) % self.num_players
        oth_cards = self.state.cards_remaining.gather(1, oth_idx)
        logits = self.opp_rank_logits.gather(1, oth_idx.unsqueeze(-1).expand(-1, -1, self.cards_per_player))
        weights = torch.softmax(logits, dim=2)
        weighted = remaining.unsqueeze(1) * weights
        norm = weighted.sum(dim=2)
        safe_norm = norm.clamp(min=1e-6)
        scale = oth_cards.float() / safe_norm
        scale = torch.where(norm > 0, scale, torch.zeros_like(scale))
        belief = weighted * scale.unsqueeze(-1)
        belief_flat = belief.reshape(B, -1) / 4.0
        oth_norm = oth_cards.float() / float(self.cards_per_player)
        played_norm = self.public_played_counts / 4.0
        
        aug = torch.cat([obs, belief_flat, oth_norm, played_norm], dim=1)
        return aug, mask

    def play_human(self, card_indices):
        if not self.game_started:
            return False, "Game not started"
        if self.game_over:
            return False, "Game is already over"
        curr = int(self.state.current_player.item())
        if curr != self.human_seat:
            return False, "Not your turn"
        
        _, mask = self.env.get_obs_and_mask(self.state)
        
        if not card_indices:
            # Pass
            if not mask[0, 0]:
                return False, "Cannot pass: must play (new lead or forced)"
            self.step_action(0, [])
            return True, ""
            
        # Validate cards are unique, in range, and owned by human
        if len(card_indices) != len(set(card_indices)):
            return False, "Duplicate cards"
        if any((c < 0 or c >= 52) for c in card_indices):
            return False, "Card index out of range"
        hand_mask = self.state.hands[0, self.human_seat]
        hand_tensor = hand_mask.nonzero(as_tuple=False).squeeze(-1)
        if hand_tensor.dim() == 0:
            hand_cards = {int(hand_tensor.item())} if hand_tensor.numel() > 0 else set()
        else:
            hand_cards = set(hand_tensor.tolist())
        if any(c not in hand_cards for c in card_indices):
            return False, "Card not in hand"

        # Helper call
        act = self._action_from_cards(card_indices, mask)
        if act is None:
            return False, "Invalid move"
        
        self.step_action(act, card_indices)
        return True, ""

    def play_ai_step(self):
        if not self.game_started:
            return False
        if self.game_over: 
            return False
        curr = int(self.state.current_player.item())
        if curr == self.human_seat: return False # Human turn
        
        aug_obs, mask = self.get_aug_obs()
        policy = self.policies.get(curr)
        
        with torch.no_grad():
            if self.mode == "policy":
                if self.cfg["use_recurrent"]:
                    seq = self.history_buffer.get_sequence(torch.tensor([0], device=self.device))
                    act, _, _ = policy.get_action(aug_obs, mask, seq)
                else:
                    act, _, _ = policy.get_action(aug_obs, mask)
                action_idx = int(act.item())
            else:
                action_idx = int(policy.select_actions(aug_obs, mask).item())
        
        self.step_action(action_idx, [])
        return True

# ==============================================================================
# Web Server
# ==============================================================================

app = FastAPI()
# Robust path resolution relative to this script
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CHECKPOINT_ROOT = Path("checkpoints").resolve()

SESSION_TTL_SECONDS = int(os.getenv("RL_POKER_SESSION_TTL", "3600"))
SESSION_CLEANUP_INTERVAL = int(os.getenv("RL_POKER_SESSION_CLEANUP", "60"))
COOKIE_SECURE = os.getenv("RL_POKER_SECURE_COOKIE", "").lower() in ("1", "true", "yes")
API_TOKEN = os.getenv("RL_POKER_API_TOKEN")
_last_cleanup = 0.0

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

sessions: dict[str, GameSession] = {}
sessions_lock = threading.Lock()


def _list_checkpoints() -> Dict[str, Path]:
    if not CHECKPOINT_ROOT.exists():
        return {}
    paths = sorted(CHECKPOINT_ROOT.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {str(p.relative_to(CHECKPOINT_ROOT)): p for p in paths}


def _prune_sessions(now: float, keep_sid: Optional[str] = None) -> None:
    global _last_cleanup
    if SESSION_TTL_SECONDS <= 0:
        return
    if now - _last_cleanup < SESSION_CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    expired = []
    for sid, sess in sessions.items():
        if keep_sid and sid == keep_sid:
            continue
        last_access = getattr(sess, "last_access", now)
        if now - last_access > SESSION_TTL_SECONDS:
            expired.append(sid)
    for sid in expired:
        del sessions[sid]


def _get_session(request: Request) -> GameSession:
    sid = request.state.session_id
    with sessions_lock:
        return sessions[sid]


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    if API_TOKEN and request.url.path.startswith("/api"):
        token = request.headers.get("X-API-Token") or request.cookies.get("rl_poker_api_token")
        if token != API_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    sid = request.cookies.get("rl_poker_session")
    new = False
    now = time.time()
    with sessions_lock:
        if not sid or sid not in sessions:
            sid = uuid.uuid4().hex
            sessions[sid] = GameSession()
            new = True
        sessions[sid].touch()
        _prune_sessions(now, keep_sid=sid)
    request.state.session_id = sid
    response = await call_next(request)
    if new:
        max_age = SESSION_TTL_SECONDS if SESSION_TTL_SECONDS > 0 else None
        response.set_cookie(
            "rl_poker_session",
            sid,
            httponly=True,
            samesite="lax",
            secure=COOKIE_SECURE,
            max_age=max_age,
        )
    return response

@app.get("/")
def get_index(request: Request):
    _ = _get_session(request)
    response = FileResponse(WEB_DIR / "index.html")
    if API_TOKEN:
        token = request.query_params.get("token")
        if token == API_TOKEN:
            max_age = SESSION_TTL_SECONDS if SESSION_TTL_SECONDS > 0 else None
            response.set_cookie(
                "rl_poker_api_token",
                token,
                httponly=True,
                samesite="lax",
                secure=COOKIE_SECURE,
                max_age=max_age,
            )
    return response

@app.get("/style.css")
def get_style(request: Request):
    _ = _get_session(request)
    return FileResponse(WEB_DIR / "style.css")

@app.get("/script.js")
def get_script(request: Request):
    _ = _get_session(request)
    return FileResponse(WEB_DIR / "script.js")

@app.get("/favicon.svg")
def get_favicon(request: Request):
    return FileResponse(WEB_DIR / "favicon.svg")

@app.get("/favicon.ico")
def get_favicon_ico(request: Request):
    # Fallback to serving SVG, modern browsers often handle it
    return FileResponse(WEB_DIR / "favicon.svg")

@app.get("/api/state")
def api_state(request: Request):
    session = _get_session(request)
    with session.lock:
        return session.get_frontend_state()

class ActionRequest(BaseModel):
    action: str # "pass" or "play"
    cards: Optional[List[int]] = None

@app.post("/api/action")
def api_action(req: ActionRequest, request: Request):
    session = _get_session(request)
    with session.lock:
        if req.action == "pass":
            success, msg = session.play_human([])
            if not success:
                raise HTTPException(status_code=400, detail=msg or "Cannot pass")
        elif req.action == "play" and req.cards:
            success, msg = session.play_human(req.cards)
            if not success:
                raise HTTPException(status_code=400, detail=msg or "Invalid move")
        else:
            raise HTTPException(status_code=400, detail="Invalid action payload")
    
    # After human moves, we may want to auto-trigger AI if we want "Fast" play?
    # Or frontend polls. Frontend polling 'state' will see current_player changed.
    return {"status": "ok"}

@app.post("/api/reset")
def api_reset(request: Request):
    session = _get_session(request)
    with session.lock:
        session.reset_game(started=False)
    return {"status": "reset"}

@app.post("/api/start")
def api_start(request: Request):
    session = _get_session(request)
    with session.lock:
        ok = session.start_game()
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot start")
    return {"status": "started"}

@app.post("/api/deal")
def api_deal(request: Request):
    session = _get_session(request)
    with session.lock:
        if session.game_started:
            raise HTTPException(status_code=400, detail="Game already started")
        session.reset_game(started=False)
    return {"status": "dealt"}

@app.get("/api/ai_move")
def api_ai_move(request: Request):
    session = _get_session(request)
    # Advance one AI step
    with session.lock:
        did_move = session.play_ai_step()
    return {"did_move": did_move}

class ConfigRequest(BaseModel):
    checkpoint: Optional[str] = None
    mode: str

@app.get("/api/config")
def api_get_config(request: Request):
    session = _get_session(request)
    # Scan checkpoints
    ckpt_map = _list_checkpoints()
    ckpts = list(ckpt_map.keys())

    current_ckpt = None
    if hasattr(session, "ckpt_path") and session.ckpt_path:
        try:
            current_ckpt = str(Path(session.ckpt_path).resolve().relative_to(CHECKPOINT_ROOT))
        except Exception:
            current_ckpt = None
    
    return {
        "checkpoints": ckpts,
        "modes": ["policy", "random"],
        "current_checkpoint": current_ckpt,
        "current_mode": session.mode
    }

@app.post("/api/config")
def api_set_config(req: ConfigRequest, request: Request):
    sid = request.state.session_id
    if req.mode not in {"policy", "random"}:
        raise HTTPException(status_code=400, detail="Invalid mode")
    ckpt_map = _list_checkpoints()
    ckpt_path = None
    if req.mode == "policy":
        if req.checkpoint:
            if req.checkpoint not in ckpt_map:
                raise HTTPException(status_code=400, detail="Invalid checkpoint selection")
            ckpt_path = ckpt_map[req.checkpoint]
    print(f"Reconfiguring session: {req.mode}, {ckpt_path}")
    with sessions_lock:
        sessions[sid] = GameSession(checkpoint_path=str(ckpt_path) if ckpt_path else None, mode=req.mode)
    return {"status": "ok"}

if __name__ == "__main__":
    # Auto-play loop logic for AI?
    # Better: Frontend calls ai_move repeatedly.
    try:
        host = os.getenv("RL_POKER_HOST", "0.0.0.0")
        port = int(os.getenv("RL_POKER_PORT", "8000"))
        print(f"Starting server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        pass

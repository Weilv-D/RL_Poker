# Learnings

## Notepad Notes

(Initial notepad created - no learnings yet)

## Task 1: Project Scaffolding

### Package Structure
- Standard Python package layout with `rl_poker/` as main package
- Subpackages: `rules/`, `moves/`, `engine/`, `envs/`, `agents/`, `scripts/`, `utils/`
- Tests in separate `tests/` directory at root level

### Dependencies
- Using CPU-only torch for faster installation: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Full CUDA torch takes 900+ MB download; CPU version is ~190 MB
- Core deps: torch>=2.0.0, pettingzoo>=1.24.0, gymnasium>=0.29.0, numpy>=1.24.0

### Seeding Utility
- `set_seed(seed: Optional[int])` sets Python random, NumPy, and PyTorch seeds
- Returns the seed used (useful when None is passed to generate random seed)
- Also sets `torch.backends.cudnn.deterministic = True` and `benchmark = False` for full determinism

### pyproject.toml
- Using modern PEP 621 format with `[project]` section
- Dev dependencies separated in `[project.optional-dependencies]`
- Pytest configured in `[tool.pytest.ini_options]`

### Environment
- System Python 3.14 requires virtual environment (externally managed)
- Created `.venv/` for project-specific dependencies
- Activate with: `source .venv/bin/activate`

## Task 2: Rules Engine

### Rank Implementation
- `Rank` is an `IntEnum` where the integer value reflects rank strength (higher = stronger)
- Order: THREE(0) < FOUR(1) < ... < KING(10) < ACE(11) < TWO(12)
- This allows natural Python comparison operators (`<`, `>`) to work correctly

### Sequence Constraints
- `SEQUENCE_RANKS` is a `frozenset` containing only 3-K ranks
- `is_valid_sequence_rank(rank)` provides fast O(1) membership testing
- Critical rule: A and 2 are STRICTLY PROHIBITED in straights and consecutive pairs

### Hand Parsing Strategy
- `parse_hand()` returns `Optional[Hand]` - None for invalid hands, Hand for valid
- This design avoids exceptions for expected "invalid hand" cases
- Order of checks matters: 3+2 and 4+3 are checked before straight to avoid misclassification

### Comparison Logic
- `can_beat()` enforces same type AND same size for comparison
- 3+2 and 4+3: `main_rank` is the triple/quad, kickers are ignored for comparison
- Sequences: `main_rank` is the highest card in the sequence
- Four-of-a-kind is NOT a bomb - it can only be played as 4+3

### Testing Approach
- Tests organized by hand type in separate test classes
- Each hand type has: valid construction, invalid cases, comparison tests
- Special tests for sequence bounds (3-K only) and no A/2 enforcement
- Explicit test that four-of-a-kind cannot beat a single (no bomb mechanics)

## Task 4: Game Engine

### Game State Design
- `GameState` is the central class managing all game state
- `PlayerState` dataclass tracks individual player state (hand, passed, finished, rank)
- Game phases: DEALING → PLAYING → FINISHED
- Scoring constants: RANK_SCORES = {1: +2, 2: +1, 3: -1, 4: -2} (zero-sum)

### Turn Management
- Lead player identified by Heart 3 ownership at game start
- First move MUST contain Heart 3 (filtered in get_legal_moves)
- Turn order is clockwise: (current + 1) % 4
- When all active players except lead pass: new lead round starts
- When lead player finishes but others passed: next active player becomes new lead

### Key Methods
- `new_game(seed)`: Creates game with shuffled deck
- `from_hands(hands)`: Creates game from pre-defined hands (for testing)
- `get_legal_moves()`: Returns moves filtered by first-move constraint if applicable
- `apply_move(move)`: Validates, executes move, handles player exit and game end
- `_advance_turn()`: Complex logic for turn progression and new lead scenarios

### Edge Cases Handled
- Player finishes while holding lead → next active player gets new lead
- All active players passed → reset passes and start new lead round
- Third player finishes → fourth player auto-assigned rank 4, game ends

### Testing Approach
- Deterministic seeding for reproducible tests
- `play_random_game(seed)` helper for full game simulations
- Tests verify: initialization, lead player, first move, turn order, PASS, new lead, exit, ranking, scoring

## Task 6: Baseline Agents + Evaluation Hooks

### Agent Interface Design
- All agents follow a common interface: `act(observation, action_mask, rng) -> action_index`
- `observation` is a dict with game state info (legal_moves, hand, current_move, etc.)
- `action_mask` is a boolean numpy array indicating legal actions
- Agents should implement `reset()` for state cleanup between episodes

### Random Agent
- Simplest baseline: uniformly samples from legal actions
- Uses numpy random generator for reproducibility (`np.random.default_rng(seed)`)
- Completely ignores game state - useful as lower bound baseline

### Heuristic Agent
- Rule-based agent with domain knowledge:
  1. When leading: play lowest single that doesn't break pairs
  2. When following: find smallest move that beats current
  3. Prefer shedding cards when safe
  4. Avoid breaking pairs (preserve for later)
- Outperforms random by significant margin (verified in evaluation)

### Policy Pool Design
- Manages model checkpoints for evaluation and self-play
- Registry stored as JSON file (`registry.json`) in pool directory
- CheckpointInfo dataclass tracks: name, path, step, metadata, created_at
- Supports custom loader functions for different model architectures
- PooledPolicyAgent wraps pool for consistent agent interface

### Evaluation Script
- Entry point: `python -m rl_poker.scripts.evaluate`
- Key flags: `--episodes`, `--opponents` (comma-separated), `--seed`, `--verbose`
- Creates observation dict with: legal_moves, hand, current_move, player_card_counts
- Tracks: average score, win rate (top 2 finish), rank distribution
- Verified heuristic beats random consistently

## Task 5: PettingZoo AEC Environment

### Environment Design
- `RLPokerAECEnv` extends `pettingzoo.AECEnv` for multi-agent turn-based play
- 4 agents named `player_0` through `player_3`
- Action space: `Discrete(MAX_ACTIONS=500)` with action masking
- Observation space: Dict with hand, last_move, remaining_cards, action_mask

### Observation Encoding
- `hand`: 52-element one-hot vector (1 for each card present)
- `last_move`: 17-element vector encoding current move to beat
  - [0]: is valid move, [1]: move type, [2]: is exemption, [3]: num cards (normalized)
  - [4:17]: 13 rank presence indicators (any suit counts)
- `remaining_cards`: 4-element vector with card counts per player
- `action_mask`: 500-element boolean array aligned with MAX_ACTIONS

### Reward Structure
- Sparse rewards only at game end
- Rewards: +2 (1st), +1 (2nd), -1 (3rd), -2 (4th)
- Zero-sum: all rewards sum to 0

### Seeding & Determinism
- `reset(seed=N)` creates deterministic game state
- Same seed produces identical initial hands and starting player
- Tested: two envs with same seed + same actions → identical rewards

### Agent Iteration Pattern
- `agent_iter()` yields current agent until all agents done
- Key: must clear `self.agents = []` when game ends to stop iteration
- `_current_agent` tracks which agent's turn it is
- Agent selection synced with underlying `GameState.current_player`

### PettingZoo AEC Gotchas
- Do NOT use `agent_selector.agent_selection` - attribute doesn't exist
- `agent_iter()` stops when `self.agents` list is empty
- Must handle "dead step" when agent is terminated (action=None)
- `_was_dead_step()` pattern for terminated agent processing

### Testing
- `smoke_env.py` validates observation shapes, action masks, episode completion
- Run: `python -m rl_poker.scripts.smoke_env --episodes 5`
- Verbose mode shows each agent's action

## Task 8: Evaluation Metrics + Policy Pool Integration

### Elo Rating System
- Standard Elo formula adapted for 4-player games
- Uses pairwise comparison: each player compared against all others
- `calculate_expected_score(rating_a, rating_b)`: returns expected win probability (0-1)
- `update_elo_ratings(ratings, ranks, k_factor)`: updates all ratings after a game
- Default K-factor: 32, Default starting Elo: 1000
- Total Elo approximately conserved (zero-sum for equally rated players)

### EloTracker Class
- Maintains ratings and rating_history
- `update(ranks)`: update ratings after episode
- `get_rating(position)`: current rating for position
- `get_rating_change(position)`: total change from starting rating
- Serializable with `to_dict()` and `from_dict()`

### EvaluationStats Enhancements
- Now includes `elo_tracker` (auto-initialized)
- `get_elo_rating(position)`: retrieve current Elo
- Summary output now includes Elo ratings with change indicator

### Policy Pool Integration
- New opponent type: `policy_pool`
- Requires `--pool-dir` argument to specify checkpoint directory
- Falls back to random agent if pool is empty (with warning)
- Uses `PooledPolicyAgent` from `rl_poker.agents.policy_pool`
- Agent type validation updated to include `policy_pool`

### CLI Updates
- `--opponents` now accepts: random, heuristic, policy_pool
- `--pool-dir`: specify policy pool checkpoint directory
- Error handling: requires --pool-dir when using policy_pool

### Test Coverage
- `tests/test_metrics.py`: 29 tests covering:
  - Metric aggregation (single/multiple episodes, empty stats, edge cases)
  - Elo calculations (expected score, rating updates, K-factor effects)
  - EloTracker (initialization, history, serialization)
  - Integration (stats + Elo, summary output)
  - Edge cases (tied ranks, rating divergence over many episodes)

## Task 3: Legal Move Enumeration + Action Encoding

### Legal Moves Design
- `Move` is a frozen dataclass with `move_type`, `cards`, `hand`, `is_exemption`, and `standard_type`
- `PASS_MOVE` singleton for pass actions (empty frozenset of cards)
- `MoveContext` dataclass holds game state for legal move determination

### Move Enumeration Strategy
- `enumerate_all_hands()` generates all standard hands using combinations
- Combinations grouped by size: singles (1), pairs (2), straights (5-13), consecutive pairs (6+), 3+2 (5), 4+3 (7)
- Exemption hands enumerated separately in `enumerate_exemption_hands()`

### Tail-Hand Exemption Rules
- 3+2 exemptions: 3+1 (4 cards) and 3+0 (3 cards)
- 4+3 exemptions: 4+2 (6 cards), 4+1 (5 cards), 4+0 (4 cards)
- Exemptions only valid when `is_tail_hand=True`
- `is_exemption=True` flag marks exemption moves with `standard_type` indicating original hand type

### Strict Follow-Up Matching
- When `previous_used_exemption=True`, next player MUST play full standard size
- Example: If prev played 3+1 (4 cards), next must play 5-card 3+2
- Cannot counter exemption with another exemption
- If cannot match full size, forced to PASS

### Action Space Design
- `MAX_ACTIONS = 1000` (increased from initial 500 after testing)
- `ActionSpace` dataclass holds legal moves, action mask, and bidirectional mappings
- Action mask is np.ndarray of shape (MAX_ACTIONS,) with dtype=bool
- Overflow guard raises `ActionEncodingError` if legal moves exceed MAX_ACTIONS

### Testing Findings
- 13-card hand with multiple quads can generate 700+ legal moves
- MAX_ACTIONS = 1000 provides safe margin
- `LegalMoveStats` class tracks max observed for tuning

## Task 7: PPO Training Script

### TensorBoard Logging
- Added `--log-dir` argument to `rl_poker/scripts/train.py`
- Uses `torch.utils.tensorboard.writer.SummaryWriter` when available
- Logs learning rate, policy loss, value loss, and mean reward

### GPU Device Handling
- `_process_obs` now places tensors on model device to avoid CPU/GPU mismatch

### Training Script Notes
- `num_updates` uses `ceil(total_timesteps / num_steps)` to avoid zero updates
- Training verified on GPU with `--cuda True`

### Logging Note
- Episode reward logging is now tracked for a single agent (`--stats-agent`, default `player_0`) to avoid zero-sum averaging across all agents

### Shell Runner
- `scripts/train_gpu.sh` starts TensorBoard (if available) and runs GPU training
- Supports flags for steps, rollout length, LR, seed, and log/checkpoint dirs

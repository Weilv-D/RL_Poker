# RL Poker RL Project Plan

## TL;DR

> **Quick Summary**: Build a clean, from-scratch simulator and PettingZoo AEC environment for the custom 4-player rules, then train a shared-policy self-play PPO agent and evaluate it against random/heuristic/opponent-pool baselines.
> 
> **Deliverables**:
> - `rl_poker/` package with rules engine, legal move generator, game engine, and AEC environment
> - Baseline agents (random + heuristic) and policy pool evaluation
> - Training and evaluation scripts using PyTorch + CleanRL-style PPO
> - Pytest-based tests for rules/move legality/env invariants (tests-after implementation)
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Rules engine → Legal move generator → Game engine → PettingZoo env → PPO training

---

## Context

### Original Request
Design a reinforcement learning project for the provided four-player, zero-sum, custom poker-like rules. Train a model and choose the most suitable technical path. Optimize average score per game with independent episodes.

### Interview Summary
**Key Discussions**:
- Objective: maximize average score per game, with each game as its own episode.
- Stack: Python + PyTorch + PettingZoo (AEC) + CleanRL.
- Training approach: self-play PPO with shared policy and action masking; maintain a policy pool for evaluation.
- Tests: YES (tests written after implementation).
- Baselines: random + heuristic + policy pool.

**Research Findings**:
- Repo is currently empty; no existing structure or patterns to follow.

### Metis Review
**Identified Gaps (addressed)**:
- No additional gaps returned by Metis.

---

## Work Objectives

### Core Objective
Implement a complete RL-ready environment and training pipeline for the custom 4-player rules, and demonstrate learning progress against defined baselines while maximizing average score per game.

### Concrete Deliverables
- `pyproject.toml` with dependencies (torch, pettingzoo, gymnasium, numpy, pytest)
- `rl_poker/rules/` for rank/hand logic and comparisons
- `rl_poker/moves/` for legal action enumeration + action encoding/masking
- `rl_poker/engine/` for turn flow, PASS, ranking, and scoring
- `rl_poker/envs/rl_poker_aec.py` PettingZoo AEC environment
- `rl_poker/agents/` with random + heuristic agents
- `rl_poker/scripts/train.py` and `rl_poker/scripts/evaluate.py`
- `tests/` for rules/moves/env invariants

### Definition of Done
- `python -m rl_poker.scripts.train --steps 1000 --self-play` runs without error and saves a checkpoint
- `python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic,policy_pool` prints average score and rank distribution
- `python -m pytest` passes all tests

### Must Have
- Exact rule compliance (card ranking, no bombs, 3-K sequences only, tail-hand exemption with strict matching)
- Action masking for illegal moves
- Deterministic seeding for reproducibility
- Self-play PPO with shared policy

### Must NOT Have (Guardrails)
- No UI/client or networked multiplayer
- No rule deviations (e.g., A/2 in sequences, bomb mechanics)
- No manual-only verification steps

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO
- **User wants tests**: YES (tests after implementation)
- **Framework**: pytest

### Tests-After Strategy
Each implementation task includes immediate post-implementation tests and automated verification commands. No manual verification steps.

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately):
├── Task 1: Project scaffolding + dependencies
└── Task 2: Rules engine (rank/hand/compare) + tests

Wave 2 (After Wave 1):
├── Task 3: Legal move enumeration + action encoding/mask + tests
└── Task 4: Game engine (turns, PASS, ranking) + tests

Wave 3 (After Wave 2):
├── Task 5: PettingZoo AEC env + smoke tests
├── Task 6: Baseline agents + evaluation hooks
└── Task 7: PPO self-play training script

Wave 4 (After Wave 3):
└── Task 8: Evaluation metrics + policy pool integration + tests

Critical Path: Task 2 → Task 3 → Task 4 → Task 5 → Task 7 → Task 8

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|----------------------|
| 1 | None | 2, 3, 4, 5, 6, 7, 8 | 2 |
| 2 | 1 | 3, 4, 5 | 1 |
| 3 | 2 | 4, 5 | 4 |
| 4 | 2, 3 | 5 | 3 |
| 5 | 3, 4 | 6, 7 | 6, 7 |
| 6 | 5 | 8 | 7 |
| 7 | 5 | 8 | 6 |
| 8 | 6, 7 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | delegate_task(category="unspecified-high", load_skills=[...], run_in_background=true) |
| 2 | 3, 4 | delegate_task(category="unspecified-high", load_skills=[...], run_in_background=true) |
| 3 | 5, 6, 7 | delegate_task(category="unspecified-high", load_skills=[...], run_in_background=true) |
| 4 | 8 | delegate_task(category="unspecified-high", load_skills=[...], run_in_background=true) |

---

## TODOs

- [x] 1. Project scaffolding + dependencies

  **What to do**:
  - Create package layout: `rl_poker/`, `rl_poker/rules/`, `rl_poker/moves/`, `rl_poker/engine/`, `rl_poker/envs/`, `rl_poker/agents/`, `rl_poker/scripts/`, `tests/`
  - Add `pyproject.toml` with dependencies: torch, pettingzoo, gymnasium, numpy, pytest
  - Add minimal `rl_poker/__init__.py` and version constant
  - Add deterministic seeding utility (numpy/torch/random)

  **Must NOT do**:
  - No UI or deployment artifacts

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: multi-file project initialization and configuration
  - **Skills**: (none)
    - No specialized UI/browser/git skill needed
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 2–8
  - **Blocked By**: None (can start immediately)

  **References**:
  - New file: `pyproject.toml` - define dependencies and project metadata
  - New path: `rl_poker/` - root Python package for all modules

  **Acceptance Criteria**:
  - [ ] `python -m pip install -e .` succeeds
  - [ ] `python -c "import rl_poker"` exits with code 0

  **Commit**: NO (repo not initialized)

- [x] 2. Rules engine (rank/hand/compare) + tests

  **What to do**:
  - Implement rank order: 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
  - Implement hand parsing for: single, pair, straight (>=5), consecutive pairs (>=3), 3+2, 4+3
  - Enforce sequence constraints: 3-K only, no A/2 in straights or consecutive pairs
  - Implement comparison logic (by main rank only for 3+2, 4+3; by highest rank for sequences)
  - Write pytest tests for valid/invalid hands and comparisons

  **Must NOT do**:
  - No bomb mechanics; four-of-a-kind is not special

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: core game rules and correctness-heavy logic
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3–5
  - **Blocked By**: Task 1

  **References**:
  - New file: `rl_poker/rules/ranks.py` - rank order and helpers
  - New file: `rl_poker/rules/hands.py` - hand classification and comparison
  - New file: `tests/test_rules.py` - unit tests for rank/hand logic
  - External: https://pettingzoo.farama.org/ (general env patterns; no direct rule logic)

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_rules.py` passes
  - [ ] Tests cover: sequence bounds (3-K only), no A/2 in sequences, main-rank comparison for 3+2/4+3

  **Commit**: NO (repo not initialized)

- [x] 3. Legal move enumeration + action encoding/mask + tests

  **What to do**:
  - Enumerate all legal moves from a given hand, including PASS
  - Implement tail-hand exemption for last move (3+1/3+0 and 4+2/4+1/4+0)
  - Implement strict follow-up matching (next player must match full standard size when previous used exemption)
  - Define action encoding as padded list of legal moves with fixed MAX_ACTIONS and action mask
  - Add overflow guard: assert legal moves <= MAX_ACTIONS; record max observed in tests
  - Write pytest tests for exemption + strict matching and action mask validity

  **Must NOT do**:
  - No truncation of legal actions without explicit error

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: complex legality logic and action-space design
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Tasks 4–5
  - **Blocked By**: Task 2

  **References**:
  - New file: `rl_poker/moves/legal_moves.py` - legal move generator and exemption rules
  - New file: `rl_poker/moves/action_encoding.py` - move list → fixed action space + mask
  - New file: `tests/test_moves.py` - tests for legality/exemption/mask
  - External: https://pettingzoo.farama.org/api/aec/ (AEC patterns for action masks)

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_moves.py` passes
  - [ ] Tests include: exemption example (333+4 → next must play 5-card 3+2), and forced PASS when unable
  - [ ] Action mask length equals MAX_ACTIONS; invalid moves are masked out

  **Commit**: NO (repo not initialized)

- [x] 4. Game engine (turns, PASS, ranking) + tests

  **What to do**:
  - Implement dealing, lead player (red heart 3) and mandatory first move containing that card
  - Implement turn order, PASS logic, and “new lead” when all others pass
  - Implement player exit, ranking, and stop condition when third player finishes
  - Apply scoring: +2, +1, -1, -2 with zero-sum per game
  - Write tests for a deterministic deck to verify rank order and score assignment

  **Must NOT do**:
  - Do not continue game beyond third finish

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: multi-agent turn flow and edge-case handling
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 2, 3

  **References**:
  - New file: `rl_poker/engine/game_state.py` - state, turn management, and scoring
  - New file: `tests/test_engine.py` - deterministic game flow tests

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_engine.py` passes
  - [ ] Deterministic test verifies finish order and scores (+2/+1/-1/-2)

  **Commit**: NO (repo not initialized)

- [x] 5. PettingZoo AEC environment + smoke tests

  **What to do**:
  - Implement AEC env wrapper over engine: `reset`, `step`, `observe`, `agent_iter`
  - Observation includes: hand encoding, last move encoding, remaining cards per player, action mask
  - Reward only at game end (per agent): +2/+1/-1/-2
  - Add seeding support and deterministic reset for tests
  - Add a smoke test script to run N episodes without error

  **Must NOT do**:
  - No environment-level shaping rewards (keep sparse end reward)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: multi-agent env interface and action masking
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6, 7)
  - **Blocks**: Tasks 6–7
  - **Blocked By**: Tasks 3, 4

  **References**:
  - New file: `rl_poker/envs/rl_poker_aec.py` - PettingZoo AEC environment
  - New file: `rl_poker/scripts/smoke_env.py` - episode smoke test
  - External: https://pettingzoo.farama.org/api/aec/ (AEC env API)

  **Acceptance Criteria**:
  - [ ] `python -m rl_poker.scripts.smoke_env --episodes 5` exits 0
  - [ ] Observation includes `action_mask` aligned with MAX_ACTIONS

  **Commit**: NO (repo not initialized)

- [x] 6. Baseline agents + evaluation hooks

  **What to do**:
  - Implement random legal-move agent
  - Implement heuristic agent (e.g., prefer shortest winning move, avoid breaking pairs, prefer shedding)
  - Add policy pool interface to load/save past checkpoints for evaluation
  - Integrate baselines into evaluation script interface

  **Must NOT do**:
  - No hardcoding to specific model architecture

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: baseline policy logic and evaluation integration
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 7)
  - **Blocks**: Task 8
  - **Blocked By**: Task 5

  **References**:
  - New file: `rl_poker/agents/random_agent.py` - random legal action
  - New file: `rl_poker/agents/heuristic_agent.py` - heuristic policy
  - New file: `rl_poker/agents/policy_pool.py` - checkpoint pool for evaluation

  **Acceptance Criteria**:
  - [ ] `python -m rl_poker.scripts.evaluate --episodes 10 --opponents random,heuristic` runs and prints scores

  **Commit**: NO (repo not initialized)

- [x] 7. PPO self-play training script (CleanRL-style)

  **What to do**:
  - Implement shared-policy PPO for AEC env with action masking
  - Support self-play by mirroring the shared policy across all agents
  - Log training metrics and save checkpoints
  - Add CLI args for steps, seed, checkpoint dir

  **Must NOT do**:
  - No external orchestration frameworks beyond PyTorch + standard libs

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: RL training loop and model plumbing
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 6)
  - **Blocks**: Task 8
  - **Blocked By**: Task 5

  **References**:
  - New file: `rl_poker/scripts/train.py` - PPO training entry point
  - External: https://cleanrl.dev/ (PPO reference implementation style)
  - External: https://pytorch.org/docs/stable/ (optimizer/model APIs)

  **Acceptance Criteria**:
  - [ ] `python -m rl_poker.scripts.train --steps 1000 --self-play --checkpoint-dir .sisyphus/checkpoints` exits 0
  - [ ] Checkpoint files created in `.sisyphus/checkpoints`

  **Commit**: NO (repo not initialized)

- [x] 8. Evaluation metrics + policy pool integration + tests

  **What to do**:
  - Implement evaluation loop: average score, rank distribution, win-rate vs baselines
  - Add Elo or similar rating tracking across runs
  - Add tests for metric aggregation (deterministic fake results)
  - Integrate policy pool as evaluation opponent option

  **Must NOT do**:
  - No manual verification of results

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: metrics, evaluation orchestration, and tests
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser automation
    - `frontend-ui-ux`: no UI work
    - `dev-browser`: no web navigation
    - `git-master`: no git operations requested

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: None
  - **Blocked By**: Tasks 6, 7

  **References**:
  - New file: `rl_poker/scripts/evaluate.py` - evaluation runner
  - New file: `tests/test_metrics.py` - aggregation and Elo tests

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_metrics.py` passes
  - [ ] `python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic,policy_pool` prints average score and rank distribution

  **Commit**: NO (repo not initialized)

---

## Commit Strategy

No git repository initialized; commits are not planned. If git is initialized later, use per-task commits with test commands from each task’s acceptance criteria.

---

## Success Criteria

### Verification Commands
```bash
python -m pytest
python -m rl_poker.scripts.train --steps 1000 --self-play --checkpoint-dir .sisyphus/checkpoints
python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic,policy_pool
```

### Final Checklist
- [ ] Rules engine strictly matches spec (no A/2 in sequences, no bombs, tail-hand exemption enforced)
- [ ] Legal move generator produces valid action masks
- [ ] AEC environment runs deterministically with seeding
- [ ] PPO self-play runs and produces checkpoints
- [ ] Evaluation metrics report average score and rank distribution

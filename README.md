# RL Poker

GPU-accelerated reinforcement learning for a 4-player Big Two-style poker variant. The project includes a full CPU rules engine, a PettingZoo AEC environment, and a GPU vectorized training system with opponent pools and PSRO-lite sampling.

English (default) | [中文](README_CN.md)

## Highlights

- Full rules engine with tail-hand exemptions and ranking-based scoring
- PettingZoo AEC environment with action masking
- GPU vectorized PPO training (fixed learner seat + dynamic opponent pool)
- PSRO-lite sampling that prioritizes opponents who currently beat you
- Optional GPU heuristic opponents for style diversity
- Behavior-based belief features + GRU history for memory
- GPU evaluation and automated checkpoint selection

## Rules Summary

- 4 players, 52 cards, 13 each, no jokers
- Rank order: 2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
- Suits do not affect comparison
- No bombs; four-of-a-kind only appears as 4+3

Legal hands:
- Single
- Pair
- Straight (5+ cards, 3–K only)
- Consecutive pairs (3+ pairs, 3–K only)
- Three plus two
- Four plus three

Tail-hand exemptions (only when finishing your hand):
- 3+2 can be played as 3+1 (4 cards) or 3+0 (3 cards)
- 4+3 can be played as 4+2 (6 cards), 4+1 (5 cards), or 4+0 (4 cards)
- After an exemption, the next player must respond with the full standard size

Flow and scoring:
- Holder of the Heart 3 leads and must include it on the first move
- Passing is allowed when following; 3 consecutive passes reset the lead
- Game ends when 3 players finish
- Score: +2 (1st), +1 (2nd), -1 (3rd), -2 (4th)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
# Minimal smoke training
python -m rl_poker.scripts.train --total-timesteps 100000

# Full GPU run (uses scripts/train_gpu.sh)
./scripts/train_gpu.sh --total-timesteps 2000000
```

### 8GB GPU Suggested Settings

```bash
python -m rl_poker.scripts.train \
  --num-envs 64 \
  --rollout-steps 64 \
  --total-timesteps 2000000 \
  --hidden-size 256
```

## Training Notes

- Learner seat is fixed to `player_0`; opponents are sampled from the pool.
- The opponent pool includes random, GPU heuristics, and historical checkpoints.
- PSRO-lite sampling increases probability of opponents that currently exploit you.
- Shaping reward adds a small EV-difference term and anneals to 0.
- Belief features are computed from public actions and an approximate action-space posterior.
- GRU-based history is enabled by default; disable with `--no-recurrent`.

## Evaluation

GPU evaluation (same environment semantics as training):

```bash
python -m rl_poker.scripts.eval_gpu \
  --checkpoint checkpoints/xxx.pt \
  --episodes 200 \
  --num-envs 128
```

CPU rules evaluation (AEC environment):

```bash
python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic
```

### Multi-Checkpoint Evaluation and Pruning

```bash
# Evaluate a directory (writes eval_results.json)
python -m rl_poker.scripts.eval_gpu --checkpoint-dir checkpoints --episodes 100

# Keep top-K and delete the rest
python scripts/select_checkpoints.py \
  --eval-json checkpoints/eval_results.json \
  --keep 5 --metric mean_score --delete
```

## Belief & Memory

Belief features are built from public play history and a lightweight posterior:
- Plays update per-rank logits based on the action’s required ranks.
- Passes penalize ranks implicated by any legal response to the previous action.
- A temperature-controlled smoothing kernel can be tuned via `--belief-temp`.

## Testing

```bash
python -m pytest
```

Notable tests:
- `tests/test_rule_parity.py` checks step-by-step alignment between the CPU engine and GPU env at the rank level.

## Project Structure

```
rl_poker/
├── rl/                 # GPU training components (env/policy/opponents/history)
├── rules/              # Core rules (ranks, hand parsing)
├── moves/              # Action space + legality (CPU/GPU)
├── engine/             # Full CPU game engine
├── envs/               # PettingZoo AEC environment
├── scripts/            # Training/evaluation entrypoints
└── agents/             # Baseline agents

scripts/
└── train_gpu.sh

tests/
```

## Design Notes

- The GPU environment is rank-centric for composite actions. Suits are not compared; Heart 3 is enforced on the first move. The CPU engine remains fully card-level. Parity tests validate rank counts, turn flow, and finishing order.

## License

MIT License

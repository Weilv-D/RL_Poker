# RL Poker

GPU-accelerated reinforcement learning for a 4-player Big Two-style poker variant, combining a CPU rules engine, a PettingZoo AEC environment, and GPU vectorized training.

English | [中文](README_CN.md)

![RL Poker overview](assets/rl_poker_overview.svg)

## Highlights

- GPU vectorized training built for multi-agent scale
- Dynamic opponent pool with PSRO-lite sampling to chase exploiters
- Behavior-based belief features paired with GRU memory
- Automated GPU evaluation with checkpoint pruning

## Project Layout

- `rl_poker/rl`: GPU env, PPO, opponents, belief, memory
- `rl_poker/engine`: CPU rules engine for correctness
- `rl_poker/envs`: PettingZoo AEC wrapper
- `rl_poker/moves` + `rl_poker/rules`: legal moves and hand logic
- `rl_poker/agents`: random, heuristic, policy pool
- `scripts/`: training/evaluation helpers and utilities

## Rules & Scoring (Big Two Variant)

- 4 players, 13 cards each (standard 52-card deck)
- Heart 3 leads the first trick; the first move must contain Heart 3
- Allowed hand types: single, pair, straight (5+ consecutive, 3-K only), consecutive pairs (3+ pairs, 3-K only), 3+2, 4+3
- Tail-hand exemptions: 3+0/3+1 and 4+0/4+1/4+2 only when the move uses all remaining cards
- If the previous player used an exemption, the next player must answer with the full standard size (5 for 3+2, 7 for 4+3) or pass
- Game ends when 3 players finish; scores are +2 (1st), +1 (2nd), -1 (3rd), -2 (4th)

## Architecture Overview

- CPU rules engine (`rl_poker/engine`) for correctness and PettingZoo AEC integration
- GPU vectorized environment (`rl_poker/rl/gpu_env.py`) for high-throughput training
- Fixed GPU action space + mask (`rl_poker/moves/gpu_action_mask.py`) to avoid per-step CPU enumeration
- Dynamic action encoding for CPU AEC env (`rl_poker/moves/action_encoding.py`, `MAX_ACTIONS=1000`)
- Parity tests to validate GPU legality against CPU rules (`tests/test_gpu_parity.py`)

## Observation & Action Encoding

- GPU base observation: 52-card hand + 13 rank counts + 5 context features
- Optional public opponent rank counts (`reveal_opponent_ranks`)
- Training augments observations with belief features (3×13), opponents’ remaining cards (3), and public played ranks (13)
- GPU actions are a fixed pre-enumerated table (PASS, singles, pairs, straights, consecutive pairs, 3+2, 4+3, exemptions), masked per state on GPU
- PettingZoo AEC uses dynamic legal-move enumeration and an action mask inside the observation dict

## Docs

- Technical report (Chinese): `docs/TECHNICAL_REPORT_CN.md`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Playground (Web & TUI)

Web UI (FastAPI):

```bash
python playground/web_server.py
```

Open `http://127.0.0.1:8000` and use the in-game `重发 / 开局` controls to shuffle and start. Optional env vars:

- `RL_POKER_HOST` / `RL_POKER_PORT` to bind a different address/port
- `RL_POKER_API_TOKEN` to require a token for `/api/*` (open `/?token=...` once)

TUI (Rich):

```bash
python playground/play_tui.py --tui
```

If checkpoints are present under `checkpoints/`, the UI will offer them; otherwise it falls back to random opponents.

Playground extras (not in core deps): `pip install fastapi uvicorn pydantic rich`.

## Training (First Run)

Recommended GPU preset:

```bash
RUN_NAME=star ./scripts/train_gpu.sh --quality
```

Custom short run (fast smoke test):

```bash
RUN_NAME=star ./scripts/train_gpu.sh --total-timesteps 1000000 --num-envs 256 --rollout-steps 64
```

Resume from the latest checkpoint:

```bash
python -m rl_poker.scripts.train --resume checkpoints/star/latest.pt
```

## Training

Override any parameter on the command line:

```bash
RUN_NAME=star ./scripts/train_gpu.sh --num-envs 1024 --rollout-steps 64 --total-timesteps 65536000
```

Notes:
- Keep `total_timesteps` divisible by `num_envs * rollout_steps` for clean updates.
- For memory pressure, reduce `--rollout-steps` or `--history-window`.
- Recurrent GRU + history buffer are enabled by default; disable with `--no-recurrent`.
- Belief features and shaping rewards can be tuned via `--belief-*` and `--shaping-*` flags.

## Checkpoint Naming

When you pass `--run-name <name>`, checkpoints are saved under:

```
checkpoints/<name>/<name>_###_step_<N>.pt
```

Examples:

```
checkpoints/star/star_001_step_14068672.pt
checkpoints/garlic/garlic_002_step_55498804.pt
```

The `###` increments by save order (not by step). Resuming from a checkpoint continues numbering from the latest in that folder. A `latest.pt` symlink is kept in each run folder.

## Logs

Training logs:

```
runs/<run-name>/<run-name>_###.log
```

Evaluation logs (GPU/CPU):

```
runs/<run-name>/<run-name>_eval_###.log
```

## Evaluation

Evaluate the latest checkpoint:

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/star/latest.pt --episodes 200 --run-name star
```

Evaluate all checkpoints in a run:

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint-dir checkpoints/star --episodes 200 --run-name star
```

CPU evaluation against a pool:

```bash
python -m rl_poker.scripts.evaluate --pool-dir checkpoints/star --run-name star --episodes 200
```

Metrics reported by `eval_gpu` include mean score, win rate, average rank, and Elo ratings.


## Utilities

- `./scripts/train_gpu.sh` preset GPU training
- `python scripts/select_checkpoints.py --run-name star` keep top checkpoints from eval json
- `python scripts/run_tests.py` run pytest without ROS plugin interference

## Testing

```bash
python scripts/run_tests.py -k rule_parity
```

## License

MIT License

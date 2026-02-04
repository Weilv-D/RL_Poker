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

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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


## Human vs AI (TUI)

```bash
python scripts/play_human_vs_ai.py --tui
```

Tips:
- `Enter` / `p` / `pass` = pass
- `0 1 2` = play by hand indices
- `3H 4D` = play by card names

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

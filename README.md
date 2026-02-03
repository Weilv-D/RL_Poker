# RL Poker

GPU-accelerated reinforcement learning for a 4-player Big Two-style poker variant, combining a CPU rules engine, a PettingZoo AEC environment, and GPU vectorized training.

GitHub short description: `GPU-accelerated multi-agent RL for Big Two poker with PSRO-lite opponents and belief-augmented PPO.`

English | [中文](README_CN.md)

![RL Poker overview](assets/rl_poker_overview.svg)

## Highlights

- GPU vectorized training built for multi-agent scale
- Dynamic opponent pool with PSRO-lite sampling to chase exploiters
- Behavior-based belief features paired with GRU memory
- Automated GPU evaluation with checkpoint pruning

## Modules at a Glance

- `rl_poker/rl`: GPU env, PPO, opponents, belief, memory
- `rl_poker/engine`: CPU rules engine for correctness
- `rl_poker/envs`: PettingZoo AEC wrapper
- `rl_poker/moves` + `rl_poker/rules`: legal moves and hand logic
- `rl_poker/agents`: random, heuristic, policy pool

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
python -m rl_poker.scripts.train --total-timesteps 100000
```

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/xxx.pt --episodes 200
```

## Training

Default GPU training preset (1024 envs) is embedded in the script:

```bash
./scripts/train_gpu.sh
```

Override any parameter on the command line:

```bash
./scripts/train_gpu.sh --num-envs 1024 --rollout-steps 64 --total-timesteps 65536000
```

Notes:
- Keep `total_timesteps` divisible by `num_envs * rollout_steps` for clean updates.
- For memory pressure, reduce `--rollout-steps` or `--history-window`.

## Evaluation

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/xxx.pt --episodes 200
```

```bash
python -m rl_poker.scripts.evaluate --checkpoint checkpoints/xxx.pt --episodes 200
```

## Utilities

- `./scripts/train_gpu.sh` preset GPU training
- `python scripts/select_checkpoints.py` prune checkpoints
- `python scripts/run_tests.py` run pytest without ROS plugin interference

## Testing

```bash
python scripts/run_tests.py -k rule_parity
```

## License

MIT License

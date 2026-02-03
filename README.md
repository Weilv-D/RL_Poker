# RL Poker

GPU-accelerated reinforcement learning for a 4-player Big Two-style poker variant, combining a CPU rules engine, a PettingZoo AEC environment, and GPU vectorized training.

English | [中文](README_CN.md)

![RL Poker overview](assets/rl_poker_overview.svg)

## Innovations

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

## Utilities

- `./scripts/train_gpu.sh` preset GPU training
- `python scripts/select_checkpoints.py` prune checkpoints

## License

MIT License

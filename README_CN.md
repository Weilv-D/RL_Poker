# RL Poker

面向四人 Big Two 风格扑克的强化学习项目，结合 CPU 规则引擎、PettingZoo AEC 环境与 GPU 向量化训练。

GitHub 简介建议：`面向 Big Two 的多智能体 GPU 强化学习，含 PSRO-lite 对手池与信念增强 PPO。`

[English](README.md) | 中文

![RL Poker 概览](assets/rl_poker_overview.svg)

## 核心亮点

- 面向多智能体规模的 GPU 向量化训练
- 动态对手池 + PSRO-lite 采样，追踪能压制当前策略的对手
- 行为信念特征与 GRU 记忆结合，覆盖长时序信息
- GPU 评估与 checkpoint 自动筛选

## 模块速览

- `rl_poker/rl`: GPU 环境、PPO、对手池、信念、记忆
- `rl_poker/engine`: CPU 规则引擎，保证正确性
- `rl_poker/envs`: PettingZoo AEC 封装
- `rl_poker/moves` + `rl_poker/rules`: 合法出牌与牌型逻辑
- `rl_poker/agents`: 随机、启发式、策略池基线

## 快速开始

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

## 训练

GPU 训练默认预设（1024 env）已经写入脚本：

```bash
./scripts/train_gpu.sh
```

需要覆盖参数时：

```bash
./scripts/train_gpu.sh --num-envs 1024 --rollout-steps 64 --total-timesteps 65536000
```

说明：
- 建议保证 `total_timesteps` 可被 `num_envs * rollout_steps` 整除。
- 显存紧张时优先降低 `--rollout-steps` 或 `--history-window`。

## 评估

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/xxx.pt --episodes 200
```

```bash
python -m rl_poker.scripts.evaluate --checkpoint checkpoints/xxx.pt --episodes 200
```

## 工具脚本

- `./scripts/train_gpu.sh` GPU 训练预设
- `python scripts/select_checkpoints.py` 自动筛选 checkpoint
- `python scripts/run_tests.py` 跳过 ROS 插件干扰的 pytest 启动器

## 测试

```bash
python scripts/run_tests.py -k rule_parity
```

## License

MIT License

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
python -m rl_poker.scripts.train --run-name star --total-timesteps 100000
```

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/star/star_001_step_100000.pt --episodes 200
```

## 训练

GPU 训练默认预设（1024 env）已经写入脚本：

```bash
./scripts/train_gpu.sh
```

推荐（显式指定训练名）：

```bash
RUN_NAME=star ./scripts/train_gpu.sh --quality
```

需要覆盖参数时：

```bash
RUN_NAME=star ./scripts/train_gpu.sh --num-envs 1024 --rollout-steps 64 --total-timesteps 65536000
```

说明：
- 建议保证 `total_timesteps` 可被 `num_envs * rollout_steps` 整除。
- 显存紧张时优先降低 `--rollout-steps` 或 `--history-window`。

### Checkpoint 命名与目录

当使用 `--run-name <name>` 时，checkpoint 会保存到：

```
checkpoints/<name>/<name>_###_step_<N>.pt
```

示例：

```
checkpoints/star/star_001_step_14068672.pt
checkpoints/garlic/garlic_002_step_55498804.pt
```

编号 `###` 按保存顺序递增（不是按 step）。从 checkpoint 恢复训练时会继续递增。

### 日志

训练日志保存到：

```
runs/<run-name>/<run-name>_###.log
```

示例：

```
runs/star/star_001.log
```

评估日志（GPU/CPU）保存到：

```
runs/<run-name>/<run-name>_eval_###.log
```

## 评估

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/star/star_001_step_14068672.pt --episodes 200
```

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint-dir checkpoints/star --episodes 200
```

```bash
python -m rl_poker.scripts.evaluate --pool-dir checkpoints/star --episodes 200
```

## 人机对弈（TUI）

```bash
python scripts/play_human_vs_ai.py --tui
```

提示：
- `Enter` / `p` / `pass` = 过牌
- `0 1 2` = 按手牌序号出牌
- `3H 4D` = 按牌面出牌

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

# RL Poker

面向四人 Big Two 风格扑克的强化学习项目，结合 CPU 规则引擎、PettingZoo AEC 环境与 GPU 向量化训练。


[English](README.md) | 中文

![RL Poker 概览](assets/rl_poker_overview.svg)

## 核心亮点

- 面向多智能体规模的 GPU 向量化训练
- 动态对手池 + PSRO-lite 采样，追踪能压制当前策略的对手
- 行为信念特征与 GRU 记忆结合，覆盖长时序信息
- GPU 评估与 checkpoint 自动筛选

## 项目结构

- `rl_poker/rl`: GPU 环境、PPO、对手池、信念、记忆
- `rl_poker/engine`: CPU 规则引擎，保证正确性
- `rl_poker/envs`: PettingZoo AEC 封装
- `rl_poker/moves` + `rl_poker/rules`: 合法出牌与牌型逻辑
- `rl_poker/agents`: 随机、启发式、策略池基线
- `scripts/`: 训练/评估辅助脚本

## 规则与计分（Big Two 变体）

- 4 人对局，每人 13 张牌（标准 52 张）
- 首出必须包含红桃 3（Heart 3），由持有者先手
- 牌型：单张、对子、顺子（5+ 连牌，仅 3-K）、连对（3+ 连对，仅 3-K）、三带二、四带三
- 尾牌豁免：三带二可用 3+0/3+1，四带三可用 4+0/4+1/4+2，且必须一次出完手牌
- 若上一手使用豁免，下一手必须用标准牌型（3+2 为 5 张，4+3 为 7 张）或不出
- 3 人出完牌后结束，计分为 +2（第 1）、+1（第 2）、-1（第 3）、-2（第 4）

## 架构概览

- CPU 规则引擎（`rl_poker/engine`）用于正确性与 PettingZoo AEC 交互
- GPU 向量化环境（`rl_poker/rl/gpu_env.py`）用于高吞吐训练
- GPU 固定动作空间与掩码（`rl_poker/moves/gpu_action_mask.py`）避免每步 CPU 枚举
- CPU AEC 动态动作编码（`rl_poker/moves/action_encoding.py`，`MAX_ACTIONS=1000`）
- GPU/CPU 合法性一致性测试（`tests/test_gpu_parity.py`）

## 观测与动作编码

- GPU 基础观测：手牌 52 位 + 13 维牌阶计数 + 5 维上下文
- 可选公开对手牌阶计数（`reveal_opponent_ranks`）
- 训练时拼接信念特征（3×13）、对手剩余牌数（3）、公共已出牌阶计数（13）
- GPU 动作空间为预枚举固定表（PASS、单张、对子、顺子、连对、三带二、四带三、豁免），按状态在 GPU 上生成掩码
- PettingZoo AEC 使用动态合法出牌枚举，并在观测中返回动作掩码

## 文档

- 技术报告（中文）：`docs/TECHNICAL_REPORT_CN.md`

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 交互式游玩（Playground）

Web UI（FastAPI）：

```bash
python playground/web_server.py
```

打开 `http://127.0.0.1:8000`，使用页面内的 `重发 / 开局` 控制牌局。可选环境变量：

- `RL_POKER_HOST` / `RL_POKER_PORT`：绑定指定地址/端口
- `RL_POKER_API_TOKEN`：启用 `/api/*` 访问令牌（首次打开 `/?token=...`）

TUI（Rich）：

```bash
python playground/play_tui.py --tui
```

若 `checkpoints/` 下存在模型，则会提供选择；否则回退为随机对手。

Playground 额外依赖（不在核心依赖中）：`pip install fastapi uvicorn pydantic rich`。

## 训练（第一次）

推荐 GPU 预设：

```bash
RUN_NAME=star ./scripts/train_gpu.sh --quality
```

快速短跑（验证环境）：

```bash
RUN_NAME=star ./scripts/train_gpu.sh --total-timesteps 1000000 --num-envs 256 --rollout-steps 64
```

从最新 checkpoint 续训：

```bash
python -m rl_poker.scripts.train --resume checkpoints/star/latest.pt
```

## 训练

需要覆盖参数时：

```bash
RUN_NAME=star ./scripts/train_gpu.sh --num-envs 1024 --rollout-steps 64 --total-timesteps 65536000
```

说明：
- 建议保证 `total_timesteps` 可被 `num_envs * rollout_steps` 整除。
- 显存紧张时优先降低 `--rollout-steps` 或 `--history-window`。
- 默认启用 GRU + 历史序列；可使用 `--no-recurrent` 关闭。
- 行为信念与 shaping 奖励可用 `--belief-*` 和 `--shaping-*` 参数调整。

## Checkpoint 命名

当使用 `--run-name <name>` 时，checkpoint 会保存到：

```
checkpoints/<name>/<name>_###_step_<N>.pt
```

示例：

```
checkpoints/star/star_001_step_14068672.pt
checkpoints/garlic/garlic_002_step_55498804.pt
```

编号 `###` 按保存顺序递增（不是按 step）。从 checkpoint 恢复训练时会继续递增。每个训练目录会维护 `latest.pt` 软链接指向最新模型。

## 日志

训练日志：

```
runs/<run-name>/<run-name>_###.log
```

评估日志（GPU/CPU）：

```
runs/<run-name>/<run-name>_eval_###.log
```

## 评估

评估最新模型：

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint checkpoints/star/latest.pt --episodes 200 --run-name star
```

评估整个 run 目录：

```bash
python -m rl_poker.scripts.eval_gpu --checkpoint-dir checkpoints/star --episodes 200 --run-name star
```

CPU 评估（对手池）：

```bash
python -m rl_poker.scripts.evaluate --pool-dir checkpoints/star --run-name star --episodes 200
```

`eval_gpu` 会输出平均得分、胜率、平均名次与 Elo 评级等指标。

## 工具脚本

- `./scripts/train_gpu.sh` GPU 训练预设
- `python scripts/select_checkpoints.py --run-name star` 自动筛选 checkpoint
- `python scripts/run_tests.py` 跳过 ROS 插件干扰的 pytest 启动器

## 测试

```bash
python scripts/run_tests.py -k rule_parity
```

## License

MIT License

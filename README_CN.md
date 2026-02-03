# RL Poker

面向四人扑克的强化学习项目，包含完整 CPU 规则引擎、PettingZoo AEC 环境，以及 GPU 向量化训练系统（对手池 + PSRO-lite）。

## 亮点

- 完整规则引擎，包含尾牌豁免与名次结算
- PettingZoo AEC 轮流出牌 + 动作掩码
- GPU 向量化 PPO 训练（固定学习位 + 动态对手池）
- PSRO-lite 采样，优先挑战能压制当前策略的对手
- GPU 启发式对手（可选，提供风格多样性）
- 行为后验信念 + GRU 历史序列
- GPU 评估与 checkpoint 自动筛选

## 规则摘要

- 4 人，52 张牌，每人 13 张；不含大小王
- 点数大小：2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
- 不比花色
- 无炸弹：四张相同仅作为四带三

合法牌型：
- 单张
- 对子
- 顺子：5 张及以上，仅 3–K
- 连对：3 对及以上，仅 3–K
- 三带二
- 四带三

尾牌豁免（仅尾牌）：
- 三带二可用 3+1（4 张）或 3+0（3 张）
- 四带三可用 4+2（6 张）、4+1（5 张）、4+0（4 张）
- 上家使用豁免后，下家必须用标准张数才能压

流程与结算：
- 红桃 3 首出，首手必须包含红桃 3
- 不可压则 PASS
- 第 3 名出完牌即结束
- 计分：第 1 名 +2，第 2 名 +1，第 3 名 -1，第 4 名 -2

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

可选：

```bash
pip install tensorboard
```

## 快速开始

```bash
# 轻量测试
python -m rl_poker.scripts.train --total-timesteps 100000

# 完整 GPU 训练
./scripts/train_gpu.sh --total-timesteps 2000000
```

### 8GB GPU 建议配置

```bash
python -m rl_poker.scripts.train \
  --num-envs 64 \
  --rollout-steps 64 \
  --total-timesteps 2000000 \
  --hidden-size 256
```

## 训练说明

- 固定学习位为 `player_0`，其余三位从对手池采样
- 对手池包含随机对手、GPU 启发式对手、历史快照
- PSRO-lite 采样偏向当前能压制你的对手
- 塑形奖励为终局奖励加微小 EV 差分，并退火到 0
- 信念特征由公开行为构建，使用近似动作空间后验更新
- 默认启用 GRU 历史序列，可用 `--no-recurrent` 关闭

## 评估

GPU 一致环境评估：

```bash
python -m rl_poker.scripts.eval_gpu \
  --checkpoint checkpoints/xxx.pt \
  --episodes 200 \
  --num-envs 128
```

CPU 规则引擎评估（AEC）：

```bash
python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic
```

### 多 checkpoint 评估与筛选

```bash
# 评估目录内所有 checkpoint（输出 eval_results.json）
python -m rl_poker.scripts.eval_gpu --checkpoint-dir checkpoints --episodes 100

# 选择 Top-K 并删除其余
python scripts/select_checkpoints.py \
  --eval-json checkpoints/eval_results.json \
  --keep 5 --metric mean_score --delete
```

## 信念与记忆

信念特征来自公开出牌历史与近似后验：
- 出牌根据动作所需点数更新 logits
- PASS 根据“可响应动作空间”对相关点数做惩罚
- `--belief-temp` 控制平滑强度

## 测试

```bash
python -m pytest
```

重点测试：
- `tests/test_rule_parity.py`：GPU 环境与 CPU 引擎的逐步对齐（点数级别）

## 项目结构

```
rl_poker/
├── rl/                 # GPU 训练组件（环境/策略/对手池/记忆）
├── rules/              # 规则定义
├── moves/              # 动作空间与合法性
├── engine/             # CPU 规则引擎
├── envs/               # PettingZoo AEC 环境
├── scripts/            # 训练/评估入口
└── agents/             # 基线对手

scripts/
└── train_gpu.sh

tests/
```

## 设计说明

- GPU 环境以“点数”为核心抽象处理复合牌型；不比较花色，仅在首手强制红桃 3。CPU 引擎保持完整牌级别。对齐测试验证点数计数、回合流转与名次一致。

## License

MIT License

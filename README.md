# RL Poker

四人扑克规则的强化学习项目，包含完整规则引擎、PettingZoo AEC 环境，以及面向实战的 GPU 向量化训练系统。

## 特性

- **规则引擎**: 严格实现牌型、出牌、豁免与结算逻辑
- **AEC 环境**: PettingZoo AEC 轮流出牌 + 动作掩码
- **PPO 训练**: GPU 向量化训练系统（固定学习位 + 动态对手池）
- **PSRO-lite 对手池**: 历史快照 + 随机/启发式对手，优先采样能压制当前策略的对手
- **对抗性塑形奖励**: 终局奖励 + 相对对手池基准的微小 EV 差分（退火到 0）
- **GPU 简易启发式**: 可选对手，风格明确，帮助稳定训练分布
- **TensorBoard**: 训练日志可视化

## 游戏规则（完整规则引擎与 AEC 环境）

- 4 人，52 张牌，每人 13 张；不含大小王
- 点数大小：2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
- 不比花色，只比点数
- 无炸弹：四张相同仅作为四带三使用

**合法牌型**
- 单张
- 对子
- 顺子：5 张及以上，仅 3-K（A/2 禁止）
- 连对：3 对及以上，仅 3-K（A/2 禁止）
- 三带二
- 四带三

**尾牌豁免（仅尾牌）**
- 三带二：剩 4 张可出 3+1；剩 3 张可出 3+0
- 四带三：剩 6 张可出 4+2；剩 5 张可出 4+1；剩 4 张可出 4+0
- 上家豁免后，下家必须补足标准张数才能压（否则必须 PASS）

**流程与结算**
- 红桃 3 持有者首出，且首手必须包含红桃 3
- 轮流出牌，不能压则 PASS
- 第三名出完牌即结束
- 积分：第 1 名 +2，第 2 名 +1，第 3 名 -1，第 4 名 -2

## 训练脚本规则说明（GPU 向量化训练）

`rl_poker/scripts/train.py` 使用 GPU 向量化环境以追求高吞吐，规则与完整引擎一致：

- **胜负**：第三名出完牌即结束
- **奖励**：第 1 名 +2，第 2 名 +1，第 3 名 -1，第 4 名 -2
- **尾牌豁免**：三带二与四带三的尾牌豁免规则已实现
- **顺子/连对长度**：顺子 5-11；连对 3-11 对（仅 3-K）
- **首手限制**：红桃 3 持有者首出，且首手必须包含红桃 3
- **学习位设定**：固定学习位为 `player_0`，其余三位从对手池采样

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install tensorboard  # 可选
```

## 训练

```bash
# 快速测试
python -m rl_poker.scripts.train --total-timesteps 100000

# 带 TensorBoard 的训练脚本（自动启动 TensorBoard）
./scripts/train_gpu.sh --total-timesteps 2000000
```

### 启动脚本使用方式（train_gpu.sh）

```bash
# 直接运行（默认使用 /home/weilv/workspace/prjforfun/RL_poker/.venv）
./scripts/train_gpu.sh

# 自定义参数
./scripts/train_gpu.sh --num-envs 256 --rollout-steps 256 --total-timesteps 50000000 \
    --learning-rate 3e-4 --num-minibatches 8 --ppo-epochs 4 --log-interval 5 --save-interval 50

# 指定 venv（可选）
PYTHON=/home/weilv/workspace/prjforfun/RL_poker/.venv/bin/python \
    ./scripts/train_gpu.sh --total-timesteps 2000000

# 查看帮助
./scripts/train_gpu.sh --help
```

### 监控训练

```bash
tensorboard --logdir runs
# http://localhost:6006
```

### 训练参数（train.py）

```
--num-envs N           并行环境数量 (默认: 128)
--total-timesteps N    总训练步数 (默认: 2000000)
--rollout-steps N      每次 rollout 的步数 (默认: 128)
--hidden-size N        网络隐藏层大小 (默认: 256)
--learning-rate LR     学习率 (默认: 3e-4)
--gamma GAMMA          折扣因子 (默认: 0.99)
--gae-lambda LAM       GAE lambda (默认: 0.95)
--ppo-epochs N         PPO 更新轮数 (默认: 4)
--num-minibatches N    小批次数量 (默认: 8)
--clip-coef C          PPO clip 系数 (默认: 0.2)
--ent-coef C           熵奖励系数 (默认: 0.01)
--vf-coef C            价值损失系数 (默认: 0.5)
--max-grad-norm N      梯度裁剪 (默认: 0.5)
--pool-max-size N      对手池最大容量 (默认: 16)
--pool-ema-beta B      对手 EV EMA 更新速率 (默认: 0.05)
--pool-psro-beta B     PSRO-lite 采样温度 (默认: 3.0)
--pool-min-prob P      最小采样概率 (默认: 0.05)
--pool-add-interval N  快照加入对手池的频率 (默认: 10)
--pool-seed N          对手池随机种子 (默认: 42)
--pool-no-random       禁用随机对手
--pool-no-heuristic    禁用 GPU 启发式对手
--pool-heuristic-styles S  启发式风格 (默认: conservative,aggressive)
--shaping-alpha A      塑形奖励系数 (默认: 0.1)
--shaping-anneal-updates N  塑形奖励退火步数 (默认: 200)
--seed N               随机种子 (默认: 42)
--checkpoint-dir DIR   检查点目录 (默认: checkpoints)
--log-dir DIR          日志目录 (默认: runs)
--no-cuda              禁用 CUDA
```

## 训练架构（对手池 + PSRO-lite）

- 固定学习位 `player_0`，其余三位为对手池采样对手
- 对手池由随机对手、GPU 启发式对手、历史快照构成
- PSRO-lite 采样权重偏向“当前能压制你的对手”
- 训练中定期加入策略快照，形成动态对抗分布

## 对抗性塑形奖励

- 终局奖励保持不变
- 额外塑形奖励：`alpha * (episode_adv - baseline)`，其中 `episode_adv = learner_score - mean(opponent_scores)`，`baseline` 为当前对手组合的 EV EMA 均值
- `alpha` 线性退火到 0，避免改变最优解

## 记忆/推理模块（预留）

- `rl_poker/rl/history.py` 提供历史序列缓存接口
- `rl_poker/rl/recurrent.py` 提供 GRU 版策略网络骨架
- 默认不启用，仅作为 Phase 3 的接入点

## 评估（基于 AEC 完整规则）

```bash
python -m rl_poker.scripts.evaluate --episodes 50 --opponents random,heuristic
```

## 项目结构

```
rl_poker/
├── rl/                 # GPU 训练核心组件（环境/策略/对手池）
├── rules/              # 游戏规则定义
│   ├── ranks.py        # 牌点和花色定义
│   └── hands.py        # 牌型判断和比较
├── moves/              # 动作空间与掩码
│   ├── legal_moves.py  # 合法动作枚举（完整规则）
│   ├── action_encoding.py
│   └── gpu_action_mask.py  # GPU 向量化训练动作掩码
├── engine/             # 游戏引擎（完整规则）
│   └── game_state.py
├── envs/               # 环境实现
│   └── rl_poker_aec.py
├── scripts/            # 训练和评估脚本
│   ├── train.py
│   ├── evaluate.py
│   └── smoke_env.py
├── agents/
│   ├── random_agent.py
│   └── heuristic_agent.py
└── utils/
    └── seeding.py

scripts/
└── train_gpu.sh

tests/
```

## 开发

```bash
python -m pytest
```

## License

MIT License

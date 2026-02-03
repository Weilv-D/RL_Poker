# RL Poker

GPU-accelerated reinforcement learning for a 4-player Big Two-style poker variant.

English  | [中文](README_CN.md)

---

## Innovations

- GPU vectorized training with a dynamic opponent pool tuned for adversarial coverage
- PSRO-lite sampling that actively prioritizes opponents that exploit the current policy
- Belief feature modeling from public play, paired with GRU memory for long-horizon context
- Automated checkpoint evaluation and pruning to keep only the most competitive policies

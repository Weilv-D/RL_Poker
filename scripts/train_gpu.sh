#!/usr/bin/env bash
# GPU-native PPO training for RL Poker
# Usage: ./scripts/train_gpu.sh [options]
set -euo pipefail

# Defaults
# Resolve project root for stable absolute paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Prefer the active virtualenv if available; otherwise fall back to `python`.
# Local override example (do not commit hard-coded paths):
# PYTHON="/home/yourname/path/to/.venv/bin/python"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PYTHON="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON="python"
    fi
fi
NUM_ENVS="${NUM_ENVS:-1024}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-65536000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"
SEED="${SEED:-42}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-}"
QUALITY=0
QUALITY_EXTRA_ARGS=()
DEFAULT_EXTRA_ARGS=(
    --ppo-epochs 4
    --num-minibatches 64
    --gru-hidden 128
    --history-window 32
    --pool-max-size 32
    --pool-add-interval 5
    --pool-psro-beta 5.0
    --pool-min-prob 0.03
    --pool-heuristic-styles conservative,aggressive,rush,counter,variance
    --shaping-alpha 0.1
    --shaping-anneal-updates 500
    --log-interval 1
    --save-interval 50
)

usage() {
    cat << 'EOF'
RL Poker GPU Training Script

Usage: ./scripts/train_gpu.sh [options]

Options (key only):
    --quality              Quality-first preset for 8GB GPUs (override defaults)
    --num-envs N           Number of parallel environments (default: 1024)
    --total-timesteps N    Total training timesteps (default: 65536000)
    --rollout-steps N      Steps per rollout (default: 64)
    --hidden-size N        Network hidden size (default: 256)
    --learning-rate LR     Learning rate (default: 2.5e-4)
    --seed N               Random seed (default: 42)
    --pool-max-size N      Opponent pool size cap (default: 16)
    --pool-add-interval N  Snapshot add interval (default: 10)
    --pool-no-random       Disable random opponents
    --pool-no-heuristic    Disable heuristic opponents
    --pool-heuristic-styles S  Heuristic styles (default: conservative,aggressive)
    --shaping-alpha A      Shaping reward coefficient (default: 0.1)
    --shaping-anneal-updates N  Shaping anneal updates (default: 200)
    --no-recurrent         Disable GRU/history
    --history-window N     History window length (default: 16)
    --gru-hidden N         GRU hidden size (default: 128)
    --reveal-opponent-ranks  Debug: reveal opponents' ranks
    --belief-no-behavior   Disable behavior-based belief updates
    --belief-decay D       Belief decay (default: 0.98)
    --belief-play-bonus B  Belief play bonus (default: 0.5)
    --belief-pass-penalty P  Belief pass penalty (default: 0.3)
    --belief-temp T        Belief temperature (default: 2.0)
    -h, --help             Show this help

Examples:
  # Quick test (5 minutes)
  ./scripts/train_gpu.sh --total-timesteps 1000000 --num-envs 256 --rollout-steps 64

  # Quality-first (recommended)
  ./scripts/train_gpu.sh --quality

  # Standard training (1 hour)
  ./scripts/train_gpu.sh --total-timesteps 10000000 --num-envs 512 --rollout-steps 64

  # Long training (7 hours) with larger network
  ./scripts/train_gpu.sh --total-timesteps 100000000 --hidden-size 512 --num-envs 1024

Performance:
  - Expected SPS: ~10,000-15,000 on modern GPUs
  - 1M steps: ~1-2 minutes
  - 10M steps: ~15-20 minutes
  - 100M steps: ~2-3 hours
EOF
}

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quality)
            QUALITY=1
            NUM_ENVS="1024"
            TOTAL_TIMESTEPS="210763776"
            ROLLOUT_STEPS="64"
            HIDDEN_SIZE="256"
            LEARNING_RATE="2.5e-4"
            QUALITY_EXTRA_ARGS=(
                --ppo-epochs 4
                --num-minibatches 64
                --gru-hidden 128
                --history-window 32
                --pool-max-size 16
                --pool-add-interval 5
                --pool-psro-beta 5.0
                --pool-min-prob 0.03
                --pool-heuristic-styles conservative,aggressive,rush,counter,variance
                --shaping-alpha 0.1
                --shaping-anneal-updates 500
                --log-interval 1
                --save-interval 50
            )
            shift
            ;;
        --num-envs) NUM_ENVS="$2"; shift 2;;
        --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2;;
        --hidden-size) HIDDEN_SIZE="$2"; shift 2;;
        --learning-rate|--lr) LEARNING_RATE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        --) shift; EXTRA_ARGS+=("$@"); break;;
        *) EXTRA_ARGS+=("$1"); shift;;
    esac
done

# Create directories
mkdir -p "$CHECKPOINT_DIR"

# Print configuration
echo "🎮 RL Poker GPU Training"
echo "========================"
echo "Environments: $NUM_ENVS"
echo "Total steps:  $TOTAL_TIMESTEPS"
echo "Rollout:      $ROLLOUT_STEPS"
echo "Hidden size:  $HIDDEN_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Seed:         $SEED"
echo ""

# Run training
CMD=(
    "$PYTHON" -m rl_poker.scripts.train
    --num-envs "$NUM_ENVS"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --rollout-steps "$ROLLOUT_STEPS"
    --hidden-size "$HIDDEN_SIZE"
    --learning-rate "$LEARNING_RATE"
    --seed "$SEED"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --log-dir "$LOG_DIR"
)
if [[ -n "$RUN_NAME" ]]; then
    CMD+=(--run-name "$RUN_NAME")
fi
CMD+=("${DEFAULT_EXTRA_ARGS[@]}" "${QUALITY_EXTRA_ARGS[@]}" "${EXTRA_ARGS[@]}")

"${CMD[@]}"

echo ""
echo "✅ Training complete!"
echo "Checkpoints: $CHECKPOINT_DIR"

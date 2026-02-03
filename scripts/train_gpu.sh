#!/usr/bin/env bash
# GPU-native PPO training for RL Poker
# Usage: ./scripts/train_gpu.sh [options]
set -euo pipefail

# Defaults
PYTHON="${PYTHON:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-256}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SEED="${SEED:-42}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-runs}"
START_TB="${START_TB:-1}"
TB_HOST="${TB_HOST:-127.0.0.1}"
TB_PORT="${TB_PORT:-6006}"

usage() {
    cat << 'EOF'
RL Poker GPU Training Script

Usage: ./scripts/train_gpu.sh [options]

Options:
  --num-envs N           Number of parallel environments (default: 256)
  --total-timesteps N    Total training timesteps (default: 1000000)
  --rollout-steps N      Steps per rollout (default: 128)
  --hidden-size N        Network hidden size (default: 256)
  --learning-rate LR     Learning rate (default: 3e-4)
  --seed N               Random seed (default: 42)
  --checkpoint-dir DIR   Checkpoint directory (default: checkpoints)
  --log-dir DIR          TensorBoard log directory (default: runs)
  --no-tensorboard       Don't start TensorBoard
  --tb-host HOST         TensorBoard host (default: 127.0.0.1)
  --tb-port PORT         TensorBoard port (default: 6006)
  -h, --help             Show this help

Examples:
  # Quick test (5 minutes)
  ./scripts/train_gpu.sh --total-timesteps 100000

  # Standard training (1 hour)
  ./scripts/train_gpu.sh --total-timesteps 10000000 --num-envs 256

  # Long training (7 hours) with larger network
  ./scripts/train_gpu.sh --total-timesteps 100000000 --hidden-size 512

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
        --num-envs) NUM_ENVS="$2"; shift 2;;
        --total-timesteps) TOTAL_TIMESTEPS="$2"; shift 2;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2;;
        --hidden-size) HIDDEN_SIZE="$2"; shift 2;;
        --learning-rate|--lr) LEARNING_RATE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2;;
        --log-dir) LOG_DIR="$2"; shift 2;;
        --no-tensorboard) START_TB="0"; shift;;
        --tb-host) TB_HOST="$2"; shift 2;;
        --tb-port) TB_PORT="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        --) shift; EXTRA_ARGS+=("$@"); break;;
        *) EXTRA_ARGS+=("$1"); shift;;
    esac
done

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# TensorBoard management
TB_PID=""
start_tensorboard() {
    if [[ "$START_TB" == "0" ]]; then
        return 0
    fi
    
    # Check if tensorboard is available
    if "$PYTHON" -c "import tensorboard" 2>/dev/null; then
        "$PYTHON" -m tensorboard.main --logdir "$LOG_DIR" --host "$TB_HOST" --port "$TB_PORT" \
            >"$LOG_DIR/tensorboard.log" 2>&1 &
        TB_PID=$!
        echo "📊 TensorBoard: http://$TB_HOST:$TB_PORT"
    else
        echo "⚠️  TensorBoard not installed. Install with: pip install tensorboard"
    fi
}

cleanup() {
    if [[ -n "$TB_PID" ]]; then
        kill "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

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

# Start TensorBoard
start_tensorboard

# Run training
"$PYTHON" -m rl_poker.scripts.train \
    --num-envs "$NUM_ENVS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --hidden-size "$HIDDEN_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "✅ Training complete!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"

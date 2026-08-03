#!/bin/bash
# Monitor Round 2 experiments and auto-launch baseline when a GPU frees up.
# Run with: nohup bash monitor_and_launch_baseline.sh &
#
# Baseline config:
#   - Isaac-FullDOF-TRPO-v0 (uniform entropy_coef=0.003)
#   - kl_ub=0.06 (from config.py, same as Round 2 experiments)
#   - num_envs=2048, max_iterations=5000, seed=30

LOG=/workspace/isaaclab/logs/monitor_baseline.log
ISAACLAB_DIR=/workspace/isaaclab

echo "========================================" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor started" >> "$LOG"
echo "  Watching for: perdiment_kl06 (GPU0), armonly_kl06 (GPU1)" >> "$LOG"
echo "  Will launch: Isaac-FullDOF-TRPO-v0 baseline_kl06" >> "$LOG"
echo "========================================" >> "$LOG"

LAUNCHED=0
CHECK_INTERVAL=60  # seconds

while [ $LAUNCHED -eq 0 ]; do
    sleep $CHECK_INTERVAL

    # Check GPU 0 (perdiment_kl06)
    if ! pgrep -f "train.py.*perdiment_kl06" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] perdiment_kl06 process not found. GPU 0 free." >> "$LOG"
        GPU_ID=0
        LAUNCHED=1
    fi

    # Check GPU 1 (armonly_kl06)
    if [ $LAUNCHED -eq 0 ] && ! pgrep -f "train.py.*armonly_kl06" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] armonly_kl06 process not found. GPU 1 free." >> "$LOG"
        GPU_ID=1
        LAUNCHED=1
    fi

    # Still waiting
    if [ $LAUNCHED -eq 0 ]; then
        # Log every 10 minutes (every 10th check)
        COUNT=$((${COUNT:-0} + 1))
        if [ $((COUNT % 10)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still waiting... (check #$COUNT)" >> "$LOG"
        fi
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching baseline on GPU $GPU_ID..." >> "$LOG"

cd "$ISAACLAB_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID nohup ./isaaclab.sh -p \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-FullDOF-TRPO-v0 \
    --num_envs 2048 \
    --max_iterations 5000 \
    --headless \
    --logger wandb \
    --log_project_name full_dof_trpo \
    --run_name baseline_kl06 \
    >> "$LOG" 2>&1 &

BASELINE_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Baseline launched! PID=$BASELINE_PID, GPU=$GPU_ID" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor complete. Check training with:" >> "$LOG"
echo "  python3 /workspace/.claude/skills/train-analyze/analyze_training.py 0 --tier 2" >> "$LOG"

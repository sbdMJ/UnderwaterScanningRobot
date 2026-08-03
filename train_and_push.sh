#!/usr/bin/env bash
# 학습 실행 → 완료 시 최신 체크포인트를 repo에 커밋·push.
# 사용: ./train_and_push.sh --task Isaac-PKRC-WallScan-Train-Direct-v0 --num_envs 4096 --headless --max_iterations 3000 --run_name myrun [...]
set -euo pipefail
ISAACLAB=/root/home/rl_ws/isaaclab
REPO=/root/home/rl_ws/UnderwaterScanningRobot
LOGROOT=$ISAACLAB/logs/rsl_rl/pkrc_wallscan

cd $ISAACLAB
./isaaclab.sh -p /root/home/rl_ws/marinelab/scripts/train.py "$@"

RUN_DIR=$(ls -td $LOGROOT/*/ | head -1)
RUN_NAME=$(basename "$RUN_DIR")
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)

mkdir -p "$REPO/checkpoints/$RUN_NAME"
cp "$CKPT" "$RUN_DIR/params/agent.yaml" "$RUN_DIR/params/env.yaml" "$REPO/checkpoints/$RUN_NAME/"

cd $REPO
git add "checkpoints/$RUN_NAME"
git commit -m "checkpoint: $RUN_NAME ($(basename "$CKPT"))"
git push origin main
echo "PUSHED $RUN_NAME $(basename "$CKPT")"

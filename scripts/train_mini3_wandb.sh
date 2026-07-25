#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_mini3_wandb.sh
#
# Optional environment overrides:
#   WANDB_ENTITY=your_user_or_team
#   WANDB_PROJECT=bfmzero-mini3
#   WANDB_GROUP=mini3-isaacsim
#   WANDB_RUN_NAME=mini3-$(date +%Y%m%d-%H%M%S)
#   WORK_DIR=results/bfmzero-mini3-isaac
#   ONLINE_PARALLEL_ENVS=1024
#   NUM_ENV_STEPS=384000000
#
# Before first online run:
#   uv run wandb login

WANDB_PROJECT="${WANDB_PROJECT:-bfmzero-mini3}"
WANDB_GROUP="${WANDB_GROUP:-mini3-isaacsim}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mini3-$(date +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-results/bfmzero-mini3-isaac}"
ONLINE_PARALLEL_ENVS="${ONLINE_PARALLEL_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-384000000}"

cmd=(
  uv run python -m humanoidverse.train
  --robot-profile mini3
  --work-dir "${WORK_DIR}"
  --online-parallel-envs "${ONLINE_PARALLEL_ENVS}"
  --num-env-steps "${NUM_ENV_STEPS}"
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "${WANDB_RUN_NAME}"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi

"${cmd[@]}"

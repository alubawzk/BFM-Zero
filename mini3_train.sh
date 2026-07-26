#!/usr/bin/env bash
set -euo pipefail

# Mini3 training launcher with W&B logging.
#
# Default usage:
#   bash mini3_train.sh
#
# Common overrides:
#   ONLINE_PARALLEL_ENVS=512 NUM_ENV_STEPS=100000000 bash mini3_train.sh
#   WORK_DIR=results/mini3-run-001 WANDB_RUN_NAME=mini3-run-001 bash mini3_train.sh
#   SMOKE_TEST=1 bash mini3_train.sh
#
# W&B auth:
#   - If the server is not logged in, run once:
#       uv run wandb login
#     or export WANDB_API_KEY before running this script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"

export WANDB_ENTITY="${WANDB_ENTITY:-ricardo_wzk-soochow-university}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

WANDB_PROJECT="${WANDB_PROJECT:-bfmzero-mini3}"
WANDB_GROUP="${WANDB_GROUP:-mini3-isaacsim}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mini3-$(date +%Y%m%d-%H%M%S)}"

WORK_DIR="${WORK_DIR:-results/bfmzero-mini3-isaac}"
ONLINE_PARALLEL_ENVS="${ONLINE_PARALLEL_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-384000000}"
SMOKE_TEST="${SMOKE_TEST:-0}"

mkdir -p "${MPLCONFIGDIR}" "${WORK_DIR}"

train_cmd=(
  uv run --frozen python -m humanoidverse.train
  --robot-profile mini3
  --work-dir "${WORK_DIR}"
  --online-parallel-envs "${ONLINE_PARALLEL_ENVS}"
  --num-env-steps "${NUM_ENV_STEPS}"
  --use-wandb
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "${WANDB_RUN_NAME}"
)

if [[ "${SMOKE_TEST}" == "1" ]]; then
  train_cmd+=(--smoke-test)
fi

echo "Starting Mini3 training..."
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "WORK_DIR=${WORK_DIR}"
echo "ONLINE_PARALLEL_ENVS=${ONLINE_PARALLEL_ENVS}"
echo "NUM_ENV_STEPS=${NUM_ENV_STEPS}"

"${train_cmd[@]}"

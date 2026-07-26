#!/usr/bin/env bash
set -euo pipefail

# Mini3 training launcher with TensorBoard and local CSV/text logging.
#
# Default usage:
#   bash mini3_train.sh
#
# Common overrides:
#   ONLINE_PARALLEL_ENVS=512 NUM_ENV_STEPS=100000000 bash mini3_train.sh
#   WORK_DIR=results/mini3-run-001 bash mini3_train.sh
#   TENSORBOARD=0 bash mini3_train.sh
#   SMOKE_TEST=1 bash mini3_train.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"

WORK_DIR="${WORK_DIR:-results/bfmzero-mini3-isaac}"
ONLINE_PARALLEL_ENVS="${ONLINE_PARALLEL_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-384000000}"
TENSORBOARD="${TENSORBOARD:-1}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${WORK_DIR}/tensorboard}"
SMOKE_TEST="${SMOKE_TEST:-0}"

mkdir -p "${MPLCONFIGDIR}" "${WORK_DIR}"

train_cmd=(
  uv run --frozen python -m humanoidverse.train
  --robot-profile mini3
  --work-dir "${WORK_DIR}"
  --online-parallel-envs "${ONLINE_PARALLEL_ENVS}"
  --num-env-steps "${NUM_ENV_STEPS}"
)

if [[ "${TENSORBOARD}" == "1" ]]; then
  train_cmd+=(
    --use-tensorboard
    --tensorboard-log-dir "${TENSORBOARD_LOG_DIR}"
  )
fi

if [[ "${SMOKE_TEST}" == "1" ]]; then
  train_cmd+=(--smoke-test)
fi

echo "Starting Mini3 training..."
echo "WORK_DIR=${WORK_DIR}"
echo "ONLINE_PARALLEL_ENVS=${ONLINE_PARALLEL_ENVS}"
echo "NUM_ENV_STEPS=${NUM_ENV_STEPS}"
echo "TENSORBOARD=${TENSORBOARD}"
if [[ "${TENSORBOARD}" == "1" ]]; then
  echo "TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR}"
fi

"${train_cmd[@]}"

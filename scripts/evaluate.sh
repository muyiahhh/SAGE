#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a PSA checkpoint}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the processed dataset directory}"
DATASET="${DATASET:-Beauty}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation}"
GPUS="${GPUS:-0}"
NUM_BEAMS="${NUM_BEAMS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_HISTORY="${MAX_HISTORY:-20}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
EVAL_PROMPT_STYLE="${EVAL_PROMPT_STYLE:-0}"
SEED="${SEED:-42}"
FORCE_EVAL="${FORCE_EVAL:-0}"

case "${EVAL_SPLIT}" in
  valid|test) ;;
  *) echo "EVAL_SPLIT must be valid or test, got ${EVAL_SPLIT}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}"
RESULT_BASE="${OUTPUT_DIR}/${EVAL_SPLIT}_predictions"
RESULT_JSON="${RESULT_BASE}_merged.json"
METRICS_JSON="${OUTPUT_DIR}/${EVAL_SPLIT}_metrics.json"

if [[ -s "${METRICS_JSON}" && "${FORCE_EVAL}" != "1" ]]; then
  echo "[resume] Metrics already exist: ${METRICS_JSON}"
  exit 0
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
N_SHARDS="${#GPU_ARRAY[@]}"
COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --dataset "${DATASET}"
  --split "${EVAL_SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --num_beams "${NUM_BEAMS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_history "${MAX_HISTORY}"
  --length_penalty "${LENGTH_PENALTY}"
  --seed "${SEED}"
)
if [[ "${EVAL_PROMPT_STYLE}" == "1" ]]; then
  COMMON_ARGS+=(--eval_prompt_style)
fi

echo "Model=${MODEL_PATH} dataset=${DATASET} split=${EVAL_SPLIT} beams=${NUM_BEAMS}"
echo "Prompt=$([[ "${EVAL_PROMPT_STYLE}" == "1" ]] && echo eval || echo train) GPUs=${GPUS}"

if [[ "${N_SHARDS}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PYTHON}" -m sage.eval.evaluate \
    "${COMMON_ARGS[@]}" \
    --output_path "${RESULT_JSON}" \
    --local_rank 0
else
  PIDS=()
  SHARD_PATHS=()
  for ((SHARD = 0; SHARD < N_SHARDS; SHARD++)); do
    SHARD_PATH="${RESULT_BASE}_shard${SHARD}.json"
    SHARD_PATHS+=("${SHARD_PATH}")
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${SHARD}]}" "${PYTHON}" -m sage.eval.evaluate \
      "${COMMON_ARGS[@]}" \
      --output_path "${SHARD_PATH}" \
      --local_rank 0 \
      --shard "${SHARD}" \
      --n_shards "${N_SHARDS}" &
    PIDS+=("$!")
  done

  FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
      FAILED=1
    fi
  done
  if [[ "${FAILED}" == "1" ]]; then
    echo "At least one evaluation shard failed" >&2
    exit 1
  fi
  "${PYTHON}" -m sage.eval.calc \
    --path "${SHARD_PATHS[@]}" \
    --index_path "${DATA_DIR}/${DATASET}.index.json" \
    --merge "${RESULT_JSON}" \
    --merge_only
fi

"${PYTHON}" -m sage.eval.calc \
  --path "${RESULT_JSON}" \
  --index_path "${DATA_DIR}/${DATASET}.index.json" \
  --save_path "${METRICS_JSON}"

echo "Metrics saved to ${METRICS_JSON}"

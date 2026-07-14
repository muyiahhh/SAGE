#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

MODE="${1:-all}"
PYTHON="${PYTHON:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-Beauty}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/Beauty}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen3_0.6b_beauty}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN="${DRY_RUN:-0}"

STAGE1A_CONFIG="${STAGE1A_CONFIG:-${REPO_ROOT}/configs/beauty_0.6b_stage1a.yaml}"
STAGE1B_CONFIG="${STAGE1B_CONFIG:-${REPO_ROOT}/configs/beauty_0.6b_stage1b.yaml}"
STAGE1A_DIR="${OUTPUT_ROOT}/stage1a"
STAGE1A_MODEL="${STAGE1A_DIR}/aligned_base_model"
TARGET_CACHE="${OUTPUT_ROOT}/targets/stage1b_targets.pt"
STAGE1B_DIR="${OUTPUT_ROOT}/stage1b"
STAGE1B_MODEL="${STAGE1B_DIR}/epoch_3"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"
FIRST_GPU="${GPU_ARRAY[0]}"

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

is_checkpoint() {
  local path="$1"
  [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] && \
    { [[ -f "${path}/model.safetensors" ]] || \
      [[ -f "${path}/model.safetensors.index.json" ]] || \
      [[ -f "${path}/pytorch_model.bin" ]] || \
      [[ -f "${path}/pytorch_model.bin.index.json" ]]; }
}

check_data() {
  run "${PYTHON}" scripts/check_data.py --data_dir "${DATA_DIR}" --dataset "${DATASET}"
}

run_stage1a() {
  if is_checkpoint "${STAGE1A_MODEL}" && [[ -f "${STAGE1A_MODEL}/jepa_heads.pt" ]]; then
    echo "[resume] Stage 1a checkpoint: ${STAGE1A_MODEL}"
    return
  fi
  run env CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m urr.train.stage1a_jepa \
    --config "${STAGE1A_CONFIG}" \
    --model_name "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${STAGE1A_DIR}"
}

build_targets() {
  if [[ -s "${TARGET_CACHE}" ]]; then
    echo "[resume] Stage 1b targets: ${TARGET_CACHE}"
    return
  fi
  if [[ "${DRY_RUN}" != "1" ]] && ! is_checkpoint "${STAGE1A_MODEL}"; then
    echo "Missing Stage 1a checkpoint: ${STAGE1A_MODEL}" >&2
    exit 1
  fi
  run env CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON}" \
    -m urr.train.build_stage1b_targets \
    --stage1a_model "${STAGE1A_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --dataset "${DATASET}" \
    --output "${TARGET_CACHE}" \
    --device cuda:0
}

run_stage1b() {
  if is_checkpoint "${STAGE1B_MODEL}"; then
    echo "[resume] Stage 1b checkpoint: ${STAGE1B_MODEL}"
    return
  fi
  if [[ "${DRY_RUN}" != "1" && ! -s "${TARGET_CACHE}" ]]; then
    echo "Missing Stage 1b target cache: ${TARGET_CACHE}" >&2
    exit 1
  fi
  run env CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="$((MASTER_PORT + 1))" \
    -m urr.train.stage1b \
    --config "${STAGE1B_CONFIG}" \
    --aligned_base_model "${STAGE1A_MODEL}" \
    --target_cache "${TARGET_CACHE}" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${STAGE1B_DIR}"
}

run_eval() {
  local split="$1"
  if [[ "${DRY_RUN}" != "1" ]] && ! is_checkpoint "${STAGE1B_MODEL}"; then
    echo "Missing Stage 1b checkpoint: ${STAGE1B_MODEL}" >&2
    exit 1
  fi
  run env \
    PYTHON="${PYTHON}" \
    MODEL_PATH="${STAGE1B_MODEL}" \
    DATA_DIR="${DATA_DIR}" \
    DATASET="${DATASET}" \
    EVAL_SPLIT="${split}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/evaluation" \
    GPUS="${GPUS}" \
    EVAL_PROMPT_STYLE=0 \
    bash scripts/evaluate.sh
}

case "${MODE}" in
  all)
    check_data
    run_stage1a
    build_targets
    run_stage1b
    run_eval valid
    run_eval test
    ;;
  check) check_data ;;
  stage1a) check_data; run_stage1a ;;
  targets) build_targets ;;
  stage1b) run_stage1b ;;
  valid) run_eval valid ;;
  test) run_eval test ;;
  *)
    echo "Usage: $0 [all|check|stage1a|targets|stage1b|valid|test]" >&2
    exit 2
    ;;
esac

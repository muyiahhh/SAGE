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

ISG_CONFIG="${ISG_CONFIG:-${REPO_ROOT}/configs/beauty_qwen3_0.6b_isg.yaml}"
PSA_CONFIG="${PSA_CONFIG:-${REPO_ROOT}/configs/beauty_qwen3_0.6b_psa.yaml}"
ISG_DIR="${OUTPUT_ROOT}/isg"
ISG_MODEL="${ISG_DIR}/grounded_model"
ANCHOR_CACHE="${OUTPUT_ROOT}/anchors/item_anchors.pt"
PSA_DIR="${OUTPUT_ROOT}/psa"
PSA_MODEL="${PSA_DIR}/epoch_3"

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

run_isg() {
  if is_checkpoint "${ISG_MODEL}" && [[ -f "${ISG_MODEL}/isg_heads.pt" ]]; then
    echo "[resume] ISG checkpoint: ${ISG_MODEL}"
    return
  fi
  run env CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m sage.train.isg \
    --config "${ISG_CONFIG}" \
    --model_name "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${ISG_DIR}"
}

build_anchors() {
  if [[ -s "${ANCHOR_CACHE}" ]]; then
    echo "[resume] Item-anchor bank: ${ANCHOR_CACHE}"
    return
  fi
  if [[ "${DRY_RUN}" != "1" ]] && ! is_checkpoint "${ISG_MODEL}"; then
    echo "Missing ISG checkpoint: ${ISG_MODEL}" >&2
    exit 1
  fi
  run env CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON}" \
    -m sage.train.build_anchor_bank \
    --isg_model "${ISG_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --dataset "${DATASET}" \
    --output "${ANCHOR_CACHE}" \
    --device cuda:0
}

run_psa() {
  if is_checkpoint "${PSA_MODEL}"; then
    echo "[resume] PSA checkpoint: ${PSA_MODEL}"
    return
  fi
  if [[ "${DRY_RUN}" != "1" && ! -s "${ANCHOR_CACHE}" ]]; then
    echo "Missing item-anchor bank: ${ANCHOR_CACHE}" >&2
    exit 1
  fi
  run env CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="$((MASTER_PORT + 1))" \
    -m sage.train.psa \
    --config "${PSA_CONFIG}" \
    --grounded_model "${ISG_MODEL}" \
    --anchor_cache "${ANCHOR_CACHE}" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${PSA_DIR}"
}

run_eval() {
  local split="$1"
  if [[ "${DRY_RUN}" != "1" ]] && ! is_checkpoint "${PSA_MODEL}"; then
    echo "Missing PSA checkpoint: ${PSA_MODEL}" >&2
    exit 1
  fi
  run env \
    PYTHON="${PYTHON}" \
    MODEL_PATH="${PSA_MODEL}" \
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
    run_isg
    build_anchors
    run_psa
    run_eval valid
    run_eval test
    ;;
  check) check_data ;;
  isg) check_data; run_isg ;;
  anchors) build_anchors ;;
  psa) run_psa ;;
  valid) run_eval valid ;;
  test) run_eval test ;;
  *)
    echo "Usage: $0 [all|check|isg|anchors|psa|valid|test]" >&2
    exit 2
    ;;
esac

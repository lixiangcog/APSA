#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_ROOT="${BASE_ROOT:?Set BASE_ROOT to the VPTTA OPTIC directory}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the fundus dataset root}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the source checkpoints}"
PYTHON="${PYTHON:-python}"
RUNNER="${REPO_ROOT}/apsa/fundus/apsa_optic.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/fundus}"
GPU="${GPU:-0}"
SOURCES="${SOURCES:-RIM_ONE_r3 REFUGE ORIGA REFUGE_Valid Drishti_GS}"

for path in "${BASE_ROOT}" "${DATASET_ROOT}" "${MODEL_ROOT}" "${RUNNER}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

COMMON=(
  --dataset_root "${DATASET_ROOT}"
  --model_root "${MODEL_ROOT}"
  --device cuda:0
  --eval_split train_test
  --image_size 512
  --batch_size 1
  --iters 1
  --rounds 3
  --num_workers 0
  --enable_gate
  --use_prompt_ema
  --postprocess_cc
  --w_prior 0.03
  --w_cup_prior 0.01
  --w_inside 0.25
  --w_clip 0.0
  --prior_sigma 0.22
  --prior_use_roi_weight 1
  --roi_w_min 0.2
  --roi_w_gamma 1.0
  --geom_sigma_scale 0.55
  --pp_disc_thr 0.45
  --pp_disc_target_ratio 0.08
  --pp_cup_thr 0.0
  --pp_cup_target_ratio 0.18
  --pp_disc_cc_mode largest
  --pp_cup_cc_mode largest
  --cup_prior_sigma_scale 0.55
)

run_source() {
  local source="$1"
  local gate_margin="$2"
  local gate_min_improve="$3"
  local outdir="${OUTPUT_ROOT}/${source}"
  [[ ! -e "${outdir}" ]] || { echo "Refusing to overwrite ${outdir}" >&2; return 2; }
  mkdir -p "${outdir}"
  echo "[run] ${source} on visible GPU ${GPU}"
  (
    cd "${BASE_ROOT}"
    env CUDA_VISIBLE_DEVICES="${GPU}" \
      PYTHONPATH="${BASE_ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${RUNNER}" \
      --Source_Dataset "${source}" \
      --path_save_log "${outdir}" \
      --gate_margin "${gate_margin}" \
      --gate_min_improve "${gate_min_improve}" \
      "${COMMON[@]}"
  )
}

mkdir -p "${OUTPUT_ROOT}"
for source in ${SOURCES}; do
  case "${source}" in
    RIM_ONE_r3|ORIGA|Drishti_GS) run_source "${source}" 0.00 0.01 ;;
    REFUGE|REFUGE_Valid) run_source "${source}" 0.02 0.00 ;;
    *) echo "Unknown source domain: ${source}" >&2; exit 2 ;;
  esac
done

echo "[done] ${OUTPUT_ROOT}"

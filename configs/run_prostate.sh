#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_ROOT="${BASE_ROOT:?Set BASE_ROOT to a PASS checkout}"
PYTHON="${PYTHON:-python}"
RUNNER="${REPO_ROOT}/apsa/prostate/apsa_prostate.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/prostate}"
CLIP_PRIOR_ROOT="${CLIP_PRIOR_ROOT:-${BASE_ROOT}/data/MRI_prostate/clip_priors}"
GPU="${GPU:-0}"
DOMAINS="${DOMAINS:-BIDMC BMC HK I2CVB UCL}"
EXP_NAME="${EXP_NAME:-apsa_frozen_reference}"
PYTHON_PREFIX="$("${PYTHON}" -c 'import sys; print(sys.prefix)')"
RUNTIME_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for path in "${BASE_ROOT}" "${RUNNER}" "${CLIP_PRIOR_ROOT}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

BASE_ARGS=(
  --model Tent_EMA
  --arch unet_2d_sptta
  --override_batch_size 16
  --override_optim_steps 3
  --reset_mode from_target
  --use_vida --vida_rank 64 --vida_hka_hidden 128
  --train_prompt --train_data_adaptor --train_vida
  --lr_prompt 1e-3 --lr_data 1e-3 --lr_vida 1e-4 --lr_bn 1e-2
  --clip_guided
  --clip_prior_root "${CLIP_PRIOR_ROOT}"
  --clip_prior_kind ellip --clip_ellip_sym
  --clip_loc_lambda 0.02
  --clip_bce_lambda 0.01
  --clip_bce_gamma 1.2
  --clip_entropy_weighted --clip_weight_min 0.05
  --case_rollback
  --case_rollback_metric clip_align
  --case_rollback_margin 0.00
  --case_guard_area_min 0.002 --case_guard_area_max 0.10
  --safe_branch_selector
  --safe_branch_selector_adaptive
  --safe_branch_selector_final_min_gain 0.02
  --manualseed 100
  --save_pre True
  --override_model_episodic 1
  --no-clip_auto_blend --no-clip_auto_weight
  --clip_slice_sigma 3.0 --clip_slice_radius 7
  --post_lcc
)

mkdir -p "${OUTPUT_ROOT}"
for domain in ${DOMAINS}; do
  outdir="${OUTPUT_ROOT}/Tent_EMA_Prostate_RUNMC2${domain}_${EXP_NAME}_sptta"
  log="${OUTPUT_ROOT}/${domain}.log"
  if find "${outdir}/metrics" -name '*_results.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] ${domain}"
    continue
  fi
  echo "[run] ${domain} on visible GPU ${GPU}"
  (
    cd "${BASE_ROOT}"
    env CUDA_VISIBLE_DEVICES="${GPU}" \
      LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" \
      PYTHONPATH="${BASE_ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${RUNNER}" \
      --gpu 0 \
      --domain "${domain}" \
      --log_folder "${OUTPUT_ROOT}" \
      --exp_name "${EXP_NAME}" \
      "${BASE_ARGS[@]}"
  ) >"${log}" 2>&1
done

echo "[done] ${OUTPUT_ROOT}"

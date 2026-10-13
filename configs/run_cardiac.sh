#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_ROOT="${BASE_ROOT:?Set BASE_ROOT to an A3-TTA checkout}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the Source-A model}"
PYTHON="${PYTHON:-python}"
RUNNER="${REPO_ROOT}/apsa/cardiac/apsa_mms.py"
CFG="${SCRIPT_DIR}/cardiac_base.yaml"
CALIBRATOR="${SCRIPT_DIR}/cardiac_source_calibrator.json"
POLICY="${SCRIPT_DIR}/cardiac_rpr_policy.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/cardiac}"
SURFACE_ASSD_PROTOCOL="${SURFACE_ASSD_PROTOCOL:-sictta_capped}"
SELECTED_INTERVAL=3

for path in "${BASE_ROOT}" "${CHECKPOINT}" "${RUNNER}" "${CFG}" "${CALIBRATOR}" "${POLICY}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

"${PYTHON}" - "${POLICY}" "${SELECTED_INTERVAL}" <<'PY'
import json
import sys
from pathlib import Path

policy = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
if policy.get("target_labels_used") is not False:
    raise SystemExit("RPR policy selection must be target-label free")
if int(policy.get("selected_interval", -1)) != expected:
    raise SystemExit("Frozen Source-A checkpoint interval mismatch")
PY

COMMON=(
  --cfg "${CFG}"
  --adapter tent
  --use-apssa 1
  --batch_size 10
  --num_workers 8
  --seed 100
  --tent_lr 0.000050
  --tent_train_scope bn
  --disable_adaptation_dropout
  --surface_assd_protocol "${SURFACE_ASSD_PROTOCOL}"
  --source_fusion 0.3
  --source_fusion_classes 0.20,0.25,0.15
  --adaptive_class_fusion
  --adaptive_fusion_strength 0.05
  --adaptive_fusion_max 0.55
  --candidate_bank_v2
  --candidate_bank_fusion_scales 0.5,0.7,0.85,1.0,1.2
  --candidate_bank_no_refined
  --candidate_bank_tta_flips h,v,hv
  --candidate_bank_min_gain 0.003
  --candidate_bank_name_bias clean_final:0.04,clean_adapted:-0.004,final:0.004,fusion_x0.50:-0.004
  --candidate_cleanup_v2
  --candidate_cleanup_sources final,adapted
  --cleanup_min_area_abs 32
  --cleanup_min_area_ratio 0.001
  --cleanup_logit_bias 5.0
  --cleanup_keep_largest_classes 1
  --cleanup_fill_hole_classes 1
  --final_bias 0.05
  --final_min_gain 0.0
  --topology_prior_strength 0.0
  --mre_use_entropy false
  --mre_use_topology false
  --final_takeover_v2
  --final_takeover_score_gain 0.0
  --final_takeover_geometry_gain -0.03
  --final_takeover_align_gain -0.002
  --no_source_output
  --prior_refine_steps 5
  --prior_refine_lr 0.00020
  --apl_online_weight 1.00
  --prior_entropy_lambda 0.45
  --prior_loc_lambda 0.075
  --prior_bce_lambda 0.028
  --prior_ts_lambda 0.035
  --prior_source_consistency_lambda 0.02
  --preupdate_rpr
  --rpr_calibrator "${CALIBRATOR}"
  --risk_aware_reversion
)

run_variant() {
  local gpu="$1"
  local order="$2"
  local policy_name="$3"
  local interval
  local domains=()
  local index

  [[ "${order}" =~ ^[BCD]{3}$ ]] || { echo "ORDER must contain three B/C/D letters" >&2; return 2; }
  for ((index=0; index<${#order}; index++)); do
    domains+=("${order:index:1}")
  done

  case "${policy_name}" in
    current) interval=1 ;;
    plus) interval="${SELECTED_INTERVAL}" ;;
    *) echo "POLICY must be current or plus" >&2; return 2 ;;
  esac

  local save_dir="${OUTPUT_ROOT}/${order}_${policy_name}"
  [[ ! -e "${save_dir}" ]] || { echo "Refusing to overwrite ${save_dir}" >&2; return 2; }

  (
    cd "${BASE_ROOT}"
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${BASE_ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${RUNNER}" "${COMMON[@]}" \
      --domains "${domains[@]}" \
      --rpr_checkpoint_interval "${interval}" \
      --save_dir "${save_dir}" \
      MODEL.CKPT_DIR "${CHECKPOINT}"
  )
}

if [[ "${1:-}" == "--run" ]]; then
  run_variant "${2:?GPU is required}" "${3:?ORDER is required}" "${4:?POLICY is required}"
else
  echo "Usage: BASE_ROOT=... CHECKPOINT=... bash $0 --run GPU ORDER POLICY" >&2
  exit 2
fi
